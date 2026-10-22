"""First-party `mindie-knowledge` MCP server and `G3` closed-loop capture/indexing adapter.

[Role]
Runs inside the selected `mindie-knowledge` component interpreter to provide:
1. `knowledge-mcp`: stdio FastMCP server exposing `knowledge_query` and `knowledge_explain`
   backed by `MaterialStore` (SQLite + ReMe retrieval).
2. `capture`: `G3` post-trial native Stop-event ingestion, `gitleaks` redaction, small-model
   summarization (`finish_indexes`), corpus linting (`audit_packages`), local bare-Git publication,
   and independent consumer readback verification.

[Invariants]
- `MaterialStore` owns SQLite state and ReMe's synchronous `asyncio.Runner`; all store operations
  execute on a dedicated single-worker `ThreadPoolExecutor(max_workers=1)` outside FastMCP's loop.
- Any `summary_status == "outcome_unknown"` or `usage["unknown_calls"]` raises
  `UnknownExternalEffects` immediately without retrying paid indexer calls.
- Publication is strictly confined to `private_root / "local_knowledge.git"` (`local-only`).

[Timeouts & Locks]
- Single-thread executor serializes all `MaterialStore` SQLite/ReMe access in `serve()`.
- `finish_indexes` bounds native summarization to `max_batches=256` deterministic iterations.

[Side-Effect Policy]
- Writes retrieval telemetry to `state / "retrieval-observations.jsonl"` and full-loop capture
  receipts to `profile / "full-loop-receipt.json"`, installing verified Markdown packages into
  `private_root / "corpus/tasks"`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mindie_bench.evaluation.timing import StageRecorder
from mindie_bench.models import UnknownExternalEffects


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def packages(corpus):
    from mindie_knowledge.materials.store import validate_package_files

    for path in sorted((corpus / "tasks").iterdir()):
        files = {p.relative_to(path).as_posix(): p.read_text() for p in path.rglob("*.md")}
        yield validate_package_files(files, "vllm-ascend")


def consumer(corpus, state, revision):
    from mindie_bench.runner.graded_view import is_graded

    if is_graded(corpus):
        from mindie_bench.integrations.graded_consumer import GradedConsumer

        return GradedConsumer(corpus, state)
    from mindie_knowledge.materials.store import MaterialStore

    store = MaterialStore(state, "vllm-ascend")
    try:
        pointers = store._pointers() if hasattr(store, "_pointers") else {}
        if pointers.get("source_revision") != revision or not pointers.get("feed"):
            store.install_packages(packages(corpus), source_revision=revision)
        if hasattr(store, "refresh_index"):
            store.refresh_index()
    except BaseException as exc:
        try:
            store.close()
        except Exception as cleanup:
            exc.add_note(f"consumer cleanup also failed: {type(cleanup).__name__}: {cleanup}")
        raise
    return store


def explain(store, ref):
    from mindie_bench.integrations.graded_consumer import GradedConsumer

    if isinstance(store, GradedConsumer):
        return store.explain(ref)
    from mindie_knowledge.materials.references import parse_read_ref

    selected = parse_read_ref(ref, domain="vllm-ascend")
    current = store.current_revisions(selected["task_id"])
    source = "feed"
    revision = current[source]
    return store.read_current(
        selected["task_id"],
        source=source,
        revision=revision,
        block_id=selected.get("block_id"),
        sha256=selected.get("sha256"),
    )


def serve(args):
    from mcp.server.fastmcp import FastMCP

    log = Path(args.state) / "retrieval-observations.jsonl"

    def record(kind, **values):
        with log.open("a") as stream:
            stream.write(json.dumps(dict(kind=kind, **values), ensure_ascii=False) + "\n")

    # MaterialStore owns SQLite state and ReMe's synchronous asyncio.Runner.
    # Its whole lifecycle must stay on one thread outside FastMCP's event loop.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindie-knowledge") as worker:
        with contextlib.redirect_stdout(sys.stderr):
            store = worker.submit(
                consumer, Path(args.corpus), Path(args.state), "isolated-benchmark-view"
            ).result()
        server = FastMCP("mindie-bench-first-party-knowledge")

        async def invoke(kind, operation, **request):
            started = time.perf_counter()
            try:
                result = await asyncio.wrap_future(worker.submit(operation))
            except Exception as exc:
                try:
                    record(
                        kind,
                        **request,
                        elapsed_sec=time.perf_counter() - started,
                        error=dict(type=type(exc).__name__, message=str(exc)),
                    )
                except Exception as logging_error:
                    exc.add_note(
                        "retrieval error logging also failed: "
                        f"{type(logging_error).__name__}: {logging_error}"
                    )
                raise
            record(kind, **request, elapsed_sec=time.perf_counter() - started, result=result)
            return result

        @server.tool()
        async def knowledge_query(
            query: str | None = None,
            limit: int = 5,
            continuation: str | None = None,
            conditions: dict | None = None,
            include_unknown: bool = False,
            domain_path: str | None = None,
        ) -> dict | list[dict]:
            """Search reference experience; retain failures and check current applicability."""
            request = dict(query=query, limit=limit, continuation=continuation)
            if conditions is not None or include_unknown:
                request.update(conditions=conditions, include_unknown=include_unknown)
            if domain_path is not None:
                request["domain_path"] = domain_path
            return await invoke(
                "query",
                lambda: store.search(**request),
                **request,
            )

        @server.tool()
        async def knowledge_explain(ref: str) -> dict:
            """Read the exact current block returned by knowledge_query."""
            return await invoke("explain", lambda: explain(store, ref), ref=ref)

        try:
            server.run(transport="stdio")
        except BaseException as exc:
            try:
                worker.submit(store.close).result()
            except Exception as cleanup:
                exc.add_note(f"consumer cleanup also failed: {type(cleanup).__name__}: {cleanup}")
            raise
        else:
            try:
                worker.submit(store.close).result()
            except Exception as cleanup:
                # The MCP server already completed its primary protocol work.
                # Surface the cleanup failure without replacing that outcome
                # with a transport error.
                print(
                    "knowledge MCP store cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}",
                    file=sys.stderr,
                )


def git(*arguments, cwd=None):
    executable = os.environ["MINDIE_BENCH_GIT"]
    try:
        return subprocess.run(
            [executable, "-c", "protocol.file.allow=always", *map(str, arguments)],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        error.add_note("Git stderr (at most 16 KiB): " + (error.stderr or "")[:16_384])
        raise


def preflight_git(destination: Path) -> dict:
    """Exercise the capture's actual local publication route before model admission."""
    destination = destination.resolve()
    destination.mkdir(exist_ok=False)
    remote, producer, cloned = [destination / name for name in ["remote.git", "producer", "consumer"]]
    git("init", "--template=", "--bare", "-q", remote)
    git("init", "--template=", "-q", "-b", "main", producer)
    body = b"Owned zero-model local Git publication preflight.\n"
    (producer / "material.md").write_bytes(body)
    git("add", "material.md", cwd=producer)
    git(
        "-c",
        "user.name=MindIE Benchmark",
        "-c",
        "user.email=benchmark@example.invalid",
        "commit",
        "-qm",
        "Local Git preflight",
        cwd=producer,
    )
    revision = git("rev-parse", "HEAD", cwd=producer)
    git("push", str(remote), "HEAD:refs/heads/main", cwd=producer)
    git("clone", "--template=", "-q", "--branch", "main", remote, cloned)
    if git("rev-parse", "HEAD", cwd=cloned) != revision or (cloned / "material.md").read_bytes() != body:
        raise RuntimeError("local Git preflight consumer differs from the producer")
    return dict(
        status="completed",
        executable=os.environ["MINDIE_BENCH_GIT"],
        exec_path=os.environ["GIT_EXEC_PATH"],
        revision=revision,
        body_sha256=hashlib.sha256(body).hexdigest(),
        model_calls=0,
        publication="owned local bare Git only",
    )


def finish_indexes(engine, summarize, *, max_batches=256):
    """Drain native due batches; failed or unknown attempts never get a retry."""
    for _ in range(max_batches):
        tasks = [
            dict(row)
            for row in engine.store.db.execute(
                "SELECT entry_id,summary_status,summary_detail FROM transcript_tasks"
            )
        ]
        if tasks and all(row["summary_status"] == "complete" for row in tasks):
            return tasks
        if any(row["summary_status"] == "outcome_unknown" for row in tasks):
            raise UnknownExternalEffects("native indexer outcome is unknown: " + json.dumps(tasks))
        if not tasks or any(row["summary_status"] not in {"complete", "pending"} for row in tasks):
            raise RuntimeError("native indexer did not complete: " + json.dumps(tasks))
        if not engine.index_queue.has_due():
            raise RuntimeError("native indexer has deferred work: " + json.dumps(tasks))
        summarize(engine)
    raise RuntimeError("native indexer exceeded its deterministic batch bound")


def close_capture_resources(engine, store, timing, *, primary=None):
    failures = []
    for stage, cleanup in [
        ("capture operation shutdown", engine.operations.shutdown),
        ("capture store close", store.close),
    ]:
        try:
            with timing.measure(stage, record_start=False):
                cleanup()
        except BaseException as error:
            failures.append((stage, error))
    if not failures:
        return []
    if primary is None:
        # The capture result is already complete: return the secondary
        # failures so the caller can attach them without discarding it.
        return failures
    for stage, error in failures:
        primary.add_note(f"{stage} also failed: {type(error).__name__}: {error}")
    return []


def native_capture_inputs(args, native, root):
    """Select the installed host adapter and its exact native Stop artifact."""
    harness = native.get("harness")
    if harness == "codex":
        if not args.codex_scripts or not args.codex_bin:
            raise ValueError("Codex capture requires explicit native scripts and executable")
        scripts = Path(args.codex_scripts).resolve(strict=True)
        sys.path.insert(0, str(scripts))
        import bridge
        import codex_transcript

        event = bridge.hook_event(Path(native["stop_event_path"]).read_bytes())
        adapter = codex_transcript
        worker = "agent_worker.py"
        os.environ["MINDIE_CODEX_BIN"] = args.codex_bin
        catalog = json.loads((root / "model-catalog.json").read_text())
        selected = [model for model in catalog["models"] if model.get("slug") == args.summary_model]
        if len(selected) != 1:
            raise ValueError("capture model requires exactly one explicit native catalog record")
    elif harness == "claude-code":
        from mindie_bench.integrations.claude_stop import read_stop

        if not args.claude_scripts or not args.claude_bin:
            raise ValueError("Claude capture requires explicit native scripts and executable")
        profile = Path(os.environ["CLAUDE_CONFIG_DIR"]).resolve(strict=True)
        if profile != root / "claude":
            raise ValueError("Claude capture requires this trial's owned native profile")
        scripts = Path(args.claude_scripts).resolve(strict=True)
        config = root / "benchmark-cc-adapter.json"
        write_json(config, {"claude_config_dir": str(profile)})
        os.environ["MINDIE_CC_CONFIG"] = str(config)
        os.environ["MINDIE_CC_BIN"] = args.claude_bin
        os.environ["ANTHROPIC_MODEL"] = args.summary_model
        sys.path.insert(0, str(scripts))
        import transcript

        actual = read_stop(native["stop_event_path"])
        event = {**actual, "turn_id": actual["prompt_id"]}
        adapter = transcript
        worker = "organizer.py"
    else:
        raise ValueError("capture requires an actual Codex or Claude Code transcript and Stop event")
    if (
        event["session_id"] != native["session_id"]
        or Path(event["transcript_path"]).resolve() != Path(native["transcript_path"]).resolve()
        or Path(event["cwd"]).resolve() != Path(native["cwd"]).resolve()
    ):
        raise ValueError("native Stop event and completed foreground task differ")
    source = adapter.capture_source(event["transcript_path"], event["session_id"])
    if Path(source["project_root"]).resolve() != Path(native["cwd"]).resolve():
        raise ValueError("native transcript project scope differs from the completed foreground")
    if not (scripts / worker).is_file():
        raise ValueError("selected native summary worker is unavailable")
    return event, source, adapter, scripts, worker


def native_summary_command(component_package, scripts, worker):
    """Load the exact shared package without granting its sibling source trees."""
    package = str(Path(component_package).resolve(strict=True))
    loader = (
        "import sys,runpy,importlib.util;"
        f"spec=importlib.util.spec_from_file_location('mindie_knowledge',"
        f"{str(Path(package) / '__init__.py')!r},submodule_search_locations={[package]!r});"
        "module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module);"
        f"sys.path.insert(0,{str(scripts)!r});"
        f"runpy.run_path({str(scripts / worker)!r},run_name='__main__')"
    )
    return [sys.executable, "-I", "-c", loader]


def capture(args):
    import mindie_knowledge
    from mindie_knowledge import consent_store
    from mindie_knowledge.loop.activation import Admission
    from mindie_knowledge.loop.engine import Engine
    from mindie_knowledge.loop.store import Store
    from mindie_knowledge.loop.transcript_capture import summarize_due
    from mindie_knowledge.materials.summarizer import SummaryLedger

    payload = json.load(sys.stdin)
    native = payload["native"]
    root = Path.cwd().resolve(strict=True)
    timing = StageRecorder(root / "capture-stages.json")
    corpus = Path(payload["destination"]).resolve(strict=True)
    local_remote = Path(payload["local_remote"]).resolve(strict=True)
    private_root = Path(os.environ["MINDIE_BENCH_KNOWLEDGE_PRIVATE_ROOT"]).resolve(strict=True)
    if private_root != root.parent / "knowledge-private" or root.name != "profile":
        raise ValueError("capture knowledge root must be this trial's owned private sibling")
    if corpus != private_root / "corpus" or local_remote != private_root / "local_knowledge.git":
        raise ValueError("capture or publication escapes the owned knowledge private root")
    if git("--git-dir", local_remote, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("publication target must be this profile's bare Git repository")
    event, source, adapter, scripts, worker = native_capture_inputs(args, native, root)
    consent = root / "benchmark-consent.json"
    consent_store.record_choice(consent, "contribute")
    settings = root / "benchmark-community.json"
    write_json(
        settings,
        dict(
            schema="mindie-community-config/1",
            enabled=True,
            generation="benchmark-current-native-task",
            enabled_at=source["created_at"],
            repository="benchmark/local-only",
            branch="main",
            project_roots=[source["project_root"]],
            idle_seconds=30,
            consent_config=str(consent),
        ),
    )
    admission = Admission(root / "benchmark-admission.sqlite3")
    admission.associate(
        event["session_id"],
        project_root=source["project_root"],
        not_before=source["created_at"],
    )
    store = Store(root / "capture-state", "vllm-ascend")
    engine = Engine(
        store,
        settings_path=settings,
        admission=admission,
        transcript_adapter=adapter,
        redactor_executable=args.gitleaks,
        summary_command=native_summary_command(Path(mindie_knowledge.__file__).parent, scripts, worker),
    )
    capture_failure = None
    result = None
    consumer_cleanup_errors = []
    try:
        with timing.measure("native_capture_admission"):
            receipt = engine.capture(
                session_id=event["session_id"],
                turn_id=event["turn_id"],
                transcript_path=event["transcript_path"],
                cwd=event["cwd"],
                harness="claude" if native["harness"] == "claude-code" else "codex",
            )
        if "id" not in receipt:
            raise RuntimeError("native capture was not admitted: " + json.dumps(receipt))
        with timing.measure("body_capture_and_redaction"):
            for _ in range(256):
                engine._process(receipt["id"])
                row = store.capture_row(receipt["id"])
                if row["status"] in {"organized", "no-new-material"}:
                    break
                if row["status"] not in {"pending", "queued", "deferred"} or engine.errors:
                    raise RuntimeError(
                        "capture failed: " + json.dumps(dict(row=row, errors=engine.errors))
                    )
            else:
                raise RuntimeError("native capture exceeded the deterministic page bound")
        body_receipt = store.capture_material_receipt(receipt["id"])
        if not body_receipt:
            raise RuntimeError("native capture produced no body receipt")
        with timing.measure("native_index_due_wait"):
            time.sleep(3.1)
        with timing.measure("indexing"):
            tasks = finish_indexes(engine, summarize_due)
        usage = SummaryLedger(store.db).usage_totals()
        if usage["unknown_calls"]:
            raise UnknownExternalEffects(
                "native indexer has unknown usage; its call must not be replayed"
            )
        staging = root / "publication"
        staging.mkdir()
        exported = []
        with timing.measure("package_export"):
            for task in tasks:
                package = store.materials.export_task(task["entry_id"], source="draft")
                exported.append(package)
                for name, text in package["files"].items():
                    path = staging / "tasks" / package["task_id"] / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
        from mindie_bench.evaluation.corpus_linter import CorpusLintRejected, audit_packages

        with timing.measure("corpus_lint"):
            corpus_lint = audit_packages(exported)
            if corpus_lint["status"] != "passed":
                raise CorpusLintRejected(corpus_lint)
        with timing.measure("local_git_publish"):
            git("init", "--template=", "-q", "-b", "main", staging)
            git("add", ".", cwd=staging)
            git(
                "-c",
                "user.name=MindIE Benchmark",
                "-c",
                "user.email=benchmark@example.invalid",
                "commit",
                "-qm",
                "Publish current redacted benchmark material locally",
                cwd=staging,
            )
            revision = git("rev-parse", "HEAD", cwd=staging)
            git("push", str(local_remote), "HEAD:refs/heads/main", cwd=staging)
        cloned = root / "independent-consumer"
        with timing.measure("consumer_git_readback"):
            git("clone", "--template=", "-q", "--branch", "main", local_remote, cloned)
            if git("rev-parse", "HEAD", cwd=cloned) != revision:
                raise RuntimeError("independent consumer did not obtain the published revision")
            fetched = list(packages(cloned))
            if {p["package_hash"] for p in fetched} != {p["package_hash"] for p in exported}:
                raise RuntimeError("published material changed between producer and consumer")
        reader = None
        consumer_failure = None
        try:
            with timing.measure("consumer_index"):
                reader = consumer(cloned, root / "consumer-index", revision)
            with timing.measure("consumer_query"):
                hits = reader.search(args.query)
            if not hits:
                raise RuntimeError("the independent consumer did not retrieve captured material")
            # search returns a page with ranked results in the installed component contract.
            rows = hits["results"] if isinstance(hits, dict) else hits
            if not rows:
                raise RuntimeError("the independent consumer returned an empty retrieval page")
            with timing.measure("consumer_explain"):
                reads = [dict(ref=hit["ref"], value=explain(reader, hit["ref"])) for hit in rows]
        except BaseException as error:
            consumer_failure = error
            raise
        finally:
            if reader is not None:
                try:
                    with timing.measure("consumer_close", record_start=False):
                        reader.close()
                except BaseException as error:
                    if consumer_failure is None:
                        consumer_cleanup_errors.append(error)
                    else:
                        consumer_failure.add_note(
                            f"consumer close also failed: {type(error).__name__}: {error}"
                        )
        with timing.measure("warm_corpus_install"):
            for package in fetched:
                target = corpus / "tasks" / package["task_id"]
                target.mkdir(exist_ok=False)
                for name, text in package["files"].items():
                    path = target / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
        total = usage["input_tokens"] + usage["cached_input_tokens"] + usage["output_tokens"]
        result = dict(
            status="completed",
            tokens=dict(total=total, complete=True),
            native_stop_delivered=True,
            native_harness=native["harness"],
            native_session_id=event["session_id"],
            native_turn_id=event["turn_id"],
            body_capture=body_receipt,
            indexer_usage=usage,
            publication=dict(kind="local-git", revision=revision),
            packages=[dict(task_id=p["task_id"], package_hash=p["package_hash"]) for p in fetched],
            corpus_lint=corpus_lint,
            consumer=dict(revision=revision, hits=hits, reads=reads),
            stage_observations=timing.stages.copy(),
            stage_scope="adapter stages; native foreground Stop-hook latency is measured separately",
        )
        if consumer_cleanup_errors:
            result["cleanup_errors"] = [
                dict(type=type(error).__name__, message=str(error))
                for error in consumer_cleanup_errors
            ]
            result["cleanup_warning"] = "consumer close failed after completed readback"
    except BaseException as error:
        capture_failure = error
        raise
    finally:
        failures = close_capture_resources(engine, store, timing, primary=capture_failure)
        if result is not None and failures:
            result.setdefault("cleanup_errors", []).extend(
                dict(stage=stage, type=type(error).__name__, message=str(error))
                for stage, error in failures
            )
            result["cleanup_warning"] = "capture resource cleanup failed after completed capture"
        if result is not None:
            write_json(root / "full-loop-receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mcp = commands.add_parser("knowledge-mcp")
    mcp.add_argument("--corpus", required=True)
    mcp.add_argument("--state", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--codex-scripts")
    cap.add_argument("--codex-bin")
    cap.add_argument("--claude-scripts")
    cap.add_argument("--claude-bin")
    cap.add_argument("--summary-model", required=True)
    cap.add_argument("--gitleaks", required=True)
    cap.add_argument("--query", required=True)
    args = parser.parse_args()
    if args.command == "knowledge-mcp":
        serve(args)
    else:
        from mindie_bench.evaluation.corpus_linter import CorpusLintRejected

        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = capture(args)
        except CorpusLintRejected as exc:
            print(json.dumps(dict(status="failed", corpus_lint=exc.report), ensure_ascii=False))
            raise SystemExit(1) from exc
        except UnknownExternalEffects as exc:
            print(json.dumps(dict(status="unknown", error=str(exc))))
            raise SystemExit(2) from exc
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
