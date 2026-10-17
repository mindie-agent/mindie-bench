"""Use installed MindIE capture, LangMem indexing and ReMe retrieval.

Run this module with the selected component interpreter. Component dependencies
remain in that runtime; the benchmark never substitutes its own summarizer or
search algorithm. Publication is confined to a fresh local bare Git repository.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
    from mindie_knowledge.materials.store import MaterialStore

    store = MaterialStore(state, "vllm-ascend")
    try:
        store.install_packages(packages(corpus), source_revision=revision)
    except BaseException as exc:
        try:
            store.close()
        except Exception as cleanup:
            exc.add_note(f"consumer cleanup also failed: {type(cleanup).__name__}: {cleanup}")
        raise
    return store


def explain(store, ref):
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
            query: str | None = None, limit: int = 5, continuation: str | None = None
        ) -> list[dict]:
            """Search reference experience; retain failures and check current applicability."""
            return await invoke(
                "query",
                lambda: store.search(query=query, limit=limit, continuation=continuation),
                query=query,
                limit=limit,
                continuation=continuation,
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
            worker.submit(store.close).result()


def git(*arguments, cwd=None):
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *map(str, arguments)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        return
    if primary is None:
        _, primary = failures.pop(0)
        for stage, error in failures:
            primary.add_note(f"{stage} also failed: {type(error).__name__}: {error}")
        raise primary
    for stage, error in failures:
        primary.add_note(f"{stage} also failed: {type(error).__name__}: {error}")


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
    if native["harness"] != "codex":
        raise ValueError("this adapter requires an actual Codex transcript and Stop event")
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
    sys.path.insert(0, str(Path(args.codex_scripts).resolve(strict=True)))
    import bridge
    import codex_transcript

    event = bridge.hook_event(Path(native["stop_event_path"]).read_bytes())
    if (
        event["session_id"] != native["session_id"]
        or Path(event["transcript_path"]).resolve() != Path(native["transcript_path"]).resolve()
        or Path(event["cwd"]).resolve() != Path(native["cwd"]).resolve()
    ):
        raise ValueError("native Stop event and completed foreground task differ")
    source = codex_transcript.capture_source(event["transcript_path"], event["session_id"])
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
    # Use complete native metadata. A slug-only catalog cannot be consumed by
    # Codex and would make the strict native worker abort before inference.
    catalog = json.loads((root / "model-catalog.json").read_text())
    selected = [model for model in catalog["models"] if model.get("slug") == args.summary_model]
    if len(selected) != 1:
        raise ValueError("capture model requires exactly one explicit native catalog record")
    os.environ["MINDIE_CODEX_BIN"] = args.codex_bin
    store = Store(root / "capture-state", "vllm-ascend")
    scripts = str(Path(args.codex_scripts).resolve(strict=True))
    component_root = str(Path(mindie_knowledge.__file__).resolve().parent.parent)
    summary_loader = (
        f"import sys,runpy;sys.path[:0]={[component_root, scripts]!r};"
        f"runpy.run_path({str(Path(scripts) / 'agent_worker.py')!r},run_name='__main__')"
    )
    engine = Engine(
        store,
        settings_path=settings,
        admission=admission,
        transcript_adapter=codex_transcript,
        redactor_executable=args.gitleaks,
        summary_command=[sys.executable, "-I", "-c", summary_loader],
    )
    capture_failure = None
    try:
        with timing.measure("native_capture_admission"):
            receipt = engine.capture(
                session_id=event["session_id"],
                turn_id=event["turn_id"],
                transcript_path=event["transcript_path"],
                cwd=event["cwd"],
                harness="codex",
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
        with timing.measure("local_git_publish"):
            git("init", "-q", "-b", "main", staging)
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
            git("clone", "-q", "--branch", "main", local_remote, cloned)
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
                        raise
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
        total = usage["input_tokens"] + usage["output_tokens"]
        result = dict(
            status="completed",
            tokens=dict(total=total, complete=True),
            native_stop_delivered=True,
            native_session_id=event["session_id"],
            body_capture=body_receipt,
            indexer_usage=usage,
            publication=dict(kind="local-git", revision=revision),
            packages=[dict(task_id=p["task_id"], package_hash=p["package_hash"]) for p in fetched],
            consumer=dict(revision=revision, hits=hits, reads=reads),
            stage_observations=timing.stages.copy(),
            stage_scope="adapter stages; native foreground Stop-hook latency is measured separately",
        )
        write_json(root / "full-loop-receipt.json", result)
        return result
    except BaseException as error:
        capture_failure = error
        raise
    finally:
        close_capture_resources(engine, store, timing, primary=capture_failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mcp = commands.add_parser("knowledge-mcp")
    mcp.add_argument("--corpus", required=True)
    mcp.add_argument("--state", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--codex-scripts", required=True)
    cap.add_argument("--codex-bin", required=True)
    cap.add_argument("--summary-model", required=True)
    cap.add_argument("--gitleaks", required=True)
    cap.add_argument("--query", required=True)
    args = parser.parse_args()
    if args.command == "knowledge-mcp":
        serve(args)
    else:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = capture(args)
        except UnknownExternalEffects as exc:
            print(json.dumps(dict(status="unknown", error=str(exc))))
            raise SystemExit(2) from exc
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
