"""[Role] Close G3 from the completed native DSH artifact through existing C6.

[Invariants] A verified business outcome supplies task_state independently of
Stop. The original compressed artifact, public extraction and selected inputs
are bound separately. Unknown paid calls are never retried. Publication and
merge use only the trial's owned local bare Git; no GitHub write is performed.
[Timeouts & Locks] C6 owns 30s scans/10s identity/315s worker and WAL/5s writes;
local Git commands are 30s, publication transactions 120s, feed reads 60s.
[Side-Effect Policy] Writes owned capture receipts, local C5 Store/outbox and
the local corpus, then verifies a separate consumer's exact publication read.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindie_bench.evaluation.timing import StageRecorder
from mindie_bench.integrations.first_party import git, write_json
from mindie_bench.models import UnknownExternalEffects


def select_transcript(native, scripts):
    """Read only the exact, finished owned artifact through its installed adapter."""
    if native.get("harness") == "codex":
        path = Path(native["transcript_path"]).resolve(strict=True)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        sys.path.insert(0, str(scripts))
        import codex_transcript

        source = codex_transcript.capture_source(path, native["session_id"])
        if Path(source["project_root"]).resolve(strict=True) != Path(native["cwd"]).resolve(strict=True):
            raise ValueError("native artifact project_root differs from the completed foreground")
        turn_id = native.get("turn_id") or native["session_id"]
        stop_file = Path(native.get("stop_event_path") or "")
        if stop_file.is_file() and not stop_file.is_symlink():
            import bridge

            event = bridge.hook_event(stop_file.read_bytes())
            if (
                event["session_id"] != native["session_id"]
                or Path(event["transcript_path"]).resolve(strict=True) != path
                or Path(event["cwd"]).resolve(strict=True) != Path(native["cwd"]).resolve(strict=True)
            ):
                raise ValueError("native Stop event and completed foreground task differ")
            turn_id = event.get("turn_id") or turn_id
        selected = codex_transcript.read_material(
            path, 0, session_id=native["session_id"], max_text_bytes=224 * 1024, max_seconds=5
        )
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError("native artifact changed during completed-task capture")
        if (
            selected["status"] != "ok"
            or selected["partial"]
            or selected["more"]
            or not selected["text"].strip()
        ):
            raise ValueError("native public transcript is incomplete; select a smaller verified task scope")
        return selected["text"], dict(
            artifact_sha256=before,
            extracted_public_sha256=selected["digest"],
            session_id=native["session_id"],
            turn_id=turn_id,
            public_records=selected["records"],
            artifact_unchanged=True,
            private_reasoning_selected=False,
            tool_results_selected=False,
        )
    if native.get("harness") == "kimi":
        import importlib.util

        kimi_home = Path(os.environ.get("KIMI_CODE_HOME", "")).resolve(strict=True)
        sessions = (kimi_home / "sessions").resolve(strict=True)
        path = Path(native["transcript_path"]).resolve(strict=True)
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.is_relative_to(sessions)
            or path.parents[2].name != native["session_id"]
        ):
            raise ValueError("Kimi transcript is outside the owned native profile")
        state_path = path.parents[2] / "state.json"
        if not state_path.is_file() or state_path.is_symlink():
            raise ValueError("Kimi session state.json is missing or invalid")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or (
            state.get("cwd") is not None
            and Path(str(state["cwd"])).resolve(strict=True) != Path(native["cwd"]).resolve(strict=True)
        ):
            raise ValueError("native artifact state.json differs from the completed foreground")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("kimi_transcript", Path(scripts) / "transcript.py")
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        selected = adapter.read_material(
            path, 0, session_id=native["session_id"], max_text_bytes=224 * 1024, max_seconds=5
        )
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError("native artifact changed during completed-task capture")
        if (
            selected["status"] != "ok"
            or selected["partial"]
            or selected["more"]
            or not selected["text"].strip()
        ):
            raise ValueError("native public transcript is incomplete; select a smaller verified task scope")
        return selected["text"], dict(
            artifact_sha256=before,
            extracted_public_sha256=selected["digest"],
            session_id=native["session_id"],
            turn_id=native.get("turn_id") or native["session_id"],
            public_records=selected["records"],
            artifact_unchanged=True,
            private_reasoning_selected=False,
            tool_results_selected=False,
        )
    import zstandard

    if native.get("harness") != "dsh" or native.get("event_source") != "observed_headless_turn_end":
        raise ValueError("graded capture requires the actual completed DSH descriptor")
    path = Path(native["transcript_path"]).resolve(strict=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with (
        path.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as stream,
    ):
        header_bytes = stream.read(16 * 1024).split(b"\n", 1)[0]
    header = json.loads(header_bytes)
    if (header.get("type"), header.get("version"), header.get("id")) != (
        "session",
        4,
        native["session_id"],
    ) or Path(header["cwd"]).resolve(strict=True) != Path(native["cwd"]).resolve(strict=True):
        raise ValueError("native artifact header differs from the completed foreground")
    sys.path.insert(0, str(scripts))
    import dsh_transcript

    selected = dsh_transcript.read_material(
        path, 0, session_id=native["session_id"], max_text_bytes=224 * 1024, max_seconds=5
    )
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("native compressed artifact changed during completed-task capture")
    if (
        selected["status"] != "ok"
        or selected["partial"]
        or selected["more"]
        or not selected["text"].strip()
    ):
        raise ValueError("native public transcript is incomplete; select a smaller verified task scope")
    return selected["text"], dict(
        artifact_sha256=before,
        extracted_public_sha256=selected["digest"],
        session_id=native["session_id"],
        turn_id=native["turn_id"],
        public_records=selected["records"],
        artifact_unchanged=True,
        private_reasoning_selected=False,
        tool_results_selected=False,
    )


def settings(root, remote, *, name, generation, scope, repository):
    value = dict(
        schema="mindie-community-config/1",
        enabled=True,
        consent_config=str(root / "consent.json"),
        generation=generation,
        enabled_at=1_700_000_000,
        repository=repository,
        branch="main",
        project_roots=[str(scope)],
        idle_seconds=300,
        transport="file",
        dev_remotes={repository: str(remote)},
        transaction_seconds=120,
        operation_limit=60,
        token_env="GH_TOKEN",
        bot={},
    )
    target = root / name
    write_json(target, value)
    return target


def _stage_records(stages, started_at, usage):
    """Add factual stage ids/absolute times and the in-capture C6 usage.

    Existing relative `started_sec`/`ended_sec` keys are preserved. Absolute
    times use the wall-clock anchor read immediately after StageRecorder creation;
    no time is reconstructed from an unrelated receipt.
    """
    records = []
    for stage in stages:
        row = dict(stage)
        row["stage_id"] = row.get("name")
        for prefix, key in (("started", "started_at"), ("ended", "ended_at")):
            if f"{prefix}_sec" in row:
                row[key] = (started_at + timedelta(seconds=row[f"{prefix}_sec"])).isoformat()
        if row.get("name") == "graded_distillation" and usage:
            known_total = sum(
                usage.get(key, 0)
                for key in ("input_tokens", "cached_input_tokens", "output_tokens")
            )
            row["usage"] = {
                "model_calls": usage.get("model_calls"),
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "tokens_total": known_total,
                "tokens_basis": "complete" if usage.get("unknown_calls") == 0 else "lower_bound",
                "npu_runs": 0,
                "estimated_cost_usd": None,
                "price_basis": "unknown",
                "complete": usage.get("unknown_calls") == 0,
            }
        records.append(row)
    return records


def _satisfy_condition_value(key: str, expected):
    if expected == "any":
        return "any"
    if key == "framework" and isinstance(expected, list):
        from packaging.requirements import InvalidRequirement, Requirement

        frameworks = {}
        for item in expected:
            try:
                req = Requirement(str(item))
                version = "1.0.0"
                for spec in req.specifier:
                    if spec.operator in {"==", "===", ">=", ">", "~="}:
                        version = spec.version
                        break
                frameworks[req.name] = version
            except InvalidRequirement:
                frameworks[str(item)] = "1.0.0"
        return frameworks
    if isinstance(expected, dict):
        if set(expected) == {"exact_set"}:
            return list(expected["exact_set"])
        return {k: _satisfy_condition_value(k, v) for k, v in expected.items()}
    if isinstance(expected, list):
        return [expected[0]] if expected else []
    if isinstance(expected, str) and expected.startswith(("==", ">=", "<=", "~=", "!=", ">", "<")):
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version

        raw = re.sub(r"^(?:===|==|>=|<=|~=|!=|>|<)\s*", "", expected) or "1.0.0"
        try:
            spec = SpecifierSet(expected)
            if Version(raw) in spec:
                return raw
            for candidate in (f"{raw}.1", "9999.0.0", "2", "0.0.1", "0"):
                if Version(candidate) in spec:
                    return candidate
        except (InvalidSpecifier, InvalidVersion):
            pass
        return raw
    return expected


def _derive_readback_probe(consumer, admitted_ids, explicit_query: str = "", explicit_conditions=None):
    index = consumer.graded._load()
    target_id = next((uid for uid in admitted_ids if uid in index.units), None)
    if target_id is None:
        raise RuntimeError("admitted unit is absent from synchronized consumer index")
    unit = index.units[target_id]
    if explicit_conditions:
        conditions = explicit_conditions
    else:
        conditions = {}
        for ancestor_id in [*index.closure.get(target_id, ()), target_id]:
            ancestor = index.units.get(ancestor_id)
            if ancestor is None:
                continue
            for key, expected in ancestor.metadata.get("conditions", {}).items():
                val = _satisfy_condition_value(key, expected)
                if key == "framework" and isinstance(val, dict) and isinstance(conditions.get("framework"), dict):
                    conditions["framework"] = {**conditions["framework"], **val}
                else:
                    conditions[key] = val
    query_text = explicit_query.strip() if isinstance(explicit_query, str) else ""
    if not query_text or query_text.lower() == "placeholder":
        triggers = unit.metadata.get("triggers", {})
        parts = [
            unit.metadata.get("title", ""),
            *triggers.get("error_signatures", ())[:2],
            *triggers.get("keywords", ())[:4],
        ]
        query_text = " ".join(part for part in parts if part).strip() or target_id
    return query_text, conditions


def capture(args, payload):
    from mindie_knowledge import consent_store
    from mindie_knowledge.community.common import Deadline
    from mindie_knowledge.community.transport import FileTransport
    from mindie_knowledge.loop.engine import Engine
    from mindie_knowledge.loop.feed import Feed
    from mindie_knowledge.loop.store import Store
    from mindie_knowledge.materials.summarizer import SummaryLedger
    from mindie_knowledge.publication_contract import parse_contract

    root = Path.cwd().resolve(strict=True)
    private = Path(os.environ["MINDIE_BENCH_KNOWLEDGE_PRIVATE_ROOT"]).resolve(strict=True)
    corpus = Path(payload["destination"]).resolve(strict=True)
    remote = Path(payload["local_remote"]).resolve(strict=True)
    if (
        root.name != "profile"
        or private != root.parent / "knowledge-private"
        or (corpus != private / "corpus" or remote != private / "local_knowledge.git")
    ):
        raise ValueError("graded capture must use this trial's owned local publication route")
    if payload.get("business_terminal_status") not in {"completed_pass", "completed_fail"}:
        raise ValueError("C6 task_state requires an independently verified business terminal result")
    contract_bytes = (corpus / "publication-contract.json").read_bytes()
    domain = json.loads(contract_bytes)["domain"]
    contract = parse_contract(contract_bytes, domain)["contract"]
    if contract["schema"] != "mindie-publication-contract/2":
        raise ValueError("G3 graded capture requires publication-contract/2")
    output = root / "graded-capture"
    output.mkdir()
    timing = StageRecorder(output / "stages.json")
    clock_started_at = datetime.now(timezone.utc)
    report = dict(
        status="running",
        protocol="graded-c6",
        publication_scope="local-only",
        github_api=False,
        task_state="completed",
        corpus_lint={"status": "pending"},
    )

    def save():
        report["capture_started_at"] = clock_started_at.isoformat()
        report["stage_observations"] = _stage_records(
            timing.stages, clock_started_at, report.get("indexer_usage")
        )
        write_json(root / "full-loop-receipt.json", report)

    save()
    producer = consumer = engine = None
    primary = None
    try:
        harness_name = payload["native"].get("harness", "dsh")
        if harness_name == "kimi":
            scripts = Path(getattr(args, "kimi_scripts", None) or args.dsh_scripts).resolve(strict=True)
            if getattr(args, "kimi_bin", None):
                os.environ["MINDIE_KIMI_BIN"] = str(Path(args.kimi_bin).resolve(strict=True))
            summary_worker = scripts / "organizer.py"
        elif harness_name == "codex":
            scripts = Path(getattr(args, "codex_scripts", None) or args.dsh_scripts).resolve(strict=True)
            if getattr(args, "codex_bin", None):
                os.environ["MINDIE_CODEX_BIN"] = str(Path(args.codex_bin).resolve(strict=True))
            if getattr(args, "summary_model", None):
                os.environ["MINDIE_SUMMARY_MODEL"] = str(args.summary_model)
            summary_worker = scripts / "agent_worker.py"
        else:
            scripts = Path(args.dsh_scripts).resolve(strict=True)
            summary_worker = scripts / "dsh_organizer.py"
        with timing.measure("native_artifact_selection"):
            text, binding = select_transcript(payload["native"], scripts)
            report["native_artifact"] = binding
        seed = output / "seed"
        shutil.copytree(corpus, seed)
        with timing.measure("seed_local_publication"):
            git("init", "--template=", "-q", "-b", "main", seed)
            git("add", ".", cwd=seed)
            git(
                "-c",
                "user.name=MindIE Benchmark",
                "-c",
                "user.email=benchmark@example.invalid",
                "commit",
                "-qm",
                "Seed exact graded benchmark view",
                cwd=seed,
            )
            git("push", str(remote), "HEAD:refs/heads/main", cwd=seed)
            git("--git-dir", remote, "symbolic-ref", "HEAD", "refs/heads/main")
        repository = "benchmark/local-graded"
        consent_store.record_choice(output / "consent.json", "contribute")
        scope = Path(payload["native"]["cwd"])
        generation = "benchmark-native-" + binding["artifact_sha256"][:24]
        config = settings(
            output,
            remote,
            name="producer.json",
            generation=generation,
            scope=scope,
            repository=repository,
        )
        producer = Store(output / "producer", domain)
        consumer = Store(output / "consumer", domain)
        feed_config = dict(domain=domain, repository=repository, url=str(remote))
        producer_feed, consumer_feed = Feed(producer, feed_config), Feed(consumer, feed_config)
        for feed in (producer_feed, consumer_feed):
            synced = feed.sync(force=True)
            if synced["status"] != "synced":
                raise RuntimeError("owned graded seed feed did not synchronize: " + json.dumps(synced))
        engine = Engine(
            producer,
            settings_path=config,
            state_dir=output / "producer-state",
            redactor_executable=args.gitleaks,
            summary_command=[sys.executable, "-I", str(summary_worker)],
        )
        locator = "benchmark/native-task@" + binding["artifact_sha256"]
        manifest = payload["task_manifest"]
        inputs = [
            dict(
                kind="authorized_transcript",
                outcome="observed",
                content=text,
                source_locator=locator + ":public-transcript.md",
                evidence_refs=["native-public:" + binding["extracted_public_sha256"]],
                repetitions=1,
            ),
            dict(
                kind="run_manifest",
                outcome=(
                    "verified"
                    if payload["business_terminal_status"] == "completed_pass"
                    else "measurement_invalid"
                ),
                content=json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                source_locator=locator + ":run-manifest.json",
                evidence_refs=[
                    "verifier:"
                    + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                ],
                repetitions=1,
            ),
        ]
        # This receipt remains local. Engine performs the established redaction
        # before any selected transcript bytes reach the native organizer.
        write_json(output / "selected-inputs.private.json", inputs)
        with timing.measure("graded_distillation"):
            distilled = engine.distill_selected(
                inputs,
                session_digest=hashlib.sha256(
                    (binding["artifact_sha256"] + ":" + binding["turn_id"]).encode()
                ).hexdigest(),
                task_state="completed",
                authorization=dict(
                    generation=generation, session=payload["native"]["session_id"], scope=str(scope)
                ),
            )
        report["distill"] = distilled
        usage = SummaryLedger(producer.db).usage_totals()
        report["indexer_usage"] = usage
        known_total = sum(usage[key] for key in ("input_tokens", "cached_input_tokens", "output_tokens"))
        report["tokens"] = dict(
            total=known_total if not usage["unknown_calls"] else None,
            known_lower_bound=known_total,
            complete=not usage["unknown_calls"],
        )
        report["usage_attribution"] = {
            "capture_ingestion": {
                "stage_id": "capture_ingestion",
                "model_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "tokens_total": 0,
                "tokens_basis": "complete",
                "npu_runs": 0,
                "estimated_cost_usd": None,
                "price_basis": "unknown",
                "complete": True,
                "evidence_ref": "capture#capture_ingestion",
            },
            "organizer": {
                "stage_id": "graded_distillation",
                "model_calls": usage.get("model_calls"),
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "tokens_total": known_total,
                "tokens_basis": "complete" if usage["unknown_calls"] == 0 else "lower_bound",
                "npu_runs": 0,
                "estimated_cost_usd": None,
                "price_basis": "unknown",
                "complete": usage["unknown_calls"] == 0,
                "evidence_ref": "capture#graded_distillation",
            },
            "tokens_complete": not usage["unknown_calls"],
            "price_basis": "unknown",
        }
        save()
        if usage["unknown_calls"] or distilled["status"] in {"outcome_unknown", "invoking"}:
            raise UnknownExternalEffects("native C6 outcome or accounting is unknown; no model retry")
        if (
            payload.get("role") == "eval"
            and distilled["status"] in {"noop", "review_or_unchanged"}
            and not distilled.get("admitted_unit_ids")
        ):
            report.update(
                status="completed",
                native_harness=harness_name,
                native_session_id=binding["session_id"],
                native_turn_id=binding["turn_id"],
                publication=dict(
                    kind="eval-already-covered-or-review",
                    status=distilled["status"],
                    units=[],
                ),
                corpus_lint=dict(
                    status="passed",
                    protocol="C6 validation and contract/2 feed validation (eval covered by seed)",
                    units=[],
                    findings=[],
                ),
            )
            save()
            return report
        if distilled["status"] != "applied" or not distilled.get("admitted_unit_ids"):
            raise RuntimeError("C6 did not admit reusable units: " + distilled["status"])
        with timing.measure("local_publication_and_merge"):
            engine.publisher.flush()
            row = producer.db.execute("SELECT * FROM outbox ORDER BY updated DESC LIMIT 1").fetchone()
            if row is None or row["status"] != "submitted":
                raise RuntimeError("C6 local publication was not submitted")
            report["outbox"] = {key: row[key] for key in ("batch_id", "status", "detail")}
            transport = FileTransport(
                output / "producer-state" / "dev-github.json", {repository: str(remote)}
            )
            pull = transport.get_pull_request(repository, 1, deadline=Deadline(20, 10))
            transport.merge_pull_request(
                repository, 1, sha=pull["head"]["sha"], method="squash", deadline=Deadline(20, 10)
            )
        with timing.measure("independent_consumer_readback"):
            synced = consumer_feed.sync(force=True)
            if synced["status"] != "synced":
                raise RuntimeError("independent graded consumer did not synchronize")
            admitted = set(distilled["admitted_unit_ids"])
            query_text, conditions = _derive_readback_probe(
                consumer,
                sorted(admitted),
                explicit_query=getattr(args, "query", ""),
                explicit_conditions=payload.get("capture_conditions"),
            )
            hits = consumer.query(query_text, conditions=conditions)
            matched = [hit for hit in hits["results"] if hit["id"] in admitted]
            if not matched:
                raise RuntimeError(
                    "independent consumer cannot retrieve a new unit under actual conditions"
                )
            reads = [consumer.explain(hit["ref"]) for hit in matched]
            report["consumer"] = dict(
                sync=synced, query=query_text, conditions=conditions, hits=hits, reads=reads
            )
        with timing.measure("warm_view_install"):
            cloned = output / "published"
            git("clone", "--template=", "-q", "--branch", "main", remote, cloned)
            revision = git("rev-parse", "HEAD", cwd=cloned)
            from mindie_bench.runner.graded_view import units

            published = units(cloned)
            if not admitted <= set(published):
                raise RuntimeError("local publication omitted an admitted graded unit")
            for path in corpus.rglob("unit-*.md"):
                path.unlink()
            for path, _ in published.values():
                target = corpus / path.relative_to(cloned)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            report["publication"] = dict(kind="local-git", revision=revision, units=sorted(admitted))
        report.update(
            status="completed",
            native_harness=harness_name,
            native_session_id=binding["session_id"],
            native_turn_id=binding["turn_id"],
            corpus_lint=dict(
                status="passed",
                protocol="C6 validation and contract/2 feed validation",
                units=sorted(admitted),
                findings=[],
            ),
        )
        save()
        return report
    except BaseException as error:
        primary = error
        report.update(
            status="unknown" if isinstance(error, UnknownExternalEffects) else "failed",
            error=dict(type=type(error).__name__, message=str(error)),
        )
        save()
        raise
    finally:
        cleanup_errors = []
        cleanup_operations = [
            ("engine", engine.operations.shutdown if engine else None),
            ("consumer", consumer.close if consumer else None),
            ("producer", producer.close if producer else None),
        ]
        for name, close in cleanup_operations:
            if close is not None:
                try:
                    close()
                except Exception as cleanup:
                    cleanup_errors.append(cleanup)
                    if primary is not None:
                        primary.add_note(f"graded capture {name} cleanup also failed: {cleanup}")
        report["cleanup"] = dict(
            status="failed" if cleanup_errors else "completed",
            errors=[dict(type=type(error).__name__, message=str(error)) for error in cleanup_errors],
        )
        if cleanup_errors and primary is None:
            # The graded result is already complete. Preserve it and attach the
            # secondary cleanup failure instead of raising from `finally`.
            report["cleanup_warning"] = (
                "graded capture cleanup failed after recorded result: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            )
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsh-scripts")
    parser.add_argument("--kimi-scripts")
    parser.add_argument("--kimi-bin")
    parser.add_argument("--codex-scripts")
    parser.add_argument("--codex-bin")
    parser.add_argument("--summary-model")
    parser.add_argument("--gitleaks", required=True)
    parser.add_argument("--query", default="")
    args = parser.parse_args()
    if not (args.dsh_scripts or args.kimi_scripts or args.codex_scripts):
        parser.error("one of --dsh-scripts, --kimi-scripts, or --codex-scripts is required")
    payload = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = capture(args, payload)
    except BaseException:
        receipt = Path.cwd() / "full-loop-receipt.json"
        if receipt.is_file():
            print(json.dumps(json.loads(receipt.read_text()), ensure_ascii=False))
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
