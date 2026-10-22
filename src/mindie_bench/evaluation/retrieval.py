# [Role] Runs frozen source-bound retrieval queries against a first-party store.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Execute frozen, source-bound queries against the selected first-party store."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

from mindie_bench.corpus import read_package
from mindie_bench.io import digest, safe_relative, tree_hash, write_json
from mindie_bench.scorer.metrics import retrieval_metrics


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)

    def quantile(fraction):
        index = (len(ordered) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return dict(
        samples=len(values),
        mean_sec=mean(values) if values else None,
        p50_sec=quantile(0.5) if values else None,
        p95_sec=quantile(0.95) if len(values) >= 20 else None,
        p95_reason=None if len(values) >= 20 else "fewer_than_20_observations",
        quantile_method="linear_interpolation; repeated reads, not independent workload draws",
    )


def load_labels(path: Path, corpus: Path, revision: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("retrieval workload must contain unique, nonempty samples")
    if any(not isinstance(row["sample_id"], str) or not row["sample_id"].strip() for row in rows):
        raise ValueError("retrieval sample IDs must be nonempty text")
    queries = [
        " ".join(row["query"].casefold().split()) if isinstance(row["query"], str) else ""
        for row in rows
    ]
    if len(set(queries)) != len(rows):
        raise ValueError("duplicate query text cannot count as independent retrieval samples")
    for row in rows:
        if row["source_commit"] != revision:
            raise ValueError("retrieval labels refer to a different corpus commit")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError("retrieval query must be nonempty text")
        if row["label_scope"] != "positive_only_target_recovery":
            raise ValueError("declare positive-only target recovery labels explicitly")
        for field in ["relevant_task_ids", "relevant_block_ids"]:
            values = row[field]
            if not values or len(values) != len(set(values)):
                raise ValueError("retrieval targets must be nonempty and unique")
        expected_paths = set()
        for task_id in row["relevant_task_ids"]:
            if safe_relative(task_id) != task_id or "/" in task_id:
                raise ValueError("invalid retrieval task ID")
            package = read_package(corpus / "tasks" / task_id)
            expected_paths.add(package.navigation.relative_to(corpus).as_posix())
        for block in row["relevant_block_ids"]:
            task_id, block_id = block.split(":", 1)
            if task_id not in row["relevant_task_ids"] or "/" in safe_relative(block_id):
                raise ValueError("block target must belong to a declared task")
            path = corpus / "tasks" / task_id / "blocks" / (block_id + ".md")
            expected_paths.add(path.relative_to(corpus).as_posix())
        if not expected_paths <= row["source_files_sha256"].keys():
            raise ValueError("retrieval labels must bind their navigation and exact target blocks")
        for relative, expected in row["source_files_sha256"].items():
            relative = safe_relative(relative)
            source = corpus / relative
            if source.is_symlink() or not source.resolve().is_relative_to(corpus.resolve()):
                raise ValueError("retrieval label source escapes the corpus")
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("retrieval relevance evidence changed")
    return rows


def _selected_identity(hit):
    from mindie_knowledge.materials.references import parse_read_ref

    parsed = parse_read_ref(hit["ref"], domain="vllm-ascend")
    return parsed["task_id"], parsed.get("block_id")


def _body(corpus, task_id, block_id):
    path = corpus / "tasks" / safe_relative(task_id) / "blocks" / (safe_relative(block_id) + ".md")
    return path.read_text().split("\n---\n\n", 1)[1]


def evaluate_retrieval(
    corpus: Path, labels: Path, output: Path, *, repetitions: int = 3, k: int = 3
) -> dict:
    import mindie_knowledge

    from mindie_bench.integrations.first_party import consumer, explain

    if repetitions < 1 or k < 1:
        raise ValueError("retrieval repetitions and k must be positive")
    corpus, labels = corpus.resolve(strict=True), labels.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(corpus), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(corpus), "status", "--porcelain"], text=True, timeout=30)
    if dirty:
        raise ValueError("component retrieval requires a clean, pinned corpus")
    rows = load_labels(labels, corpus, revision)
    component_root = Path(mindie_knowledge.__file__).resolve().parent
    component_sources = {
        path.relative_to(component_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(component_root.rglob("*.py"))
    }
    component_identity = dict(
        interpreter=sys.executable,
        python_version=sys.version,
        source_sha256=digest(component_sources),
        source_python_files=len(component_sources),
        benchmark_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    original_hash = tree_hash(corpus)
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    (output / "labels.jsonl").write_bytes(labels.read_bytes())
    write_json(
        output / "workload.json",
        dict(
            schema="mindie-bench-retrieval-workload/1",
            source_commit=revision,
            corpus_sha256=original_hash,
            labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
            distinct_queries=len(rows),
            repetitions=repetitions,
            k=k,
            label_scope="positive_only_target_recovery",
            component_identity=component_identity,
        ),
        exclusive=True,
    )
    observations, summaries = [], []
    reader = None
    primary = None
    result = None
    phase = "install"
    started = time.perf_counter()
    try:
        reader = consumer(corpus, output / "state", revision)
        installed_sec = time.perf_counter() - started
        phase = "index"
        started = time.perf_counter()
        readiness = reader.refresh_index()
        index_sec = time.perf_counter() - started
        if readiness.get("complete") is not True:
            raise RuntimeError("first-party retrieval index is not ready")
        for row in rows:
            samples = []
            for repetition in range(1, repetitions + 1):
                phase = "query"
                started = time.perf_counter()
                page = reader.search(query=row["query"], limit=k)
                query_sec = time.perf_counter() - started
                ranked = page["results"] if isinstance(page, dict) else page
                if not isinstance(ranked, list):
                    raise ValueError("first-party search did not return ranked results")
                identities = [_selected_identity(hit) for hit in ranked]
                tasks = list(dict.fromkeys(task_id for task_id, _ in identities))
                blocks = [f"{task}:{block}" for task, block in identities if block is not None]
                sample = dict(
                    sample_id=row["sample_id"],
                    repetition=repetition,
                    request_kind=(
                        "first_query_for_this_label" if repetition == 1 else "repeat_same_query"
                    ),
                    query_sec=query_sec,
                    ranked_task_ids=tasks,
                    ranked_block_ids=blocks,
                    task_metrics=retrieval_metrics(tasks, set(row["relevant_task_ids"]), k=k),
                    block_metrics=retrieval_metrics(blocks, set(row["relevant_block_ids"]), k=k),
                    query_result_utf8_bytes=len(json.dumps(page, ensure_ascii=False).encode("utf-8")),
                    explain_sec=None,
                    returned_body_sha256=None,
                )
                if ranked:
                    phase = "explain"
                    started = time.perf_counter()
                    value = explain(reader, ranked[0]["ref"])
                    sample["explain_sec"] = time.perf_counter() - started
                    task_id, block_id = identities[0]
                    if block_id is not None:
                        source = _body(corpus, task_id, block_id)
                        if value["content"] != source:
                            raise RuntimeError("component explain body differs from canonical Markdown")
                        sample["returned_body_sha256"] = hashlib.sha256(source.encode()).hexdigest()
                        sample["returned_body_utf8_bytes"] = len(source.encode())
                    sample["explain_result_utf8_bytes"] = len(
                        json.dumps(value, ensure_ascii=False).encode("utf-8")
                    )
                observations.append(sample)
                samples.append(sample)
                with (output / "observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            summaries.append(
                dict(
                    sample_id=row["sample_id"],
                    query_kind=row.get("query_kind", "unspecified"),
                    task_metrics={
                        key: mean(sample["task_metrics"][key] for sample in samples)
                        for key in samples[0]["task_metrics"]
                    },
                    block_metrics={
                        key: mean(sample["block_metrics"][key] for sample in samples)
                        for key in samples[0]["block_metrics"]
                    },
                    ranking_varied=len({tuple(sample["ranked_block_ids"]) for sample in samples}) > 1,
                )
            )
        phase = "source_integrity"
        if tree_hash(corpus) != original_hash:
            raise RuntimeError("source corpus changed during component evaluation")
        result = dict(
            schema="mindie-bench-component-retrieval/1",
            status="completed",
            role="component_target_recovery_not_end_to_end_benefit",
            distinct_queries=len(rows),
            measured_query_calls=len(observations),
            repetitions=repetitions,
            k=k,
            source_commit=revision,
            component_identity=component_identity,
            source_corpus_unchanged=True,
            install_sec=installed_sec,
            index_build_sec=index_sec,
            query_latency=latency_summary([row["query_sec"] for row in observations]),
            explain_latency=latency_summary(
                [row["explain_sec"] for row in observations if row["explain_sec"] is not None]
            ),
            task_metrics={
                key: mean(row["task_metrics"][key] for row in summaries)
                for key in summaries[0]["task_metrics"]
            },
            block_metrics={
                key: mean(row["block_metrics"][key] for row in summaries)
                for key in summaries[0]["block_metrics"]
            },
            per_query=summaries,
            token_reduction=None,
            token_reduction_reason="UTF-8 payload bytes are observed; no model tokenizer supplied",
            limits=[
                "Positive-only target labels are not exhaustive relevance judgments.",
                "Repeated requests do not increase the number of independent queries.",
                "Component latency under this workload is not a production SLA.",
                "No native Agent, Stop, paid model, network or NPU request is made by this runner.",
            ],
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_error = None
        if reader is not None:
            try:
                reader.close()
            except BaseException as cleanup:
                cleanup_error = cleanup
                if primary is not None:
                    primary.add_note(
                        f"retrieval store close also failed: {type(cleanup).__name__}: {cleanup}"
                    )
        failure = primary if primary is not None else cleanup_error
        if primary is None and cleanup_error is not None and result is not None:
            # The retrieval evaluation already completed. Keep its measured
            # observations and attach the close failure as secondary evidence.
            result["cleanup_errors"] = [
                dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
            ]
            result["cleanup_warning"] = "retrieval store close failed after completed evaluation"
            failure = None
            cleanup_error = None
        if failure is not None:
            try:
                write_json(
                    output / "failure.json",
                    dict(
                        status="failed",
                        phase=phase if primary is not None else "close",
                        error=dict(type=type(failure).__name__, message=str(failure)),
                        notes=getattr(failure, "__notes__", []),
                        cleanup_error=(
                            dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
                            if cleanup_error is not None
                            else None
                        ),
                        completed_observations=len(observations),
                    ),
                )
            except BaseException as logging_error:
                failure.add_note(
                    "retrieval failure recording also failed: "
                    f"{type(logging_error).__name__}: {logging_error}"
                )
        if primary is None and cleanup_error is not None:
            raise cleanup_error
    write_json(output / "summary.json", result, exclusive=True)
    return result
