"""No network upload; atomic HF-compatible artifacts after two fail-closed gates."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from mindie_bench.hf_sync.leaderboard import render
from mindie_bench.io import digest, write_json
from mindie_bench.scorer.comparator import grouped_summaries, load_receipts

REDACT = """
import json, sys
from mindie_knowledge.loop.transcript_redaction import redact
from mindie_knowledge.redact import scan_text
request=json.load(sys.stdin)
clean,rules=redact(request["text"], executable=request["gitleaks"],
    key=bytes.fromhex(request["key"]), private_paths=request["private_paths"])
remaining=scan_text(clean)
if remaining:
    raise RuntimeError("redaction left unresolved sensitive findings")
print(json.dumps({"clean":clean,"rules":rules}))
"""

PUBLIC_FINGERPRINT = {
    "host_os",
    "host_arch",
    "harness_name",
    "harness_version",
    "harness_binary_sha256",
    "base_model",
    "reasoning_effort",
    "model_catalog_sha256",
    "skills_sha256",
    "source_projection",
    "benchmark_version",
    "benchmark_source_sha256",
    "base_commit",
    "npu_model",
    "npu_count",
    "remote_environment",
    "component_revisions",
    "corpus_commit",
    "corpus_view_sha256",
    "excluded_task_hashes",
    "components",
    "publication_policy",
    "backend",
    "budget",
    "evidence",
}


def public_row(receipt):
    return dict(
        schema_version=receipt.schema_version,
        trial_id=receipt.trial_id,
        instance_id=receipt.instance_id,
        repetition=receipt.repetition,
        tier=receipt.tier,
        ablation_group=receipt.ablation_group.value,
        knowledge_gradient=receipt.knowledge_gradient,
        task_role=receipt.task_role,
        cluster=receipt.cluster,
        evidence=receipt.evidence,
        terminal_status=receipt.terminal_status.value,
        task_digest=receipt.task_digest,
        experiment_digest=receipt.experiment_digest,
        environment_fingerprint={
            key: value
            for key, value in receipt.environment_fingerprint.items()
            if key in PUBLIC_FINGERPRINT
        },
        primary_metrics=receipt.primary_metrics,
        # Raw exceptions, prompts, reasoning, command arguments, native
        # payloads, home paths, and profile configs are never exported.
        error_stages=sorted({error["stage"] for error in receipt.errors}),
    )


def export_hf(source: Path, output: Path, *, knowledge_python: Path, gitleaks: Path) -> dict:
    if output.exists():
        raise FileExistsError("export destination already exists")
    if not knowledge_python.is_absolute() or not gitleaks.is_absolute():
        raise ValueError("use explicit absolute redactor runtime and scanner paths")
    if not knowledge_python.is_file() or not gitleaks.is_file():
        raise ValueError("required redactor runtime or gitleaks is unavailable")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export requires uv sync --extra export") from exc
    receipts = load_receipts(source)
    payload = {"receipts": [public_row(r) for r in receipts], "leaderboard": grouped_summaries(receipts)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mindie-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir(mode=0o700)
        request = dict(
            text=json.dumps(payload, ensure_ascii=False),
            gitleaks=str(gitleaks),
            key=os.urandom(32).hex(),
            private_paths=[str(Path.home()), str(source.resolve())],
        )
        # Use the installed first-party deterministic redactor, not a new,
        # weaker regex implementation or a summarizing model.
        result = subprocess.run(
            [str(knowledge_python), "-I", "-c", REDACT],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError("required mindie_knowledge redaction failed; no export was published")
        clean = json.loads(result.stdout)
        data = json.loads(clean["clean"])
        write_json(staging / "receipts.json", data["receipts"], exclusive=True)
        write_json(staging / "leaderboard.json", data["leaderboard"], exclusive=True)
        (staging / "index.html").write_text(render(data["leaderboard"]))
        scanner_config = Path(temporary) / "gitleaks.toml"
        scanner_config.write_text("[extend]\nuseDefault = true\n")
        scan = subprocess.run(
            [
                str(gitleaks),
                "dir",
                str(staging),
                "--config",
                str(scanner_config),
                "--no-banner",
                "--no-color",
                "--ignore-gitleaks-allow",
                "--exit-code",
                "1",
                "--report-format",
                "json",
                "--report-path",
                str(Path(temporary) / "scan.json"),
            ],
            capture_output=True,
            timeout=120,
            env={k: v for k, v in os.environ.items() if not k.startswith("GITLEAKS_")},
        )
        if scan.returncode != 0:
            raise RuntimeError("required final Gitleaks gate failed; no export was published")
        rows = []
        for row in data["receipts"]:
            rows.append(
                {
                    key: value
                    if not isinstance(value, (dict, list))
                    else json.dumps(value, sort_keys=True, allow_nan=False)
                    for key, value in row.items()
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), staging / "trials.parquet")
        if pq.read_table(staging / "trials.parquet").num_rows != len(receipts):
            raise RuntimeError("Parquet round-trip count mismatch")
        write_json(
            staging / "eval_manifest.json",
            dict(
                schema_version="1.0",
                attempts=len(rows),
                redaction_rules=clean["rules"],
                gitleaks_gate="passed",
                payload_sha256=digest(data),
                uploaded=False,
                disclosure="Metrics preserve synthetic/replay/live and unknown/infra boundaries.",
            ),
            exclusive=True,
        )
        (staging / "README.md").write_text(
            "---\nlicense: mit\nconfigs:\n- config_name: default\n  data_files:\n"
            "  - split: trials\n    path: trials.parquet\n---\n\n"
            "# MindIE Bench evaluation export\n\n"
            "Each row identifies its tier and evidence class. Synthetic and replay scores "
            "are not live NPU results. Invalid/unknown attempts remain in the ledger. "
            "Nested metrics and fingerprints use JSON columns to preserve their schemas.\n"
        )
        os.replace(staging, output)
    return dict(output=str(output), rows=len(receipts), uploaded=False)
