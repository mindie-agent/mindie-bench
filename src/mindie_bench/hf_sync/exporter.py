"""No network upload; atomic HF-compatible artifacts after two fail-closed gates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
    "inference_contract",
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

PUBLIC_IDENTITY = (
    "schema_version",
    "trial_id",
    "instance_id",
    "repetition",
    "tier",
    "ablation_group",
    "knowledge_gradient",
    "task_role",
    "cluster",
    "evidence",
    "terminal_status",
    "task_digest",
    "experiment_digest",
)


def public_row(receipt):
    return dict(
        schema_version=receipt.schema_version,
        trial_id=hashlib.sha256(receipt.trial_id.encode("utf-8")).hexdigest(),
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


def public_task_row(task_root: Path) -> dict:
    from mindie_bench.models import TaskSpec

    spec = TaskSpec.load(task_root)
    env_dir = task_root / "environment"
    public_env_files = (
        sorted(p.relative_to(env_dir).as_posix() for p in env_dir.rglob("*") if p.is_file())
        if env_dir.is_dir()
        else []
    )
    tiers = sorted(spec.environment.keys())
    primary_tier = "l3" if "l3" in spec.environment else tiers[0]
    env_spec = spec.environment[primary_tier]
    return dict(
        schema_version=spec.schema_version,
        instance_id=spec.task.instance_id,
        domain_category=spec.task.domain_category,
        difficulty=spec.task.difficulty,
        tier=primary_tier,
        tiers=tiers,
        knowledge_gradient=spec.mindie_knowledge.knowledge_gradient,
        task_role=spec.task.role,
        cluster=spec.task.cluster,
        evidence=spec.task.evidence,
        base_commit=env_spec.base_commit,
        npu_model=env_spec.npu_model,
        npu_count=env_spec.npu_count,
        instruction=(task_root / "instruction.md").read_text(),
        public_environment_files=public_env_files,
        target_files_count=len(spec.mindie_knowledge.target_files),
        inference=spec.inference.model_dump(mode="json") if spec.inference else None,
    )


def load_public_tasks(tasks_root: Path) -> list[dict]:
    paths = (
        [tasks_root]
        if (tasks_root / "task.toml").is_file()
        else sorted(p.parent for p in tasks_root.rglob("task.toml"))
    )
    if not paths:
        raise ValueError(f"no task manifests found under {tasks_root}")
    return [public_task_row(path) for path in paths]


def validate_redacted_receipts(data, expected_rows):
    rows = data.get("receipts") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ValueError("redaction changed receipt count; no export was published")
    expected = {row["trial_id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("public receipt ID collision; no export was published")
    seen = set()
    for row in rows:
        identity = row.get("trial_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or identity in seen or identity not in expected:
            raise ValueError("redaction changed or duplicated receipt IDs; no export was published")
        original = expected[identity]
        # Preserve the typed trial contract without restoring any redacted value.
        if any(
            key not in row or type(row[key]) is not type(original[key]) or row[key] != original[key]
            for key in PUBLIC_IDENTITY
        ):
            raise ValueError("redaction changed receipt identity or contract; no export was published")
        seen.add(identity)


def _dataset_card(leaderboard: dict, *, has_tasks: bool) -> str:
    def fmt(value):
        return "unmeasured" if value is None else f"{value:.4g}"

    configs_yaml = (
        "---\nlicense: mit\nconfigs:\n"
        "- config_name: default\n  data_files:\n  - split: trials\n    path: trials.parquet\n"
    )
    if has_tasks:
        configs_yaml += (
            "- config_name: tasks\n  data_files:\n  - split: tasks\n    path: tasks.parquet\n"
        )
    configs_yaml += "---\n\n"
    lines = [
        configs_yaml + "# MindIE Bench evaluation export",
        "",
        "Each row identifies its tier and evidence class. Synthetic and replay scores "
        "are not live NPU results. Invalid/unknown attempts remain in the ledger. "
        "Nested metrics and fingerprints use JSON columns to preserve their schemas.",
        "",
        "## Leaderboard Summary",
        "",
        "| Group | Attempts | Valid | Pass@1 | Pass^k | Mean Delivery (s) | Statuses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for group_key, summary in sorted(leaderboard.items()):
        lines.append(
            f"| `{group_key}` | {summary.get('attempts')} | {summary.get('valid_attempts')} | "
            f"{fmt(summary.get('pass_at_1'))} | {fmt(summary.get('pass_pow_k'))} | "
            f"{fmt(summary.get('mean_delivery_wall_clock_sec'))} | `{summary.get('statuses')}` |"
        )
    return "\n".join(lines) + "\n"


def export_hf(
    source: Path,
    output: Path,
    *,
    knowledge_python: Path,
    gitleaks: Path,
    tasks_root: Path | None = None,
    upload_repo: str | None = None,
    private: bool = True,
) -> dict:
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
    if (source / "eval_manifest.json").is_file():
        from mindie_bench.evaluation.accounting import audit_report

        summaries = audit_report(source)["summaries"]
    else:
        summaries = grouped_summaries(receipts)
    payload = {
        "receipts": [public_row(r) for r in receipts],
        "leaderboard": summaries,
    }
    private_paths = [str(Path.home()), str(source.resolve())]
    if tasks_root is not None:
        tasks_root = tasks_root.resolve(strict=True)
        payload["tasks"] = load_public_tasks(tasks_root)
        private_paths.append(str(tasks_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mindie-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir(mode=0o700)
        request = dict(
            text=json.dumps(payload, ensure_ascii=False),
            gitleaks=str(gitleaks),
            key=os.urandom(32).hex(),
            private_paths=private_paths,
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
        validate_redacted_receipts(data, payload["receipts"])
        if "tasks" in payload:
            redacted_tasks = data.get("tasks")
            if not isinstance(redacted_tasks, list) or len(redacted_tasks) != len(payload["tasks"]):
                raise ValueError("redaction changed task count; no export was published")
            write_json(staging / "tasks.json", redacted_tasks, exclusive=True)
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
        if "tasks" in data:
            task_parquet_rows = [
                {
                    key: value
                    if not isinstance(value, (dict, list))
                    else json.dumps(value, sort_keys=True, allow_nan=False)
                    for key, value in row.items()
                }
                for row in data["tasks"]
            ]
            pq.write_table(pa.Table.from_pylist(task_parquet_rows), staging / "tasks.parquet")
            if pq.read_table(staging / "tasks.parquet").num_rows != len(data["tasks"]):
                raise RuntimeError("Tasks Parquet round-trip count mismatch")
        uploaded = bool(upload_repo)
        manifest_payload = dict(
            schema_version="1.0",
            attempts=len(rows),
            public_trial_id_scheme="sha256-utf8-native-trial-id",
            redaction_rules=clean["rules"],
            gitleaks_gate="passed",
            payload_sha256=digest(data),
            uploaded=uploaded,
            disclosure="Metrics preserve synthetic/replay/live and unknown/infra boundaries.",
        )
        if "tasks" in data:
            manifest_payload["tasks_count"] = len(data["tasks"])
        if upload_repo:
            manifest_payload["repo_id"] = upload_repo
            manifest_payload["private"] = private
        write_json(staging / "eval_manifest.json", manifest_payload, exclusive=True)
        (staging / "README.md").write_text(
            _dataset_card(data["leaderboard"], has_tasks="tasks" in data)
        )
        if upload_repo:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                for candidate in sorted(
                    (Path.home() / ".local/share/uv/tools/huggingface-hub/lib").glob(
                        "python*/site-packages"
                    )
                ):
                    if str(candidate) not in sys.path:
                        sys.path.append(str(candidate))
                try:
                    from huggingface_hub import HfApi
                except ImportError as exc:
                    raise RuntimeError("Hugging Face upload requires huggingface_hub") from exc
            try:
                import socksio  # noqa: F401
            except ImportError:
                for key in ("ALL_PROXY", "all_proxy"):
                    if os.environ.get(key, "").lower().startswith("socks"):
                        os.environ.pop(key, None)
            api = HfApi()
            api.create_repo(
                repo_id=upload_repo, repo_type="dataset", private=private, exist_ok=True
            )
            api.upload_folder(
                folder_path=str(staging),
                repo_id=upload_repo,
                repo_type="dataset",
                commit_message=f"Publish MindIE Bench dataset ({len(rows)} trial receipts)",
            )
        os.replace(staging, output)
    summary = dict(output=str(output), rows=len(receipts), uploaded=uploaded)
    if "tasks" in data:
        summary["tasks"] = len(data["tasks"])
    if upload_repo:
        summary["repo_id"] = upload_repo
        summary["private"] = private
    return summary
