"""Archived replays must judge the recorded side and retain historical unknowns."""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

from mindie_bench.evaluation.reevaluate import reevaluate, sha256


def test_remote_replay_restores_patch_and_helpers_and_does_not_rewrite_unknown(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=5)
    (repo / "candidate.py").write_text("def calculate(x):\n    return 0\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, timeout=5)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=t@local",
                    "commit", "-qm", "baseline"], check=True, timeout=5)
    base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=5).decode().strip()
    (repo / "candidate.py").write_text("from helper import double\ndef calculate(x):\n    return double(x)\n")
    patch = subprocess.check_output(["git", "-C", str(repo), "diff"], timeout=5)
    task = tmp_path / "tasks/arithmetic"
    (task / "verifier").mkdir(parents=True)
    (task / "task.toml").write_text(f'[environment.l3]\nrepo_path = "{repo}"\n')
    (task / "verifier/test.sh").write_text(
        '#!/bin/sh\nexec "$MINDIE_BENCH_PYTHON" -B -c '\
        "'import json; from candidate import calculate; "
        'ok = all(calculate(x) == x * 2 for x in [-17, 0, 93]); '
        'print(json.dumps({"reward": float(ok)})); raise SystemExit(not ok)' + "'\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "remote_final.patch").write_bytes(patch)
    (trial / "remote_cleanup.json").write_text(json.dumps({"cleanup": {
        "patch_b64": base64.b64encode(patch).decode(), "errors": [],
        "files_modified": ["candidate.py", "helper.py"],
        "untracked": {"helper.py": base64.b64encode(b"def double(x):\n    return x + x\n").decode()},
    }}))
    # This local mirror is deliberately wrong; only the remote artifacts were judged.
    local = tmp_path / "local"
    local.mkdir()
    (local / "candidate.py").write_text("raise RuntimeError('wrong source')\n")
    shutil.make_archive(str(trial / "final_workspace"), "gztar", local)
    receipt = trial / "trial_receipt.json"
    receipt.write_text(json.dumps({"terminal_status": "unknown", "primary_metrics": {
        "resolved": None, "reward": None}, "environment_fingerprint": {
        "base_commit": base, "budget": {"timeout_sec": 17}}, "auxiliary_metrics": {
        "oracle_manifest": {"files": {"verifier/test.sh": "historical-sha"}}}}))
    before = receipt.read_bytes()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"trials": [{"primary": True, "trial_id": "trial",
        "task_id": "arithmetic", "harness": "codex", "group": "G2", "tier": "l3",
        "oracle_scope": "ast_only_oracle", "receipt_path": str(receipt),
        "receipt_sha256": sha256(receipt)}]}))
    result = reevaluate(inventory, task.parent, tmp_path / "report", Path(sys.executable))
    row = result["trials"][0]
    assert row["reevaluation"]["reward"] == 1.0
    assert row["original_status"] == "unknown" and row["original_resolved"] is None
    assert row["reevaluation"]["speedup_ratio"] is None
    assert row["candidate_source"]["untracked_files"] == ["helper.py"]
    assert receipt.read_bytes() == before
    assert result["summary"]["primary_trials"] == 1
    assert result["comparisons"][0]["causal_performance_comparison"] == "not_established_by_zero_model_replay"

    # Real postflight writes this marker only after all untracked bytes were captured.
    # A partial snapshot is missing evidence, even when the available patch would pass.
    cleanup_path = trial / "remote_cleanup.json"
    cleanup = json.loads(cleanup_path.read_text())
    cleanup["cleanup"].pop("files_modified")
    cleanup["cleanup"]["errors"] = ["untracked artifact bound exceeded"]
    cleanup_path.write_text(json.dumps(cleanup))
    partial = reevaluate(inventory, task.parent, tmp_path / "partial-report", Path(sys.executable))
    observation = partial["trials"][0]["reevaluation"]
    assert observation["status"] == "not_evaluated" and observation["reward"] is None
    assert observation["failure_category"] == "artifact_or_execution_error"
    saved = json.loads((tmp_path / "partial-report/trial/result.json").read_text())
    assert saved["original_receipt_unchanged"] is True
    assert receipt.read_bytes() == before
