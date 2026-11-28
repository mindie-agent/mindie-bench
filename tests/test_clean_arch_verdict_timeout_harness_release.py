"""Orthogonal behavioral tests for `mindie-bench` verdict honesty, verifier timeout arithmetic, boundary provenance, and `mindie-harness` R3 release support.

[Role]
  Verify via real subprocess, trial runner, and consumer release-validator execution that:
  1. Settled `PASS`/`FAIL` oracle verdicts are never overwritten to `INFRA` when
     post-verifier trajectory scoring or workspace/service cleanup raises (`cleanup_warning`,
     `metrics_incomplete=True`).
  2. `remote_verifier` expands `outer_timeout = inner + 20.0`, `private_remote_judge.py`
     grants the full inner budget without subtracting `inventory_grace`, and `RemoteSlot.verify`
     accepts `timeout > slot.timeout`.
  3. `run_matrix` reuses verified runtime-import provenance across identical cells in one run
     while persisting `runtime-preflight.json` in every trial directory.
  4. `mindie-agent/harness` formal `0.1.17` baseline (`v0.1.4`) in `release-policy.json` and
     `BIRTH_COMPONENTS` consumer compatibility (`prepare_plugin_metadata`, `validate_release`)
     pass against `knowledge@3c478a19`.

[Invariants]
  * Runs under sandboxed `tmp_path` isolation (`conftest.py`).

[Timeouts & Locks]
  * Completes within Tier 1 (`< 15s` per test).

[Side-Effect Policy]
  * Local temporary files and subprocesses only; zero network calls.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_remote_verifier import LocalJudgeTransport

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.models import Group, OracleContract, RemoteConfig, TerminalStatus
from mindie_bench.runner import matrix_runner, orchestrator, verifier
from mindie_bench.runner.remote_slot import RemoteSlot


def test_post_verifier_scoring_and_cleanup_errors_preserve_confirmed_pass_verdict(
    case, tmp_path, monkeypatch
):
    run_root = tmp_path / "verdict-honesty-run"
    run_root.mkdir()

    def boom_cleanup(self):
        raise OSError("simulated worktree cleanup EBUSY")

    def boom_score(**_kwargs):
        raise RuntimeError("simulated auxiliary trajectory parser failure")

    monkeypatch.setattr(orchestrator.WorktreeSlot, "cleanup", boom_cleanup)
    monkeypatch.setattr(orchestrator, "score_trial_trajectory", boom_score)

    receipt = orchestrator.run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=run_root
    )
    assert receipt.terminal_status == TerminalStatus.PASS
    assert receipt.primary_metrics["resolved"] is True
    assert receipt.primary_metrics["reward"] == 1.0
    assert receipt.auxiliary_metrics["business_terminal_status"] == TerminalStatus.PASS.value
    assert receipt.auxiliary_metrics["metrics_incomplete"] is True
    assert "workspace_cleanup" in receipt.auxiliary_metrics["cleanup_warning"]
    stages = {err["stage"] for err in receipt.errors}
    assert {"scoring", "workspace_cleanup"} <= stages


def test_remote_verifier_grants_full_inner_budget_and_slot_verify_allows_outer_margin(
    case, tmp_path, monkeypatch
):
    root = tmp_path / "slow-judge-task"
    (root / "verifier").mkdir(parents=True)
    (root / "oracle").mkdir()
    # Sleep 1.2s with inner timeout=2.0s: under the old `overall_deadline - min(5, timeout/2)`
    # (= 1.0s) bug, this valid judge timed out prematurely!
    (root / "verifier/test.sh").write_text(
        f'"{Path(sys.executable).as_posix()}" -c "import time; time.sleep(1.2)"\n'
        'printf \'{"reward":1}\\n\'\n',
        encoding="utf-8",
        newline="\n",
    )
    (root / "oracle/golden.py").write_text("ANSWER = 42\n", encoding="utf-8", newline="\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    LocalJudgeTransport.tmp = tmp_path / "private-tmp"
    LocalJudgeTransport.tmp.mkdir()
    from mindie_bench.runner.host_platform import write_command_alias

    write_command_alias(LocalJudgeTransport.tmp / "bin", "python3", [sys.executable, "-X", "utf8"])
    LocalJudgeTransport.workspace = tmp_path / "workspace"
    LocalJudgeTransport.workspace.mkdir()

    transport = LocalJudgeTransport()
    contract = OracleContract()
    frozen = freeze_oracle(root, contract)
    val, _rec = verifier.remote_verifier(root, transport, trial, 2.0, frozen, contract, None)
    assert val.reward == 1.0
    assert transport.calls[0][2] == 22.0  # inner 2.0s + 20.0s outer margin

    # RemoteSlot.verify must allow verifier outer timeout > slot.timeout (agent budget)
    cfg = RemoteConfig(
        host="fixture", container="fixture", repo_path="/fixture",
        slots_root=str(tmp_path / "slots"), devices=[0], probe_command=["probe"],
    )
    env = case["spec"].environment["l2"].model_copy(update={"npu_model": "Ascend910", "npu_count": 1})
    slot = RemoteSlot(cfg, env, "trial-slot", [0], 5, artifact_dir=tmp_path)
    seen_req = {}

    def fake_request(payload, wait_timeout, **_kw):
        seen_req["timeout"] = payload["timeout"]
        seen_req["wait"] = wait_timeout
        return {"exit_code": 0, "stdout_b64": "", "stderr_b64": ""}

    monkeypatch.setattr(slot, "_request", fake_request)
    slot.verify(b"echo ok\n", timeout=25.0)
    assert seen_req == {"timeout": 25.0, "wait": 50.0}


def test_matrix_runner_reuses_runtime_import_boundary_across_cells(case, tmp_path, monkeypatch):
    calls = []

    def counting_probe(profile, read_paths, trial_dir, expected_digest):
        calls.append(trial_dir.name)
        from mindie_bench.io import write_json

        observed = {
            "benchmark_source_sha256": expected_digest,
            "executable": sys.executable,
            "package_root": str(Path(orchestrator.__file__).resolve().parents[1]),
            "modules": {},
        }
        write_json(trial_dir / "runtime-preflight.json", {"status": "completed", "observed": observed})
        return observed

    monkeypatch.setattr(orchestrator, "probe_runtime_import", counting_probe)
    monkeypatch.setattr(orchestrator, "_DEFAULT_PROBE_RUNTIME_IMPORT", counting_probe)

    out = tmp_path / "matrix-out"
    receipts = matrix_runner.run_matrix(
        [case["task"]], case["config"],
        tier="l2", groups=[Group.G0], repetitions=3, concurrency=1, output=out,
    )
    assert len(receipts) == 3
    assert len(calls) == 1, "identical runtime boundary in one matrix run should probe once"
    for r in receipts:
        pf = json.loads((Path(r.artifacts["private_trial_dir"]) / "runtime-preflight.json").read_text())
        assert pf["status"] == "completed"


def test_mindie_harness_birth_release_gate_and_pr_checklist_bench_profile(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / ".github" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import consumer_contract
        import prepare_plugin_metadata

        policy = json.loads((repo_root / "release-policy.json").read_text(encoding="utf-8"))
        harness_entry = policy["repositories"]["mindie-agent/harness"]
        assert harness_entry == {
            "package": "mindie-harness",
            "version": "0.1.4",
            "revision": "14313b8ba58c888b20c9c216ec41d14f5e62f4f7",
        }
        assert len(policy["repositories"]) == 18
        assert all(
            row.get("version") and len(str(row.get("revision") or "")) == 40
            for row in policy["repositories"].values()
        )

        spec_mk = importlib.util.find_spec("mindie_knowledge")
        consumer_src = (
            Path(spec_mk.origin).resolve().parents[1]
            if spec_mk and spec_mk.origin
            else repo_root.parents[1] / "knowledge"
        )
        if not (consumer_src / "mindie_knowledge").is_dir():
            pytest.skip("mindie_knowledge consumer package unavailable")

        # 1. prepare_plugin_metadata treats mindie-agent/harness as a shared wheel component (returns None)
        assert (
            prepare_plugin_metadata.prepare(
                consumer=str(consumer_src),
                repository="mindie-agent/harness",
                version="0.1.0",
                revision="a" * 40,
                source=str(tmp_path),
                assets=str(tmp_path),
            )
            is None
        )

        # 2. validate_release validates mindie-agent/harness v0.1.0 via Catalog.latest and restores COMPONENTS
        rev = "b" * 40
        pyproject_bytes = b'[project]\nname = "mindie-harness"\nversion = "0.1.0"\n'
        git_blob_sha = hashlib.sha1(
            b"blob " + str(len(pyproject_bytes)).encode() + b"\0" + pyproject_bytes
        ).hexdigest()
        bench_json = json.dumps({
            "schema_version": "mindie-release-benchmark/2",
            "repository": "mindie-agent/harness",
            "version": "0.1.0",
            "release_gate": True,
            "blocking_reasons": [],
            "source": {"revision": rev, "dirty": False},
            "source_after": {"revision": rev, "dirty": False},
        }).encode()
        asset_bytes = {
            "mindie_harness-0.1.0-py3-none-any.whl": b"wheel-bytes",
            "mindie_harness-0.1.0.tar.gz": b"sdist-bytes",
            "benchmark-results.json": bench_json,
            "benchmark-results.md": b"# benchmark\n",
            "benchmark-SHA256SUMS": b"sums\n",
        }
        sums_txt = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in sorted(asset_bytes.items())
        ).encode()
        asset_bytes["SHA256SUMS"] = sums_txt
        release_payload = {
            "tag_name": "v0.1.0",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": k, "size": len(v)} for k, v in asset_bytes.items()],
        }

        def fake_fetch(path: str):
            if path == "repos/mindie-agent/harness/git/ref/tags/v0.1.0":
                return {"object": {"type": "commit", "sha": rev}}
            if path == f"repos/mindie-agent/harness/contents/pyproject.toml?ref={rev}":
                return {
                    "encoding": "base64",
                    "size": len(pyproject_bytes),
                    "sha": git_blob_sha,
                    "content": base64.b64encode(pyproject_bytes).decode(),
                }
            raise AssertionError(f"unexpected GitHub API path: {path}")

        _, components_ref = consumer_contract.load_consumer(consumer_src)
        had_before = "mindie-harness" in components_ref
        selected = consumer_contract.validate_release(
            consumer_src, "mindie-agent/harness", "0.1.0", rev,
            release_payload, lambda name: asset_bytes[name], fake_fetch,
        )
        assert selected == {
            "repository": "mindie-agent/harness",
            "version": "0.1.0",
            "revision": rev,
        }
        assert ("mindie-harness" in components_ref) == had_before
    finally:
        sys.path.remove(str(scripts_dir))

    # 3. Pre-PR Checklist bench profile gate
    gate_path = scripts_dir / "check_pr_checklist.py"
    spec = importlib.util.spec_from_file_location("bench_pr_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gate)
    raw_tpl = (repo_root / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert gate.validate_pr_body(raw_tpl, profile="bench")
