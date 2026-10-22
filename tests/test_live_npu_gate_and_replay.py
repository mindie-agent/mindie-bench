"""Tier 2 live A2/A3 qualification-gate and mechanism-replay verification.

The tests only create `/tmp/mindie-npu-test-<uuid>` sandboxes, read
`npu-smi`, compile throwaway C artifacts and inspect lock PIDs. They never
launch a model, reset a device or touch an existing workspace.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from mindie_bench.evaluation.knowledge_tiered import distill_selected
from mindie_bench.evaluation.knowledge_tiered_support import GradedUnit, Store
from mindie_bench.evaluation.mechanism_replay import (
    DECISION_SCHEMA,
    MAX_MERGED_BODY_BYTES,
    RELATED_BODY_BUDGET_BYTES,
    collect_m6_repository_evidence,
    evaluate_mechanism_replay,
    prepare_learner_prompt_view,
)
from mindie_bench.evaluation.strategy_selection import (
    load_combined_routing_index,
    prepare_strategy_selector_view,
    run_strategy_selection_pilot,
)
from mindie_bench.runner.qualification_gate import (
    CommandReceipt,
    G0Spec,
    G1Spec,
    GateCommand,
    GateState,
    QualificationGateRunner,
    incremental_build_command,
)

pytestmark = pytest.mark.live_npu

A2_HOST = os.environ.get("MINDIE_BENCH_A2_HOST", "root@173.125.1.2")
A3_HOST = os.environ.get("MINDIE_BENCH_A3_HOST", "root@192.168.13.165")
A2_CONTAINER_ENV = "MINDIE_BENCH_A2_CONTAINER"
A3_CONTAINER_ENV = "MINDIE_BENCH_A3_CONTAINER"
SSH_OPTIONS = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ConnectTimeout=8",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=30",
    "-o", "ControlPath=/tmp/mindie-bench-ssh-%C",
)


@dataclass(frozen=True)
class LiveTarget:
    host: str
    container: str


def _ssh(host: str, command: str, *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _remote_sh(target: LiveTarget, script: str, *, timeout: float = 30.0):
    return _ssh(
        target.host,
        shlex.join(["docker", "exec", "-i", target.container, "/bin/bash", "-lc", script]),
        timeout=timeout,
    )


def _discover_container(host: str, env_name: str) -> str:
    preferred = os.environ.get(env_name)
    listed = _ssh(host, "docker ps --format '{{.Names}}'", timeout=15)
    if listed.returncode != 0:
        pytest.skip(f"live host {host} is unavailable: {listed.stderr[-400:]}")
    names = [line for line in listed.stdout.splitlines() if line.strip()]
    candidates = ([preferred] if preferred else []) + names
    for name in dict.fromkeys(candidates):
        probe = _ssh(
            host,
            shlex.join([
                "docker", "exec", "-i", name, "/bin/bash", "-lc",
                "test -d /usr/local/Ascend && command -v python3 >/dev/null "
                "&& npu-smi info >/dev/null 2>&1",
            ]),
            timeout=15,
        )
        if probe.returncode == 0:
            return name
    pytest.skip(f"no live NPU container with Ascend runtime found on {host}")


@pytest.fixture(scope="session")
def live_targets():
    return {
        "A2": LiveTarget(A2_HOST, _discover_container(A2_HOST, A2_CONTAINER_ENV)),
        "A3": LiveTarget(A3_HOST, _discover_container(A3_HOST, A3_CONTAINER_ENV)),
    }


@pytest.fixture
def sandbox(live_targets):
    target = live_targets["A2"]
    root = f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex}"
    assert root.startswith("/tmp/mindie-npu-test-bench-")
    created = _remote_sh(target, f"mkdir -p {shlex.quote(root)}/build {shlex.quote(root)}/config")
    if created.returncode != 0:
        pytest.skip(f"could not create isolated sandbox on {target.host}: {created.stderr[-300:]}")
    try:
        yield target, root
    finally:
        _remote_sh(
            target,
            f"if [ -f {shlex.quote(root)}/live.lock ]; then "
            f"kill \"$(cat {shlex.quote(root)}/live.lock)\" 2>/dev/null || true; fi; "
            f"rm -rf -- {shlex.quote(root)}",
            timeout=20,
        )


class RemoteReader:
    """Prefetch a bounded remote artifact set into this test process."""

    def __init__(self, target: LiveTarget, paths: list[str]):
        code = (
            "import base64,json,os,sys\n"
            "out={}\n"
            "for p in json.loads(sys.argv[1]):\n"
            "    if os.path.isfile(p):\n"
            "        out[p]=base64.b64encode(open(p,'rb').read()).decode()\n"
            "print(json.dumps(out))\n"
        )
        command = shlex.join(["python3", "-c", code, json.dumps(paths)])
        result = _remote_sh(target, command)
        if result.returncode != 0:
            raise RuntimeError(f"remote artifact prefetch failed: {result.stderr[-500:]}")
        self._data = {
            key: base64.b64decode(value) for key, value in json.loads(result.stdout).items()
        }

    def exists(self, path: str) -> bool:
        return path in self._data

    def read_bytes(self, path: str) -> bytes:
        return self._data[path]


class RemoteExecutor:
    def __init__(self, target: LiveTarget):
        self.target = target

    def __call__(self, command: GateCommand) -> CommandReceipt:
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in command.env.items()
        )
        invocation = shlex.join(list(command.argv))
        script = f"{prefix} {invocation}" if prefix else invocation
        if command.cwd:
            script = f"cd {shlex.quote(command.cwd)} && {script}"
        try:
            result = _remote_sh(self.target, script, timeout=command.timeout_sec + 8)
        except subprocess.TimeoutExpired:
            return CommandReceipt(command.argv, None, timed_out=True,
                                  error="SSH transport timed out")
        return CommandReceipt(
            command.argv, result.returncode, result.stdout[-4_000_000:],
            result.stderr[-4_000_000:],
        )

    def pid_alive(self, pid: int) -> bool | None:
        result = _remote_sh(self.target, f"kill -0 {pid} 2>/dev/null", timeout=10)
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        return None


def _stage_native_artifacts(target: LiveTarget, root: str) -> dict[str, str]:
    script = f"""
set -eu
cd {shlex.quote(root)}
cat > build/flashmla.c <<'EOF'
int flashmla_kernel(int x) {{ return x + 1; }}
EOF
cc -c -fPIC -o build/flashmla.o build/flashmla.c
cat > build/flashmla_aicpu.json <<'EOF'
{{"FlashMLA":{{"opInfo":{{"engine":"DNN_VM_AICPU","kernelSo":"libtransformer_aicpu_kernels.so","functionName":"RunCpuKernel","computeCost":"100","workspaceSize":"100"}}}}}}
EOF
cat > operators.json <<'EOF'
{{"schema":"mindie-native-operator-manifest/1","operators":[{{"name":"FlashMLA","aicore":"build/flashmla.o","aicpu":"build/flashmla_aicpu.json"}}]}}
EOF
cat > config/op.json <<'EOF'
{{"op":"FlashMLA","tiling":{{"block":32}}}}
EOF
printf '{{"op":"FlashMLA"}}{{"stale":"tail"}}' > config/corrupt.json
cat > libkernel.c <<'EOF'
int flashmla_kernel(int x);
int exported(int x) {{ return flashmla_kernel(x); }}
EOF
cc -shared -fPIC -o libkernel.so build/flashmla.o libkernel.c
printf 'pid=99999999 pgid=99999999\\n' > stale.lock
nohup sleep 60 </dev/null >/dev/null 2>&1 &
echo $! > live.lock
cat > Makefile <<'EOF'
native-gate-opp:
	@case "$$MAKEFLAGS" in *-j*) ;; *) echo MAKEFLAGS_MISSING; exit 9;; esac
	@test "$$CMAKE_BUILD_PARALLEL_LEVEL" = "3"
	@test "$$OPS_CPU_NUMBER" = "3"
	@echo G1_BUILD_OK
EOF
printf 'READY\\n'
"""
    result = _remote_sh(target, script, timeout=60)
    assert result.returncode == 0, result.stderr
    return {
        "manifest": f"{root}/operators.json",
        "object": f"{root}/build/flashmla.o",
        "aicpu": f"{root}/build/flashmla_aicpu.json",
        "valid_config": f"{root}/config/op.json",
        "corrupt_config": f"{root}/config/corrupt.json",
        "library": f"{root}/libkernel.so",
        "stale_lock": f"{root}/stale.lock",
        "live_lock": f"{root}/live.lock",
    }


def _artifact_paths(artifacts: dict[str, str]) -> list[str]:
    return [
        artifacts[name]
        for name in (
            "manifest", "object", "aicpu", "valid_config", "corrupt_config", "library",
            "stale_lock", "live_lock",
        )
    ]


def _gate0_spec(artifacts: dict[str, str], *, config: str) -> G0Spec:
    return G0Spec(
        contract_tests=(
            GateCommand(
                ("python3", "-c", "print('CPU_CONTRACT_OK')"), timeout_sec=10,
                expect_stdout=("CPU_CONTRACT_OK",),
            ),
        ),
        manifests=(artifacts["manifest"],),
        generated_json=(config,),
        ldd_targets=(artifacts["library"],),
    )


def test_live_a2_preflight_g0_artifacts_and_g1_lock_and_parallelism(sandbox):
    target, root = sandbox
    preflight = _remote_sh(
        target, "npu-smi info | head -20; python3 -c 'import torch_npu; print(torch_npu.__version__)'"
    )
    assert preflight.returncode == 0, preflight.stderr
    assert "910B4" in preflight.stdout
    artifacts = _stage_native_artifacts(target, root)
    runner = QualificationGateRunner(
        executor=RemoteExecutor(target),
        reader=RemoteReader(target, _artifact_paths(artifacts)),
        pid_alive=RemoteExecutor(target).pid_alive,
    )
    g0 = runner.verify_g0_artifact_and_contract(_gate0_spec(artifacts, config=artifacts["valid_config"]))
    assert g0.state is GateState.COMPLETED, g0.to_dict()
    corrupt = runner.verify_g0_artifact_and_contract(
        _gate0_spec(artifacts, config=artifacts["corrupt_config"])
    )
    assert corrupt.state is GateState.FAILED
    assert any("trailing or malformed JSON" in reason for reason in corrupt.reasons)

    locked = runner.build_and_verify_g1_incremental_gate(
        G1Spec(
            source_root=root,
            jobs=3,
            build_sentinel=("G1_BUILD_OK",),
            debug_lock_paths=(artifacts["stale_lock"], artifacts["live_lock"]),
            eager_qualification=GateCommand(
                ("python3", "-c", "print('EAGER_OK')"), timeout_sec=10,
                expect_stdout=("EAGER_OK",),
            ),
            aclgraph_qualification=GateCommand(
                ("python3", "-c", "print('ACLGRAPH_OK')"), timeout_sec=10,
                expect_stdout=("ACLGRAPH_OK",),
            ),
        )
    )
    assert locked.state is GateState.FAILED
    assert locked.metrics["locked"] is True
    assert next(lock for lock in locked.stages[0].evidence["locks"]
                if lock["path"] == artifacts["stale_lock"])["state"] == "completed"
    assert next(lock for lock in locked.stages[0].evidence["locks"]
                if lock["path"] == artifacts["live_lock"])["state"] == "failed"

    unlocked = runner.build_and_verify_g1_incremental_gate(
        G1Spec(
            source_root=root,
            jobs=3,
            build_sentinel=("G1_BUILD_OK",),
            debug_lock_paths=(artifacts["stale_lock"],),
            eager_qualification=GateCommand(
                ("python3", "-c", "print('EAGER_OK')"), timeout_sec=10,
                expect_stdout=("EAGER_OK",),
            ),
            aclgraph_qualification=GateCommand(
                ("python3", "-c", "print('ACLGRAPH_OK')"), timeout_sec=10,
                expect_stdout=("ACLGRAPH_OK",),
            ),
        )
    )
    assert unlocked.state is GateState.COMPLETED, unlocked.to_dict()
    build = incremental_build_command(source_root=root, jobs=3, timeout_sec=60)
    assert build.env == {
        "MAKEFLAGS": "-j3",
        "CMAKE_BUILD_PARALLEL_LEVEL": "3",
        "OPS_CPU_NUMBER": "3",
    }


def test_live_a3_dual_die_preflight_and_g0_artifacts(live_targets):
    target = live_targets["A3"]
    preflight = _remote_sh(target, "npu-smi info -l; npu-smi info | head -8")
    assert preflight.returncode == 0, preflight.stderr
    total = int(re.search(r"Total Count\s*:\s*(\d+)", preflight.stdout).group(1))
    chips = [int(value) for value in re.findall(r"Chip Count\s*:\s*(\d+)", preflight.stdout)]
    assert total == 8 and len(chips) == 8 and sum(chips) == 16
    assert "Ascend910" in preflight.stdout
    root = f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex}"
    assert root.startswith("/tmp/mindie-npu-test-bench-")
    try:
        created = _remote_sh(target, f"mkdir -p {shlex.quote(root)}/build {shlex.quote(root)}/config")
        assert created.returncode == 0, created.stderr
        artifacts = _stage_native_artifacts(target, root)
        runner = QualificationGateRunner(
            executor=RemoteExecutor(target),
            reader=RemoteReader(target, _artifact_paths(artifacts)),
            pid_alive=RemoteExecutor(target).pid_alive,
        )
        result = runner.verify_g0_artifact_and_contract(
            _gate0_spec(artifacts, config=artifacts["valid_config"])
        )
        assert result.state is GateState.COMPLETED, result.to_dict()
        manifest_stage = next(stage for stage in result.stages if stage.name == "manifest")
        assert manifest_stage.evidence["aicore_objects_present"] == 1
    finally:
        _remote_sh(
            target,
            f"if [ -f {shlex.quote(root)}/live.lock ]; then "
            f"kill \"$(cat {shlex.quote(root)}/live.lock)\" 2>/dev/null || true; fi; "
            f"rm -rf -- {shlex.quote(root)}",
            timeout=20,
        )


def _live_replay_bodies() -> tuple[str, str]:
    branch = (
        "Live A2 condition: zeroer remains excluded for the 910B4 decode branch when "
        "prefix_cache=off. PV2, Score2, S256 and vmula stay in the compact exclusion table.\n"
    )
    before = (branch * 90) + (
        "Human correction: the old baseline means the immediately previous run. 4:1 and "
        "parent/subslot remain superseded and inactive.\n"
    )
    after = (
        "Live A2 condition retained: the 910B4 decode branch keeps prefix_cache=off as the "
        "primary path; zeroer remains excluded and is not reopened without a changed condition.\n\n"
        "Compact exclusion table: PV2, Score2, S256 and vmula were rejected by paired runs. "
        "4:1 and parent/subslot remain superseded; a prefix-cache-enabled branch is recorded "
        "only as scoped and untested. Human corrections are not technical proof.\n"
    )
    return before, after


def test_live_a2_end_to_end_mechanism_replay_and_graded_repository(sandbox, tmp_path):
    target, _root = sandbox
    preflight = _remote_sh(target, "npu-smi info | grep -m1 -o '910B4'")
    assert preflight.returncode == 0 and "910B4" in preflight.stdout
    soc = "Ascend" + preflight.stdout.strip()
    gitleaks = shutil.which("gitleaks") or next(
        (str(path) for path in Path.home().glob(".cache/mindie-agent/gitleaks/*/*/gitleaks")),
        None,
    )
    if not gitleaks:
        pytest.skip("installed gitleaks is required for the production organizer")
    before_body, after_body = _live_replay_bodies()
    assert len(before_body.encode()) <= RELATED_BODY_BUDGET_BYTES
    body_sha = hashlib.sha256(before_body.encode()).hexdigest()
    metadata = {
        "schema": "mindie-graded-unit/1",
        "id": "unit-live-a2-history-conditioned-exclusion",
        "revision": 1,
        "tier": "L3",
        "category": "workflow",
        "evidence_strength": "single_observation",
        "domain_path": "diagnostics/live",
        "title": "Live A2 history-conditioned exclusion",
        "inherits_from": [],
        "conditions": {"soc": soc, "workload_scope": "decode", "prefix_cache": "off"},
        "triggers": {"error_signatures": [], "keywords": ["zeroer", "prefix", "live"]},
        "provenance": {
            "source_type": "session_distill",
            "source_locator": f"live-a2-container-{target.container}:/tmp/mindie-npu-test-preflight.json",
            "source_digest": body_sha,
            "evidence_refs": ["t4-11236", "t4-16041", "t4-17326"],
            "supersedes": [],
        },
    }
    unit = GradedUnit.from_parts(metadata, before_body,
                                 f"{metadata['domain_path']}/{metadata['id']}.md")
    store = Store(tmp_path / "store", "vllm-ascend")
    store.graded.install([unit])
    before_view = store.graded.explain(store.graded.ref(unit.id))
    before_body = before_view["content"]
    before_active = len(store.graded._load().units)
    evidence_content = (
        "Verified live milestone: zeroer stays excluded on the same A2 conditions; 4:1 and "
        "parent/subslot remain superseded and the previous-run baseline remains authoritative."
    )
    session = "e" * 64
    proposal_meta = deepcopy({key: value for key, value in unit.metadata.items() if key != "revision"})
    proposal_meta["provenance"]["source_digest"] = hashlib.sha256(evidence_content.encode()).hexdigest()
    proposal = {
        "schema_version": "mindie-distill-decision/1",
        "session_digest": session,
        "evidence_inputs": ["authorized_transcript"],
        "action": "propose_graded_units",
        "drop_reason": None,
        "units": [{
            "target_unit_id": unit.id,
            "target_tier": proposal_meta["tier"],
            "category": proposal_meta["category"],
            "evidence_strength": proposal_meta["evidence_strength"],
            "domain_path": proposal_meta["domain_path"],
            "operation": "merge_into_existing",
            "frontmatter": proposal_meta,
            "markdown_body": after_body,
        }],
    }
    applied = distill_selected(
        store=store,
        inputs=[{
            "kind": "authorized_transcript",
            "outcome": "verified",
            "source_locator": metadata["provenance"]["source_locator"],
            "content": evidence_content,
            "evidence_refs": list(metadata["provenance"]["evidence_refs"]),
            "repetitions": 1,
        }],
        session_digest=session,
        task_state="verified_milestone",
        proposal=proposal,
        redactor_executable=gitleaks,
    )
    assert applied["status"] == "applied", applied
    current = store.graded.explain(store.graded.ref(unit.id))
    assert current["revision"] == 2
    assert current["content"] == after_body
    assert current["frontmatter"]["conditions"] == metadata["conditions"]
    assert len(current["content"].encode()) <= MAX_MERGED_BODY_BYTES
    evidence = collect_m6_repository_evidence(
        store.graded,
        unit_id=unit.id,
        old_conditions=metadata["conditions"],
        new_query="prefix cache enabled branch scoped untested",
        old_query="zeroer excluded 4:1 parent subslot correction",
        before_body=before_body,
        before_active_units=before_active,
    )
    assert evidence["old_ref_status"] == "removed_or_superseded"
    assert evidence["new_branch_recalled"] is True
    assert evidence["old_conditions_recalled"] is True
    decision = {
        "schema_version": DECISION_SCHEMA,
        "case_id": "M6-01",
        "action": "revise_existing_unit",
        "hypothesis_id": "H1_revise_compact",
        "evidence_locators": ["T4:11236", "T4:24224", "T1:ACTIVE_STATE.md"],
        "revision": {
            "target_unit_id": unit.id,
            "previous_revision": 1,
            "operation": "merge_into_existing",
            "new_unit_created": False,
            "active_unit_count_delta": 0,
            "related_unit_snapshot": {
                "body_complete": True,
                "body_excerpt_bytes": len(before_body.encode()),
                "body_sha256": hashlib.sha256(before_body.encode()).hexdigest(),
            },
            "append_only": False,
            "rewrite_to_compact": True,
            "before_body_bytes": len(before_body.encode()),
            "after_body_bytes": len(after_body.encode()),
            "net_body_bytes": len(after_body.encode()) - len(before_body.encode()),
            "max_body_bytes": MAX_MERGED_BODY_BYTES,
            "body_sha256": hashlib.sha256(after_body.encode()).hexdigest(),
            "old_conditions": dict(metadata["conditions"]),
            "new_branch_query": "prefix cache enabled branch scoped untested",
            "old_query": "zeroer excluded 4:1 parent subslot correction",
            "superseded_layouts": ["4:1", "parent/subslot"],
            "revived_layouts": [],
            "excluded_hypotheses": ["zeroer"],
            "reopened_hypotheses": [],
            "reopen_conditions_changed": False,
        },
        "evidence": {
            "counter_evidence_retained": [
                {"claim": "zeroer excluded", "source": "T4:11236"},
                {"claim": "4:1 superseded", "source": "T4:17326"},
                {"claim": "parent/subslot superseded", "source": "T4:19868"},
            ],
            "human_corrections": ["T4:16041", "T4:24062"],
            "technical_proofs": ["T4:17326", "T4:19868"],
            "separated": True,
        },
    }
    view = prepare_learner_prompt_view("M6-01")
    assert set(view) == {
        "schema_version", "case_id", "pre_decision_facts", "competing_hypotheses",
        "provenance_locators",
    }
    result = evaluate_mechanism_replay(
        "M6-01", decision, repository=store.graded,
        revision_context={"before_body": before_body, "before_active_units": before_active,
                          "before_conditions": unit.metadata["conditions"]},
    )
    assert result["state"] == "completed", result


def test_live_a2_and_a3_strategy_selection_navigation_and_abc_cells(live_targets):
    a2_target = live_targets["A2"]
    a3_target = live_targets["A3"]
    a2_root = f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex}"
    a3_root = f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex}"
    try:
        a2_probe = _remote_sh(
            a2_target,
            f"mkdir -p {shlex.quote(a2_root)} && npu-smi info | grep -m1 -o '910B4'",
        )
        a3_probe = _remote_sh(
            a3_target,
            f"mkdir -p {shlex.quote(a3_root)} && npu-smi info -l",
        )
        assert a2_probe.returncode == 0 and "910B4" in a2_probe.stdout
        assert a3_probe.returncode == 0
        chips = [int(value) for value in re.findall(r"Chip Count\s*:\s*(\d+)", a3_probe.stdout)]
        assert len(chips) == 8 and sum(chips) == 16

        workspace_root = Path(__file__).resolve().parents[2]
        vllm_root = workspace_root / "knowledge-vllm-ascend"
        operators_root = workspace_root / "knowledge-operators"
        index = load_combined_routing_index((vllm_root, operators_root))
        view_b = prepare_strategy_selector_view("S1-F2-boundary", arm="B", routing_index=index)
        view_c = prepare_strategy_selector_view("S1-F2-boundary", arm="C", routing_index=index)
        assert "pre_decision_observations" not in view_b
        assert "pre_decision_observations" in view_c
        assert all("body" not in card for card in view_c["candidate_navigation_cards"])

        report = run_strategy_selection_pilot(corpus_roots=(vllm_root, operators_root), repetitions=2)
        assert report["readiness"]["ready"] is True
        assert report["comparisons"]["C_vs_B"]["selection_reasonable_delta"] > 0
        assert report["comparisons"]["C_vs_B"]["false_adoption_delta"] < 0
        assert report["comparisons"]["C_vs_A"]["pass_rate_delta"] > 0
    finally:
        _remote_sh(a2_target, f"rm -rf -- {shlex.quote(a2_root)}", timeout=15)
        _remote_sh(a3_target, f"rm -rf -- {shlex.quote(a3_root)}", timeout=15)

