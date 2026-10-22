"""Coordinator-backed RSI execution and measurement-evidence contracts.

These are contract tests for the executor and the TaskClient adapter. The
concurrency and genuine-ID cases use the installed coordinator's real SQLite
lease state machine in a temporary directory; they do not touch an NPU. The
TaskClient cases use a protocol double and explicitly do not claim hardware
execution.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from mindie_bench.runner.rsi_execution import (
    RSIExecutionUnavailable,
    TaskClientRSIExecutor,
    execute_rsi_compile_and_timing,
)
from mindie_bench.runner.rsi_measurement import _assigned_devices, _role_identities

coordinator_module = pytest.importorskip("mindie_coordinator.host.mindie_npu_coordination")
NpuCoordinator = coordinator_module.NpuCoordinator

FREE = {"status": "ok", "devices": [0, 1], "busy": {}, "free": [0, 1]}


from test_rsi_execution import (  # noqa: E402
    _CoordinatorAuthority,
    _timing_result,
)


def test_cpu_only_compile_waves_overlap_under_the_declared_slot_limit(tmp_path):
    barrier = threading.Barrier(4)
    observed: list[str] = []
    active = {"now": 0, "maximum": 0}
    lock = threading.Lock()

    def compile_runner(candidate):
        with lock:
            active["now"] += 1
            active["maximum"] = max(active["maximum"], active["now"])
            observed.append(candidate["candidate_id"])
        barrier.wait(timeout=5)
        with lock:
            active["now"] -= 1
        return {"compile_ok": True, "run_id": f"compile-{candidate['candidate_id']}"}

    candidates = [{"candidate_id": f"c{index}", "oracle_passed": True} for index in range(4)]
    result = execute_rsi_compile_and_timing(
        candidates, compile_runner=compile_runner,
        state_dir=tmp_path / "compile", max_compile_slots=4,
    )
    assert active["maximum"] == 4
    assert set(observed) == {"c0", "c1", "c2", "c3"}
    assert result["execution_authority"]["available"] is False
    # Compile proof alone is not a measurement: without a timing lease the
    # candidates stay inconclusive and are not distill-ready.
    assert all(row["verdict"] == "inconclusive" for row in result["evaluations"])
    assert result["timing_order"] == ["c0", "c1", "c2", "c3"]
    assert all(
        not row["gates"]["gate2_measurement_guard"]["passed"]
        for row in result["evaluations"]
    )


def test_concurrent_coordinator_ownership_blocks_duplicate_timing(tmp_path):
    authority_a = _CoordinatorAuthority(tmp_path / "coordinator")
    authority_b = _CoordinatorAuthority(tmp_path / "coordinator")
    entered = threading.Event()
    release = threading.Event()
    active = {"now": 0, "maximum": 0}
    lock = threading.Lock()

    def timing_a(candidate):
        lease = candidate["coordinator_lease"]
        with lock:
            active["now"] += 1
            active["maximum"] = max(active["maximum"], active["now"])
        entered.set()
        assert release.wait(timeout=5)
        with lock:
            active["now"] -= 1
        return _timing_result(lease)

    def timing_b(candidate):
        raise AssertionError("duplicate timing ran without coordinator ownership")

    candidate = {"candidate_id": "repro", "compile_ok": True, "oracle_passed": True}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            execute_rsi_compile_and_timing,
            [candidate], timing_runner=timing_a, lease_authority=authority_a,
            state_dir=tmp_path / "shared-state",
        )
        assert entered.wait(timeout=5), "first timing never acquired its coordinator lease"
        second = execute_rsi_compile_and_timing(
            [dict(candidate)], timing_runner=timing_b, lease_authority=authority_b,
            state_dir=tmp_path / "shared-state",
        )
        release.set()
        accepted = first.result(timeout=10)

    assert active["maximum"] == 1
    assert accepted["evaluations"][0]["verdict"] == "supported"
    assert accepted["evaluations"][0]["run_id"] == authority_a.issued[0]
    assert second["evaluations"][0]["verdict"] == "inconclusive"
    assert "coordinator" in second["evaluations"][0]["failure_reason"]
    assert second["timing_attempts"][0]["acquired"] is False


def test_timing_failure_and_lease_cleanup_remain_visible(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")
    calls = []

    def timing(candidate):
        calls.append(candidate["candidate_id"])
        raise RuntimeError("device timing fault")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "broken", "compile_ok": True, "oracle_passed": True}],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "failure",
    )
    entry = result["evaluations"][0]
    assert calls == ["broken"]
    assert entry["verdict"] == "inconclusive"
    assert "device timing fault" in entry["failure_reason"]
    assert result["timing_attempts"][0]["acquired"] is True
    assert result["timing_attempts"][0]["released"] is True
    assert authority.snapshot(authority.issued[0])["state"] == "released"


def test_missing_run_workspace_and_guard_evidence_cannot_distill(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")

    def timing(candidate):
        cid = candidate["candidate_id"]
        lease = candidate["coordinator_lease"]
        if cid == "missing-run":
            return {
                "oracle_passed": True, "measurement_passed": True,
                "speedup_ratio": 1.2, "ablation_retained": True,
            }
        if cid == "missing-workspace":
            return _timing_result(lease)
        if cid == "missing-guard-values":
            return _timing_result(
                lease, workspace_delta_bytes=0,
                guard_workloads=[{"name": "uniform-guard"}],
            )
        return _timing_result(
            lease, workspace_delta_bytes=0,
            guard_workloads=[{
                "name": "uniform-guard", "baseline_ms": 100.0,
                "candidate_ms": 101.0, "max_regression_ratio": 1.05,
            }],
        )

    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "missing-run", "compile_ok": True, "oracle_passed": True},
            {"candidate_id": "missing-workspace", "compile_ok": True, "oracle_passed": True},
            {"candidate_id": "missing-guard-values", "compile_ok": True, "oracle_passed": True},
            {"candidate_id": "measured", "compile_ok": True, "oracle_passed": True},
        ],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "evidence",
        max_workspace_delta_bytes=0,
        required_guard_workloads=["uniform-guard"],
    )
    entries = {row["candidate_id"]: row for row in result["evaluations"]}

    assert entries["missing-run"]["verdict"] == "inconclusive"
    assert entries["missing-run"]["run_id"] == ""
    assert entries["missing-run"]["distill_ready"] is False
    assert "run_id" in entries["missing-run"]["failure_reason"]
    assert entries["missing-workspace"]["verdict"] == "inconclusive"
    assert entries["missing-workspace"]["distill_ready"] is False
    assert "workspace_delta_not_measured" in entries["missing-workspace"]["failure_reason"]
    assert entries["missing-guard-values"]["verdict"] == "inconclusive"
    assert entries["missing-guard-values"]["distill_ready"] is False
    assert "guard_measurement_missing:uniform-guard" in entries["missing-guard-values"]["failure_reason"]
    assert entries["measured"]["verdict"] == "supported"
    assert entries["measured"]["run_id"] in authority.issued
    assert result["best_candidate"]["candidate_id"] == "measured"
    assert result["best_candidate"]["run_id"] == entries["measured"]["run_id"]


def test_a_reported_guard_without_values_is_not_a_pass(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")

    def timing(candidate):
        return _timing_result(
            candidate["coordinator_lease"],
            guard_workloads=[{"name": "uniform-guard", "passed": True}],
        )

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "guard", "compile_ok": True, "oracle_passed": True}],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "guard",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "guard_measurement_missing:uniform-guard" in entry["failure_reason"]


def test_prefilled_timing_claims_are_invalid_without_an_executed_lease(tmp_path):
    def timing_runner(candidate):
        raise AssertionError("no authority means no timing callback")

    result = execute_rsi_compile_and_timing(
        [{
            "candidate_id": "prefilled", "compile_ok": True, "oracle_passed": True,
            "measurement_passed": True, "run_id": "invented-timing-id",
            "speedup_ratio": 1.2, "ablation_retained": True,
            "workspace_delta_bytes": 0,
        }],
        timing_runner=timing_runner,
        state_dir=tmp_path / "prefilled",
        max_workspace_delta_bytes=0,
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert entry["run_id"] == ""
    assert entry["distill_ready"] is False
    assert "coordinator_lease_unavailable" in entry["failure_reason"]
    assert result["timing_attempts"][0]["acquired"] is False


def test_declared_workspace_cap_requires_measurement_but_unrequested_gates_are_not_invented(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")

    def timing(candidate):
        return _timing_result(candidate["coordinator_lease"])

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "repro", "compile_ok": True, "oracle_passed": True}],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "cap",
        max_workspace_delta_bytes=0,
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "workspace_delta_not_measured" in entry["failure_reason"]

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "repro", "compile_ok": True, "oracle_passed": True}],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "no-cap",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert "workspace_delta" not in entry["failure_reason"]


def test_compile_failure_is_preserved_and_never_timed(tmp_path):
    calls: list[str] = []

    def compile_runner(candidate):
        calls.append(candidate["candidate_id"])
        return {"compile_ok": False, "failure_reason": "bisheng compile error",
                "run_id": "compile-failed-1"}

    def timing_runner(candidate):
        raise AssertionError("failed compile must not be timed")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "bad", "mutation_summary": "invalid tile"}],
        compile_runner=compile_runner, timing_runner=timing_runner,
        state_dir=tmp_path / "compile-failure",
    )
    entry = result["evaluations"][0]
    assert calls == ["bad"]
    assert result["timing_order"] == []
    assert entry["verdict"] == "inconclusive"
    assert entry["gate_reached"] == 1
    assert entry["failure_reason"] == "bisheng compile error"
    assert entry["run_id"] == "compile-failed-1"


def test_device_compile_holds_a_coordinator_lease(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")

    def compile_runner(candidate):
        lease = candidate["coordinator_lease"]
        assert lease.receipt()["state"] == "running"
        return {"compile_ok": True, "oracle_passed": True,
                "run_id": lease.execution_id}

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "jit", "device_compile": True}],
        compile_runner=compile_runner, lease_authority=authority,
        state_dir=tmp_path / "device-compile-ok",
    )
    assert result["compile_order"] == ["jit"]
    assert result["candidates"][0]["compile_ok"] is True
    assert result["candidates"][0]["compile_run_id"] == authority.issued[0]


def test_device_compile_requires_a_lease_even_when_a_callback_is_available(tmp_path):
    def compile_runner(candidate):
        raise AssertionError("device compile ran without a lease")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "jit", "device_compile": True}],
        compile_runner=compile_runner, state_dir=tmp_path / "device-compile",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "coordinator" in entry["failure_reason"]


class _FakeTaskClient:
    """Protocol double shaped like the real service `_reply` payload.

    This is not production evidence; the coordinator concurrency tests use the
    real host SQLite state machine separately.
    """

    def __init__(self, *, stdout, state="succeeded", released=True, devices=(0,),
                 terminal_on_running=False, running_state="running",
                 observe_terminal_on_running=False,
                 stop_error=False, release_wait_timeout=False,
                 terminal_status_at=None, foreign_status_at=None):
        self.execution_id = uuid.uuid4().hex + uuid.uuid4().hex
        self.stdout = stdout
        self.state = state
        self.released = released
        self.devices = list(devices)
        self.terminal_on_running = terminal_on_running
        self.running_state = running_state
        self.observe_terminal_on_running = observe_terminal_on_running
        self.stop_error = stop_error
        self.release_wait_timeout = release_wait_timeout
        self.terminal_status_at = terminal_status_at
        self.foreign_status_at = foreign_status_at
        self.status_observations = 0
        self.runs: list[dict] = []
        self.stopped = False
        self.finished = False
        self.running = False

    def _environment(self):
        if not self.devices:
            return {}
        return {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, self.devices))}

    def _assignment(self):
        return {
            "runtime_ids": ["runtime-1"],
            "roles": [{
                "name": "default", "runtime_id": "runtime-1", "host": "host-a",
                "root": "/tmp/runtime-root", "rank": 0,
                "devices": ",".join(map(str, self.devices)) if self.devices else None,
                "state": self.state,
            }],
        }

    def _role_views(self):
        state = self.state if self.finished else "running"
        environment = self._environment()
        return [{
            "name": "default", "state": state, "runtime_id": "runtime-1",
            "host": "host-a", "root": "/tmp/runtime-root", "env": {},
            "target": {
                "execution_id": self.execution_id, "state": state,
                "live": not self.finished, "environment": environment,
                "service_port": None, "assignment": self._assignment(),
            },
            "environment": environment,
        }]

    def _reply(self, state, *, released):
        return {
            "execution_id": self.execution_id, "run_id": self.execution_id,
            "state": state, "resources_released": released,
            "assignment": self._assignment(), "roles": self._role_views(),
            "target": self._role_views()[0]["target"],
            "stdout": self.stdout,
            "stderr": "" if state == "succeeded" else "command failed",
        }

    def run(self, **kwargs):
        self.runs.append(dict(kwargs))
        return {"execution_id": self.execution_id, "state": "queued"}

    def wait(self, execution_id, *, until="running", timeout_seconds=30):
        assert execution_id == self.execution_id
        if until == "running":
            self.running = True
            if self.terminal_on_running:
                self.finished = True
                return self._reply(self.state, released=self.released)
            return self._reply(self.running_state, released=False)
        self.finished = True
        if self.release_wait_timeout:
            reply = self._reply(self.state, released=self.released)
            reply["wait_timed_out"] = True
            return reply
        return self._reply(self.state, released=self.released)

    def observe(self, execution_id, action="status", refresh=True):
        assert execution_id == self.execution_id
        if action == "stop":
            if self.stop_error:
                raise RuntimeError("stop transport failed")
            self.stopped = True
            self.finished = True
            self.released = True
            return self._reply("cancelled", released=True)
        self.status_observations += 1
        if self.terminal_status_at == self.status_observations:
            self.finished = True
            return self._reply(self.state, released=self.released)
        if self.foreign_status_at == self.status_observations:
            reply = self._reply("running", released=False)
            reply["execution_id"] = "f" * 64
            return reply
        if self.finished and self.released:
            return self._reply(self.state, released=True)
        if self.observe_terminal_on_running:
            return self._reply(self.state, released=self.released)
        return self._reply("running", released=False)

def test_task_client_adapter_keeps_the_real_execution_id_and_npu_resources(tmp_path):
    client = _FakeTaskClient(stdout=json.dumps({
        "oracle_passed": True, "measurement_passed": True, "speedup_ratio": 1.3,
        "ablation_retained": True, "workspace_delta_bytes": 0,
        "guard_workloads": [{
            "name": "uniform-guard", "baseline_ms": 100.0,
            "candidate_ms": 101.0, "max_regression_ratio": 1.05,
        }],
    }))
    adapter = TaskClientRSIExecutor(client)
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "real", "compile_ok": True,
          "timing_command": "python -c 'print(1)'"}],
        execution_backend=adapter, state_dir=tmp_path / "adapter",
        max_workspace_delta_bytes=0,
        required_guard_workloads=["uniform-guard"],
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == client.execution_id
    assert result["best_candidate"]["run_id"] == client.execution_id
    assert result["execution_authority"]["mode"] == "mindie_coordinator.task_client"
    assert result["execution_authority"]["evidence_class"] == "coordinator_execution"
    assert result["execution_authority"]["production_adapter"] is True
    assert result["execution_authority"]["physical_acceptance"] is False
    assert client.runs[0]["resources"] == {"devices": [0]}
    assert client.runs[0]["command"] == "python -c 'print(1)'"
    assert entry["execution_evidence"]["evidence_class"] == "coordinator_execution"
    assert entry["execution_evidence"]["physical_acceptance"] is False


def test_task_client_adapter_rejects_fabricated_ids_and_preserves_measured_payload(tmp_path):
    fabricated = _FakeTaskClient(stdout=json.dumps({
        "run_id": "invented-candidate-timing", "oracle_passed": True,
        "measurement_passed": True, "speedup_ratio": 1.3,
        "ablation_retained": True,
    }))
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "fake", "compile_ok": True, "oracle_passed": True,
          "timing_command": "echo fake"}],
        execution_backend=TaskClientRSIExecutor(fabricated),
        state_dir=tmp_path / "fabricated",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert entry["run_id"] == ""
    assert "run_id" in entry["failure_reason"]

    unreleased = _FakeTaskClient(
        stdout=json.dumps({"oracle_passed": True, "measurement_passed": True,
                           "speedup_ratio": 1.3, "ablation_retained": True}),
        state="succeeded", released=False,
    )
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "held", "compile_ok": True, "oracle_passed": True,
          "timing_command": "echo held"}],
        execution_backend=TaskClientRSIExecutor(unreleased),
        state_dir=tmp_path / "unreleased",
    )
    entry = result["evaluations"][0]
    # The terminal payload was validated before release accounting. Stop then
    # confirmed release, so the result survives with clean cleanup evidence.
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == unreleased.execution_id
    assert result["execution_authority"]["cleanup_ok"] is True
    assert result["candidates"][0]["coordinator_lease_receipt"][
        "resources_released"
    ] is True


def test_task_client_cpu_compile_uses_zero_npu_resources(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"compile_ok": True}),
        devices=(),
    )
    adapter = TaskClientRSIExecutor(client)
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "cpu", "compile_command": "make -j4", "oracle_passed": True}],
        execution_backend=adapter, state_dir=tmp_path / "cpu-compile",
    )
    assert result["compile_order"] == ["cpu"]
    assert client.runs[0]["resources"] == {"npu_count": 0}
    assert result["timing_order"] == ["cpu"]
    assert result["evaluations"][0]["gate_reached"] == 2


class _MissingValidatorAuthority:
    mode = "test_receipt_only"
    evidence_class = "test_receipt_only"

    def acquire(self, devices, *, phase, candidate, timeout_seconds):
        raise AssertionError("authority without a verifier must be rejected before execution")


def test_authority_without_independent_verifier_is_rejected(tmp_path):
    with pytest.raises(RSIExecutionUnavailable, match="validate_result"):
        execute_rsi_compile_and_timing(
            [{"candidate_id": "c", "compile_ok": True}],
            timing_runner=lambda candidate: {},
            lease_authority=_MissingValidatorAuthority(),
            state_dir=tmp_path / "no-verifier",
        )


def test_observed_roles_and_target_environment_are_the_grant():
    # Real service `_reply`: top-level role views and target.environment. An
    # assignment echo with a different device must not replace the grant.
    reply = {
        "execution_id": "a" * 64, "state": "running",
        "roles": [{
            "name": "default", "state": "running", "env": {},
            "target": {
                "state": "running",
                "environment": {"ASCEND_RT_VISIBLE_DEVICES": "4,5"},
            },
            "environment": {"ASCEND_RT_VISIBLE_DEVICES": "4,5"},
        }],
        "target": {
            "state": "running",
            "environment": {"ASCEND_RT_VISIBLE_DEVICES": "4,5"},
        },
        "assignment": {"roles": [{"name": "default", "devices": "9"}]},
    }
    assert _assigned_devices(reply) == [4, 5]
    roles = _role_identities(reply)
    assert roles == [{
        "name": "default", "host": None, "runtime_id": None,
        "state": "running", "devices": [4, 5],
        "device_source": "role.target.environment.ASCEND_RT_VISIBLE_DEVICES",
    }]
    direct = {
        "execution_id": "a" * 64, "state": "running",
        "roles": [{"name": "default", "state": "running", "devices": [6],
                   "target": {"environment": {}}}],
        "assignment": {"roles": [{"name": "default", "devices": "9"}]},
    }
    assert _assigned_devices(direct) == [6]
    assert _role_identities(direct)[0]["device_source"] == "role.devices"
    top_level = {
        "execution_id": "a" * 64, "state": "running",
        "roles": [{"name": "default", "state": "running"}],
        "target": {"state": "running", "env": {"ASCEND_RT_VISIBLE_DEVICES": "3"}},
        "assignment": {"roles": [{"name": "default", "devices": "9"}]},
    }
    assert _assigned_devices(top_level) == [3]
    cpu = {
        "execution_id": "a" * 64, "state": "running",
        "roles": [{"name": "default", "state": "running", "target": {"environment": {}}}],
        "assignment": {"roles": [{"name": "default", "devices": None}]},
    }
    assert _assigned_devices(cpu) == []


def test_completed_execution_collects_exact_result_without_invoking_callback(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 1.3, "ablation_retained": True,
        }),
        terminal_on_running=True,
    )
    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("a completed/released execution cannot authorize a callback")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "completed", "compile_ok": True,
          "oracle_passed": True, "timing_command": "real measurement"}],
        timing_runner=timing_runner,
        lease_authority=TaskClientRSIExecutor(client),
        state_dir=tmp_path / "completed",
    )
    assert called == []
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == client.execution_id
    assert result["candidates"][0]["measurement_collection"] == "completed_execution"
    assert result["timing_attempts"][0]["collection_mode"] == "completed_execution"


def test_adapter_rejects_forged_callback_result_by_querying_coordinator(tmp_path):
    client = _FakeTaskClient(stdout=json.dumps({
        "oracle_passed": True, "measurement_passed": True,
        "speedup_ratio": 1.3, "ablation_retained": True,
    }))
    adapter = TaskClientRSIExecutor(client)

    def forged(candidate):
        return {
            "run_id": candidate["coordinator_lease"].execution_id,
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 9.9, "ablation_retained": True,
        }

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "forged", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=forged, lease_authority=adapter,
        state_dir=tmp_path / "forged",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "differs from coordinator observation" in entry["failure_reason"]
    assert result["best_candidate"] is None


