"""Coordinator-backed RSI execution and measurement-evidence contracts.

These are contract tests for the executor and the TaskClient adapter. The
concurrency and genuine-ID cases use the installed coordinator's real SQLite
lease state machine in a temporary directory; they do not touch an NPU. The
TaskClient cases use a protocol double and explicitly do not claim hardware
execution.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

from mindie_bench.runner.rsi_execution import (
    RSIExecutionError,
    RSIExecutionUnavailable,
    TaskClientRSIExecutor,
    execute_rsi_compile_and_timing,
)
from mindie_bench.runner.rsi_measurement import _assigned_devices

coordinator_module = pytest.importorskip("mindie_coordinator.host.mindie_npu_coordination")
NpuCoordinator = coordinator_module.NpuCoordinator

FREE = {"status": "ok", "devices": [0, 1], "busy": {}, "free": [0, 1]}


class _CoordinatorLease:
    def __init__(self, authority, task, token, epoch):
        self.authority = authority
        self.task_id = task
        self.token = token
        self.epoch = epoch
        self.released = False

    @property
    def execution_id(self):
        return self.task_id

    def receipt(self):
        return {
            "execution_id": self.task_id,
            "task_id": self.task_id,
            "devices": [0],
            "state": "running" if not self.released else "released",
            "fence_token": self.token,
            "coordination_epoch": self.epoch,
            "authority": "mindie_coordinator.npu_host_queue",
            "released": self.released,
        }

    def verify(self):
        row = self.authority.snapshot(self.task_id)
        if row.get("state") != "active":
            raise RSIExecutionError(f"coordinator task is {row.get('state')!r}")
        return True

    def release(self):
        if self.released:
            return self.receipt()
        reply = self.authority.coordinator.release(
            self.task_id, self.token, self.authority.observed, completion_confirmed=True,
        )
        if reply.get("status") != "released":
            raise RSIExecutionError(f"coordinator retained lease: {reply.get('status')}")
        self.released = True
        return self.receipt()


class _CoordinatorAuthority:
    """Real NpuCoordinator state in an isolated temp dir; test-only wrapper."""

    mode = "mindie_coordinator.npu_host_queue"
    evidence_class = "test_coordinator_authority"

    def __init__(self, state_dir):
        self.coordinator = NpuCoordinator(state_dir)
        self.observed = dict(FREE)
        self.issued: list[str] = []

    def snapshot(self, task_id):
        rows = self.coordinator.snapshot(None, task_id=task_id)["tasks"]
        return dict(rows[0]) if rows else {}

    def acquire(self, devices, *, phase, candidate, timeout_seconds):
        if not devices:
            raise RSIExecutionUnavailable("device lease requires devices")
        task_id = "rsi-" + uuid.uuid4().hex[:12]
        self.coordinator.submit({
            "task_id": task_id, "agent_id": "rsi-test", "devices": list(devices),
        })
        granted = self.coordinator.acquire(task_id, self.observed)
        if granted.get("status") != "granted":
            self.coordinator.cancel(task_id, self.observed)
            raise RSIExecutionUnavailable(
                f"coordinator lease unavailable: {granted.get('reason') or granted.get('status')}"
            )
        token = granted["task"]["fence_token"]
        preflight = self.coordinator.preflight(task_id, token, self.observed)
        if preflight.get("status") != "starting":
            self.coordinator.cancel(task_id, self.observed)
            raise RSIExecutionUnavailable("coordinator preflight did not admit the lease")
        active = self.coordinator.activate(task_id, token, pid=os.getpid())
        if active.get("status") != "active":
            self.coordinator.cancel(task_id, self.observed)
            raise RSIExecutionUnavailable("coordinator did not activate the lease")
        epoch = self.coordinator.snapshot(None)["coordination_epoch"]
        self.issued.append(task_id)
        return _CoordinatorLease(self, task_id, token, epoch)

    def validate_result(self, lease, result, *, phase, devices):
        if result.get("run_id") != lease.execution_id:
            raise RSIExecutionError("result does not preserve the coordinator execution ID")

    def release(self, lease):
        return lease.release()


def _timing_result(lease, **values):
    payload = {
        "run_id": lease.execution_id,
        "oracle_passed": True,
        "measurement_passed": True,
        "speedup_ratio": 1.2,
        "ablation_retained": True,
    }
    payload.update(values)
    return payload




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

class _MissingValidatorAuthority:
    mode = "test_receipt_only"
    evidence_class = "test_receipt_only"

    def acquire(self, devices, *, phase, candidate, timeout_seconds):
        raise AssertionError("authority without a verifier must be rejected before execution")

def test_cleanup_failure_does_not_erase_measured_success(tmp_path):
    class FailingRelease(_CoordinatorAuthority):
        def release(self, lease):
            raise RSIExecutionError(
                "release observation failed", execution_id=lease.execution_id,
                cleanup={"released": False, "stop_error": "stop unavailable"},
            )

    authority = FailingRelease(tmp_path / "coordinator")

    def timing(candidate):
        return _timing_result(candidate["coordinator_lease"])

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "measured", "compile_ok": True, "oracle_passed": True}],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "cleanup",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == authority.issued[0]
    assert result["execution_authority"]["cleanup_ok"] is False
    assert result["execution_authority"]["cleanup_failures"]
    assert result["candidates"][0]["cleanup_failures"][0]["execution_id"] == authority.issued[0]


def test_required_guard_snapshot_survives_prefill_clear_and_global_augments(tmp_path):
    authority = _CoordinatorAuthority(tmp_path / "coordinator")

    def timing(candidate):
        # The candidate requires candidate-guard in its original declaration,
        # but the completed command omitted it. The global guard is present.
        return _timing_result(
            candidate["coordinator_lease"], workspace_delta_bytes=0,
            guard_workloads=[{
                "name": "global-guard", "baseline_ms": 100.0,
                "candidate_ms": 101.0, "max_regression_ratio": 1.05,
            }],
        )

    result = execute_rsi_compile_and_timing(
        [{
            "candidate_id": "guards", "compile_ok": True, "oracle_passed": True,
            "guard_workloads": [{
                "name": "candidate-guard", "required": True,
                "baseline_ms": 100.0, "candidate_ms": 101.0,
                "max_regression_ratio": 1.05,
            }],
        }],
        timing_runner=timing, lease_authority=authority,
        state_dir=tmp_path / "guards",
        max_workspace_delta_bytes=0,
        required_guard_workloads=["global-guard"],
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "guard_workload_missing:candidate-guard" in entry["failure_reason"]


def test_cpu_compile_does_not_inherit_gang_topology(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"compile_ok": True}), devices=(),
    )
    adapter = TaskClientRSIExecutor(
        client,
        topology={
            "host": "192.168.13.165", "die_count": 2, "scale": "S1",
            "placement_policy": "same_npu_dual_die",
            "roles": [{"name": "default", "devices": [0, 1]}],
        },
    )
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "cpu", "compile_command": "make -j4",
          "oracle_passed": True}],
        execution_backend=adapter, state_dir=tmp_path / "cpu-topology",
    )
    assert client.runs[0]["resources"] == {"npu_count": 0}
    assert client.runs[0]["topology"] == {"host": "192.168.13.165"}
    assert result["compile_order"] == ["cpu"]


def test_cpu_compile_timeout_is_bounded_and_visible(tmp_path):
    def compile_runner(candidate):
        time.sleep(0.2)
        return {"compile_ok": True, "oracle_passed": True}

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "slow", "compile_command": "actual build"}],
        compile_runner=compile_runner, state_dir=tmp_path / "slow",
        compile_timeout_seconds=0.05,
    )
    assert result["compile_timeouts"] == ["slow"]
    assert result["candidates"][0]["compile_ok"] is False
    assert "compile_callback_timeout" in result["candidates"][0]["failure_reason"]



def test_default_compile_timeout_covers_production_adapter_waits(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"compile_ok": True}), devices=(),
    )
    adapter = TaskClientRSIExecutor(
        client, command_timeout_seconds=12, lease_wait_seconds=3,
        release_wait_seconds=4, release_timeout_seconds=7,
    )
    inner = adapter.compile_wait_budget_seconds(lease_timeout_seconds=3)
    assert inner == 3 + 12 + 2 * 7 + adapter.cleanup_wait_seconds()
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "cpu", "compile_command": "make -j4",
          "oracle_passed": True}],
        execution_backend=adapter, state_dir=tmp_path / "budget",
        lease_timeout_seconds=3,
    )
    budget = result["execution_authority"]["compile_timeout_budget"]
    assert budget["source"] == "execution_authority"
    assert budget["inner_seconds"] == inner
    assert budget["grace_seconds"] == 10.0
    assert budget["outer_seconds"] == inner + 10.0
    assert result["execution_authority"]["compile_timeout_seconds"] == inner + 10.0
    assert result["compile_order"] == ["cpu"]

    with pytest.raises(ValueError, match="below the production inner budget"):
        execute_rsi_compile_and_timing(
            [{"candidate_id": "cpu", "compile_command": "make -j4",
              "oracle_passed": True}],
            execution_backend=adapter, state_dir=tmp_path / "too-small",
            lease_timeout_seconds=3, compile_timeout_seconds=inner,
        )


def test_candidate_without_declared_id_uses_the_planner_identity(tmp_path):
    def compile_runner(candidate):
        return {"compile_ok": True, "run_id": "cpu-compile-1"}

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": None, "oracle_passed": True}],
        compile_runner=compile_runner, state_dir=tmp_path / "identity",
    )
    assert result["compile_order"] == ["candidate-0"]
    assert result["candidates"][0]["candidate_id"] is None
    assert result["evaluations"][0]["candidate_id"] == "candidate-0"
    assert result["candidates"][0]["compile_ok"] is True


def test_waiting_execution_does_not_authorize_callback_and_preserves_identity(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"oracle_passed": True, "measurement_passed": True,
                           "speedup_ratio": 1.3, "ablation_retained": True}),
        running_state="waiting_for_runtime",
    )
    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("waiting lease must not authorize a callback")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "waiting", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=timing_runner,
        lease_authority=TaskClientRSIExecutor(client),
        state_dir=tmp_path / "waiting",
    )
    assert called == []
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert result["candidates"][0]["attempted_execution_id"] == client.execution_id
    assert result["candidates"][0]["attempted_cleanup"]["released"] is True
    assert result["timing_attempts"][0]["execution_id"] == client.execution_id


def test_acquire_cleanup_failure_is_propagated_and_visible(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"oracle_passed": True, "measurement_passed": True,
                           "speedup_ratio": 1.3, "ablation_retained": True}),
        running_state="waiting_for_runtime",
        released=False,
        stop_error=True,
        release_wait_timeout=True,
    )
    adapter = TaskClientRSIExecutor(client, release_timeout_seconds=0.2,
                                    release_wait_seconds=0.1)
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "cleanup", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=lambda candidate: {},
        lease_authority=adapter,
        state_dir=tmp_path / "cleanup-failure",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert result["execution_authority"]["cleanup_ok"] is False
    candidate = result["candidates"][0]
    assert candidate["attempted_execution_id"] == client.execution_id
    cleanup = candidate["attempted_cleanup"]
    assert cleanup["execution_id"] == client.execution_id
    assert "stop transport failed" in cleanup["stop_error"]
    assert cleanup["released"] is False


def test_authority_cannot_declare_physical_acceptance(tmp_path):
    class _PhysicalClaim(_CoordinatorAuthority):
        physical_acceptance = True

    with pytest.raises(RSIExecutionUnavailable, match="physical acceptance"):
        execute_rsi_compile_and_timing(
            [{"candidate_id": "c", "compile_ok": True}],
            timing_runner=lambda candidate: {},
            lease_authority=_PhysicalClaim(tmp_path / "coordinator"),
            state_dir=tmp_path / "physical",
        )


def test_cpu_compile_rejects_unexpected_device_grant(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"compile_ok": True}),
        devices=(0,),
    )
    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "cpu", "compile_command": "make -j4",
          "oracle_passed": True}],
        execution_backend=TaskClientRSIExecutor(client),
        state_dir=tmp_path / "cpu-device-grant",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "inconclusive"
    assert "unexpectedly acquired devices" in result["candidates"][0]["failure_reason"]


def test_target_env_variant_is_an_observed_grant():
    reply = {
        "execution_id": "b" * 64, "state": "running",
        "roles": [{
            "name": "default", "state": "running",
            "target": {"state": "running", "env": {"ASCEND_RT_VISIBLE_DEVICES": "7"}},
        }],
        "assignment": {"roles": [{"name": "default", "devices": "9"}]},
    }
    assert _assigned_devices(reply) == [7]


def test_device_compile_timeout_blocks_later_timing_until_cleanup_is_known(tmp_path):
    def slow_device_compile(candidate):
        time.sleep(0.2)
        return {"compile_ok": True}

    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("timing must wait for unknown device compile cleanup")

    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "device-slow", "device_compile": True,
             "oracle_passed": True},
            {"candidate_id": "preproved", "compile_ok": True, "oracle_passed": True},
        ],
        compile_runner=slow_device_compile,
        timing_runner=timing_runner,
        lease_authority=_CoordinatorAuthority(tmp_path / "coordinator"),
        state_dir=tmp_path / "device-timeout",
        compile_timeout_seconds=0.05,
    )
    assert called == []
    assert result["execution_authority"]["device_compile_cleanup_unknown"] is True
    assert result["execution_authority"]["cleanup_ok"] is False
    assert result["timing_attempts"] == [{
        "candidate_id": "preproved", "acquired": False,
        "cleanup_ok": False, "failure_reason": "device_compile_cleanup_unknown",
    }]


def test_fast_completion_between_running_wait_and_verification_is_collected_not_called(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 1.3, "ablation_retained": True,
        }),
        observe_terminal_on_running=True,
    )
    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("terminal execution cannot authorize a callback")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "fast", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=timing_runner,
        lease_authority=TaskClientRSIExecutor(client),
        state_dir=tmp_path / "fast",
    )
    assert called == []
    assert result["evaluations"][0]["verdict"] == "supported"
    assert result["candidates"][0]["measurement_collection"] == "completed_execution"


def test_terminal_transition_after_adapter_verification_is_collected_exactly(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 1.3, "ablation_retained": True,
        }),
        terminal_status_at=2,
    )
    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("a terminal execution must not authorize a callback")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "race", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=timing_runner,
        lease_authority=TaskClientRSIExecutor(client),
        state_dir=tmp_path / "terminal-race",
    )
    assert called == []
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == client.execution_id
    assert result["candidates"][0]["measurement_collection"] == "completed_execution"
    assert result["timing_attempts"][0]["execution_id"] == client.execution_id
    assert result["timing_attempts"][0]["released"] is True


def test_verification_failure_after_acquire_cleans_the_submitted_execution(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 1.3, "ablation_retained": True,
        }),
        foreign_status_at=2, stop_error=True, release_wait_timeout=True,
        released=False,
    )
    adapter = TaskClientRSIExecutor(
        client, release_wait_seconds=0.1, release_timeout_seconds=0.2,
    )
    called = []

    def timing_runner(candidate):
        called.append(candidate["candidate_id"])
        raise AssertionError("an unverifiable lease must not authorize a callback")

    result = execute_rsi_compile_and_timing(
        [{"candidate_id": "unverified", "compile_ok": True, "oracle_passed": True,
          "timing_command": "real measurement"}],
        timing_runner=timing_runner, lease_authority=adapter,
        state_dir=tmp_path / "verification-cleanup",
    )
    assert called == []
    candidate = result["candidates"][0]
    assert candidate["attempted_execution_id"] == client.execution_id
    cleanup = candidate["attempted_cleanup"]
    assert cleanup["execution_id"] == client.execution_id
    assert cleanup["released"] is False
    assert "stop transport failed" in cleanup["stop_error"]
    assert result["execution_authority"]["cleanup_ok"] is False
    assert result["timing_attempts"][0]["acquired"] is False


def test_release_accounting_timeout_preserves_timing_schema_and_blocks_later_npu(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({
            "oracle_passed": True, "measurement_passed": True,
            "speedup_ratio": 1.3, "ablation_retained": True,
        }),
        released=False, stop_error=True, release_wait_timeout=True,
    )
    adapter = TaskClientRSIExecutor(
        client, release_wait_seconds=0.1, release_timeout_seconds=0.2,
    )
    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "measured", "compile_ok": True, "oracle_passed": True,
             "timing_command": "real measurement"},
            {"candidate_id": "later", "compile_ok": True, "oracle_passed": True,
             "timing_command": "must not run"},
        ],
        execution_backend=adapter, state_dir=tmp_path / "release-timeout",
    )
    entry = result["evaluations"][0]
    assert entry["verdict"] == "supported"
    assert entry["run_id"] == client.execution_id
    assert entry["cleanup_ok"] is False
    receipt = result["candidates"][0]["coordinator_lease_receipt"]
    assert receipt["schema_version"] == "mindie-bench-rsi-execution/1"
    assert receipt["authority"] == "mindie_coordinator.task_client"
    assert receipt["execution_id"] == client.execution_id
    assert receipt["phase"] == "timing"
    assert receipt["resources_released"] is False
    assert result["candidates"][0]["cleanup_failures"][0]["execution_id"] == client.execution_id
    assert result["timing_attempts"][0]["cleanup_ok"] is False
    assert result["timing_attempts"][1]["candidate_id"] == "later"
    assert result["timing_attempts"][1]["acquired"] is False
    assert result["timing_attempts"][1]["failure_reason"] == "coordinator_cleanup_failed"
    assert len(client.runs) == 1


def test_cpu_compile_release_timeout_preserves_compile_and_blocks_timing(tmp_path):
    client = _FakeTaskClient(
        stdout=json.dumps({"compile_ok": True}), devices=(),
        released=False, stop_error=True, release_wait_timeout=True,
    )
    adapter = TaskClientRSIExecutor(
        client, release_wait_seconds=0.1, release_timeout_seconds=0.2,
    )
    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "cpu", "compile_command": "make -j4",
             "oracle_passed": True},
            {"candidate_id": "later", "compile_ok": True, "oracle_passed": True,
             "timing_command": "must not run"},
        ],
        execution_backend=adapter, state_dir=tmp_path / "compile-release-timeout",
    )
    candidate = result["candidates"][0]
    assert candidate["compile_ok"] is True
    assert candidate["compile_run_id"] == client.execution_id
    assert candidate["attempted_cleanup"]["released"] is False
    assert candidate["cleanup_failures"][0]["execution_id"] == client.execution_id
    assert result["execution_authority"]["cleanup_ok"] is False
    assert [item["candidate_id"] for item in result["timing_attempts"]] == ["cpu", "later"]
    assert all(item["acquired"] is False for item in result["timing_attempts"])
    assert all(
        item["failure_reason"] == "coordinator_cleanup_failed"
        for item in result["timing_attempts"]
    )
    assert len(client.runs) == 1
    assert client.runs[0]["resources"] == {"npu_count": 0}
