import copy
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mindie_bench.integrations.component_runtime import (
    HOST_OBSERVATION,
    ComponentCallError,
    ComponentRuntime,
    physical_observation,
)
from mindie_bench.runner.remote_slot import WORKER, RemoteSlot


def runtime(tmp_path):
    return ComponentRuntime(
        {
            "interpreters": dict.fromkeys(
                ["coordinator", "npu-top", "diagnostics", "knowledge", "remote-dev"], sys.executable
            ),
            "endpoint": {"host": "fixture", "port": 22},
            "diagnostics_root": str(tmp_path / "diagnostics"),
            "port_ranges": {"master": [30000, 30001], "hccl": [31000, 31001]},
        },
        "owned-slot",
        "owned-container",
        [8],
        3600,
        tmp_path,
    )


def telemetry():
    # A3 card 4 contains two independent physical chips. Chip 9's unrelated
    # process must not contaminate the admission decision for physical chip 8.
    parsed = {
        "devices": [
            {
                "npu_id": 4,
                "chips": [
                    {"phy_id": 8, "hbm": {"used_mb": 3000, "total_mb": 65536}, "aicore_percent": 0},
                    {"phy_id": 9, "hbm": {"used_mb": 50000, "total_mb": 65536}, "aicore_percent": 0},
                ],
                "processes": [{"pid": 99, "phy_id": 9}],
            }
        ]
    }
    authority = {
        "status": "ok",
        "devices": [8, 9],
        "busy": {"9": [{"kind": "process", "pid": 99}]},
    }
    return parsed, authority


def test_physical_chip_join_does_not_confuse_card_id_or_adjacent_workload():
    parsed, authority = telemetry()
    observed = physical_observation(parsed, authority, [8], [], 3600, idle=True)
    assert observed == {"8": {"used_mb": 3000, "aicore_percent": 0, "pids": [], "foreign_pids": []}}
    with pytest.raises(RuntimeError, match="not observed"):
        physical_observation(parsed, authority, [4], [], 3600, idle=True)


def test_owned_busy_work_is_allowed_but_foreign_and_unaccounted_hbm_are_rejected():
    parsed, authority = telemetry()
    parsed["devices"][0]["processes"].append({"pid": 42, "phy_id": 8})
    authority["busy"]["8"] = [{"kind": "process", "pid": 42}]
    assert physical_observation(parsed, authority, [8], [42], 3600, idle=False)["8"]["pids"] == [42]
    with pytest.raises(RuntimeError, match="foreign_pids=\\[42\\]"):
        physical_observation(parsed, authority, [8], [], 3600, idle=False)
    with pytest.raises(RuntimeError, match="occupied"):
        physical_observation(parsed, authority, [8], [42], 3600, idle=True)
    parsed, authority = telemetry()
    parsed["devices"][0]["chips"][0]["hbm"]["used_mb"] = 50000
    with pytest.raises(RuntimeError, match="unaccounted"):
        physical_observation(parsed, authority, [8], [], 3600, idle=False)


def test_missing_or_disagreeing_process_visibility_never_becomes_idle():
    parsed, authority = telemetry()
    for changed in [
        {"status": "failed"},
        {**authority, "busy": None},
        {**authority, "busy": {"8": [{"kind": "process", "pid": 42}]}},
    ]:
        with pytest.raises(RuntimeError):
            physical_observation(parsed, changed, [8], [], 3600, idle=True)
    parsed["devices"][0]["processes"].append({"pid": 42})
    with pytest.raises(ValueError, match="physical chip identity"):
        physical_observation(parsed, authority, [8], [], 3600, idle=True)


def test_idle_requires_measured_zero_aicore_and_vanished_pid_is_unresolved():
    parsed, authority = telemetry()
    chip = parsed["devices"][0]["chips"][0]
    chip["aicore_percent"] = None
    with pytest.raises(ValueError, match="AICore"):
        physical_observation(parsed, authority, [8], [], 3600, idle=True)
    chip["aicore_percent"] = 0.1
    with pytest.raises(RuntimeError, match="occupied"):
        physical_observation(parsed, authority, [8], [], 3600, idle=True)
    chip["aicore_percent"] = 0
    parsed["devices"][0]["processes"].append({"pid": 42, "phy_id": 8})
    authority["busy"]["8"] = [{"kind": "process", "pid": 42}]
    with pytest.raises(RuntimeError, match="vanished during sample"):
        physical_observation(parsed, authority, [8], [], 3600, idle=False, live_pids=[])


def test_namespace_ownership_survives_job_env_changes_but_not_pid_reuse(tmp_path, monkeypatch, capsys):
    import pathlib

    proc = tmp_path / "proc"
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("boot")

    def process(pid, ticks, namespace):
        root = proc / str(pid)
        (root / "ns").mkdir(parents=True, exist_ok=True)
        (root / "stat").write_text(f"{pid} (worker) " + " ".join(["S", *["0"] * 18, str(ticks)]))
        link = root / "ns/pid"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(namespace)

    process(1, 100, "pid:[55]")
    process(2, 200, "pid:[55]")  # No environment file: token equality cannot identify this child.
    process(3, 300, "pid:[66]")
    process(4, 400, "pid:[55]")
    original = pathlib.Path

    def run(*args, **kwargs):
        process(4, 401, "pid:[66]")
        return SimpleNamespace(returncode=0, stdout="actual-parser-input", stderr="")

    with monkeypatch.context() as changes:
        changes.setattr(
            pathlib,
            "Path",
            lambda value: (
                proc
                if value == "/proc"
                else boot
                if value == "/proc/sys/kernel/random/boot_id"
                else original(value)
            ),
        )
        changes.setattr(subprocess, "run", run)
        changes.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps(
                    {
                        "owner": {
                            "pid": 1,
                            "start_ticks": "100",
                            "boot_id": "boot",
                            "pid_namespace": "pid:[55]",
                        },
                        "require_owner": True,
                    }
                )
            ),
        )
        exec(compile(HOST_OBSERVATION, "actual-host-observation", "exec"), {})
    observed = json.loads(capsys.readouterr().out)
    assert observed["owned_pids"] == [1, 2]
    assert observed["live_pids"] == [1, 2, 3, 4]


def fake_components(subject, monkeypatch, calls, *, lose=None):
    def call(component, action, **payload):
        calls.append((component, action, copy.deepcopy(payload)))
        if action == "metadata":
            return {"version": "1.0", "revision": "a" * 40, "editable": False}
        if action == "coordinator":
            request = payload["request"]
            if request["action"] == lose:
                raise TimeoutError("lost host acknowledgement")
            if request["action"] == "status":
                return {"status": "ok", "coordination_epoch": "fixture-epoch"}
            kind = request["task_id"].rsplit(".", 1)[1]
            if request["action"] == "submit-acquire-preflight":
                return {
                    "status": "starting",
                    "task": {
                        "task_id": request["task_id"],
                        "state": "starting",
                        "granted_devices": [8] if kind == "master" else [],
                        "fence_token": 1 if kind == "master" else 2,
                        "granted_service_port": 30000 if kind == "master" else 31000,
                    },
                }
            return {
                "status": "released" if request["action"] == "release" else "active",
                "task": {"pid": 42},
            }
        if action == "diagnostic":
            return {"record_ref": "private-record", "logging_failed": False}
        if action == "container_id":
            return {"container_id": "exact-container-id"}
        raise AssertionError((component, action))

    monkeypatch.setattr(subject, "_call", call)
    monkeypatch.setattr(subject, "observe", lambda *args, **kwargs: {})


def test_acquire_uses_real_coordinator_contract_for_devices_and_both_ports(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    calls = []
    fake_components(subject, monkeypatch, calls)
    environment = subject.acquire()
    assert environment["MASTER_PORT"] == "30000"
    assert environment["HCCL_IF_BASE_PORT"] == "31000"
    requests = [row[2]["request"] for row in calls if row[1] == "coordinator"]
    assert [row["action"] for row in requests] == [
        "status",
        "submit-acquire-preflight",
        "submit-acquire-preflight",
    ]
    assert requests[1]["devices"] == [8]
    assert requests[2]["npu_count"] == 0
    assert all(row["coordination_epoch"] == "fixture-epoch" for row in requests[1:])
    assert all(row["allow_external_busy"] is False for row in requests[1:])
    subject.finish(cleanup_confirmed=True)
    releases = [
        row[2]["request"]
        for row in calls
        if row[1] == "coordinator" and row[2]["request"]["action"] == "release"
    ]
    assert [row["fence_token"] for row in releases] == [1, 2]
    assert all(row["completion_confirmed"] for row in releases)


def test_lost_admission_is_not_replayed_or_released_and_exact_id_remains_queryable(
    tmp_path, monkeypatch
):
    subject = runtime(tmp_path)
    calls = []
    fake_components(subject, monkeypatch, calls, lose="submit-acquire-preflight")
    with pytest.raises(TimeoutError):
        subject.acquire()
    assert subject.unknown
    with pytest.raises(RuntimeError, match="unknown"):
        subject.finish(cleanup_confirmed=True)
    requests = [row[2]["request"] for row in calls if row[1] == "coordinator"]
    assert [row["action"] for row in requests] == ["status", "submit-acquire-preflight", "status"]
    assert requests[-1]["task_id"] == "owned-slot.master"


def test_cleanup_unknown_retains_leases_without_submitting_release(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    calls = []
    fake_components(subject, monkeypatch, calls)
    subject.acquire()
    before = len(calls)
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        subject.finish(cleanup_confirmed=False)
    assert len(calls) == before


def test_finish_releases_owned_leases_but_preserves_foreign_occupancy_failure(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    calls = []
    fake_components(subject, monkeypatch, calls)
    subject.acquire()

    def occupied(*args, **kwargs):
        parsed, authority = telemetry()
        parsed["devices"][0]["processes"].append({"pid": 2478484, "phy_id": 8})
        authority["busy"]["8"] = [{"kind": "process", "pid": 2478484}]
        return physical_observation(parsed, authority, [8], [], 3600, idle=True)

    monkeypatch.setattr(subject, "observe", occupied)
    with pytest.raises(RuntimeError, match="occupied"):
        subject.finish(cleanup_confirmed=True)
    releases = [
        row[2]["request"] for row in calls
        if row[1] == "coordinator" and row[2]["request"]["action"] == "release"
    ]
    assert [row["fence_token"] for row in releases] == [1, 2]
    assert all(row["completion_confirmed"] for row in releases)

def active_lease(subject):
    subject.epoch = "fixture-epoch"
    subject.owner_identity = {"pid": 42}
    task = {
        "task_id": "owned-slot.master",
        "fence_token": 7,
        "agent_id": "mindie-bench",
        "session_id": subject.trace_id,
        "container_name": "owned-container",
        "granted_devices": [8],
        "granted_service_port": 30000,
        "pid": 42,
        "state": "active",
    }
    subject.leases = {"master": task}
    return task


def test_not_sent_heartbeat_queries_exact_owner_without_replaying_mutation(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    task = active_lease(subject)
    requests = []
    # Captured from final remote-dev's real HostQueue call to a closed loopback SSH port.
    details = {
        "type": "RemoteExecutionError",
        "category": "connection_unavailable",
        "retryable": True,
        "submission_state": "not_sent",
        "message": "SSH RPC unavailable; request was not sent: Connection refused",
    }

    def call(component, action, **payload):
        request = payload["request"]
        requests.append(request)
        if len(requests) == 1:
            raise ComponentCallError(details)
        if request["action"] == "status":
            return {"status": "ok", "coordination_epoch": subject.epoch, "tasks": [task]}
        return {"status": "active", "task": task}

    monkeypatch.setattr(subject, "_call", call)
    subject._heartbeat(task)
    subject.assert_healthy()
    assert not subject.unknown
    assert [row["action"] for row in requests] == ["heartbeat", "status"]
    assert requests[1]["task_id"] == task["task_id"]
    assert requests[1]["coordination_epoch"] == subject.epoch
    subject._heartbeat(task)  # The next scheduled heartbeat remains usable.
    assert [row["action"] for row in requests] == ["heartbeat", "status", "heartbeat"]


@pytest.mark.parametrize("changed", [{"fence_token": 8}, {"session_id": "other-owner"}])
def test_not_sent_heartbeat_cannot_continue_a_reassigned_lease(tmp_path, monkeypatch, changed):
    subject = runtime(tmp_path)
    task = active_lease(subject)
    requests = []

    def call(component, action, **payload):
        request = payload["request"]
        requests.append(request)
        if request["action"] == "heartbeat":
            raise ComponentCallError({
                "message": "connection unavailable",
                "submission_state": "not_sent",
                "retryable": True,
            })
        return {
            "status": "ok",
            "coordination_epoch": subject.epoch,
            "tasks": [{**task, **changed}],
        }

    monkeypatch.setattr(subject, "_call", call)
    with pytest.raises(RuntimeError, match="active owner could not be confirmed"):
        subject._heartbeat(task)
    assert [row["action"] for row in requests] == ["heartbeat", "status"]
    assert not subject.unknown


def test_uncertain_heartbeat_stays_unknown_and_is_not_replayed(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    task = active_lease(subject)
    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs["request"]["action"])
        raise ComponentCallError({
            "message": "RPC disconnected after submission",
            "submission_state": "uncertain",
            "retryable": False,
        })

    monkeypatch.setattr(subject, "_call", call)
    with pytest.raises(ComponentCallError):
        subject._heartbeat(task)
    assert subject.unknown
    with pytest.raises(RuntimeError, match="unknown"):
        subject._heartbeat(task)
    assert calls == ["heartbeat"]


def test_real_helper_pipe_preserves_not_sent_failure(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    helper = tmp_path / "component_error.py"
    helper.write_text(
        "import json\nprint(json.dumps({'component_error': {"
        "'message': 'SSH RPC unavailable; request was not sent', "
        "'category': 'connection_unavailable', 'submission_state': 'not_sent', "
        "'retryable': True}}))\n"
    )
    start = subprocess.Popen

    def launch(command, **kwargs):
        return start([*command[:-1], str(helper)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(ComponentCallError) as error:
        subject._authority("heartbeat", task_id="owned-slot.master", fence_token=7)
    assert error.value.submission_state == "not_sent"
    assert not subject.unknown


def test_acknowledged_helper_cleanup_uses_owner_pipe_without_signalling_a_reaped_group(
    tmp_path, monkeypatch
):
    import os

    subject = runtime(tmp_path)
    process = SimpleNamespace(
        pid=12345,
        returncode=0,
        communicate=lambda data, timeout: (b'{"acknowledged":true}', b""),
        wait=lambda timeout: 0,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    def forbidden_signal(*args):
        raise PermissionError("macOS reaped process-group signal denied")

    monkeypatch.setattr(os, "killpg", forbidden_signal)
    assert subject._call("diagnostics", "diagnostic") == {"acknowledged": True}


def test_activation_is_fenced_to_the_exact_waiting_worker(tmp_path, monkeypatch):
    subject = runtime(tmp_path)
    calls = []
    fake_components(subject, monkeypatch, calls)
    subject.acquire()
    with pytest.raises(RuntimeError, match="owner"):
        subject.activate({"process_guard": {"marker": "someone-else"}}, lambda: None)
    guard = {"marker": subject.marker, "boot_id": "boot", "retain_until_release": True}
    receipt = {
        "pid": 1,
        "start_ticks": "12345",
        "boot_id": "boot",
        "process_guard": guard,
        "pid_namespace": "pid:[55]",
    }
    subject.activate(receipt, lambda: None)
    subject.finish(cleanup_confirmed=True)
    requests = [
        row[2]["request"]
        for row in calls
        if row[1] == "coordinator" and row[2]["request"]["action"] == "activate"
    ]
    assert len(requests) == 2
    assert all(row["prepared_supervisor"]["container_id"] == "exact-container-id" for row in requests)
    assert all(row["prepared_supervisor"]["receipt"] == receipt for row in requests)


def test_slot_cache_directories_are_absolute_distinct_and_inherited(case, monkeypatch):
    commands = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(stdin=io.BytesIO()),
    )
    slots = []
    for name in ["slot-one", "slot-two"]:
        config = SimpleNamespace(
            slots_root="/owned/slots",
            runtime_read_paths=[],
            repo_path="/repo",
            probe_command=["probe"],
            host="host",
            container="container",
            memory_limit_mb=3600,
        )
        slot = RemoteSlot(config, case["spec"].environment["l2"], name, [8], 1)
        monkeypatch.setattr(slot, "_read", lambda timeout: {"ready": True})
        monkeypatch.setattr(slot, "probe", lambda: {})
        slot.prepare()
        slots.append(slot)
    for key in ["ASCEND_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]:
        assert slots[0].slot_environment[key] != slots[1].slot_environment[key]
        for slot, command in zip(slots, commands, strict=True):
            assert slot.slot_environment[key].startswith(slot.path + "/.bench-home/")
            assert f"export {key}=" in command[-1]
            assert slot.slot_environment[key] in command[-1]
    assert slots[0].slot_environment["TMPDIR"] != slots[1].slot_environment["TMPDIR"]
    assert slots[0].tmpdir_backing != slots[1].tmpdir_backing
    for slot, command in zip(slots, commands, strict=True):
        assert slot.tmpdir_backing.startswith(slot.path + "/.bench-home/")
        assert "--bind " + slot.tmpdir_backing + " " + slot.slot_environment["TMPDIR"] in command[-1]


def test_long_trial_tmpdir_leaves_room_for_native_unix_socket_uuid(case, monkeypatch):
    commands = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(stdin=io.BytesIO()),
    )
    config = SimpleNamespace(
        slots_root="/bench/slots-deployment-config-migration",
        runtime_read_paths=[],
        repo_path="/repo",
        probe_command=["probe"],
        host="host",
        container="container",
        memory_limit_mb=3600,
    )
    slot = RemoteSlot(config, case["spec"].environment["l2"], "trial-" + "x" * 96, [8], 1)
    monkeypatch.setattr(slot, "_read", lambda timeout: {"ready": True})
    monkeypatch.setattr(slot, "probe", lambda: {})
    slot.prepare()
    # vLLM's native TP MessageQueue appends a UUID; sockaddr_un has a fixed bound.
    ipc_path = slot.slot_environment["TMPDIR"] + "/" + "a" * 36
    assert len(ipc_path.encode()) < 104


def test_verifier_environment_is_literal_and_cannot_replace_slot_isolation(case, monkeypatch, tmp_path):
    import base64
    import shlex

    config = SimpleNamespace(slots_root="/owned/slots")
    slot = RemoteSlot(config, case["spec"].environment["l2"], "owned-env", [8], 1)
    marker = tmp_path / "must-not-be-created"
    value = f"$(touch {shlex.quote(str(marker))}); 'literal'\nnonce"

    def execute(command, stdin_b64):
        result = subprocess.run(
            shlex.split(command),
            input=base64.b64decode(stdin_b64),
            capture_output=True,
            check=True,
        )
        return result.stdout

    monkeypatch.setattr(slot, "execute", execute)
    observed = slot.verify(b"printf '%s' \"$BENCH_NONCE\"", env={"BENCH_NONCE": value})
    assert observed.decode() == value and not marker.exists()
    for env in [
        {"MASTER_PORT": "40000"},
        {"TMPDIR": "/someone-else"},
        {"BAD-NAME": "nonce"},
        {"BENCH_NONCE": "nul\0"},
    ]:
        with pytest.raises(ValueError, match="invalid verifier environment"):
            slot.verify(b"exit 0", env=env)


def test_judge_deadline_is_per_request_and_does_not_expand_or_mutate_business_budget(case, monkeypatch):
    slot = RemoteSlot(
        SimpleNamespace(slots_root="/owned/slots"),
        case["spec"].environment["l2"],
        "owned-deadline",
        [8],
        1200,
    )
    received = []
    monkeypatch.setattr(
        slot,
        "_request",
        lambda request, timeout: received.append((request, timeout)) or {"exit_code": 0},
    )
    slot.verify(b"exit 0", env={"BENCH_NONCE": "literal"}, timeout=240)
    assert received[-1][0]["timeout"] == 240
    assert received[-1][1] == 255
    assert slot.timeout == 1200
    slot.execute("business")
    assert received[-1][0]["timeout"] == 1200
    assert len(received) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="actual remote worker uses Linux /proc")
def test_worker_guard_receipt_is_actual_process_identity_without_npu(tmp_path):
    # A short local owned subprocess exercises the same worker startup, no SSH.
    import os
    import signal

    marker = "a" * 32
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "-c", WORKER],
        env={**os.environ, "MINDIE_BENCH_COMPONENT_GUARD": "1", "REMOTE_DEV_JOB_TOKEN": marker},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        value = json.loads(process.stdout.readline())
        owner = value["owner_receipt"]
        assert value["ready"] is True and owner["pid"] == process.pid
        assert owner["start_ticks"].isdigit()
        assert owner["process_guard"] == {
            "marker": marker,
            "boot_id": owner["boot_id"],
            "retain_until_release": True,
        }
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        for stream in [process.stdin, process.stdout, process.stderr]:
            stream.close()
