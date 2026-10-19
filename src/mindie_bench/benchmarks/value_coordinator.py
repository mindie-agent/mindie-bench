"""Exercise competing agents and live TCP listeners through the real allocator.

[Role] Compare uncoordinated check-then-bind with a shared production lease authority.
[Invariants] No NPU workload starts; hardware observations only supply device identities.
[Timeouts & Locks] Each child exchange has a ten-second deadline; SQLite owns serialization.
[Side-Effect Policy] Own temporary state, child processes and loopback sockets only.
"""

from __future__ import annotations

import multiprocessing
import socket
import time
from pathlib import Path

from .value_common import source_revisions, write_json


def _listening(ports):
    found = []
    for port in ports:
        with socket.socket() as client:
            client.settimeout(0.1)
            if client.connect_ex(("127.0.0.1", port)) == 0:
                found.append(port)
    return {"status": "ok", "ports": found}


def _ports():
    sockets = [socket.socket(), socket.socket()]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return sorted(item.getsockname()[1] for item in sockets)
    finally:
        for item in sockets:
            item.close()


def _worker(connection, task_id, state, observed, ports, coordinated):
    from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator

    coordinator = NpuCoordinator(state) if coordinated else None
    listener = None
    grant = None
    try:
        connection.send({"status": "ready"})
        while True:
            command = connection.recv()
            started = time.monotonic()
            if command == "stop":
                return
            if command == "acquire":
                grant = (
                    coordinator.acquire(task_id, observed, listening=_listening(ports))
                    if coordinator
                    else {
                        "status": "granted",
                        "task": {"granted_devices": [observed["free"][0]],
                                 "granted_service_port": ports[0]},
                    }
                )
                result = {"grant": grant}
                if grant["status"] == "granted":
                    listener = socket.socket()
                    try:
                        listener.bind(("127.0.0.1", grant["task"]["granted_service_port"]))
                        listener.listen(128)
                        result["bind"] = {"status": "listening"}
                    except OSError as error:
                        listener.close()
                        listener = None
                        result["bind"] = {"status": "failed", "errno": error.errno}
                result["elapsed_sec"] = time.monotonic() - started
            elif command == "close":
                if listener is not None:
                    listener.close()
                    listener = None
                result = {"status": "closed"}
            elif command == "release":
                result = coordinator.release(
                    task_id, grant["task"]["fence_token"], observed,
                    listening=_listening(ports),
                )
            else:
                raise ValueError(command)
            connection.send(result)
    finally:
        if listener is not None:
            listener.close()
        connection.close()


def _receive(connection):
    if not connection.poll(10):
        raise TimeoutError("coordinator scenario child did not answer within ten seconds")
    return connection.recv()


def _pair(root, observed, ports, coordinated):
    from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator

    context = multiprocessing.get_context("spawn")
    state = root / ("candidate-state" if coordinated else "baseline-state")
    coordinator = NpuCoordinator(state) if coordinated else None
    task_ids = ["value-agent-a", "value-agent-b"]
    if coordinator:
        for task_id in task_ids:
            coordinator.submit({"task_id": task_id, "agent_id": task_id, "npu_count": 1,
                                "service_port": 0, "service_ports": ports})
    children = []
    connections = []
    events = []
    started = time.monotonic()
    try:
        for task_id in task_ids:
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker, args=(child, task_id, state, observed, ports, coordinated)
            )
            process.start()
            child.close()
            children.append(process)
            connections.append(parent)
        for connection in connections:
            assert _receive(connection)["status"] == "ready"
        for connection in connections:
            connection.send("acquire")
        grants = [_receive(connection) for connection in connections]
        events.append({"phase": "simultaneous_admission", "results": grants})
        if coordinated:
            # FIFO can answer waiting to the later task before the first transaction commits.
            for index, result in enumerate(grants):
                if result["grant"]["status"] == "waiting":
                    connections[index].send("acquire")
                    grants[index] = _receive(connections[index])
                    events.append({"phase": "after_predecessor_admission", "result": grants[index]})
            assert all(row["bind"]["status"] == "listening" for row in grants)
            assert len({row["grant"]["task"]["granted_devices"][0] for row in grants}) == 2
            assert len({row["grant"]["task"]["granted_service_port"] for row in grants}) == 2
            coordinator.submit({"task_id": "value-agent-c", "agent_id": "value-agent-c",
                                "npu_count": 1, "service_port": 0, "service_ports": ports})
            blocked = coordinator.acquire("value-agent-c", observed, listening=_listening(ports))
            assert blocked["status"] == "waiting"
            connections[0].send("release")
            busy_release = _receive(connections[0])
            assert busy_release["status"] == "orphaned_busy"
            still_blocked = coordinator.acquire(
                "value-agent-c", observed, listening=_listening(ports)
            )
            assert still_blocked["status"] == "waiting"
            recovery_started = time.monotonic()
            connections[0].send("close")
            _receive(connections[0])
            connections[0].send("release")
            released = _receive(connections[0])
            assert released["status"] == "released"
            recovered = coordinator.acquire(
                "value-agent-c", observed, listening=_listening(ports)
            )
            assert recovered["status"] == "granted"
            with socket.socket() as recovered_listener:
                recovered_listener.bind(
                    ("127.0.0.1", recovered["task"]["granted_service_port"])
                )
                recovered_listener.listen()
            recovery_sec = time.monotonic() - recovery_started
            coordinator.release("value-agent-c", recovered["task"]["fence_token"],
                                observed, listening=_listening(ports))
            connections[1].send("close")
            _receive(connections[1])
            connections[1].send("release")
            assert _receive(connections[1])["status"] == "released"
            events.append({"phase": "blocked_then_recovered", "initial": blocked,
                           "busy_release": busy_release, "still_blocked": still_blocked,
                           "released": released, "recovered": recovered,
                           "recovery_sec": recovery_sec})
        else:
            recovery_sec = None
            assert sum(row["bind"]["status"] == "listening" for row in grants) == 1
        return {
            "successful_listeners": sum(row["bind"]["status"] == "listening" for row in grants),
            "duplicate_device_selections": 2 - len({
                row["grant"]["task"]["granted_devices"][0] for row in grants
            }),
            "bind_conflicts": sum(row["bind"]["status"] == "failed" for row in grants),
            "elapsed_sec": time.monotonic() - started,
            "release_to_next_success_sec": recovery_sec,
            "events": events,
        }
    finally:
        for connection in connections:
            try:
                connection.send("stop")
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        for process in children:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)


def run(output: Path, observed: dict, repetitions: int):
    free = observed.get("free", [])
    if observed.get("status") != "ok" or len(free) < 2:
        return {"id": "coordinator-contention", "component": "coordinator",
                "status": "unsupported", "scope": "two available physical device identities required",
                "baseline": None, "candidate": None, "repetitions": 0, "measurements": [],
                "limitations": ["No NPU admission experiment ran."],
                "source_revisions": source_revisions()}
    selected = {**observed, "devices": free[:2], "free": free[:2], "busy": {}}
    measurements = []
    for repetition in range(1, repetitions + 1):
        root = output / f"coordinator-{repetition}"
        root.mkdir(parents=True)
        ports = _ports()
        baseline = _pair(root, selected, ports, False)
        candidate = _pair(root, selected, ports, True)
        raw = {"occupancy_input": selected, "ports": ports,
               "baseline": baseline, "candidate": candidate}
        write_json(root / "raw.json", raw)
        measurements.append({
            "repetition": repetition, "status": "pass", "raw": str(root / "raw.json"),
            "baseline": {k: v for k, v in baseline.items() if k != "events"},
            "candidate": {k: v for k, v in candidate.items() if k != "events"},
        })
    return {
        "id": "coordinator-contention", "component": "coordinator", "status": "pass",
        "scope": "two concurrent agents, live TCP ports, production SQLite allocation and recovery",
        "baseline": "Both agents select the first free device and port from the same observation.",
        "candidate": "Both agents submit to one production NpuCoordinator; a third waits for release.",
        "repetitions": repetitions, "measurements": measurements,
        "limitations": [
            "This is the check-then-act race, not a comparison against another atomic scheduler.",
            "Device identities come from observation; no NPU compute is launched or reserved globally.",
            "Ports, child processes and allocation state are owned by this local scenario.",
            "Elapsed time includes child startup and is not a model productivity estimate.",
        ],
        "source_revisions": source_revisions(),
    }
