"""Parent-side adapters for the existing coordinator, observer and diagnostics.

The coordinator remains the sole allocation authority. These adapters retain its
epoch and fencing tokens, and never replay an uncertain mutation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from pathlib import Path

PACKAGES = {
    "coordinator": "mindie-coordinator",
    "npu-top": "npu-top",
    "diagnostics": "mindie-diagnostics",
    "knowledge": "mindie-knowledge",
    "remote-dev": "remote-dev",
}

# Execute through remote-dev's existing transport, on the physical host. Only
# matching process identities are returned; no environment contents leave it.
HOST_OBSERVATION = r"""
import json, os, pathlib, subprocess, sys, time
request=json.load(sys.stdin)
boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
owner=request.get("owner")
def snapshot():
    result={}
    for process in pathlib.Path("/proc").iterdir():
        if not process.name.isdigit(): continue
        try:
            fields=(process/"stat").read_text().rsplit(") ",1)[1].split()
            result[int(process.name)]=dict(start_ticks=fields[19],
                pid_namespace=os.readlink(process/"ns/pid"))
        except (FileNotFoundError,ProcessLookupError):
            continue
    return result
before=snapshot()
if owner and request["require_owner"]:
    if (boot_id!=owner["boot_id"] or before.get(owner["pid"])!=
        dict(start_ticks=owner["start_ticks"],pid_namespace=owner["pid_namespace"])):
        raise RuntimeError("the attested namespace owner is no longer present")
result=subprocess.run(["npu-smi","info"],capture_output=True,text=True,timeout=20)
if result.returncode:
    raise RuntimeError("npu-smi info failed: "+result.stderr[-2000:])
after=snapshot()
if owner and request["require_owner"] and after.get(owner["pid"])!=before[owner["pid"]]:
    raise RuntimeError("the attested namespace owner changed during observation")
owned=set()
if owner:
    for pid in before.keys()|after.keys():
        prior,current=before.get(pid),after.get(pid)
        # The namespace owns descendants even when remote-dev replaces its
        # per-job environment token. A recycled PID never inherits ownership.
        if prior and current and prior!=current: continue
        identity=current or prior
        if identity["pid_namespace"]==owner["pid_namespace"]: owned.add(pid)
print(json.dumps(dict(npu_smi=result.stdout,owned_pids=sorted(owned),
    live_pids=sorted(after),observed_at=time.time(),boot_id=boot_id,
    ownership_basis="attested PID namespace and start ticks; before/after sample")))
"""

CONTAINER_ID = r"""
import json, subprocess, sys
request=json.load(sys.stdin)
identity=subprocess.run(["docker","inspect","--format","{{.Id}}",request["container"]],
    capture_output=True,text=True,check=True,timeout=15).stdout.strip()
print(json.dumps(dict(container_id=identity)))
"""


def package_metadata(name, source_paths=()):
    modules = {
        "mindie-coordinator": "mindie_coordinator",
        "npu-top": "npu_top",
        "mindie-diagnostics": "mindie_diagnostics",
        "mindie-knowledge": "mindie_knowledge",
        "remote-dev": "remote_dev",
    }
    module = importlib.util.find_spec(modules[name])
    for source in source_paths:
        root = Path(source).resolve()
        if module and module.origin and Path(module.origin).resolve().is_relative_to(root):
            project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
            if project["name"] != name:
                raise ValueError("explicit source package identity differs from its project metadata")
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            return dict(
                name=name,
                version=project["version"],
                revision=revision,
                installation="explicit_source",
                metadata_path=str(root / "pyproject.toml"),
                module_origin=module.origin,
                editable=True,
            )
    distribution = importlib.metadata.distribution(name)
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "revision": direct.get("vcs_info", {}).get("commit_id"),
        "editable": direct.get("dir_info", {}).get("editable", False),
        "metadata_path": str(distribution.locate_file("")),
        "installation": "installed",
    }


def physical_observation(
    parsed, authoritative, devices, owned_pids, memory_limit, *, idle, live_pids=None
):
    """Join the two real parsers by Phy-ID, without guessing card/chip indices."""
    if authoritative.get("status") != "ok" or authoritative.get("busy") is None:
        raise RuntimeError("coordinator occupancy parser could not establish process visibility")
    physical = {}
    for card in parsed["devices"]:
        chips = card.get("chips")
        if chips:
            for chip in chips:
                identity = chip.get("phy_id")
                if type(identity) is not int or identity in physical:
                    raise ValueError("npu-top physical chip mapping is missing or ambiguous")
                processes = [row for row in card["processes"] if row.get("phy_id") == identity]
                if any("phy_id" not in row for row in card["processes"]):
                    raise ValueError("npu-top process has no physical chip identity")
                physical[identity] = dict(
                    hbm=chip["hbm"], processes=processes, aicore_percent=chip.get("aicore_percent")
                )
        else:
            identity = card["npu_id"]
            if identity in physical:
                raise ValueError("npu-top physical device mapping is ambiguous")
            physical[identity] = dict(
                hbm=card["hbm"],
                processes=card["processes"],
                aicore_percent=card.get("aicore_percent"),
            )
    selected = {}
    for device in devices:
        if device not in authoritative["devices"] or device not in physical:
            raise RuntimeError(f"physical NPU {device} was not observed by both components")
        row = physical[device]
        used = row["hbm"].get("used_mb")
        total = row["hbm"].get("total_mb")
        if type(used) is not int or type(total) is not int or used < 0 or total <= 0:
            raise ValueError(f"NPU {device} has missing HBM telemetry")
        aicore = row["aicore_percent"]
        if type(aicore) not in (int, float) or not 0 <= aicore <= 100:
            raise ValueError(f"NPU {device} has missing or invalid AICore telemetry")
        pids = {process["pid"] for process in row["processes"]}
        authority_pids = {
            reason["pid"]
            for reason in authoritative["busy"].get(str(device), [])
            if reason.get("kind") == "process"
        }
        if pids != authority_pids:
            raise RuntimeError(f"component process observations disagree for physical NPU {device}")
        unowned = pids - set(owned_pids)
        unresolved = sorted(unowned - set(live_pids)) if live_pids is not None else []
        if unresolved:
            raise RuntimeError(f"NPU {device} process ownership vanished during sample: {unresolved}")
        foreign = sorted(unowned)
        if (
            foreign
            or (idle and (pids or aicore > 0))
            or (not pids and (used >= memory_limit or aicore > 0))
        ):
            raise RuntimeError(f"NPU {device} is occupied or unaccounted: foreign_pids={foreign}")
        selected[str(device)] = dict(
            used_mb=used, aicore_percent=aicore, pids=sorted(pids), foreign_pids=foreign
        )
    return selected


def _child(request):
    source_paths = request.get("source_paths", [])
    if any(not Path(path).is_absolute() or not Path(path).is_dir() for path in source_paths):
        raise ValueError("component source paths must be explicit existing absolute directories")
    sys.path[:0] = source_paths
    action = request["action"]
    if action == "metadata":
        return package_metadata(request["package"], source_paths)
    if action == "coordinator":
        from mindie_coordinator.host_queue import HostQueue

        return HostQueue().request(request["endpoint"], request["request"])
    if action in {"host_observation", "container_id"}:
        from remote_dev.core.endpoint import resolve_endpoint
        from remote_dev.core.ssh_transport import run_remote_python

        result = run_remote_python(
            resolve_endpoint(request["endpoint"]),
            HOST_OBSERVATION if action == "host_observation" else CONTAINER_ID,
            request["payload"],
            timeout_ms=30_000,
        )
        if result.get("status") in {"failed", "timeout", "cancelled"}:
            raise RuntimeError(json.dumps(result))
        return result
    if action == "parse_observation":
        from npu_top.npu_smi import parse_npu_smi_info

        return parse_npu_smi_info(request["text"])
    if action == "parse_authority":
        from mindie_coordinator.host_queue import parse_npu_smi_info

        return parse_npu_smi_info(request["text"])
    if action == "diagnostic":
        from mindie_diagnostics.logging import Recorder

        recorder = Recorder("mindie-bench", root=request["root"])
        recorder.event("INFO", request["event"], **request["attributes"])
        result = dict(
            record_ref=recorder.record_ref,
            logging_failed=recorder.logging_failed,
            dropped_records=recorder.dropped_records,
        )
        recorder.close()
        if result["logging_failed"] or result["dropped_records"] or not result["record_ref"]:
            raise RuntimeError("required diagnostics event was not recorded")
        return result
    raise ValueError(f"unsupported component adapter action: {action}")


class ComponentRuntime:
    """A slot's component handles, not a second allocation or scheduling ledger."""

    def __init__(self, config, trial_id, container, devices, memory_limit, artifact_dir):
        self.config = config
        self.trial_id, self.container, self.devices = trial_id, container, devices
        self.memory_limit, self.artifact_dir = memory_limit, artifact_dir
        self.marker = uuid.uuid4().hex
        self.trace_id = uuid.uuid4().hex
        self.epoch = None
        self.leases = {}
        self.ports = {}
        self.unknown = False
        self.failure = None
        self.stop = threading.Event()
        self.monitor = None
        self.owner_identity = None
        self.rpc_lock = threading.Lock()
        self.journal_lock = threading.Lock()
        self._validate_config()

    def _validate_config(self):
        if self.artifact_dir is None:
            raise ValueError("component runtime requires a private trial artifact directory")
        for component in PACKAGES:
            path = self.config["interpreters"][component]
            if not Path(path).is_absolute() or not Path(path).is_file():
                raise ValueError(f"{component} interpreter must be an existing absolute path")
        root = Path(self.config["diagnostics_root"])
        if not root.is_absolute():
            raise ValueError("diagnostics_root must be an absolute private path")
        self.interval = self.config.get("heartbeat_interval_seconds", 10)
        self.ttl = self.config.get("heartbeat_ttl_seconds", 120)
        if not 1 <= self.interval <= 30 or self.ttl < 3 * self.interval:
            raise ValueError("heartbeat TTL must cover at least three bounded observation intervals")
        for name in ("master", "hccl"):
            ports = self.config["port_ranges"][name]
            if (
                not isinstance(ports, list)
                or not ports
                or len(ports) > 4096
                or any(type(port) is not int or not 1024 <= port < 65536 for port in ports)
                or len(set(ports)) != len(ports)
            ):
                raise ValueError("port_ranges must contain distinct unprivileged TCP ports")
        if set(self.config["port_ranges"]["master"]) & set(self.config["port_ranges"]["hccl"]):
            raise ValueError("MASTER_PORT and HCCL port choices must be disjoint")

    def _journal(self, event, **facts):
        row = dict(event=event, monotonic_sec=time.monotonic(), trace_id=self.trace_id, **facts)
        with self.journal_lock:
            with (self.artifact_dir / "component_runtime.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _call(self, component, action, **payload):
        request = dict(
            action=action,
            source_paths=self.config.get("python_paths", {}).get(component, []),
            **payload,
        )
        read_fd, owner_fd = os.pipe()
        process = None
        primary = None
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "mindie_bench.runner.owner",
                    str(read_fd),
                    self.config["interpreters"][component],
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(read_fd,),
                env={
                    **os.environ,
                    "REMOTE_DEV_SSH_MUX_DIR": str(self.artifact_dir / "component-ssh-mux"),
                    "MINDIE_DIAGNOSTICS_ROOT": self.config["diagnostics_root"],
                    "MINDIE_DIAGNOSTICS_CONTEXT": json.dumps(
                        {
                            "trace_id": self.trace_id,
                            "operation_id": self.trace_id,
                        }
                    ),
                },
            )
            os.close(read_fd)
            read_fd = -1
            stdout, stderr = process.communicate(json.dumps(request).encode(), timeout=60)
            if len(stdout) > 8 * 1024 * 1024 or len(stderr) > 1024 * 1024:
                raise RuntimeError("component result exceeded the evidence bound")
            if process.returncode:
                raise RuntimeError(
                    f"{component}/{action} exited {process.returncode}: "
                    + stderr.decode(errors="replace")
                )
            result = json.loads(stdout)
            if not isinstance(result, dict):
                raise ValueError("component adapter returned a non-object")
            return result
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            os.close(owner_fd)
            if process:
                try:
                    # Closing the existing owner pipe reclaims descendants.
                    # A reaped leader's process-group number is no longer an
                    # owned signal target, even if a zombie still exposes it.
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    except BaseException as cleanup:
                        if primary is not None:
                            primary.add_note(f"component helper cleanup also failed: {cleanup}")
                        else:
                            raise RuntimeError(
                                f"component helper cleanup failed: {cleanup}"
                            ) from cleanup

    def _authority(self, action, **fields):
        started = time.monotonic()
        mutation = action != "status"
        if mutation and self.unknown:
            raise RuntimeError("coordinator mutation outcome is unknown; reconcile without replay")
        request = dict(action=action, **fields)
        if self.epoch:
            request["coordination_epoch"] = self.epoch
        if self.config.get("coordinator_state_dir"):
            request["state_dir"] = self.config["coordinator_state_dir"]
        request["_mutation"] = mutation
        self._journal("coordinator_request", request=request)
        try:
            response = self._call(
                "coordinator", "coordinator", endpoint=self.config["endpoint"], request=request
            )
        except Exception as exc:
            if mutation:
                self.unknown = True
            self._journal(
                "coordinator_error",
                action=action,
                unknown=self.unknown,
                error=str(exc),
                elapsed_sec=time.monotonic() - started,
            )
            raise
        self._journal(
            "coordinator_response",
            action=action,
            response=response,
            elapsed_sec=time.monotonic() - started,
        )
        return response

    def acquire(self):
        for component, package in PACKAGES.items():
            actual = self._call(component, "metadata", package=package)
            self._journal("component_provenance", component=component, actual=actual)
            if actual.get("editable") and actual.get("installation") != "explicit_source":
                raise RuntimeError(f"{component} editable installation must declare its source path")
            for key, expected in self.config.get("expected_components", {}).get(component, {}).items():
                if key not in {"version", "revision"} or actual.get(key) != expected:
                    raise RuntimeError(
                        f"{component} installed {key} does not match the selected runtime"
                    )
        observed = self._authority("status", no_probe=True, event_limit=0)
        if not isinstance(observed.get("coordination_epoch"), str):
            raise RuntimeError("coordinator did not return its authority epoch")
        self.epoch = observed["coordination_epoch"]
        self.observe("admission", idle=True)
        for kind in ("master", "hccl"):
            task_id = self.trial_id + "." + kind
            self.leases[kind] = dict(task_id=task_id)
            response = self._authority(
                "submit-acquire-preflight",
                task_id=task_id,
                agent_id="mindie-bench",
                session_id=self.trace_id,
                container_name=self.container,
                **({"devices": self.devices} if kind == "master" else {"npu_count": 0}),
                service_port=0,
                service_ports=self.config["port_ranges"][kind],
                start_ttl_seconds=120,
                allow_external_busy=False,
            )
            task = response.get("task", {})
            # Even a queued admission is retained for this exact task identity.
            self.leases[kind] = task
            if (
                response.get("status") != "starting"
                or task.get("granted_devices") != (self.devices if kind == "master" else [])
                or type(task.get("fence_token")) is not int
                or task.get("granted_service_port") not in self.config["port_ranges"][kind]
            ):
                raise RuntimeError("coordinator did not grant the exact requested devices and port")
            self.ports[kind] = task["granted_service_port"]
        self.diagnostic("slot.lease_acquired")
        return dict(
            MASTER_PORT=str(self.ports["master"]),
            HCCL_IF_BASE_PORT=str(self.ports["hccl"]),
            REMOTE_DEV_JOB_TOKEN=self.marker,
        )

    def diagnostic(self, event):
        started = time.monotonic()
        result = self._call(
            "diagnostics",
            "diagnostic",
            root=self.config["diagnostics_root"],
            event=event,
            attributes=dict(
                trace_id=self.trace_id,
                operation_id=self.trace_id,
                selected_device_count=len(self.devices),
                unknown=self.unknown,
            ),
        )
        self._journal(
            "diagnostics", event_name=event, result=result, elapsed_sec=time.monotonic() - started
        )

    def activate(self, receipt, on_failure):
        if (
            receipt.get("process_guard", {}).get("marker") != self.marker
            or receipt.get("process_guard", {}).get("retain_until_release") is not True
            or not isinstance(receipt.get("pid_namespace"), str)
        ):
            raise RuntimeError("worker process guard does not match its allocated owner")
        identity = self._call(
            "remote-dev",
            "container_id",
            endpoint=self.config["endpoint"],
            payload=dict(container=self.container),
        )
        prepared = dict(
            container_name=self.container, container_id=identity["container_id"], receipt=receipt
        )
        host_pids = set()
        for task in self.leases.values():
            response = self._authority(
                "activate",
                task_id=task["task_id"],
                fence_token=task["fence_token"],
                process_guard=receipt["process_guard"],
                prepared_supervisor=prepared,
                heartbeat_ttl_seconds=self.ttl,
            )
            if response.get("status") != "active":
                raise RuntimeError("coordinator failed to activate the guarded slot")
            host_pids.add(response["task"]["pid"])
        if len(host_pids) != 1:
            raise RuntimeError("coordinator activation disagrees about the namespace owner")
        self.owner_identity = {
            "pid": host_pids.pop(),
            "boot_id": receipt["boot_id"],
            "start_ticks": receipt["start_ticks"],
            "pid_namespace": receipt["pid_namespace"],
        }
        self.diagnostic("slot.lease_active")

        def watch():
            while not self.stop.wait(self.interval):
                try:
                    with self.rpc_lock:
                        for task in self.leases.values():
                            result = self._authority(
                                "heartbeat",
                                task_id=task["task_id"],
                                fence_token=task["fence_token"],
                                heartbeat_ttl_seconds=self.ttl,
                            )
                            if result.get("status") != "active":
                                raise RuntimeError("coordinator no longer confirms the slot's lease")
                        self.observe("active", idle=False)
                except Exception as exc:
                    self.failure = str(exc)
                    self._journal("monitor_failed", error=self.failure, unknown=self.unknown)
                    on_failure()
                    return

        self.monitor = threading.Thread(target=watch, name="bench-component-monitor", daemon=True)
        self.monitor.start()

    def observe(self, phase, *, idle):
        started = time.monotonic()
        observed = self._call(
            "remote-dev",
            "host_observation",
            endpoint=self.config["endpoint"],
            payload=dict(owner=self.owner_identity, require_owner=not idle),
        )
        parsed = self._call("npu-top", "parse_observation", text=observed["npu_smi"])
        authority = self._call("coordinator", "parse_authority", text=observed["npu_smi"])
        try:
            selected = physical_observation(
                parsed,
                authority,
                self.devices,
                observed["owned_pids"],
                self.memory_limit,
                idle=idle,
                live_pids=observed["live_pids"],
            )
        except Exception as exc:
            self._journal(
                "npu_observation_rejected",
                phase=phase,
                error=str(exc),
                observed_at=observed["observed_at"],
                observer=parsed,
                authority=authority,
                owned_pids=observed["owned_pids"],
                selected_devices=self.devices,
                elapsed_sec=time.monotonic() - started,
            )
            raise
        self._journal(
            "npu_observation",
            phase=phase,
            observed_at=observed["observed_at"],
            selected_devices=selected,
            allocation_authority=False,
            npu_smi_sha256=hashlib.sha256(observed["npu_smi"].encode()).hexdigest(),
            basis="npu-top + coordinator parsers of the same live host sample",
            ownership_basis=observed["ownership_basis"],
            elapsed_sec=time.monotonic() - started,
        )
        return selected

    def assert_healthy(self):
        if self.failure or self.unknown:
            raise RuntimeError(
                f"component lease/observation failed: {self.failure or 'unknown mutation'}"
            )

    def stop_monitor(self):
        self.stop.set()
        if self.monitor:
            self.monitor.join(timeout=65)
            if self.monitor.is_alive():
                raise RuntimeError("component monitor did not settle; lease retained")

    def finish(self, *, cleanup_confirmed):
        self.stop_monitor()
        if not cleanup_confirmed:
            self._journal("lease_retained", reason="remote namespace cleanup is unconfirmed")
            raise RuntimeError("remote cleanup unconfirmed; coordinator lease retained")
        # A lost acknowledgement is never resolved by repeating its mutation.
        if self.unknown:
            for task in self.leases.values():
                self._authority("status", task_id=task["task_id"], no_probe=True)
            raise RuntimeError("coordinator mutation unknown; lease requires read-only reconciliation")
        self.observe("release_preflight", idle=True)
        for task in self.leases.values():
            if type(task.get("fence_token")) is int:
                result = self._authority(
                    "release",
                    task_id=task["task_id"],
                    fence_token=task["fence_token"],
                    completion_confirmed=True,
                )
                if result.get("status") != "released":
                    raise RuntimeError("coordinator retained busy or uncertain resources")
            elif task.get("state") == "queued":
                result = self._authority("cancel", task_id=task["task_id"])
                if result.get("status") != "cancelled":
                    raise RuntimeError("coordinator did not cancel this unlaunched queued task")
        self.diagnostic("slot.lease_released")
        if self.failure:
            raise RuntimeError(f"slot monitoring failed before cleanup: {self.failure}")


if __name__ == "__main__":
    print(json.dumps(_child(json.load(sys.stdin))))
