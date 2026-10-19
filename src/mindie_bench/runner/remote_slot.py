"""One SSH-owned PID/network namespace with cooperative per-device NPU leases.

[Role]
Manages an isolated remote Ascend NPU trial workspace inside a dedicated container using
`flock` per-device locks, detached Git worktrees, source-projection stripping, and `bwrap`
PID/network/IPC namespaces.

[Invariants]
- Only the public source snapshot reaches the remote slot (`.git` metadata and harness configs
  are stripped before agent execution and restored only inside `POSTFLIGHT` diff extraction).
- Any unacknowledged frame or broken SSH pipe sets `operation_unknown = True`, blocking further
  submissions on the slot and surfacing `UnknownExternalEffects` to the orchestrator.
- `TMPDIR` is bound to a short `/tmp/bench-<hex12>` path backed by `.bench-home/tmp` so vLLM
  native TP `sockaddr_un` Unix domain sockets never overflow the 104-byte path limit.

[Timeouts & Locks]
- Per-command worker timeout: `outer_timeout = float(request["timeout"]) + 10.0s` (`outer >= inner + 10s`),
  giving structured inner commands (`_PRIVATE_REMOTE_JUDGE`) a 10s grace window to finish post-timeout
  inventory before `WORKER` kills the child process.
- Host-side `_request` pipe wait: `limit + 15.0s` (`outer_timeout + 5.0s`).
- Slot startup readiness wait: `40.0s`; slot teardown (`POSTFLIGHT` + NPU HBM/PID drain) wait: `40.0s`.

[Side-Effect Policy]
- Acquires `flock -n` on `/tmp/mindie-bench-device-<id>.lock` (and optional `ComponentRuntime`
  coordinator leases) before creating the worktree; `trap POSTFLIGHT EXIT` kills remaining slot
  processes via `pidfd_send_signal`, verifies NPU HBM/PID drain, extracts `patch_b64`, and removes
  the worktree.
"""

from __future__ import annotations

import base64
import json
import os
import select
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from mindie_bench.io import is_ignored_cache_path, safe_relative
from mindie_bench.models import RemoteConfig, SourceEnvironment
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS, PROJECTOR

_RESERVED_SLOT_ENV = {
    "REMOTE_DEV_JOB_TOKEN",
    "MINDIE_BENCH_SLOT_ID",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_CACHE_PATH",
    "CCACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TMPDIR",
    "HCCL_IF_BASE_PORT",
    "MASTER_PORT",
    "MINDIE_BENCH_COMPONENT_GUARD",
}

WORKER = r"""
import base64, json, os, queue, select, subprocess, sys, threading, uuid
MAX_STREAM_BYTES=8*1024*1024
MAX_REQUEST_BYTES=96*1024*1024
requests=queue.Queue()
streams={}
def read_owner():
    while True:
        line=sys.stdin.buffer.readline(MAX_REQUEST_BYTES+1)
        if not line:
            break
        if len(line)>MAX_REQUEST_BYTES:
            requests.put({"frame_error":"remote request exceeds evidence bound"})
            while line and not line.endswith(b"\n"):
                line=sys.stdin.buffer.readline(MAX_REQUEST_BYTES+1)
            continue
        requests.put(line)
    # EOF is owner loss, including while a command is running. Exiting the
    # PID-namespace leader lets bubblewrap kill all its detached descendants.
    os._exit(0)
def write_stream(state):
    try:
        while True:
            data=state["input"].get()
            if data is None:
                state["process"].stdin.close()
                return
            try:
                remaining=memoryview(data)
                while remaining:
                    size=os.write(state["process"].stdin.fileno(),remaining[:65536])
                    if not size:
                        raise BrokenPipeError("remote stream accepted no input bytes")
                    remaining=remaining[size:]
            finally:
                with state["lock"]:
                    state["pending"]-=len(data)
    except Exception as exc:
        with state["lock"]:
            state["write_error"]=type(exc).__name__+": "+str(exc)
def open_stream(command):
    if not isinstance(command,str) or len(command.encode())>1024*1024:
        raise ValueError("invalid remote stream command")
    if len(streams)>=32:
        raise ValueError("remote stream process bound exceeded")
    process=subprocess.Popen(["/bin/bash","-c",command],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    state=dict(process=process,input=queue.Queue(maxsize=32),lock=threading.Lock(),
        pending=0,eof=False,write_error=None,pipes={})
    for name in ["stdout","stderr"]:
        pipe=getattr(process,name)
        os.set_blocking(pipe.fileno(),False)
        state["pipes"][pipe.fileno()]=(name,pipe)
    identity=uuid.uuid4().hex
    streams[identity]=state
    threading.Thread(target=write_stream,args=(state,),daemon=True).start()
    return {"stream_id":identity}
def exchange_stream(request):
    identity=request["stream_id"]
    state=streams[identity]
    data=base64.b64decode(request.get("stdin_b64",""),validate=True)
    eof=request.get("eof",False)
    if type(eof) is not bool or len(data)>MAX_STREAM_BYTES:
        raise ValueError("invalid or oversized remote stream input")
    with state["lock"]:
        if state["write_error"]:
            raise RuntimeError(state["write_error"])
        if state["eof"] and (data or eof):
            raise ValueError("remote stream input is already closed")
        needed=int(bool(data))+int(eof)
        if (state["pending"]+len(data)>MAX_STREAM_BYTES
                or state["input"].qsize()+needed>32):
            raise ValueError("remote stream pending input bound exceeded")
        state["pending"]+=len(data)
        if data:
            state["input"].put_nowait(data)
        if eof:
            state["eof"]=True
            state["input"].put_nowait(None)
    output={"stdout":[],"stderr":[]}
    size=0
    while state["pipes"] and size<MAX_STREAM_BYTES:
        ready,_,_=select.select(list(state["pipes"]),[],[],.05 if not size else 0)
        if not ready:
            break
        for fd in ready:
            try:
                chunk=os.read(fd,min(65536,MAX_STREAM_BYTES-size))
            except BlockingIOError:
                continue
            if chunk:
                output[state["pipes"][fd][0]].append(chunk)
                size+=len(chunk)
            else:
                state["pipes"].pop(fd)[1].close()
            if size>=MAX_STREAM_BYTES:
                break
    with state["lock"]:
        if state["write_error"]:
            raise RuntimeError(state["write_error"])
    code=state["process"].poll()
    if state["pipes"]:
        code=None
    elif code is not None:
        with state["lock"]:
            if not state["eof"] and not state["pending"]:
                state["eof"]=True
                state["input"].put_nowait(None)
        streams.pop(identity)
    return dict(exit_code=code,
        stdout_b64=base64.b64encode(b"".join(output["stdout"])).decode(),
        stderr_b64=base64.b64encode(b"".join(output["stderr"])).decode())
threading.Thread(target=read_owner, daemon=True).start()
ready={"ready":True}
marker=os.environ.get("REMOTE_DEV_JOB_TOKEN")
if os.environ.get("MINDIE_BENCH_COMPONENT_GUARD")=="1":
    boot_id=open("/proc/sys/kernel/random/boot_id").read().strip()
    fields=open("/proc/self/stat").read().rsplit(") ",1)[1].split()
    guard=dict(marker=marker,boot_id=boot_id,retain_until_release=True)
    ready["owner_receipt"]=dict(pid=os.getpid(),start_ticks=fields[19],
        boot_id=boot_id,process_guard=guard,pid_namespace=os.readlink("/proc/self/ns/pid"))
print(json.dumps(ready), flush=True)
while True:
    line=requests.get()
    try:
        if isinstance(line,dict):
            raise ValueError(line["frame_error"])
        request = json.loads(line)
        if request.get("control")=="quiesce":
            for item in os.listdir("/proc"):
                if item.isdigit() and int(item) not in {os.getpid(),os.getppid()}:
                    try:
                        os.kill(int(item),9)
                    except ProcessLookupError:
                        pass
            print(json.dumps({"quiesced":True}),flush=True)
            continue
        if request.get("control")=="open_stream":
            value=open_stream(request["command"])
        elif request.get("control")=="exchange_stream":
            value=exchange_stream(request)
        else:
            outer_timeout = float(request["timeout"]) + 10.0 if request.get("timeout") is not None else None
            result = subprocess.run(["/bin/bash", "-c", request["command"]],
                input=base64.b64decode(request["stdin_b64"]), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=outer_timeout)
            value = dict(exit_code=result.returncode, stdout_b64=base64.b64encode(result.stdout).decode(),
                stderr_b64=base64.b64encode(result.stderr).decode())
    except Exception as exc:
        value = dict(error=type(exc).__name__, message=str(exc))
    print(json.dumps(value), flush=True)
"""

POSTFLIGHT = r"""
import base64, json, os, pathlib, signal, subprocess, sys, time
cfg=json.loads(sys.argv[1])
path=pathlib.Path(cfg["path"])
errors=[]
diff_paths=["--",".",*[":(exclude)"+name for name in cfg["source_projection"]]]
token=("MINDIE_BENCH_SLOT_ID="+path.name).encode()
# Only this allocated slot's exact environment token is eligible for cleanup.
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():
        continue
    try:
        if proc.stat().st_uid!=os.getuid():
            continue
        fd=os.pidfd_open(int(proc.name))
    except (FileNotFoundError, ProcessLookupError):
        continue
    try:
        if token in (proc/"environ").read_bytes().split(b"\0"):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass
    except Exception as exc:
        errors.append(type(exc).__name__+": "+str(exc))
    finally:
        os.close(fd)
probe=None
probe_errors=[]
for attempt in range(20):
    try:
        probe=json.loads(subprocess.run(cfg["probe_command"],check=True,capture_output=True,
            text=True,timeout=10).stdout)
        if all(probe[str(d)]["used_mb"]<cfg["memory_limit_mb"] and not probe[str(d)]["pids"]
               for d in cfg["devices"]):
            break
    except Exception as exc:
        probe_errors.append(type(exc).__name__+": "+str(exc))
        break
    time.sleep(.5)
final={"probe":probe,"errors":errors}
if probe_errors:
    final["probe_errors"]=probe_errors
if path.is_dir():
    try:
        common=subprocess.run(["git","-C",cfg["repo_path"],"rev-parse","--git-common-dir"],
            check=True,capture_output=True,text=True).stdout.strip()
        common=pathlib.Path(common)
        if not common.is_absolute():
            common=pathlib.Path(cfg["repo_path"])/common
        matches=[p for p in (common/"worktrees").iterdir()
                 if (p/"gitdir").read_text().strip()==str(path/".git")]
        if len(matches)!=1:
            raise RuntimeError("cannot identify this slot's worktree metadata")
        gitfile=path/".git"
        if gitfile.is_symlink() or gitfile.is_file():
            gitfile.unlink()
        if gitfile.exists():
            raise RuntimeError("agent created an unexpected .git directory")
        gitfile.write_text("gitdir: "+str(matches[0])+"\n")
        patch=subprocess.run(["git","-C",str(path),"diff","--binary","--no-ext-diff","--no-textconv",
            "HEAD",*diff_paths],check=True,capture_output=True).stdout
        if len(patch)>32*1024*1024:
            raise RuntimeError("remote final patch exceeds artifact bound")
        final["patch_b64"]=base64.b64encode(patch).decode()
        changed=subprocess.run(["git","-C",str(path),"diff","--name-only","-z",
            "--no-ext-diff","--no-textconv","HEAD",*diff_paths],
            check=True,capture_output=True).stdout.split(b"\0")
        untracked=subprocess.run(["git","-C",str(path),"ls-files","--others","--exclude-standard","-z",
            *diff_paths],
            check=True,capture_output=True).stdout.split(b"\0")
        final["untracked"]={}
        size=0
        ignored_parts={"__pycache__",".pytest_cache",".ruff_cache",".mypy_cache"}
        ignored_suffixes={".pyc",".pyo"}
        for name in untracked:
            if not name or name.startswith(b".bench-home/"):
                continue
            rel_str=os.fsdecode(name)
            rel_path=pathlib.PurePosixPath(rel_str)
            if ignored_parts.intersection(rel_path.parts) or rel_path.suffix.lower() in ignored_suffixes:
                continue
            item=path/rel_str
            if item.is_symlink() or not item.resolve().is_relative_to(path.resolve()):
                raise RuntimeError("untracked artifact escapes slot")
            data=item.read_bytes()
            size+=len(data)
            if size>32*1024*1024:
                raise RuntimeError("remote untracked artifact bound exceeded")
            final["untracked"][rel_str]=base64.b64encode(data).decode()
        final["files_modified"]=sorted(set(
            [os.fsdecode(name) for name in changed if name]+list(final["untracked"])))
        subprocess.run(["git","-C",cfg["repo_path"],"worktree","remove","--force",str(path)],
            check=True,capture_output=True)
    except Exception as exc:
        final["errors"].append(type(exc).__name__+": "+str(exc))
artifact=str(path)+".final.json"
pathlib.Path(artifact).write_text(json.dumps(final))
print(json.dumps({"cleanup":final,"remote_artifact":artifact}),flush=True)
"""

KDA_BUILD_CACHE_HELPER = r"""
import hashlib, json, os, pathlib, shutil, subprocess, sys
GENERATED = {"build", "build_out", "output"}
sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def current_sources(root):
    return {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.relative_to(root).parts[0] not in GENERATED and not p.is_symlink() and p.is_file()
    }
receipt_path, csrc, cache_dir = pathlib.Path("../build-receipt.json"), pathlib.Path("csrc"), None
if receipt_path.is_file() and csrc.is_dir():
    try:
        rec = json.loads(receipt_path.read_text())
        if (rec.get("schema") == "mindie-bench-kda-source-build/1" and rec.get("status") == "prepared"
                and current_sources(csrc) == rec.get("source_files_sha256")):
            identity = {"source_files_sha256": rec["source_files_sha256"],
                        "input_archives_sha256": rec.get("input_archives_sha256"),
                        "generic_dependency": rec.get("generic_dependency"),
                        "command": rec.get("command"), "args": sys.argv[1:]}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache_dir = pathlib.Path(os.environ["HOME"]) / "cache" / "kda_build_out" / key
            manifest_path = cache_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                cached_pkg = cache_dir / manifest["name"]
                if cached_pkg.is_file() and sha256(cached_pkg) == manifest["sha256"]:
                    build_out = csrc / "build_out"
                    build_out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached_pkg, build_out / manifest["name"])
                    raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        cache_dir = None
res = subprocess.run(["/bin/bash", *sys.argv[1:]])
if res.returncode == 0 and cache_dir is not None:
    try:
        pkgs = list((csrc / "build_out").glob("*.run"))
        if len(pkgs) == 1:
            pkg = pkgs[0]
            tmp_dir = cache_dir.with_suffix(".tmp-" + str(os.getpid()))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pkg, tmp_dir / pkg.name)
            (tmp_dir / "manifest.json").write_text(json.dumps({"name": pkg.name, "sha256": sha256(pkg)}))
            if not cache_dir.exists():
                tmp_dir.rename(cache_dir)
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
raise SystemExit(res.returncode)
"""


class RemoteSlot:
    def __init__(
        self,
        config: RemoteConfig,
        environment: SourceEnvironment,
        trial_id: str,
        devices: list[int],
        timeout: float,
        artifact_dir: Path | None = None,
    ):
        self.config, self.environment = config, environment
        self.devices, self.timeout = devices, timeout
        self.path = str(PurePosixPath(config.slots_root) / trial_id)
        self.artifact_dir = artifact_dir
        self.lock = threading.Lock()
        self.process = self.preflight = self.postflight = None
        self.cleanup_receipt = self.modified_files = self.stderr_file = None
        self.component_runtime = self.slot_environment = self.tmpdir_backing = None
        self.created = self.closed = self.operation_unknown = False
        self.cleanup_confirmed = False
        self.public_files: dict[str, str] = {}
        self.read_buffer = b""

    def command(self, command: list[str]) -> list[str]:
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=20",
            self.config.host,
            shlex.join(["docker", "exec", "-i", self.config.container, *command]),
        ]

    def probe(self) -> dict:
        result = subprocess.run(
            self.command(self.config.probe_command),
            capture_output=True,
            timeout=30,
            check=True,
            text=True,
        )
        return self.validate_probe(json.loads(result.stdout))

    def validate_probe(self, data: dict) -> dict:
        environment = data.get("_environment")
        required = {"image_digest", "cann", "torch_npu", "driver", "python", "kernel"}
        if (
            not isinstance(environment, dict)
            or not required <= environment.keys()
            or any(not isinstance(environment[k], str) or not environment[k].strip() for k in required)
        ):
            raise ValueError("NPU probe lacks an exact image/runtime software fingerprint")
        for device in self.devices:
            row = data[str(device)]
            if (
                type(row.get("used_mb")) is not int
                or row["used_mb"] < 0
                or not isinstance(row.get("pids"), list)
                or not isinstance(row.get("model"), str)
            ):
                raise ValueError("invalid NPU probe; expected device -> {used_mb, pids, model}")
            if row["used_mb"] >= self.config.memory_limit_mb or row["pids"]:
                raise RuntimeError(f"NPU device {device} is occupied")
            if row["model"] != self.environment.npu_model:
                raise RuntimeError(f"NPU model mismatch for device {device}")
        return {
            **{str(device): data[str(device)] for device in self.devices},
            "_environment": {name: environment[name] for name in sorted(required)},
        }

    def prepare(self):
        quote = shlex.quote
        runtime_config = getattr(self.config, "component_runtime", None)
        extra_environment = {}
        if runtime_config is not None:
            from mindie_bench.integrations.component_runtime import ComponentRuntime

            self.component_runtime = ComponentRuntime(
                runtime_config,
                PurePosixPath(self.path).name,
                self.config.container,
                self.devices,
                self.config.memory_limit_mb,
                self.artifact_dir,
            )
            extra_environment = self.component_runtime.acquire()
            extra_environment["MINDIE_BENCH_COMPONENT_GUARD"] = "1"
        cache_root = self.path + "/.bench-home"
        self.tmpdir_backing = cache_root + "/tmp"
        self.slot_environment = {
            "ASCEND_CACHE_PATH": cache_root + "/cache/ascend",
            "CCACHE_DIR": cache_root + "/cache/ccache",
            "TORCH_EXTENSIONS_DIR": cache_root + "/cache/torch_extensions",
            "TORCHINDUCTOR_CACHE_DIR": cache_root + "/cache/torchinductor",
            "TRITON_CACHE_DIR": cache_root + "/cache/triton",
            # Native TP uses Unix sockets with fixed-length paths. This short
            # namespace-private mount still writes to the slot's owned home.
            "TMPDIR": "/tmp/bench-" + uuid.uuid4().hex[:12],
            **extra_environment,
        }
        environment_exports = "\n".join(
            f"export {name}={quote(value)}" for name, value in self.slot_environment.items()
        )
        cache_keys = (
            "ASCEND_CACHE_PATH",
            "CCACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TMPDIR",
        )
        cache_directories = " ".join(
            quote(self.tmpdir_backing if name == "TMPDIR" else self.slot_environment[name])
            for name in cache_keys
        )
        locks = "\n".join(
            f"exec {10 + i}>/tmp/mindie-bench-device-{device}.lock\nflock -n {10 + i}"
            for i, device in enumerate(self.devices)
        )
        bind = " ".join(
            "--ro-bind " + quote(path) + " " + quote(path) for path in self.config.runtime_read_paths
        )
        device_bind = (
            "for device in /dev/davinci[0-9]* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do\n"
            ' test ! -e "$device" || device_args+=(--dev-bind "$device" "$device")\n'
            "done"
        )
        cleanup_config = dict(
            path=self.path,
            repo_path=self.config.repo_path,
            devices=self.devices,
            memory_limit_mb=self.config.memory_limit_mb,
            probe_command=self.config.probe_command,
            source_projection=HARNESS_CONFIG_PATHS,
        )
        cleanup = shlex.join(["python3", "-I", "-c", POSTFLIGHT, json.dumps(cleanup_config)])
        slot_bin = cache_root + "/bin"
        bash_shim = (
            "#!/bin/bash\n"
            'if [ "${1:-}" = "csrc/build.sh" ] && [ -f "../build-receipt.json" ]; then\n'
            f'  exec python3 -I -B {quote(slot_bin + "/kda_build_cache.py")} "$@"\n'
            "fi\n"
            'exec /bin/bash "$@"\n'
        )
        command = f"""set -eu
{locks}
python3 -I -c 'import os,signal; fd=os.pidfd_open(os.getpid()); os.close(fd); assert hasattr(signal,"pidfd_send_signal")'
mkdir -p {quote(self.config.slots_root)}
trap {quote(cleanup)} EXIT
test ! -e {quote(self.path)}
git -C {quote(self.config.repo_path)} cat-file -e {quote(self.environment.base_commit + "^{commit}")}
mkdir -p {quote(self.config.slots_root)}
git -C {quote(self.config.repo_path)} worktree add --detach {quote(self.path)} {quote(self.environment.base_commit)} >/dev/null 2>&1
rm {quote(self.path + "/.git")}
{shlex.join(["python3", "-I", "-c", PROJECTOR, self.path])}
mkdir {quote(self.path + "/.bench-home")}
mkdir -p {cache_directories} {quote(slot_bin)}
python3 -I -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text(sys.argv[2]); (p.parent/"bash").write_text(sys.argv[3]); (p.parent/"bash").chmod(0o755)' {quote(slot_bin + "/kda_build_cache.py")} {quote(KDA_BUILD_CACHE_HELPER)} {quote(bash_shim)}
export PATH={quote(slot_bin)}:"$PATH"
export MINDIE_BENCH_SLOT_ID={quote(PurePosixPath(self.path).name)}
export ASCEND_RT_VISIBLE_DEVICES={quote(",".join(map(str, self.devices)))}
{environment_exports}
device_args=()
{device_bind}
bwrap --die-with-parent --unshare-pid --unshare-net --unshare-ipc --unshare-uts \\
 --cap-add ALL --new-session --proc /proc --dev /dev "${{device_args[@]}}" \\
 --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind-try /lib64 /lib64 \\
 --ro-bind /etc /etc --bind /sys /sys --tmpfs /tmp --tmpfs /var {bind} \\
 --bind {quote(self.path)} {quote(self.path)} \\
 --bind {quote(self.tmpdir_backing)} {quote(self.slot_environment["TMPDIR"])} \\
 --chdir {quote(self.path)} --setenv HOME {quote(self.path + "/.bench-home")} \\
 -- python3 -I -u -c {quote(WORKER)}
"""
        if self.artifact_dir:
            self.stderr_file = (self.artifact_dir / "remote.stderr").open("xb")
        self.process = subprocess.Popen(
            self.command(["/bin/bash", "-c", command]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file or subprocess.DEVNULL,
            start_new_session=True,
            bufsize=0,
        )
        ready = self._read(40)
        if "cleanup" in ready:
            self.cleanup_receipt = ready
        if not isinstance(ready, dict) or ready.get("ready") is not True:
            raise RuntimeError("remote worker failed to enter its isolated slot")
        self.created = True
        self.preflight = self.probe()
        if self.component_runtime:

            def stop_owned_worker():
                self.operation_unknown = True
                try:
                    self.process.stdin.close()
                except OSError as exc:
                    self.component_runtime.failure += f"; owner-pipe close: {exc}"

            self.component_runtime.activate(ready.get("owner_receipt", {}), stop_owned_worker)
        if self.artifact_dir:
            public_env = {
                k: v
                for k, v in self.slot_environment.items()
                if k not in {"REMOTE_DEV_JOB_TOKEN", "MINDIE_BENCH_COMPONENT_GUARD"}
            }
            (self.artifact_dir / "slot_isolation.json").write_text(
                json.dumps({"TMPDIR_BACKING_PATH": self.tmpdir_backing, **public_env})
            )
        return self

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.read_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("remote worker acknowledgement lost; effects unknown")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("remote worker exited without a receipt")
            self.read_buffer += chunk
            if len(self.read_buffer) > 96 * 1024 * 1024:
                raise ValueError("remote receipt exceeds evidence bound")
        line, self.read_buffer = self.read_buffer.split(b"\n", 1)
        return json.loads(line)

    def _send_frame(self, frame: bytes, deadline: float):
        descriptor = self.process.stdin.fileno()
        os.set_blocking(descriptor, False)
        remaining = memoryview(frame)
        while remaining:
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([], [descriptor], [], wait)[1]:
                raise TimeoutError("remote worker submission incomplete; effects unknown")
            try:
                written = os.write(descriptor, remaining[:65536])
            except BlockingIOError:
                continue
            if not written:
                raise BrokenPipeError("remote worker accepted no request bytes")
            remaining = remaining[written:]

    def _request(self, request, timeout, *, expected=None):
        if self.closed:
            raise RuntimeError("remote slot is closed")
        if self.operation_unknown:
            raise RuntimeError("a prior remote operation is unknown; do not submit another command")
        if self.component_runtime:
            self.component_runtime.assert_healthy()
        deadline = time.monotonic() + timeout
        frame = (json.dumps(request) + "\n").encode()
        if len(frame) > 96 * 1024 * 1024:
            raise ValueError("remote request exceeds evidence bound")
        try:
            self._send_frame(frame, deadline)
            result = self._read(max(0, deadline - time.monotonic()))
            if not isinstance(result, dict):
                raise RuntimeError("remote worker returned an invalid acknowledgement")
            if "cleanup" in result:
                self.cleanup_receipt = result
                raise RuntimeError("remote worker entered cleanup before its business acknowledgement")
            if "error" in result:
                raise RuntimeError(f"remote operation uncertain: {result['error']}: {result['message']}")
            if expected is not None and not expected(result):
                raise RuntimeError("remote worker returned an invalid acknowledgement")
            return result
        except Exception:
            self.operation_unknown = True
            raise

    def execute(self, command: str, stdin_b64: str = "", *, timeout: float | None = None) -> dict:
        limit = self.timeout if timeout is None else timeout
        if (
            not isinstance(limit, (int, float))
            or isinstance(limit, bool)
            or not 0 < limit <= self.timeout
        ):
            raise ValueError("command timeout must be positive and cannot exceed the slot budget")
        with self.lock:
            return self._request(
                dict(command=command, stdin_b64=stdin_b64, timeout=limit),
                limit + 15,
            )

    def open_stream(self, command: str) -> dict:
        with self.lock:
            return self._request(
                dict(control="open_stream", command=command),
                30,
                expected=lambda value: (
                    isinstance(value.get("stream_id"), str) and bool(value["stream_id"])
                ),
            )

    def exchange_stream(self, stream_id: str, stdin_b64: str = "", eof: bool = False) -> dict:
        with self.lock:
            return self._request(
                dict(
                    control="exchange_stream",
                    stream_id=stream_id,
                    stdin_b64=stdin_b64,
                    eof=eof,
                ),
                30,
                expected=lambda value: (
                    isinstance(value.get("stdout_b64"), str)
                    and isinstance(value.get("stderr_b64"), str)
                    and "exit_code" in value
                    and (value["exit_code"] is None or type(value["exit_code"]) is int)
                ),
            )

    def quiesce(self):
        with self.lock:
            self._request(
                dict(control="quiesce"),
                30,
                expected=lambda value: value == {"quiesced": True},
            )

    def upload_public(self, source: Path):
        if not source.exists():
            return
        files = {
            p.relative_to(source).as_posix(): base64.b64encode(p.read_bytes()).decode()
            for p in source.rglob("*")
            if p.is_file() and not is_ignored_cache_path(p.relative_to(source))
        }
        code = (
            "import base64,json,pathlib,sys; data=json.load(sys.stdin); root=pathlib.Path.cwd(); "
            "\nfor name,content in data.items():"
            "\n p=root/name"
            "\n assert p.resolve().is_relative_to(root.resolve()) and not p.exists()"
            "\n p.parent.mkdir(parents=True,exist_ok=True)"
            "\n p.write_bytes(base64.b64decode(content))"
        )
        result = self.execute(
            shlex.join(["python3", "-I", "-c", code]),
            base64.b64encode(json.dumps(files).encode()).decode(),
        )
        if result["exit_code"] != 0:
            raise RuntimeError("could not stage public task assets in the remote slot")
        self.public_files = files

    def verify(self, script: bytes, *, env: dict | None = None, timeout: float | None = None):
        variables = {}
        for name, value in (env or {}).items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or not name.isascii()
                or not isinstance(value, str)
                or "\0" in value
                or name in _RESERVED_SLOT_ENV
            ):
                raise ValueError("invalid verifier environment or reserved slot isolation variable")
            variables[name] = value
        command = (
            shlex.join(
                ["env", *[f"{name}={value}" for name, value in variables.items()], "/bin/bash", "-s"]
            )
            if variables
            else "/bin/bash -s"
        )
        if self.component_runtime:
            self.component_runtime.assert_healthy()
            self.component_runtime.observe("verifier_before", idle=False)
        if timeout is None:
            return self.execute(command, base64.b64encode(script).decode())
        return self.execute(command, base64.b64encode(script).decode(), timeout=timeout)

    def close(self):
        self.closed = True
        errors = []
        if self.component_runtime:
            try:
                self.component_runtime.stop_monitor()
            except Exception as exc:
                errors.append(f"remote monitor settlement: {type(exc).__name__}: {exc}")
        if self.process:
            try:
                self.process.stdin.close()
            except OSError as exc:
                errors.append(f"remote owner-pipe close: {type(exc).__name__}: {exc}")
            try:
                if self.cleanup_receipt is None:
                    deadline = time.monotonic() + 40
                    while self.cleanup_receipt is None:
                        receipt = self._read(max(0, deadline - time.monotonic()))
                        if "cleanup" in receipt:
                            self.cleanup_receipt = receipt
                        elif self.artifact_dir:
                            with (self.artifact_dir / "remote_late_results.jsonl").open("a") as stream:
                                stream.write(json.dumps(receipt) + "\n")
                self.process.wait(timeout=5)
            except (TimeoutError, subprocess.TimeoutExpired, RuntimeError) as exc:
                errors.append(f"remote worker cleanup unconfirmed: {type(exc).__name__}: {exc}")
                if self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, 9)
                    except ProcessLookupError:
                        pass
                    except OSError as cleanup:
                        errors.append(f"remote local process-group cleanup: {cleanup}")
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired as cleanup:
                    errors.append(f"remote local process wait: {cleanup}")
        if self.stderr_file:
            try:
                self.stderr_file.close()
            except OSError as exc:
                errors.append(f"remote stderr close: {exc}")
        if self.cleanup_receipt:
            final = self.cleanup_receipt.get("cleanup", {})
            errors.extend(final.get("errors", []))
            # Namespace completion is independent of whole-device occupancy and
            # local evidence accounting. Coordinator still checks owner/ports.
            self.cleanup_confirmed = not errors
            errors.extend(final.get("probe_errors", []))
            if "files_modified" in final:
                try:
                    names = final["files_modified"]
                    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                        raise ValueError("remote modified-file inventory must be a list of paths")
                    changed = {safe_relative(name) for name in names}
                    if self.public_files:
                        untracked = final.get("untracked")
                        if not isinstance(untracked, dict):
                            raise ValueError("remote public-file comparison needs final untracked bytes")
                        for name, original in self.public_files.items():
                            if untracked.get(name) == original:
                                changed.discard(name)
                            else:
                                changed.add(name)
                    self.modified_files = sorted(changed)
                except ValueError as exc:
                    errors.append(f"modified-file inventory: {exc}")
            try:
                self.postflight = self.validate_probe(final["probe"])
                if self.preflight and self.postflight["_environment"] != self.preflight["_environment"]:
                    raise RuntimeError("remote runtime changed during this trial")
            except Exception as exc:
                errors.append(f"postflight: {type(exc).__name__}: {exc}")
            if self.artifact_dir:
                try:
                    (self.artifact_dir / "remote_cleanup.json").write_text(
                        json.dumps(self.cleanup_receipt)
                    )
                    if "patch_b64" in final:
                        (self.artifact_dir / "remote_final.patch").write_bytes(
                            base64.b64decode(final["patch_b64"])
                        )
                except (OSError, ValueError) as exc:
                    errors.append(f"local cleanup evidence: {type(exc).__name__}: {exc}")
        else:
            errors.append(
                "no remote cleanup receipt; do not reuse this slot without read-only reconciliation"
            )
        if self.component_runtime:
            try:
                self.component_runtime.finish(
                    cleanup_confirmed=(
                        (self.process is None and not self.created)
                        or self.cleanup_confirmed
                    )
                )
            except Exception as exc:
                errors.append(f"components: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
