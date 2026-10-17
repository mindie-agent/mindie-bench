"""One SSH-owned PID/network namespace with cooperative per-device leases.

Only the public source snapshot reaches the remote slot. The worker lives for
the whole trial, so remote-dev jobs survive individual one-shot tool calls.
Destroying the namespace kills its descendants, including detached jobs.
The native Ascend driver requires the dedicated container's privileged device
view. This backend runs trusted evaluation tasks; it is not a hostile-tenant
security boundary. Source history and local personal profiles are not mounted.
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
from pathlib import Path, PurePosixPath

from mindie_bench.models import RemoteConfig, SourceEnvironment
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS, PROJECTOR

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
print(json.dumps({"ready": True}), flush=True)
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
            result = subprocess.run(["/bin/bash", "-c", request["command"]],
                input=base64.b64decode(request["stdin_b64"]), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=request["timeout"])
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
for attempt in range(20):
    try:
        probe=json.loads(subprocess.run(cfg["probe_command"],check=True,capture_output=True,
            text=True,timeout=10).stdout)
        if all(probe[str(d)]["used_mb"]<cfg["memory_limit_mb"] and not probe[str(d)]["pids"]
               for d in cfg["devices"]):
            break
    except Exception as exc:
        errors.append(type(exc).__name__+": "+str(exc))
        break
    time.sleep(.5)
final={"probe":probe,"errors":errors}
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
        untracked=subprocess.run(["git","-C",str(path),"ls-files","--others","--exclude-standard","-z",
            *diff_paths],
            check=True,capture_output=True).stdout.split(b"\0")
        final["untracked"]={}
        size=0
        for name in untracked:
            if not name or name.startswith(b".bench-home/"):
                continue
            item=path/os.fsdecode(name)
            if item.is_symlink() or not item.resolve().is_relative_to(path.resolve()):
                raise RuntimeError("untracked artifact escapes slot")
            data=item.read_bytes()
            size+=len(data)
            if size>32*1024*1024:
                raise RuntimeError("remote untracked artifact bound exceeded")
            final["untracked"][os.fsdecode(name)]=base64.b64encode(data).decode()
        subprocess.run(["git","-C",cfg["repo_path"],"worktree","remove","--force",str(path)],
            check=True,capture_output=True)
    except Exception as exc:
        final["errors"].append(type(exc).__name__+": "+str(exc))
artifact=str(path)+".final.json"
pathlib.Path(artifact).write_text(json.dumps(final))
print(json.dumps({"cleanup":final,"remote_artifact":artifact}),flush=True)
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
        self.process = None
        self.lock = threading.Lock()
        self.preflight = None
        self.postflight = None
        self.created = False
        self.closed = False
        self.artifact_dir = artifact_dir
        self.cleanup_receipt = None
        self.stderr_file = None
        self.read_buffer = b""
        self.operation_unknown = False

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
        data = json.loads(result.stdout)
        return self.validate_probe(data)

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
export MINDIE_BENCH_SLOT_ID={quote(PurePosixPath(self.path).name)}
export ASCEND_RT_VISIBLE_DEVICES={quote(",".join(map(str, self.devices)))}
device_args=()
{device_bind}
bwrap --die-with-parent --unshare-pid --unshare-net --unshare-ipc --unshare-uts \\
 --cap-add ALL --new-session --proc /proc --dev /dev "${{device_args[@]}}" \\
 --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind-try /lib64 /lib64 \\
 --ro-bind /etc /etc --bind /sys /sys --tmpfs /tmp --tmpfs /var {bind} \\
 --bind {quote(self.path)} {quote(self.path)} \\
 --chdir {quote(self.path)} --setenv HOME {quote(self.path + "/.bench-home")} \\
 -- python3 -I -u -c {quote(WORKER)}
"""
        # The worker holds the leases; the probe happens after its ready
        # acknowledgement, before any model or agent command is admitted.
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
        if ready != {"ready": True}:
            raise RuntimeError("remote worker failed to enter its isolated slot")
        self.created = True
        self.preflight = self.probe()
        return self

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.read_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("remote worker acknowledgement lost; effects unknown")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
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
        # A stalled SSH peer can fill this pipe before _read's timeout starts.
        # All writes use this method; leave the owned, unbuffered pipe nonblocking.
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
        # Every caller holds the same lock, including stream control frames.
        if self.closed:
            raise RuntimeError("remote slot is closed")
        if self.operation_unknown:
            raise RuntimeError("a prior remote operation is unknown; do not submit another command")
        deadline = time.monotonic() + timeout
        frame = (json.dumps(request) + "\n").encode()
        if len(frame) > 96 * 1024 * 1024:
            raise ValueError("remote request exceeds evidence bound")
        try:
            self._send_frame(frame, deadline)
            result = self._read(max(0, deadline - time.monotonic()))
            if not isinstance(result, dict):
                raise RuntimeError("remote worker returned an invalid acknowledgement")
            if "error" in result:
                raise RuntimeError(f"remote operation uncertain: {result['error']}: {result['message']}")
            if expected is not None and not expected(result):
                raise RuntimeError("remote worker returned an invalid acknowledgement")
            return result
        except Exception:
            self.operation_unknown = True
            raise

    def execute(self, command: str, stdin_b64: str = "") -> dict:
        with self.lock:
            return self._request(
                dict(command=command, stdin_b64=stdin_b64, timeout=self.timeout),
                self.timeout + 5,
            )

    def open_stream(self, command: str) -> dict:
        with self.lock:
            return self._request(
                dict(control="open_stream", command=command),
                10,
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
                10,
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
                10,
                expected=lambda value: value == {"quiesced": True},
            )

    def upload_public(self, source: Path):
        if not source.exists():
            return
        from mindie_bench.io import tree_hash

        tree_hash(source)
        files = {
            p.relative_to(source).as_posix(): base64.b64encode(p.read_bytes()).decode()
            for p in source.rglob("*")
            if p.is_file()
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

    def verify(self, script: bytes):
        return self.execute("/bin/bash -s", base64.b64encode(script).decode())

    def close(self):
        self.closed = True
        errors = []
        if self.process:
            self.process.stdin.close()
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
            except (TimeoutError, subprocess.TimeoutExpired, RuntimeError):
                os.killpg(self.process.pid, 9)
                self.process.wait(timeout=5)
                errors.append(
                    "remote SSH worker required forced termination; remote cleanup unconfirmed"
                )
        if self.stderr_file:
            self.stderr_file.close()
        if self.cleanup_receipt:
            final = self.cleanup_receipt.get("cleanup", {})
            errors.extend(final.get("errors", []))
            try:
                self.postflight = self.validate_probe(final["probe"])
                if self.preflight and self.postflight["_environment"] != self.preflight["_environment"]:
                    raise RuntimeError("remote runtime changed during this trial")
            except Exception as exc:
                errors.append(f"postflight: {type(exc).__name__}: {exc}")
            if self.artifact_dir:
                (self.artifact_dir / "remote_cleanup.json").write_text(json.dumps(self.cleanup_receipt))
                if "patch_b64" in final:
                    (self.artifact_dir / "remote_final.patch").write_bytes(
                        base64.b64decode(final["patch_b64"])
                    )
        else:
            errors.append(
                "no remote cleanup receipt; do not reuse this slot without read-only reconciliation"
            )
        if errors:
            raise RuntimeError("; ".join(errors))
