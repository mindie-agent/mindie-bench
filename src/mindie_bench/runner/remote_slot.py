"""One SSH-owned bubblewrap slot with cooperative per-device flock leases.

Only the public source snapshot reaches the remote slot. The worker lives for
the whole trial, so remote-dev jobs survive individual one-shot tool calls.
Destroying the namespace kills its descendants, including detached jobs.
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

WORKER = r"""
import base64, json, os, queue, subprocess, sys, threading
requests=queue.Queue()
def read_owner():
    for line in sys.stdin:
        requests.put(line)
    # EOF is owner loss, including while a command is running. Exiting the
    # PID-namespace leader lets bubblewrap kill all its detached descendants.
    os._exit(0)
threading.Thread(target=read_owner, daemon=True).start()
print(json.dumps({"ready": True}), flush=True)
while True:
    line=requests.get()
    request = json.loads(line)
    try:
        if request.get("control")=="quiesce":
            for item in os.listdir("/proc"):
                if item.isdigit() and int(item) not in {os.getpid(),os.getppid()}:
                    try:
                        os.kill(int(item),9)
                    except ProcessLookupError:
                        pass
            print(json.dumps({"quiesced":True}),flush=True)
            continue
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
token=("MINDIE_BENCH_SLOT_ID="+path.name).encode()
# Only this allocated slot's exact environment token is eligible for cleanup.
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():
        continue
    try:
        if proc.stat().st_uid!=os.getuid():
            continue
        fd=os.pidfd_open(int(proc.name))
    except ProcessLookupError:
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
            "HEAD"],check=True,capture_output=True).stdout
        if len(patch)>32*1024*1024:
            raise RuntimeError("remote final patch exceeds artifact bound")
        final["patch_b64"]=base64.b64encode(patch).decode()
        untracked=subprocess.run(["git","-C",str(path),"ls-files","--others","--exclude-standard","-z"],
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

    def command(self, command: list[str]) -> list[str]:
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
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
        cleanup_config = dict(
            path=self.path,
            repo_path=self.config.repo_path,
            devices=self.devices,
            memory_limit_mb=self.config.memory_limit_mb,
            probe_command=self.config.probe_command,
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
mkdir {quote(self.path + "/.bench-home")}
export MINDIE_BENCH_SLOT_ID={quote(PurePosixPath(self.path).name)}
export ASCEND_RT_VISIBLE_DEVICES={quote(",".join(map(str, self.devices)))}
bwrap --die-with-parent --unshare-all --new-session --proc /proc --dev-bind /dev /dev \\
 --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /lib64 /lib64 \\
 --ro-bind /etc /etc --tmpfs /tmp {bind} --bind {quote(self.path)} {quote(self.path)} \\
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

    def execute(self, command: str, stdin_b64: str = "") -> dict:
        with self.lock:
            if self.closed:
                raise RuntimeError("remote slot is closed")
            request = dict(command=command, stdin_b64=stdin_b64, timeout=self.timeout)
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            self.process.stdin.flush()
            result = self._read(self.timeout + 5)
            if "error" in result:
                raise RuntimeError(f"remote operation uncertain: {result['error']}: {result['message']}")
            return result

    def quiesce(self):
        with self.lock:
            self.process.stdin.write(b'{"control":"quiesce"}\n')
            self.process.stdin.flush()
            if self._read(10) != {"quiesced": True}:
                raise RuntimeError("remote jobs could not be quiesced before verification")

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
                    self.cleanup_receipt = self._read(40)
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
