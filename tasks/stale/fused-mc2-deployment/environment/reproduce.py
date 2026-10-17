import fcntl
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys

from ascend_runtime_config import validate_moe_execution

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as network:
    request = struct.pack("16sh", b"lo", 0)
    flags = struct.unpack("16sh", fcntl.ioctl(network.fileno(), 0x8913, request)[:18])[1]
    if not flags & 1:
        fcntl.ioctl(network.fileno(), 0x8914, struct.pack("16sh", b"lo", flags | 1))
master_port = os.environ.get("MASTER_PORT")
if not master_port:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = str(listener.getsockname()[1])
root = pathlib.Path(os.environ["HOME"]) / "mc2-probe"
root.mkdir(exist_ok=True)
vendor = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
env = dict(os.environ)
env.update(
    PROBE_ROOT=str(root),
    PYTHONPATH=str(pathlib.Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
    ASCEND_CUSTOM_OPP_PATH=vendor,
    LD_LIBRARY_PATH=vendor + "/op_api/lib:" + env.get("LD_LIBRARY_PATH", ""),
    HCCL_IF_IP="127.0.0.1",
    HCCL_SOCKET_IFNAME="lo",
    HCCL_WHITELIST_DISABLE="1",
    HCCL_CONNECT_TIMEOUT="120",
    HCCL_EXEC_TIMEOUT="30",
    HCCL_BUFFSIZE="64",
    OMP_NUM_THREADS="1",
)
subprocess.run(
    [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        "--master_port=" + master_port,
        "run_moe_worker.py",
    ],
    env=env,
    check=True,
    timeout=180,
)
for rank in [0, 1]:
    for case_id, tokens in [(0, 32), (1, 64)]:
        raw = (root / f"candidate-case-{case_id}-rank-{rank}.json").read_text()
        print(raw, flush=True)
        validate_moe_execution(json.loads(raw), tokens)

