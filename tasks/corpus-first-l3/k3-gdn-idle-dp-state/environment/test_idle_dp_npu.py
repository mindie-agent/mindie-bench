import fcntl
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import tempfile

root = pathlib.Path.cwd()
if not (root / "vllm_ascend/__init__.py").is_file():
    raise RuntimeError("run from the trial source root")
worker = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "bench_idle_dp_worker.py").resolve(
    strict=True
)
cases = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [[3, 7, 0.25, 0.375]]
# RemoteSlot owns a private network namespace. Bring up only its loopback.
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as net_socket:
    request = struct.pack("16sh", b"lo", 0)
    flags = struct.unpack("16sh", fcntl.ioctl(net_socket.fileno(), 0x8913, request)[:18])[1]
    if not flags & 1:
        fcntl.ioctl(net_socket.fileno(), 0x8914, struct.pack("16sh", b"lo", flags | 1))
ports = []
listeners = [socket.socket(), socket.socket()]
try:
    for listener in listeners:
        listener.bind(("127.0.0.1", 0))
        ports.append(listener.getsockname()[1])
finally:
    for listener in listeners:
        listener.close()

env = dict(os.environ)
vendor = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
env.update(
    ASCEND_CUSTOM_OPP_PATH=vendor,
    LD_LIBRARY_PATH=vendor + "/op_api/lib:" + env.get("LD_LIBRARY_PATH", ""),
    PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
    HCCL_IF_IP="127.0.0.1",
    HCCL_SOCKET_IFNAME="lo",
    HCCL_WHITELIST_DISABLE="1",
    HCCL_CONNECT_TIMEOUT="120",
    HCCL_EXEC_TIMEOUT="30",
    MINDIE_BENCH_CONTRACT_DP_PORT=str(ports[1]),
    MINDIE_BENCH_CONTRACT_CASES=json.dumps(cases),
    OMP_NUM_THREADS="1",
)
with tempfile.TemporaryDirectory(prefix="idle-dp-contract-", dir=os.environ["HOME"]) as generated:
    env["MINDIE_BENCH_CONTRACT_ROOT"] = generated
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            "--master_port=" + str(ports[0]),
            str(worker),
        ],
        env=env,
        timeout=210,
    )
    receipts = [
        json.loads((pathlib.Path(generated) / f"rank-{rank}.completed.json").read_text())
        for rank in (0, 1)
    ]
    for rank, receipt in enumerate(receipts):
        if (
            receipt["rank"] != rank
            or not receipt["business_completed"]
            or receipt["case_count"] != len(cases)
            or receipt["primary_error_type"] is not None
            or receipt["cleanup_errors"]
        ):
            raise RuntimeError(
                "NPU worker did not complete its business and cleanup: " + json.dumps(receipt)
            )
    passed = all(receipt["passed"] for receipt in receipts)
    if result.returncode != (0 if passed else 1):
        raise RuntimeError("native process exit contradicts acknowledged state result")
# The external judge's final stdout line has the exact VerifierResult schema.
print(json.dumps({"reward": 1.0 if passed else 0.0}), flush=True)
raise SystemExit(0 if passed else 1)
