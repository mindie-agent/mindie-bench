import copy
import fcntl
import importlib.util
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import tempfile


def valid_observation(row, tokens):
    return (
        row.get("model") == "Qwen3MoeForCausalLM"
        and row.get("config_effective") == 0
        and row.get("ep_size") == 2
        and row.get("tp_size") == 2
        and row.get("hccl_rank_sum") == 3
        and row.get("output_shape") == [tokens, 1024]
        and row.get("output_finite") is True
        and row.get("output_norm", 0) > 0
        and "npu::npu_grouped_matmul" in row.get("actual_operator_keys", [])
        and not any(
            "dispatch_ffn_combine" in name.lower() or "dispatchffncombine" in name.lower()
            for name in row.get("actual_operator_keys", [])
        )
    )


def self_test():
    row = dict(
        model="Qwen3MoeForCausalLM",
        config_effective=0,
        ep_size=2,
        tp_size=2,
        hccl_rank_sum=3,
        output_shape=[32, 1024],
        output_finite=True,
        output_norm=1,
        actual_operator_keys=["npu::npu_grouped_matmul"],
    )
    assert valid_observation(row, 32)
    for changed in [
        dict(config_effective=1),
        dict(actual_operator_keys=["npu::npu_grouped_matmul", "_C_ascend::dispatch_ffn_combine"]),
        dict(actual_operator_keys=[]),
        dict(output_norm=0),
        dict(output_finite=False),
        dict(ep_size=1),
        dict(hccl_rank_sum=1),
        dict(output_shape=[64, 1024]),
    ]:
        bad = copy.deepcopy(row)
        bad.update(changed)
        assert not valid_observation(bad, 32)
    print("stale-mc2-negative-controls:passed")


if sys.argv[1:] == ["--self-test"]:
    self_test()
    raise SystemExit(0)


def check_launcher():
    # Exercise the actual submitted module against independent input dictionaries.
    path = pathlib.Path.cwd() / "deployment.py"
    spec = importlib.util.spec_from_file_location("submitted_deployment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observations = []
    for flag in [0, 1]:
        original = {
            "enable_fused_mc2": flag,
            "enable_cpu_binding": False,
            "weight_nz_mode": 0,
            "unrelated_deployment_option": ["preserve", flag],
        }
        saved = copy.deepcopy(original)
        result = module.configure(original)
        observations.append(
            result.get("enable_fused_mc2") == 0
            and original == saved
            and {k: v for k, v in result.items() if k != "enable_fused_mc2"}
            == {k: v for k, v in saved.items() if k != "enable_fused_mc2"}
        )
    return all(observations)


with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as network:
    request = struct.pack("16sh", b"lo", 0)
    flags = struct.unpack("16sh", fcntl.ioctl(network.fileno(), 0x8913, request)[:18])[1]
    if not flags & 1:
        fcntl.ioctl(network.fileno(), 0x8914, struct.pack("16sh", b"lo", flags | 1))
vendor = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
env = dict(os.environ)
env.update(
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
worker = pathlib.Path(__file__).with_name("worker.py")
passed = check_launcher()
max_abs = 0.0
with tempfile.TemporaryDirectory(prefix="mc2-private-", dir=os.environ["HOME"]) as temporary:
    env["PROBE_ROOT"] = temporary
    for phase in ["reference", "candidate"]:
        env["MINDIE_STALE_PRIVATE_PHASE"] = phase
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "--master_addr=127.0.0.1",
                "--master_port=" + os.environ["MASTER_PORT"],
                str(worker),
            ],
            env=env,
            check=True,
            timeout=180,
        )
    import torch

    for case_id, tokens in enumerate([32, 64]):
        for rank in [0, 1]:
            prefix = pathlib.Path(temporary)
            candidate = prefix / f"candidate-case-{case_id}-rank-{rank}"
            reference = prefix / f"reference-case-{case_id}-rank-{rank}"
            observed = json.loads(candidate.with_suffix(".json").read_text())
            reference_row = json.loads(reference.with_suffix(".json").read_text())
            if not valid_observation(reference_row, tokens):
                raise RuntimeError("current-config reference did not execute the required NPU path")
            actual = torch.load(candidate.with_suffix(".pt"), weights_only=True)
            expected = torch.load(reference.with_suffix(".pt"), weights_only=True)
            error = float((actual - expected).abs().max())
            close = bool(torch.allclose(actual, expected, rtol=0.03, atol=0.003))
            valid = valid_observation(observed, tokens) and close
            max_abs = max(max_abs, error)
            passed = passed and valid
            print(
                json.dumps(
                    dict(
                        stage="private_case",
                        case_id=case_id,
                        rank=rank,
                        tokens=tokens,
                        candidate=observed,
                        reference=reference_row,
                        max_abs_error=error,
                        numerical_close=close,
                        passed=valid,
                    )
                ),
                flush=True,
            )
print(
    json.dumps(
        dict(
            reward=1.0 if passed else 0.0,
            max_abs_error=max_abs,
            failure_reason=None
            if passed
            else "current deployment compatibility override was not applied",
        )
    ),
    flush=True,
)
raise SystemExit(0 if passed else 1)
