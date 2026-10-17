"""Run finite mixed-length inputs through the freshly built native KDA operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    receipt = json.loads((args.build / "build-receipt.json").read_text())
    if receipt["status"] != "completed":
        raise RuntimeError("build and install the current source before running this example")
    prefix = Path(receipt["opp_prefix"]).resolve()
    library = prefix / "op_api" / "lib" / "libcust_opapi.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != receipt["op_api_sha256"]:
        raise RuntimeError("the freshly built API library changed")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(prefix)
    os.environ["LD_LIBRARY_PATH"] = (
        str(prefix / "op_api" / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if not args.native:
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--build",
                str(args.build.resolve()),
                "--native",
            ],
            os.environ,
        )
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    if torch.npu.device_count() != 1 or os.environ.get("ASCEND_CUSTOM_OPP_PATH") != str(prefix):
        raise RuntimeError("one admitted NPU and the newly built OPP route are required")
    torch.manual_seed(1127)
    heads, dim, chunk = 6, 128, 64
    lengths = ([16, 32, 64] * 43)[:127]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    shape = (1, offsets[-1], heads, dim)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1).to(torch.bfloat16)
    v = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    gate = torch.randn(shape) * 0.1
    beta = torch.sigmoid(torch.randn(shape[:-1]))
    metadata = [
        value
        for index, length in enumerate(lengths)
        for part in range((length + chunk - 1) // chunk)
        for value in (index, part)
    ]
    try:
        outputs = torch.ops._C_ascend.chunk_kda_fwd(
            q.npu(),
            k.npu(),
            v.npu(),
            gate.npu(),
            beta.npu(),
            dim**-0.5,
            chunk,
            initial_state=torch.zeros((127, heads, dim, dim), device="npu"),
            output_final_state=True,
            cu_seqlens=offsets,
            chunk_indices=metadata,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=torch.zeros(heads, device="npu"),
            dt_bias=torch.zeros(heads * dim, device="npu"),
            disable_recompute=False,
            state_v_first=True,
        )
        torch.npu.synchronize()
        finite = [None if item is None else bool(torch.isfinite(item).all()) for item in outputs]
        loaded = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcust_opapi" in line
        }
        if not loaded or any(not Path(path).resolve().is_relative_to(prefix) for path in loaded):
            raise RuntimeError("native loader did not use only the freshly built API library")
        print(
            json.dumps(
                {"sequence_count": 127, "all_outputs_finite": finite, "loaded_api": sorted(loaded)}
            ),
            flush=True,
        )
        if not all(value is None or value for value in finite):
            raise SystemExit(1)
    finally:
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
