"""Judge a freshly compiled native operator using the entire historical test module."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

TEST_SHA256 = "3dea5a1523accc1f7f3bc7102e35cb5e455c7b4eb246f37f1aafebbf6fb48370"
CASES = [
    (seq_count, length, recompute)
    for recompute in (False, True)
    for seq_count, length in [(127, None), (128, None), (1, 80), (1, 129)]
]


def emit(record: dict) -> None:
    print(json.dumps(record), flush=True)


def main() -> None:
    private = Path(sys.argv[1]).resolve()
    built = json.loads((private / "build" / "build-receipt.json").read_text())
    if built["status"] != "completed" or not built["source_unchanged"]:
        raise RuntimeError("source build did not complete with a stable input snapshot")
    prefix = Path(built["opp_prefix"]).resolve()
    library = prefix / "op_api" / "lib" / "libcust_opapi.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != built["op_api_sha256"]:
        raise RuntimeError("newly built API library changed")
    test = private / "upstream_test.py"
    if hashlib.sha256(test.read_bytes()).hexdigest() != TEST_SHA256:
        raise RuntimeError("independent complete upstream numerical test bytes changed")
    spec = importlib.util.spec_from_file_location("private_complete_kda_test", test)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH") != str(prefix):
        raise RuntimeError("runtime bootstrap replaced the newly built custom-OPP route")
    torch = module.torch
    torch.set_num_threads(8)
    if torch.npu.device_count() != 1:
        raise RuntimeError("the task requires exactly one admitted real NPU")
    emit(
        {
            "stage": "native_import",
            "source_files_sha256": built["source_files_sha256"],
            "input_archives_sha256": built["input_archives_sha256"],
            "package_sha256": built["package_sha256"],
            "op_api_sha256": built["op_api_sha256"],
            "upstream_test_sha256": TEST_SHA256,
            "device_name": torch.npu.get_device_name(0),
            "schema": str(torch.ops._C_ascend.chunk_kda_fwd.default._schema),
        }
    )
    rows = []
    for count, length, recompute in CASES:
        row = {
            "stage": "numerical_case",
            "sequence_count": count,
            "fixed_length": length,
            "disable_recompute": recompute,
        }
        try:
            module.test_chunk_kda_fwd_padded_workspace_matches_reference(count, length, recompute)
        except AssertionError as error:
            row.update(status="business_failed", assertion=str(error))
        except BaseException as error:
            row.update(status="infrastructure_failure", error=type(error).__name__ + ": " + str(error))
            emit(row)
            raise
        else:
            row["status"] = "passed"
        finally:
            primary = sys.exception()
            try:
                module._cleanup_npu()
            except BaseException as cleanup:
                if primary is not None:
                    primary.add_note("NPU cleanup also failed: " + repr(cleanup))
                else:
                    emit(row)
                    raise
        rows.append(row)
        emit(row)
    loaded = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libcust_opapi" in line and "/" in line.split()[-1]
    }
    if not loaded or any(not Path(path).resolve().is_relative_to(prefix) for path in loaded):
        raise RuntimeError("native process did not exclusively load its freshly compiled OPP")
    passed = all(row["status"] == "passed" for row in rows)
    emit(
        {
            "stage": "business_complete",
            "case_count": len(rows),
            "loaded_api_libraries": sorted(loaded),
            "cleanup_completed": True,
            "hardware_kernel_invocation_count": None,
            "judge": "finite outputs plus independent FP32 CPU output/final-state comparisons",
        }
    )
    emit({"reward": 1.0 if passed else 0.0})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
