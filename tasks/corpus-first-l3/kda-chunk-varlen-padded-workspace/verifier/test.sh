#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
test ! -e /vllm-workspace/vllm-ascend/csrc
test ! -e /vllm-workspace/vllm-ascend/tests
test ! -e /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
private=$(mktemp -d "$HOME/.kda-verifier.XXXXXX")
trap 'rm -rf -- "$private"' EXIT
cat > "$private/build_kda.py" <<'PRIVATE_KDA_ASSET_0'
"""Build the current complete CANN source tree into a fresh, owned OPP prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

INPUTS = {
    "generic-protobuf-abseil.tar.gz": (
        "13c73902399a1c3524ad8acd5f5386ee48b6c4a2a6a19774ed287c728d5142ea"
    ),
    "catlass-41bf90.tar.gz": "5f2705722d814365b5bfbf95651e4502eab170a1beb875bf7a90d5d5d8cb39f3",
    "makeself-release-2.5.0-patch1.tar.gz": (
        "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a"
    ),
    "abseil-cpp-20230802.1.tar.gz": ("987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed"),
    "protobuf-25.1.tar.gz": "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
    "include.zip": "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
}
BUILD_COMMAND = [
    "bash",
    "csrc/build.sh",
    "--pkg",
    "--ops=chunk_kda_fwd,kda_layout_swap12",
    "--soc=ascend910_93",
    "-j8",
]
GENERATED = {"build", "build_out", "output"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs(root: Path, *, cold_generic: bool = False) -> dict[str, str]:
    result = {}
    for name, expected in INPUTS.items():
        if cold_generic and name == "generic-protobuf-abseil.tar.gz":
            continue
        path = root / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"dependency bytes differ from the frozen input: {name}")
        result[name] = actual
    return result


def source_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in GENERATED:
            continue
        if path.is_symlink():
            raise ValueError(f"source symlink is outside this task's source contract: {relative}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path)
    if not (root / "build.sh").is_file():
        raise ValueError("run from a complete source worktree containing csrc/build.sh")
    return result


def unpack_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")


def generic_prefix(root: Path) -> dict:
    manifest = json.loads((root / "generic-manifest.json").read_text())
    if (
        manifest.get("compiled_generic_inputs") != ["protobuf/bin/protoc"]
        or manifest.get("compiled_kda_inputs") != []
    ):
        raise ValueError("generic dependency manifest has an unexpected compiled input")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("generic dependency contains a symlink")
        if path.is_file() and path.name != "generic-manifest.json":
            name = path.relative_to(root).as_posix()
            value = path.read_bytes()
            compiled = (
                path.suffix.lower() in {".so", ".a", ".o", ".run", ".pyc"}
                or value.startswith(b"\x7fELF")
                or value.startswith(b"!<arch>")
            )
            if compiled and name != "protobuf/bin/protoc":
                raise ValueError("generic dependency includes an unexpected native object: " + name)
            if b"chunk_kda_fwd" in value:
                raise ValueError("generic dependency includes target operator material: " + name)
            files[name] = hashlib.sha256(value).hexdigest()
    if files != manifest["files_sha256"]:
        raise ValueError("generic dependency file identity differs from its frozen manifest")
    for relative in [
        "protobuf/include/google/protobuf/message.h",
        "protobuf/bin/protoc",
        "protobuf/bin/options-pinned.h",
        "abseil/absl/log/absl_log.h",
    ]:
        if relative not in files:
            raise ValueError("generic dependency is incomplete: " + relative)
    return manifest


def prepare(source: Path, inputs: Path, output: Path, *, cold_generic: bool = False) -> dict:
    dependencies = checked_inputs(inputs, cold_generic=cold_generic)
    files = source_files(source / "csrc")
    output.mkdir(parents=True, exist_ok=False)
    copied = output / "source" / "csrc"

    def ignore(directory, names):
        return GENERATED & set(names) if Path(directory) == source / "csrc" else set()

    shutil.copytree(source / "csrc", copied, ignore=ignore)
    third_party = copied / "third_party"
    third_party.mkdir(exist_ok=True)
    for name in ["catlass", "makeself", "json", "abseil-cpp", "ascend_protobuf", "pkg", "generic-build"]:
        path = third_party / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    unpack_tar(inputs / "catlass-41bf90.tar.gz", third_party / "catlass")
    generic_record = {
        "reuse": "none; explicitly requested original upstream cold generic source build",
        "compiled_kda_inputs": [],
    }
    if not cold_generic:
        generic = third_party / "generic-build"
        unpack_tar(inputs / "generic-protobuf-abseil.tar.gz", generic)
        generic_manifest = generic_prefix(generic)
        shutil.copytree(generic / "abseil", third_party / "abseil-cpp")
        cache = copied / "build"
        cache.mkdir()
        (cache / "CMakeCache.txt").write_text(
            "ASCEND_PROTOBUF_SHARED_INCLUDE:PATH=" + str(generic / "protobuf" / "include") + "\n"
            "ASCEND_PROTOC:PATH=" + str(generic / "protobuf" / "bin") + "\n"
        )
        generic_record = {
            "manifest_sha256": sha256(generic / "generic-manifest.json"),
            "protoc_sha256": generic_manifest["files_sha256"]["protobuf/bin/protoc"],
            "protoc_version": generic_manifest["protoc_version"],
            "reuse": "upstream ASCEND_PROTOBUF_SHARED_INCLUDE/ASCEND_PROTOC native CMake path",
            "compiled_kda_inputs": [],
        }
    makeself = third_party / "makeself"
    unpack_tar(inputs / "makeself-release-2.5.0-patch1.tar.gz", makeself)
    if not (makeself / "makeself.sh").is_file():
        children = list(makeself.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("unexpected makeself release layout")
        nested = children[0]
        for child in list(nested.iterdir()):
            child.rename(makeself / child.name)
        nested.rmdir()
    package_dir = third_party / "pkg"
    package_dir.mkdir()
    for name in ["abseil-cpp-20230802.1.tar.gz", "protobuf-25.1.tar.gz"]:
        shutil.copyfile(inputs / name, package_dir / name)
    json_dir = third_party / "json"
    json_dir.mkdir()
    with zipfile.ZipFile(inputs / "include.zip") as archive:
        for member in archive.infolist():
            target = json_dir / member.filename
            if not target.resolve().is_relative_to(json_dir.resolve()):
                raise ValueError("JSON dependency archive contains an escaping path")
        archive.extractall(json_dir)
    return {
        "schema": "mindie-bench-kda-source-build/1",
        "status": "prepared",
        "source_files_sha256": files,
        "input_archives_sha256": dependencies,
        "generic_dependency": generic_record,
        "command": BUILD_COMMAND,
        "source_copy": str(output / "source"),
        "opp_prefix": str(output / "opp" / "vendors" / "custom_transformer"),
        "model_calls": 0,
        "npu_tensor_calls_during_build": 0,
        "started": time.time(),
    }


def build(
    source: Path, inputs: Path, output: Path, *, prepare_only: bool = False, cold_generic: bool = False
) -> dict:
    record = prepare(source.resolve(), inputs.resolve(), output.resolve(), cold_generic=cold_generic)
    receipt = output / "build-receipt.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if prepare_only:
        return record
    source_copy = Path(record["source_copy"])
    env = dict(os.environ, MAX_JOBS="8", CMAKE_BUILD_PARALLEL_LEVEL="8", CCACHE_PROGRAM="false")
    with (output / "build.stdout").open("xb") as stdout:
        with (output / "build.stderr").open("xb") as stderr:
            compiled = subprocess.run(
                BUILD_COMMAND, cwd=source_copy, env=env, stdout=stdout, stderr=stderr
            )
    record.update(build_exit_code=compiled.returncode, finished_build=time.time())
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if compiled.returncode:
        sys.stdout.buffer.write((output / "build.stdout").read_bytes())
        sys.stderr.buffer.write((output / "build.stderr").read_bytes())
        raise RuntimeError(f"normal CANN source build failed; see {output / 'build.stderr'}")
    packages = list((source_copy / "csrc" / "build_out").glob("*.run"))
    if len(packages) != 1:
        raise RuntimeError("normal CANN source build did not produce exactly one package")
    package = packages[0]
    command = ["bash", str(package), "--quiet", "--nox11", "--install-path=" + str(output / "opp")]
    with (output / "install.stdout").open("xb") as stdout:
        with (output / "install.stderr").open("xb") as stderr:
            installed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    record.update(
        package_sha256=sha256(package),
        install_command=command,
        install_exit_code=installed.returncode,
    )
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if installed.returncode:
        sys.stdout.buffer.write((output / "install.stdout").read_bytes())
        sys.stderr.buffer.write((output / "install.stderr").read_bytes())
        raise RuntimeError(f"normal OPP installation failed; see {output / 'install.stderr'}")
    library = Path(record["opp_prefix"]) / "op_api" / "lib" / "libcust_opapi.so"
    record.update(
        status="completed",
        op_api_sha256=sha256(library),
        finished=time.time(),
        source_unchanged=source_files(source / "csrc") == record["source_files_sha256"],
    )
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if not record["source_unchanged"]:
        raise RuntimeError("input source changed while the copied source was building")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(os.environ.get("MINDIE_KDA_INPUTS", "/bench/kda-runtime-inputs")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--cold-generic",
        action="store_true",
        help="explicit operator bootstrap: compile generic Protobuf from the five source archives",
    )
    args = parser.parse_args()
    record = build(
        args.source,
        args.inputs,
        args.output,
        prepare_only=args.prepare_only,
        cold_generic=args.cold_generic,
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "receipt_path": str(args.output.resolve() / "build-receipt.json"),
                "source_file_count": len(record["source_files_sha256"]),
                "opp_prefix": record["opp_prefix"],
                "package_sha256": record.get("package_sha256"),
                "elapsed_sec": record["finished"] - record["started"] if "finished" in record else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
PRIVATE_KDA_ASSET_0
cat > "$private/verify.py" <<'PRIVATE_KDA_ASSET_1'
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
PRIVATE_KDA_ASSET_1
cat > "$private/upstream_test.py" <<'PRIVATE_KDA_ASSET_2'
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import gc
import math
from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

DETERMINISM_REPEATS = 20
CHUNK_KDA_OUTPUT_NAMES = (
    "o",
    "final_state",
    "g",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)


@dataclass
class ChunkKdaReferenceResult:
    o: torch.Tensor
    final_state: torch.Tensor | None


def _lower_inverse(mat: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def chunk_kda_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> ChunkKdaReferenceResult:
    bsz, total_t, hq, kdim = q.shape
    _, _, hv, vdim = v.shape
    group = hv // hq
    out_dtype = v.dtype
    device = q.device
    o = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=out_dtype)
    state = (
        torch.zeros((bsz, hv, kdim, vdim), device=device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32).clone()
    )

    for b in range(bsz):
        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)
            cur_t = end - start
            for ihv in range(hv):
                ih = ihv // group
                q_blk = q[b, start:end, ih].to(torch.float32)
                k_blk = k[b, start:end, ih].to(torch.float32)
                v_blk = v[b, start:end, ihv].to(torch.float32)
                g_blk = gk[b, start:end, ihv].to(torch.float32)
                beta_blk = beta[b, start:end, ihv].to(torch.float32)

                causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril()
                strict_causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril(diagonal=-1)
                rel = g_blk[:, None, :] - g_blk[None, :, :]
                rel = rel.masked_fill(~causal[:, :, None], 0.0)
                gate = torch.exp2(rel)
                qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, gate) * float(scale)
                kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, gate)
                tril_qk = torch.where(causal, qk, torch.zeros_like(qk))
                tril_kk = torch.where(strict_causal, kk * beta_blk[:, None], torch.zeros_like(kk))
                inv_akk = _lower_inverse(tril_kk)

                k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
                v_beta = v_blk * beta_blk[:, None]
                w_blk = inv_akk @ k_beta_g
                u_blk = inv_akk @ v_beta

                last_g = g_blk[cur_t - 1]
                qg_blk = q_blk * torch.exp2(g_blk)
                kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)
                h_prev = state[b, ihv].clone()
                v_new_blk = u_blk - w_blk @ h_prev
                state[b, ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

                o_inter = qg_blk @ h_prev * float(scale)
                o_local = tril_qk @ v_new_blk
                o[b, start:end, ihv] = (o_inter + o_local).to(out_dtype)

    return ChunkKdaReferenceResult(o=o, final_state=state if output_final_state else None)


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _is_ascend_950() -> bool:
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


def _assert_close(name, actual, expected, rtol=5e-2, atol=5e-2):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol, msg=name)


def _snapshot_outputs(outputs):
    torch.npu.synchronize()
    return tuple(None if output is None else output.detach().cpu().contiguous() for output in outputs)


def _assert_outputs_bitwise_equal(reference, actual, repeat):
    assert len(reference) == len(actual) == len(CHUNK_KDA_OUTPUT_NAMES)
    for name, expected, current in zip(CHUNK_KDA_OUTPUT_NAMES, reference, actual):
        if expected is None or current is None:
            assert expected is None and current is None, (
                f"repeat={repeat} output={name} changed between None and Tensor"
            )
            continue
        same_metadata = expected.shape == current.shape and expected.dtype == current.dtype
        same_bits = same_metadata and torch.equal(expected.view(torch.uint8), current.view(torch.uint8))
        if same_bits:
            continue

        expected_float = expected.float()
        current_float = current.float()
        finite = torch.isfinite(expected_float) & torch.isfinite(current_float)
        max_abs_diff = (
            (expected_float[finite] - current_float[finite]).abs().max().item() if finite.any() else float("nan")
        )
        expected_nonfinite = (~torch.isfinite(expected_float)).sum().item()
        current_nonfinite = (~torch.isfinite(current_float)).sum().item()
        raise AssertionError(
            f"repeat={repeat} output={name} is not bitwise deterministic: "
            f"expected_nonfinite={expected_nonfinite}, current_nonfinite={current_nonfinite}, "
            f"max_abs_diff={max_abs_diff}"
        )


def _gate_cumsum_reference(g, chunk_size, cu_seqlens=None):
    g_cpu = g.detach().cpu().to(torch.float32)
    ref = torch.empty_like(g_cpu)
    rcp_ln2 = 1.4426950408889634
    if cu_seqlens is None:
        cu_seqlens = [0, g_cpu.shape[1]]
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            ref[:, chunk_start:chunk_end] = torch.cumsum(g_cpu[:, chunk_start:chunk_end] * rcp_ln2, dim=1)
    return ref


def _layout_swap12_reference(x):
    return x.transpose(1, 2).contiguous()


def test_kda_torch_bindings_have_shape_correct_meta_kernels():
    q = torch.empty((1, 64, 1, 128), device="meta", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, 64, 2, 256), device="meta", dtype=torch.bfloat16)
    raw_gate = torch.empty((1, 64, 2, 128), device="meta", dtype=torch.bfloat16)
    beta = torch.empty((1, 64, 2), device="meta", dtype=torch.float32)
    a_log = torch.empty((2,), device="meta", dtype=torch.float32)
    dt_bias = torch.empty((2 * 128,), device="meta", dtype=torch.float32)

    gk = torch.ops._C_ascend.kda_gate_cumsum(raw_gate, 64, layout="BSND")
    outputs = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        128**-0.5,
        64,
        layout="BSND",
        output_final_state=True,
        safe_gate=True,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    swapped = torch.ops._C_ascend.kda_layout_swap12(raw_gate)

    assert gk.shape == raw_gate.shape
    assert gk.dtype == torch.float32
    assert [tuple(output.shape) for output in outputs[:-1]] == [
        (1, 64, 2, 256),
        (1, 2, 128, 256),
        (1, 2, 64, 128),
        (1, 2, 64, 64),
        (1, 2, 64, 64),
        (1, 2, 64, 128),
        (1, 2, 64, 256),
        (1, 2, 64, 128),
        (1, 2, 64, 128),
        (1, 2, 64, 256),
        (1, 1, 2, 128, 256),
    ]
    assert outputs[11] is None
    assert outputs[0].dtype == torch.bfloat16
    assert outputs[1].dtype == torch.float32
    assert swapped.shape == (1, 2, 64, 128)
    assert swapped.dtype == torch.bfloat16


@torch.inference_mode()
def test_chunk_kda_fwd_matches_reference_bsnd():
    torch.manual_seed(20260720)

    bsz, total_t, hq, hv, kdim, vdim = 1, 64, 1, 1, 128, 128
    dtype = torch.float16
    q = (torch.randn(bsz, total_t, hq, kdim, dtype=dtype) * 0.05).npu()
    k = (torch.randn(bsz, total_t, hq, kdim, dtype=dtype) * 0.05).npu()
    v = (torch.randn(bsz, total_t, hv, vdim, dtype=dtype) * 0.05).npu()
    g = (-torch.rand(bsz, total_t, hv, kdim, dtype=torch.float32) * 0.05).npu()
    beta = torch.sigmoid(torch.randn(bsz, total_t, hv, dtype=torch.float32)).npu()
    gk = torch.ops._C_ascend.kda_gate_cumsum(g, 64, layout="BSND")
    initial_state = (torch.randn(bsz, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    got = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        gk.cpu(),
        beta.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    assert torch.isfinite(got[0]).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _cleanup_npu()


@pytest.mark.parametrize(
    ("shape", "dtype", "with_dependency"),
    [
        ((1, 64, 2, 128), torch.float32, False),
        ((1, 64, 2, 128), torch.float16, True),
        ((1, 64, 2, 128), torch.bfloat16, False),
    ],
)
@torch.inference_mode()
def test_kda_layout_swap12_matches_reference(shape, dtype, with_dependency):
    torch.manual_seed(20260720 + len(shape) + shape[-1])

    x = (torch.randn(*shape, dtype=dtype) * 0.04).npu()
    expected = _layout_swap12_reference(x.cpu())
    dependency = torch.empty_like(expected).npu() if with_dependency else None

    got = torch.ops._C_ascend.kda_layout_swap12(x, dependency=dependency)

    assert torch.isfinite(got).all().item()
    _assert_close("layout_swap12", got, expected, rtol=0, atol=0)
    _cleanup_npu()


@pytest.mark.parametrize(
    ("total_t", "hq", "hv", "kdim", "vdim", "dtype"),
    [
        (64, 1, 1, 128, 128, torch.float16),
        (128, 1, 2, 128, 256, torch.float16),
        (128, 2, 2, 128, 256, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fwd_c128_v256_path(total_t, hq, hv, kdim, vdim, dtype):
    torch.manual_seed(20260720 + total_t + hq + hv + vdim)

    q = (torch.randn(1, total_t, hq, kdim, dtype=dtype) * 0.04).npu()
    k = (torch.randn(1, total_t, hq, kdim, dtype=dtype) * 0.04).npu()
    v = (torch.randn(1, total_t, hv, vdim, dtype=dtype) * 0.04).npu()
    g = (-torch.rand(1, total_t, hv, kdim, dtype=torch.float32) * 0.04).npu()
    beta = torch.sigmoid(torch.randn(1, total_t, hv, dtype=torch.float32)).npu()
    initial_state = (torch.randn(1, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    gk = torch.ops._C_ascend.kda_gate_cumsum(g, 64, layout="BSND")

    def run_chunk_kda_fwd():
        return torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True,
            return_intermediate_states=True,
        )

    is_a5_determinism_case = (
        total_t == 128 and hq == 2 and hv == 2 and kdim == 128 and vdim == 256 and dtype == torch.bfloat16
    )
    if is_a5_determinism_case:
        run_chunk_kda_fwd()
        torch.npu.synchronize()
        got = run_chunk_kda_fwd()
        reference_outputs = _snapshot_outputs(got)
        for repeat in range(1, DETERMINISM_REPEATS):
            current_outputs = _snapshot_outputs(run_chunk_kda_fwd())
            _assert_outputs_bitwise_equal(reference_outputs, current_outputs, repeat)
    else:
        got = run_chunk_kda_fwd()
    ref = chunk_kda_forward_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        gk.cpu(),
        beta.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    assert torch.isfinite(got[0]).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", got[0], ref.o)
    _assert_close("final_state", got[1], ref.final_state)
    _cleanup_npu()


@pytest.mark.parametrize(
    ("total_t", "disable_recompute"),
    [
        pytest.param(15, False, id="single-tail-model-mode"),
        pytest.param(15, True, id="single-tail-all-outputs"),
        pytest.param(65, True, id="full-chunk-plus-tail-all-outputs"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fwd_tail_is_bitwise_deterministic(total_t, disable_recompute):
    torch.manual_seed(20260820 + total_t + int(disable_recompute))

    shape = (1, total_t, 6, 128)
    q = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    k = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    v = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    raw_gate = (-7.0 + torch.randn(shape) * 0.03).to(torch.float32).npu()
    beta = (torch.rand((1, total_t, 6)) * 0.2 + 0.05).to(torch.float32).npu()
    initial_state = (torch.randn((1, 6, 128, 128)) * 0.01).to(torch.float32).npu()
    a_log = torch.zeros(6, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros(6 * 128, dtype=torch.float32, device="npu")
    chunk_indices = _canonical_chunk_indices([0, total_t], 64)

    def run_chunk_kda_fwd():
        return torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v,
            raw_gate,
            beta,
            128**-0.5,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=[0, total_t],
            chunk_indices=chunk_indices,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=disable_recompute,
            return_intermediate_states=True,
            state_v_first=True,
        )

    run_chunk_kda_fwd()
    reference_outputs = _snapshot_outputs(run_chunk_kda_fwd())
    for repeat in range(1, DETERMINISM_REPEATS):
        current_outputs = _snapshot_outputs(run_chunk_kda_fwd())
        _assert_outputs_bitwise_equal(reference_outputs, current_outputs, repeat)

    _cleanup_npu()


@torch.inference_mode()
def test_kda_gate_cumsum_matches_reference():
    torch.manual_seed(20260720)

    g = (-torch.rand(1, 96, 2, 128, dtype=torch.float32) * 0.05).npu()
    cu_seqlens = [0, 31, 96]
    out = torch.ops._C_ascend.kda_gate_cumsum(g, 64, cu_seqlens=cu_seqlens, layout="BSND")
    ref = _gate_cumsum_reference(g, 64, cu_seqlens)

    assert torch.isfinite(out).all().item()
    _assert_close("gk", out, ref, rtol=2e-3, atol=2e-3)
    _cleanup_npu()


@torch.inference_mode()
def test_chunk_kda_fwd_bnsd_layout_matches_reference():
    torch.manual_seed(20260720)

    total_t, hq, hv, kdim, vdim = 64, 1, 1, 128, 128
    q_bsnd = (torch.randn(1, total_t, hq, kdim, dtype=torch.float16) * 0.04).npu()
    k_bsnd = (torch.randn(1, total_t, hq, kdim, dtype=torch.float16) * 0.04).npu()
    v_bsnd = (torch.randn(1, total_t, hv, vdim, dtype=torch.float16) * 0.04).npu()
    g_bsnd = (-torch.rand(1, total_t, hv, kdim, dtype=torch.float32) * 0.04).npu()
    beta_bsn = torch.sigmoid(torch.randn(1, total_t, hv, dtype=torch.float32)).npu()
    initial_state = (torch.randn(1, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    q_bnsd = q_bsnd.transpose(1, 2).contiguous()
    k_bnsd = k_bsnd.transpose(1, 2).contiguous()
    v_bnsd = v_bsnd.transpose(1, 2).contiguous()
    g_bnsd = g_bsnd.transpose(1, 2).contiguous()
    beta_bns = beta_bsn.transpose(1, 2).contiguous()

    gk_bnsd = torch.ops._C_ascend.kda_gate_cumsum(g_bnsd, 64, layout="BNSD")
    got = torch.ops._C_ascend.chunk_kda_fwd(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        g_bnsd,
        beta_bns,
        scale,
        64,
        layout="BNSD",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    gk_bsnd = gk_bnsd.transpose(1, 2).contiguous()
    ref = chunk_kda_forward_reference(
        q_bsnd.cpu(),
        k_bsnd.cpu(),
        v_bsnd.cpu(),
        gk_bsnd.cpu(),
        beta_bsn.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    out_bsnd = got[0]
    assert torch.isfinite(out_bsnd).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", out_bsnd, ref.o)
    _assert_close("final_state", got[1], ref.final_state)
    _cleanup_npu()


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return None
    return [
        value
        for seq_id, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size)
        for value in (seq_id, chunk_id)
    ]


@pytest.mark.parametrize("disable_recompute", [False, True])
@pytest.mark.parametrize("seq_count,fixed_length", [(127, None), (128, None), (1, 80), (1, 129)])
def test_chunk_kda_fwd_padded_workspace_matches_reference(seq_count, fixed_length, disable_recompute):
    """Full chunks must not overwrite U seeds when other chunks contain padding."""
    torch.manual_seed(1000 + seq_count)
    heads, head_dim, chunk_size = 6, 128, 64
    lengths = (
        [fixed_length] if fixed_length is not None else [16, 32, 64] * (seq_count // 3) + [16, 32, 64][: seq_count % 3]
    )
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    shape = (1, cu_seqlens[-1], heads, head_dim)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1).to(torch.bfloat16)
    v = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    raw_gate = torch.randn(shape) * 0.1
    beta = torch.sigmoid(torch.randn(shape[:-1]))
    initial_state = torch.zeros((seq_count, heads, head_dim, head_dim), dtype=torch.float32)
    metadata = cu_seqlens if fixed_length is None else None

    outputs = torch.ops._C_ascend.chunk_kda_fwd(
        q.npu(),
        k.npu(),
        v.npu(),
        raw_gate.npu(),
        beta.npu(),
        head_dim**-0.5,
        chunk_size,
        initial_state=initial_state.npu(),
        output_final_state=True,
        cu_seqlens=metadata,
        chunk_indices=_canonical_chunk_indices(metadata, chunk_size),
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=torch.zeros(heads, device="npu", dtype=torch.float32),
        dt_bias=torch.zeros(heads * head_dim, device="npu", dtype=torch.float32),
        disable_recompute=disable_recompute,
        state_v_first=True,
    )
    actual = _snapshot_outputs(outputs)
    for name, output in zip(CHUNK_KDA_OUTPUT_NAMES, actual):
        if output is not None:
            assert torch.isfinite(output).all(), name

    gate = _gate_cumsum_reference(-5.0 * torch.sigmoid(raw_gate), chunk_size, cu_seqlens)
    for seq, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        reference = chunk_kda_forward_reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            head_dim**-0.5,
            chunk_size,
            output_final_state=True,
        )
        _assert_close(f"sequence {seq} output", actual[0][:, start:end], reference.o, rtol=5e-2, atol=5e-4)
        assert reference.final_state is not None
        _assert_close(
            f"sequence {seq} final state",
            actual[1][seq : seq + 1],
            reference.final_state.transpose(-1, -2),
            rtol=5e-2,
            atol=3e-3,
        )
    _cleanup_npu()


def _run_chunk_kda_fwd_a5_case(
    layout,
    tokens,
    batch_size,
    query_heads,
    value_heads,
    key_dim,
    value_dim,
    chunk_size,
    dtype,
    cu_seqlens,
):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    if cu_seqlens is not None:
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == tokens
    is_tnd = layout == "TND"
    q_shape = (tokens, query_heads, key_dim) if is_tnd else (batch_size, tokens, query_heads, key_dim)
    v_shape = (tokens, value_heads, value_dim) if is_tnd else (batch_size, tokens, value_heads, value_dim)
    g_shape = (tokens, value_heads, key_dim) if is_tnd else (batch_size, tokens, value_heads, key_dim)
    beta_shape = (tokens, value_heads) if is_tnd else (batch_size, tokens, value_heads)
    q = torch.full(
        q_shape,
        1.0 / math.sqrt(key_dim),
        dtype=dtype,
        device=device,
    )
    k = torch.full_like(q, 1.0 / math.sqrt(key_dim))
    v = torch.zeros(v_shape, dtype=dtype, device=device)
    raw_gate = torch.full(
        g_shape,
        -0.005 * math.log(2.0),
        dtype=torch.float32,
        device=device,
    )
    beta_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    beta = torch.full(beta_shape, 0.5, dtype=beta_dtype, device=device)
    chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)

    torch.npu.synchronize()
    outputs = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        key_dim**-0.5,
        chunk_size,
        layout=layout,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        disable_recompute=False,
        return_intermediate_states=False,
    )
    torch.npu.synchronize()

    sequence_count = len(cu_seqlens) - 1 if cu_seqlens is not None else batch_size
    expected_output_shape = (tokens, value_heads, value_dim) if is_tnd else (batch_size, tokens, value_heads, value_dim)
    expected_gk_shape = (value_heads, tokens, key_dim) if is_tnd else (batch_size, value_heads, tokens, key_dim)
    assert len(outputs) == len(CHUNK_KDA_OUTPUT_NAMES)
    assert outputs[0].shape == expected_output_shape
    assert outputs[1].shape == (sequence_count, value_heads, key_dim, value_dim)
    assert outputs[2].shape == expected_gk_shape
    assert torch.count_nonzero(outputs[0]).item() == 0
    assert torch.count_nonzero(outputs[1]).item() == 0
    assert torch.isfinite(outputs[2]).all().item()

    _cleanup_npu()


@pytest.mark.parametrize(
    ("layout", "cu_seqlens"),
    [
        pytest.param("BSND", None, id="BSND-dense"),
        pytest.param("TND", [0, 2047, 4096, 8191], id="TND-varlen"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda_fwd_a5_profile_t8191(layout, cu_seqlens):
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    _run_chunk_kda_fwd_a5_case(
        layout=layout,
        tokens=8191,
        batch_size=1,
        query_heads=16,
        value_heads=32,
        key_dim=128,
        value_dim=128,
        chunk_size=64,
        dtype=torch.bfloat16,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize(
    (
        "layout",
        "tokens",
        "batch_size",
        "query_heads",
        "value_heads",
        "key_dim",
        "value_dim",
        "chunk_size",
        "dtype",
        "cu_seqlens",
    ),
    [
        pytest.param(
            "BSND",
            600,
            1,
            6,
            12,
            128,
            256,
            128,
            torch.bfloat16,
            [0, 127, 383, 600],
            id="BSND-varlen-bf16",
        ),
        pytest.param(
            "TND",
            257,
            1,
            6,
            12,
            128,
            256,
            128,
            torch.bfloat16,
            None,
            id="TND-dense-bf16",
        ),
        pytest.param(
            "TND",
            300,
            1,
            6,
            12,
            128,
            128,
            64,
            torch.bfloat16,
            [0, 63, 191, 300],
            id="TND-varlen-bf16",
        ),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda_fwd_a5_generalized_layouts(
    layout,
    tokens,
    batch_size,
    query_heads,
    value_heads,
    key_dim,
    value_dim,
    chunk_size,
    dtype,
    cu_seqlens,
):
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    _run_chunk_kda_fwd_a5_case(
        layout=layout,
        tokens=tokens,
        batch_size=batch_size,
        query_heads=query_heads,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk_size,
        dtype=dtype,
        cu_seqlens=cu_seqlens,
    )
PRIVATE_KDA_ASSET_2
python3 -I -B "$private/build_kda.py" --source "$PWD" --output "$private/build"
export ASCEND_CUSTOM_OPP_PATH="$private/build/opp/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=8
python3 -I -B "$private/verify.py" "$private"
