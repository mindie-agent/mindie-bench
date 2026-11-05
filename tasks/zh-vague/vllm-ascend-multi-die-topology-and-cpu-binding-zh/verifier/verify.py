# [Role] Independent CPU behavior oracle for Ascend multi-Die topology & CPU binding (Stream C - C1).
# [Invariants] Only executed behavioral assertions establish success; no NPU hardware required.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources in isolated temp files; emits a JSON verdict only.

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
ALLOWED_FILES = {"vllm_ascend/cpu_binding.py", "vllm_ascend/platform.py"}

SAMPLE_NPU_SMI_M = """\
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend910_9392
0                              1                              1                              Ascend910_9392
1                              0                              2                              Ascend910_9392
1                              1                              3                              Ascend910_9392
2                              0                              4                              Ascend910_9392
2                              1                              5                              Ascend910_9392
3                              0                              6                              Ascend910_9392
3                              1                              7                              Ascend910_9392
4                              0                              8                              Ascend910_9392
4                              1                              9                              Ascend910_9392
5                              0                              10                             Ascend910_9392
5                              1                              11                             Ascend910_9392
6                              0                              12                             Ascend910_9392
6                              1                              13                             Ascend910_9392
7                              0                              14                             Ascend910_9392
7                              1                              15                             Ascend910_9392
"""

SAMPLE_NPU_SMI_INFO = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc3                 Version: 24.1.rc3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 10001         | vllm_worker              | 32000                   |
| 0       1                 | 10002         | vllm_worker              | 32000                   |
| 1       0                 | 10003         | vllm_worker              | 32000                   |
| 1       1                 | 10004         | vllm_worker              | 32000                   |
| 4       0                 | 10009         | vllm_worker              | 32000                   |
| 4       1                 | 10010         | vllm_worker              | 32000                   |
+===========================+===============+====================================================+
"""

SAMPLE_TOPO_8_BOARDS = """\
        NPU0    NPU1    NPU2    NPU3    NPU4    NPU5    NPU6    NPU7    CPU Affinity
NPU0    X       HCCS    HCCS    HCCS    NA      NA      NA      NA      0-47
NPU1    HCCS    X       HCCS    HCCS    NA      NA      NA      NA      0-47
NPU2    HCCS    HCCS    X       HCCS    NA      NA      NA      NA      48-95
NPU3    HCCS    HCCS    HCCS    X       NA      NA      NA      NA      48-95
NPU4    NA      NA      NA      NA      X       HCCS    HCCS    HCCS    96-143
NPU5    NA      NA      NA      NA      HCCS    X       HCCS    HCCS    96-143
NPU6    NA      NA      NA      NA      HCCS    HCCS    X       HCCS    144-191
NPU7    NA      NA      NA      NA      HCCS    HCCS    HCCS    X       144-191
"""

SAMPLE_TOPO_16_DIES = "\n".join(
    ["        CPU Affinity"]
    + [f"NPU{i}   {(i // 4) * 48}-{((i // 4) + 1) * 48 - 1}" for i in range(16)]
)

SAMPLE_LSCPU = "\n".join(
    ["CPU NODE"] + [f"{cpu} {cpu // 48}" for cpu in range(192)]
)


def check_diff_scope() -> None:
    if not (ROOT / ".git").exists():
        return
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    disallowed = {
        p for p in changed if p not in ALLOWED_FILES and not p.startswith("tests/") and not p.startswith("docs/")
    }
    if disallowed:
        raise AssertionError(f"Modified files outside allowed scope {sorted(ALLOWED_FILES)}: {sorted(disallowed)}")


def load_cpu_binding_module():
    if "regex" not in sys.modules:
        sys.modules["regex"] = re
    if "psutil" not in sys.modules:
        psutil_mod = types.ModuleType("psutil")
        psutil_mod.Process = lambda: types.SimpleNamespace(pid=12345)
        sys.modules["psutil"] = psutil_mod
    if "vllm" not in sys.modules:
        vllm_mod = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm_mod
    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.logger = types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            warning_once=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        sys.modules["vllm.logger"] = logger_mod
    if "vllm_ascend" not in sys.modules:
        pkg = types.ModuleType("vllm_ascend")
        pkg.__path__ = [str(ROOT / "vllm_ascend")]
        sys.modules["vllm_ascend"] = pkg
    if "vllm_ascend.utils" not in sys.modules:
        utils_mod = types.ModuleType("vllm_ascend.utils")

        class AscendDeviceType:
            A2 = "A2"
            A3 = "A3"
            _310P = "310P"
            A5 = "A5"

        utils_mod.AscendDeviceType = AscendDeviceType
        utils_mod.get_ascend_device_type = lambda: AscendDeviceType.A2
        sys.modules["vllm_ascend.utils"] = utils_mod

    path = ROOT / "vllm_ascend/cpu_binding.py"
    spec = importlib.util.spec_from_file_location("vllm_ascend.cpu_binding", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_platform_helpers():
    path = ROOT / "vllm_ascend/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if any(name in {"DCMI_THROTTLING_BIT_REASONS"} for name in targets):
                wanted_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name not in {"register"}:
            wanted_nodes.append(node)

    ns = {"Any": object, "os": os, "re": re}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod_ast = ast.Module(body=[future, *wanted_nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod_ast), str(path), "exec"), ns)
    if "decode_dcmi_throttling_mask" not in ns or "select_multi_die_gang_devices" not in ns:
        raise AssertionError(
            "vllm_ascend/platform.py must define decode_dcmi_throttling_mask and select_multi_die_gang_devices"
        )
    return ns["decode_dcmi_throttling_mask"], ns["select_multi_die_gang_devices"]


def verify_cpu_binding(cb_mod) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        status_file = Path(tmpdir) / "status"
        isolated_file = Path(tmpdir) / "isolated"
        status_file.write_text("Name:\tvllm\nCpus_allowed_list:\t0-191\n", encoding="utf-8")
        isolated_file.write_text("0-3,48-51,96-99,144-147\n", encoding="utf-8")

        cb_mod.ALLOWED_CPUS_PATH = str(status_file)
        cb_mod.ISOLATED_CPUS_PATH = str(isolated_file)

        current_topo = [SAMPLE_TOPO_8_BOARDS]

        def fake_execute_command(cmd: list[str]) -> tuple[str, int]:
            if cmd == ["npu-smi", "info", "-m"]:
                return SAMPLE_NPU_SMI_M, 0
            if cmd == ["npu-smi", "info"]:
                return SAMPLE_NPU_SMI_INFO, 0
            if cmd == ["npu-smi", "info", "-t", "topo"]:
                return current_topo[0], 0
            if cmd == ["lscpu", "-e=CPU,NODE"]:
                return SAMPLE_LSCPU, 0
            return "", 0

        cb_mod.execute_command = fake_execute_command

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
        cb_mod.ASCEND_RT_VISIBLE_DEVICES = "0,1"
        alloc_a = cb_mod.CpuAlloc(rank_id=0)
        alloc_a.build_cpu_node_map()
        alloc_a._build_topo_affinity_cpu_pool()

        isolated_set = {0, 1, 2, 3, 48, 49, 50, 51, 96, 97, 98, 99, 144, 145, 146, 147}
        assert not (set(alloc_a.device_info.allowed_cpus) & isolated_set), (
            f"allowed_cpus failed to exclude kernel-isolated CPUs: "
            f"{sorted(set(alloc_a.device_info.allowed_cpus) & isolated_set)}"
        )
        assert len(alloc_a.device_info.allowed_cpus) == 176, (
            f"Expected 176 usable CPUs after subtracting 16 isolated cores, got {len(alloc_a.device_info.allowed_cpus)}"
        )
        assert set(alloc_a.device_info.npu_affinity.keys()) == set(range(16)), (
            f"Dual-Die 8-board topo must expand to 16 logical Phy-IDs 0..15, got {sorted(alloc_a.device_info.npu_affinity.keys())}"
        )

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        cb_mod.ASCEND_RT_VISIBLE_DEVICES = "2,3"
        alloc_b = cb_mod.CpuAlloc(rank_id=0)
        alloc_b.build_cpu_node_map()
        alloc_b._build_topo_affinity_cpu_pool()

        pool_0 = set(alloc_a.npu_cpu_pool[0])
        pool_1 = set(alloc_a.npu_cpu_pool[1])
        pool_2 = set(alloc_b.npu_cpu_pool[2])
        pool_3 = set(alloc_b.npu_cpu_pool[3])
        assert not (pool_0 & pool_2) and not (pool_1 & pool_3) and not (pool_0 & pool_1), (
            f"Co-tenant instances on NUMA 0 have overlapping CPU pools: "
            f"0={sorted(pool_0)}, 1={sorted(pool_1)}, 2={sorted(pool_2)}, 3={sorted(pool_3)}"
        )
        assert pool_0 | pool_1 | pool_2 | pool_3 == (set(range(4, 48)) | set(range(52, 96))), (
            "NUMA 0+1 extended non-isolated CPUs must be partitioned across logical Phy-IDs 0,1,2,3"
        )

        current_topo[0] = SAMPLE_TOPO_16_DIES
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "8,9"
        cb_mod.ASCEND_RT_VISIBLE_DEVICES = "8,9"
        alloc_c = cb_mod.CpuAlloc(rank_id=0)
        alloc_c.build_cpu_node_map()
        alloc_c._build_topo_affinity_cpu_pool()

        expected_numa2_ext = set(range(100, 144)) | set(range(148, 192))
        for phy_id in (8, 9):
            cpus = alloc_c.npu_cpu_pool[phy_id]
            assert set(cpus).issubset(expected_numa2_ext) and any(100 <= c <= 143 for c in cpus), (
                f"Logical Phy-ID {phy_id} must bind to NUMA 2 (+NUMA 3 extended) non-isolated CPUs, got {cpus}"
            )
            assert not any(c < 96 for c in cpus), (
                f"Logical Phy-ID {phy_id} wrongly bound to NUMA 0/1 CPUs: {cpus}"
            )


def verify_platform_helpers(decode_fn, select_fn) -> None:
    ok = decode_fn(0x0, 0)
    assert ok == {"healthy": True, "throttled": False, "uncorrectable_ecc": False, "reasons": []}, (
        f"Unexpected healthy decode: {ok}"
    )

    overtemp = decode_fn(0x1, 0)
    assert overtemp["healthy"] is False and overtemp["throttled"] is True, f"bit 0 failed: {overtemp}"
    assert overtemp["uncorrectable_ecc"] is False and overtemp["reasons"] == ["overtemp_throttling"], (
        f"bit 0 reasons mismatch: {overtemp}"
    )

    pcap = decode_fn(0x2, 0)
    assert pcap["healthy"] is False and pcap["reasons"] == ["power_cap_throttling"], (
        f"bit 1 reasons mismatch: {pcap}"
    )

    ecc = decode_fn(0x8, 0)
    assert ecc["healthy"] is False and ecc["uncorrectable_ecc"] is True and ecc["reasons"] == [
        "hbm_ecc_uncorrectable"
    ], f"bit 3 ECC mismatch: {ecc}"

    combo = decode_fn(0x1 | 0x4 | 0x8, 0)
    assert combo["reasons"] == ["overtemp_throttling", "unknown_bit_2", "hbm_ecc_uncorrectable"], (
        f"Combined bitmask decode mismatch: {combo}"
    )

    err_ret = decode_fn(0x0, -8005)
    assert err_ret["healthy"] is False and err_ret["throttled"] is True and err_ret["reasons"] == [
        "dcmi_query_error(-8005)"
    ], f"Negative ret_code decode mismatch: {err_ret}"

    sel1 = select_fn(4, [0, 2, 3, 4, 5, 8, 9], {})
    assert sel1 == [2, 3, 4, 5], f"Expected complete sibling pairs [2, 3, 4, 5] on Ring 0, got {sel1}"

    sel2 = select_fn(4, [0, 2, 4, 6, 8, 9, 10, 11], {})
    assert sel2 == [8, 9, 10, 11], (
        f"Expected Ring 1 complete sibling pairs [8, 9, 10, 11] over Ring 0 singletons, got {sel2}"
    )

    sel3 = select_fn(2, [0, 1, 2, 3, 4, 5], {1: 0x1, 5: 0x8})
    assert sel3 == [2, 3], f"Expected healthy sibling pair [2, 3] after filtering throttled/ECC dies, got {sel3}"


def main() -> None:
    try:
        check_diff_scope()
        cb_mod = load_cpu_binding_module()
        decode_fn, select_fn = load_platform_helpers()
        verify_cpu_binding(cb_mod)
        verify_platform_helpers(decode_fn, select_fn)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "correctness_verified": False,
                    "error_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "validation_scope": "cpu_behavior",
                    "performance_status": "unmeasured",
                }
            )
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "cpu_behavior",
                "performance_status": "unmeasured",
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
            }
        )
    )


if __name__ == "__main__":
    main()
