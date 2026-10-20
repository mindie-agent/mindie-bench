"""Production S0-S3 topology, stress and multi-rank benchmark contracts.

Tier 0 uses captured A2/A3 `npu-smi` layouts and pure evaluation evidence.
Tier 1 validates all five Section 5.4 task specs and binds their conditions to
the real `GradedRepository`. Tier 2 probes the standing A2/A3 hosts without
starting, stopping or resetting a workload.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_bench.evaluation.knowledge_tiered import ConsultedKnowledgeRef
from mindie_bench.evaluation.knowledge_tiered_support import Store
from mindie_bench.evaluation.production_topology import (
    GangLeasePlan,
    NpuTopology,
    TopologySpec,
    evaluate_distributed_invariance_and_sla,
    load_production_benchmark_task,
    parse_npu_smi_info,
    plan_topology_gang_lease,
    probe_npu_smi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks/production-multi-die"
OPERATORS = Path("/Volumes/data/code/knowledge-operators")
A2_HOST = os.environ.get("MINDIE_BENCH_LIVE_A2_HOST", "root@173.125.1.2")
A3_HOSTS = tuple(
    os.environ.get(
        "MINDIE_BENCH_LIVE_A3_HOSTS", "root@192.168.13.165,root@192.168.13.156"
    ).split(",")
)

A2_HEADER = """+------------------------------------------------------------------------------------------------+
| npu-smi 25.2.2                   Version: 25.2.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
"""
A3_HEADER = """+------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                 Version: 26.0.rc1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
"""
INFO_L_A2 = "\n".join(
    ["\tTotal Count                    : 8", ""]
    + [line for npu in range(8) for line in
       (f"\tNPU ID                         : {npu}", "\tChip Count                     : 1", "")]
) + "\n"
INFO_L_A3 = "\n".join(
    ["\tTotal Count                    : 8", ""]
    + [line for npu in range(8) for line in
       (f"\tNPU ID                         : {npu}", "\tChip Count                     : 2", "")]
) + "\n"


def _npu_smi_capture(*, dual_die: bool, busy_npus: tuple[int, ...] = ()) -> str:
    """Render the captured A2/A3 table shape with explicit busy/idle packages."""
    lines = [A3_HEADER if dual_die else A2_HEADER]
    name = "Ascend910" if dual_die else "910B4"
    for npu in range(8):
        for chip in range(2 if dual_die else 1):
            phy = npu * 2 + chip if dual_die else npu
            hbm = "60232 / 65536" if npu in busy_npus else "3119 / 65536"
            health = "OK"
            lines.append(
                f"| {npu}     {name:<18} | {health:<13} | 170.0        51                     0    / 0                |"
            )
            lines.append(
                f"| {chip}     {phy:<19} | 0000:{npu:02X}:00.0  | 0                    0    / 0                {hbm:<20} |"
            )
            lines.append(
                "+------------------------------------------------------------------------------------------------+"
            )
    lines.append(
        "+---------------------------+---------------+----------------------------------------------------+"
    )
    lines.append(
        "| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |"
    )
    lines.append(
        "+===========================+===============+====================================================+"
    )
    for npu in range(8):
        if npu not in busy_npus:
            lines.append(f"| No running processes found in NPU {npu}".ljust(100) + "|")
            lines.append(
                "+===========================+===============+====================================================+"
            )
            continue
        for chip in range(2 if dual_die else 1):
            pid = 200000 + npu * 2 + chip
            lines.append(
                f"| {npu:<8}{chip:<18}| {pid:<13} | VLLMWorker_TP            | 57000                   |"
            )
        lines.append(
            "+===========================+===============+====================================================+"
        )
    return "\n".join(lines) + "\n"


def _snapshot(host: str, *, dual_die: bool, busy: tuple[int, ...] = (),
              if_ip: str | None = None, shm: tuple[str, ...] = ()) -> NpuTopology:
    return parse_npu_smi_info(
        _npu_smi_capture(dual_die=dual_die, busy_npus=busy),
        info_l_text=INFO_L_A3 if dual_die else INFO_L_A2,
        host=host,
        hccl_if_ip=if_ip or host.split("@")[-1],
        hccl_socket_ifname="enp210s0f0",
        shm_hccl_entries=shm,
    )


def _task_plan(task, inventory) -> GangLeasePlan:
    return task.plan(inventory, hosts=task.topology.hosts)


# ---------------------------------------------------------------- Tier 0 data
def _stress_evidence() -> dict:
    return {
        "reference": {
            "tp": 1,
            "dp": 1,
            "mode": "eager",
            "rank_state_checksums": {"0": "state-a"},
        },
        "variants": [
            {
                "name": "tp2dp2-eager",
                "tp": 2,
                "dp": 2,
                "mode": "eager",
                "kl_divergence": 0.0001,
                "top1_agreement": 0.9999,
                "rank_state_checksums": {"0": "state-a", "1": "state-a", "2": "state-a", "3": "state-a"},
            },
            {
                "name": "tp2dp2-aclgraph",
                "tp": 2,
                "dp": 2,
                "mode": "aclgraph",
                "kl_divergence": 0.0002,
                "top1_agreement": 0.9998,
                "rank_state_checksums": {"0": "state-a", "1": "state-a", "2": "state-a", "3": "state-a"},
            },
        ],
        "stress": {
            "asymmetric_dp": {
                "cases": [
                    {"name": "left", "batch_per_rank": [8, 0], "collective_participants": [0, 1],
                     "conv_state_unchanged": True, "ssm_state_unchanged": True},
                    {"name": "right", "batch_per_rank": [0, 8], "collective_participants": [0, 1],
                     "conv_state_unchanged": True, "ssm_state_unchanged": True},
                    {"name": "one", "batch_per_rank": [1, 8], "collective_participants": [0, 1],
                     "conv_state_unchanged": True, "ssm_state_unchanged": True},
                    {"name": "both-idle", "batch_per_rank": [0, 0],
                     "collective_participants": [0, 1], "conv_state_unchanged": True,
                     "ssm_state_unchanged": True},
                ]
            },
            "moe_skew": {
                "rank_step_ms": {"0": 100.0, "1": 100.0, "2": 100.0, "3": 120.0},
                "expert_counts": [100, 100, 10, 10],
            },
            "varlen_continuous": {
                "varlen_prefill": {"lengths": [128, 512, 2048]},
                "continuous_batching_decode": {"tpot_ms": 45.0},
                "lengths": [128, 512, 2048],
            },
            "concurrency_ladder": {
                "levels": {
                    1: {"success": True, "goodput_tokens_per_sec": 80.0},
                    16: {"success": True, "goodput_tokens_per_sec": 900.0},
                    64: {"success": True, "goodput_tokens_per_sec": 1400.0},
                }
            },
        },
        "metrics": {
            "ttft_p50_ms": 120.0,
            "ttft_p95_ms": 240.0,
            "tpot_p50_ms": 40.0,
            "tpot_p95_ms": 50.0,
            "goodput_tokens_per_sec": 500.0,
            "comm_ratio": 0.2,
            "overlap_ratio": 0.5,
            "rank_workspace_delta_bytes": {"0": 1024, "1": 2048, "2": 1024, "3": 1536},
        },
        "guard_workloads": [
            {"name": "uniform-guard", "baseline_ms": 100.0, "candidate_ms": 101.0,
             "max_regression_ratio": 1.05}
        ],
        "max_workspace_delta_bytes": 4096,
    }


# ------------------------------------------------------------------- Tier 0
@pytest.mark.tier0
def test_a2_a3_golden_captures_identify_phy_ids_and_busy_idle_dies():
    a2 = _snapshot("173.125.1.2", dual_die=False)
    assert a2.die_count == 8 and a2.chip_counts == {npu: 1 for npu in range(8)}
    assert a2.idle_phy_ids == tuple(range(8))
    assert a2.dual_die_packages == ()

    idle = _snapshot("192.168.13.165", dual_die=True)
    assert idle.die_count == 16 and idle.chip_counts == {npu: 2 for npu in range(8)}
    assert idle.idle_phy_ids == tuple(range(16))
    assert idle.dual_die_packages == tuple(range(8))
    assert all(die.sibling_phy_id == phy ^ 1 for phy, die in enumerate(idle.dies))

    busy = _snapshot("192.168.13.156", dual_die=True, busy=(0, 1, 4, 5, 6, 7))
    assert busy.idle_phy_ids == (4, 5, 6, 7)
    assert busy.busy_phy_ids == (0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15)
    assert all("VLLMWorker_TP" in die.process_names for die in busy.busy_dies)

    full = _snapshot("192.168.13.154", dual_die=True, busy=tuple(range(8)))
    assert full.idle_phy_ids == () and full.busy_phy_ids == tuple(range(16))

    # Residual HBM after a completed job must not hide an explicit no-process Die.
    residual = parse_npu_smi_info(
        _npu_smi_capture(dual_die=True, busy_npus=(0, 1, 4, 5, 6, 7)).replace(
            "3119 / 65536", "60232 / 65536"
        ),
        info_l_text=INFO_L_A3,
        host="192.168.13.156",
    )
    assert residual.idle_phy_ids == (4, 5, 6, 7)
    assert all(die.hbm_used_mb > 4096 for die in residual.idle_dies)


@pytest.mark.tier0
def test_same_npu_dual_die_allocates_sibling_pairs_and_rejects_straddling():
    busy = _snapshot("192.168.13.156", dual_die=True, busy=(0, 1, 4, 5, 6, 7))
    pair = plan_topology_gang_lease(
        TopologySpec("S1", 2, "same_npu_dual_die"), busy
    )
    assert pair.ok and pair.visible_phy_ids == (4, 5)
    assert pair.allocations[0].sibling_pairs == ((4, 5),)
    assert pair.allocations[0].ascend_rt_visible_devices == "4,5"

    four = plan_topology_gang_lease(
        TopologySpec("S1", 4, "same_npu_dual_die"), busy
    )
    assert four.ok and four.visible_phy_ids == (4, 5, 6, 7)
    assert set(four.allocations[0].sibling_pairs) == {(4, 5), (6, 7)}

    straddle = plan_topology_gang_lease(
        TopologySpec("S1", 2, "same_npu_dual_die", visible_phy_ids=(1, 2)),
        _snapshot("192.168.13.165", dual_die=True),
    )
    assert not straddle.ok and straddle.cross_package_degradation
    assert straddle.allocations == ()
    assert any("cross_package_degradation" in row for row in straddle.blockers)

    a2 = _snapshot("173.125.1.2", dual_die=False)
    assert not plan_topology_gang_lease(
        TopologySpec("S1", 2, "same_npu_dual_die"), a2
    ).ok
    with pytest.raises(ValueError, match="2..4"):
        TopologySpec("S1", 5, "intra_node_hccs")


@pytest.mark.tier0
def test_intra_node_and_pd_disaggregation_keep_instance_visibility_disjoint():
    idle = _snapshot("192.168.13.165", dual_die=True)
    four = plan_topology_gang_lease(TopologySpec("S1", 4, "intra_node_hccs"), idle)
    assert four.ok and four.visible_phy_ids == (0, 1, 2, 3)

    pd = plan_topology_gang_lease(
        TopologySpec("S2", 16, "disaggregated_pd_pair"), idle
    )
    assert pd.ok and pd.prefill and pd.decode
    assert pd.prefill.visible_phy_ids == tuple(range(8))
    assert pd.decode.visible_phy_ids == tuple(range(8, 16))
    assert set(pd.prefill.visible_phy_ids).isdisjoint(pd.decode.visible_phy_ids)
    assert pd.prefill.environment["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert pd.decode.environment["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    assert pd.prefill.environment["MINDIE_BENCH_PD_ROLE"] == "prefill"
    assert pd.decode.environment["MINDIE_BENCH_PD_ROLE"] == "decode"

    busy = _snapshot("192.168.13.156", dual_die=True, busy=(0, 1, 4, 5, 6, 7))
    assert not plan_topology_gang_lease(
        TopologySpec("S2", 16, "disaggregated_pd_pair"), busy
    ).ok


@pytest.mark.tier0
def test_multi_node_roce_requires_two_hccl_hosts_and_clean_shm():
    host_a = _snapshot("192.168.13.165", dual_die=True, if_ip="192.168.13.165")
    host_b = _snapshot("192.168.13.156", dual_die=True, if_ip="192.168.13.156")
    spec = TopologySpec(
        "S3", 16, "multi_node_roce", hosts=("192.168.13.165", "192.168.13.156")
    )
    plan = plan_topology_gang_lease(
        spec, inventories={"192.168.13.165": host_a, "192.168.13.156": host_b}
    )
    assert plan.ok, plan.blockers
    assert [row.visible_phy_ids for row in plan.allocations] == [
        tuple(range(8)),
        tuple(range(8)),
    ]
    assert plan.hccl_interfaces["192.168.13.165"]["HCCL_SOCKET_IFNAME"] == "enp210s0f0"

    missing_interface = replace(host_b, hccl_if_ip=None)
    assert not plan_topology_gang_lease(
        spec,
        inventories={"192.168.13.165": host_a, "192.168.13.156": missing_interface},
    ).ok
    dirty = replace(host_b, shm_hccl_entries=("hccl_abc",))
    rejected = plan_topology_gang_lease(
        spec, inventories={"192.168.13.165": host_a, "192.168.13.156": dirty}
    )
    assert not rejected.ok
    assert any("residual /dev/shm/hccl*" in row for row in rejected.blockers)
    assert not plan_topology_gang_lease(spec, host_a).ok


@pytest.mark.tier0
def test_distributed_invariance_and_sla_accepts_the_production_evidence():
    result = evaluate_distributed_invariance_and_sla(_stress_evidence())
    assert result["passed"], result["failed_gates"]
    assert result["metrics"]["rank_skew_ratio"] == 1.2
    assert result["max_rank_workspace_delta_bytes"] == 2048
    assert result["l4_03"]["passed"] is True
    assert result["parallelism_invariance"]["variants"][1]["mode"] == "aclgraph"
    assert result["metrics"]["goodput_tokens_per_sec"] == 500.0


@pytest.mark.tier0
def test_asymmetric_dp_detects_timeout_idle_participation_and_state_corruption():
    evidence = _stress_evidence()
    cases = evidence["stress"]["asymmetric_dp"]["cases"]
    cases[0]["error"] = "HCCL_EXEC_TIMEOUT while waiting on rank 1"
    cases[1]["idle_rank_participated"] = False
    cases[2]["conv_state_corruption"] = True
    cases[3]["ssm_state_unchanged"] = False
    result = evaluate_distributed_invariance_and_sla(evidence)
    assert not result["passed"]
    failures = result["stages"]["asymmetric_dp"]["failures"]
    assert "HCCL_EXEC_TIMEOUT" in failures
    assert "missing_idle_rank_collective_participation" in failures
    assert "conv_state_corruption" in failures
    assert "ssm_state_corruption" in failures
    assert "asymmetric_dp" in result["failed_gates"]


@pytest.mark.tier0
def test_parallelism_invariance_rejects_kl_top1_checksum_and_missing_aclgraph():
    evidence = _stress_evidence()
    evidence["variants"][0]["kl_divergence"] = 0.5
    evidence["variants"][0]["top1_agreement"] = 0.5
    evidence["variants"][1]["rank_state_checksums"]["2"] = "corrupt"
    result = evaluate_distributed_invariance_and_sla(evidence)
    assert not result["passed"]
    rows = result["parallelism_invariance"]["variants"]
    assert "kl_divergence_exceeded" in rows[0]["failures"]
    assert "top1_agreement_below_floor" in rows[0]["failures"]
    assert "rank_state_checksum_mismatch" in rows[1]["failures"]

    evidence = _stress_evidence()
    evidence["variants"] = [evidence["variants"][0]]
    result = evaluate_distributed_invariance_and_sla(evidence)
    assert "aclgraph_variant_not_measured" in result["parallelism_invariance"]["failures"]


@pytest.mark.tier0
def test_logit_kl_top1_and_alternate_evidence_keys_are_computed_not_assumed():
    evidence = _stress_evidence()
    evidence["candidates"] = evidence.pop("variants")
    evidence["stress_inputs"] = evidence.pop("stress")
    evidence["sla"] = {"metrics": evidence.pop("metrics")}
    evidence["reference"]["logits"] = [1.0, 2.0, 3.0]
    evidence["candidates"][0]["logits"] = [1.0, 2.0, 3.0]
    evidence["candidates"][1]["logits"] = [1.0, 2.0, 3.0]
    for variant in evidence["candidates"]:
        variant.pop("kl_divergence")
        variant.pop("top1_agreement")
    passed = evaluate_distributed_invariance_and_sla(evidence)
    assert passed["passed"], passed["failed_gates"]
    assert passed["parallelism_invariance"]["variants"][0]["kl_divergence"] == 0.0
    assert passed["parallelism_invariance"]["variants"][0]["top1_agreement"] == 1.0

    evidence["candidates"][1]["logits"] = [3.0, 2.0, 1.0]
    failed = evaluate_distributed_invariance_and_sla(evidence)
    assert "top1_agreement_below_floor" in failed["parallelism_invariance"]["variants"][1]["failures"]


@pytest.mark.tier0
def test_moe_skew_varlen_concurrency_and_l4_guards_reject_bad_production_runs():
    evidence = _stress_evidence()
    evidence["stress"]["moe_skew"]["rank_step_ms"] = {
        "0": 100.0, "1": 100.0, "2": 100.0, "3": 300.0
    }
    evidence["stress"]["concurrency_ladder"]["levels"].pop(64)
    evidence["metrics"]["comm_ratio"] = 0.95
    evidence["metrics"]["overlap_ratio"] = 0.01
    evidence["metrics"]["rank_workspace_delta_bytes"]["3"] = 8192
    evidence["guard_workloads"] = [
        {"name": "uniform-guard", "baseline_ms": 100.0, "candidate_ms": 140.0}
    ]
    result = evaluate_distributed_invariance_and_sla(evidence)
    assert not result["passed"]
    assert "L4-03" in result["failed_gates"]
    assert "max_rank_workspace_delta_above_limit" in result["l4_03"]["failures"]
    assert "guard_workload_regressed" in result["l4_03"]["failures"]
    assert "straggler_rank_skew" in result["stages"]["moe_skew"]["failures"]
    assert "missing_concurrency:64" in result["stages"]["concurrency_ladder"]["failures"]
    assert "sla" in result["failed_gates"]


# ------------------------------------------------------------------- Tier 1
EXPECTED_TASKS = {
    "a3-dual-die-tp2-allreduce-placement": ("S1", 2, "same_npu_dual_die"),
    "qwen35-gdn-dp2-tp2-asymmetric-aclgraph": ("S1", 4, "intra_node_hccs"),
    "qwen3-moe-tp2-ep4-mc2-eplb-skew": ("S2", 8, "intra_node_hccs"),
    "a3-16die-pd-disaggregated-mooncake-handoff": ("S2", 16, "disaggregated_pd_pair"),
    "a3-2node-roce-hccl-dp4-ep8-canary": ("S3", 16, "multi_node_roce"),
}


@pytest.mark.tier1
def test_five_production_tasks_match_section_5_4_contract():
    directories = sorted(path for path in TASKS_ROOT.iterdir() if path.is_dir())
    assert [path.name for path in directories] == sorted(EXPECTED_TASKS)
    loaded = {}
    for path in directories:
        task = load_production_benchmark_task(path)
        scale, dies, placement = EXPECTED_TASKS[path.name]
        assert task.instance_id == path.name
        assert (task.topology.scale, task.topology.die_count, task.topology.placement) == (
            scale,
            dies,
            placement,
        )
        environment = task.task["environment"]["l3"]
        assert environment["npu_count"] == dies
        assert environment["topology_scale"] == scale
        assert environment["placement"] == placement
        assert task.query and "instruction" not in task.query
        loaded[path.name] = task
    assert len({task.instance_id for task in loaded.values()}) == 5
    assert loaded["a3-2node-roce-hccl-dp4-ep8-canary"].topology.hosts == (
        "192.168.13.165",
        "192.168.13.156",
    )


@pytest.mark.tier1
def test_task_topology_conditions_bind_to_graded_repository_and_tiered_refs(tmp_path):
    if not OPERATORS.is_dir():
        pytest.skip(f"real graded corpus is unavailable: {OPERATORS}")
    store = Store(tmp_path / "operators", "operators")
    store.graded.install_root(OPERATORS)
    try:
        for path in sorted(TASKS_ROOT.iterdir()):
            if not path.is_dir():
                continue
            task = load_production_benchmark_task(path)
            payload = store.graded.query(
                task.query,
                conditions={
                    **task.conditions,
                    "soc": "Ascend910B4",
                    "dtype": "bfloat16",
                    "npu_count": task.topology.die_count,
                },
                include_unknown=True,
                navigation_only=True,
                limit=5,
            )
            assert payload["page"] == "graded_units"
            cards = payload.get("cards") or payload.get("results")
            assert cards
            ref = ConsultedKnowledgeRef.from_value(cards[0])
            assert ref.ref.startswith("mindie://")
            view = store.graded.explain(
                refs=[ref.ref],
                conditions={**task.conditions, "soc": "Ascend910B4", "dtype": "bfloat16"},
                token_budget=4096,
            )
            assert view["domain"] == "operators"
            if cards[0].get("match_status") == "matched":
                assert view["units"] or view["requires"]
            else:
                assert view["unresolved"]
    finally:
        store.close()


@pytest.mark.tier1
def test_each_production_task_plans_against_its_golden_inventory():
    idle = _snapshot("192.168.13.165", dual_die=True, if_ip="192.168.13.165")
    second = _snapshot("192.168.13.156", dual_die=True, if_ip="192.168.13.156")
    for path in sorted(TASKS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        task = load_production_benchmark_task(path)
        inventory = (
            {"192.168.13.165": idle, "192.168.13.156": second}
            if task.topology.placement == "multi_node_roce"
            else idle
        )
        plan = _task_plan(task, inventory)
        assert plan.ok, (task.instance_id, plan.blockers)
        assert plan.visible_phy_ids
        if task.topology.placement == "disaggregated_pd_pair":
            assert plan.prefill and plan.decode
            assert set(plan.prefill.visible_phy_ids).isdisjoint(plan.decode.visible_phy_ids)


# ------------------------------------------------------------------- Tier 2
def _ssh(host: str, script: str, *, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            host, shlex.join(["bash", "-lc", script]),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _exercise_sandbox(host: str, plans: tuple[GangLeasePlan, ...]) -> str:
    sandbox = f"/tmp/mindie-npu-test-{uuid.uuid4().hex[:12]}"
    quoted = shlex.quote(sandbox)
    payload = json.dumps([plan.to_dict() for plan in plans], sort_keys=True)
    script = (
        "set -eu\n"
        f"sandbox={quoted}\n"
        "trap 'rm -rf \"$sandbox\"' EXIT\n"
        "mkdir -p \"$sandbox\"\n"
        f"printf '%s\\n' {shlex.quote(payload)} > \"$sandbox/plan.json\"\n"
        "test -s \"$sandbox/plan.json\"\n"
        "python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \"$sandbox/plan.json\"\n"
        "rm -rf \"$sandbox\"\n"
        "trap - EXIT\n"
    )
    completed = _ssh(host, script)
    assert completed.returncode == 0, completed.stderr
    return sandbox


@pytest.mark.live_npu
def test_live_a2_a3_inventory_and_real_idle_s1_s2_gang_leases():
    hosts = (A2_HOST, *A3_HOSTS)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        snapshots = list(pool.map(lambda host: probe_npu_smi(host, timeout=12.0), hosts))
    reachable = [snapshot for snapshot in snapshots if snapshot.reachable]
    if not reachable:
        pytest.skip("no standing A2/A3 host is reachable for the live topology probe")

    a2 = snapshots[0]
    a3_rows = [
        snapshot for snapshot in snapshots[1:]
        if snapshot.reachable and snapshot.dual_die_packages
    ]
    work: dict[str, list[GangLeasePlan]] = {}
    assertions = []
    if a2.reachable and len(a2.idle_phy_ids) >= 4:
        plan = plan_topology_gang_lease(
            TopologySpec("S1", 4, "intra_node_hccs", hosts=(a2.host,)), a2
        )
        if plan.ok:
            work.setdefault(a2.host, []).append(plan)
            assertions.append(("A2 S1 4-Die", plan))
    for snapshot in a3_rows:
        for die_count in (2, 4):
            if len(snapshot.idle_phy_ids) < die_count:
                continue
            plan = plan_topology_gang_lease(
                TopologySpec(
                    "S1",
                    die_count,
                    "same_npu_dual_die" if die_count == 2 else "intra_node_hccs",
                    hosts=(snapshot.host,),
                ),
                snapshot,
            )
            if plan.ok:
                work.setdefault(snapshot.host, []).append(plan)
                assertions.append((f"{snapshot.host} S1 {die_count}-Die", plan))
        if len(snapshot.idle_phy_ids) >= 16:
            plan = plan_topology_gang_lease(
                TopologySpec("S2", 16, "disaggregated_pd_pair", hosts=(snapshot.host,)),
                snapshot,
            )
            if plan.ok:
                work.setdefault(snapshot.host, []).append(plan)
                assertions.append((f"{snapshot.host} S2 16-Die PD", plan))
    if len(a3_rows) >= 2 and all(len(row.idle_phy_ids) >= 8 for row in a3_rows[:2]):
        plan = plan_topology_gang_lease(
            TopologySpec(
                "S3",
                16,
                "multi_node_roce",
                hosts=(a3_rows[0].host, a3_rows[1].host),
            ),
            inventories={row.host: row for row in a3_rows[:2]},
        )
        if plan.ok:
            for row in plan.allocations:
                work.setdefault(row.host, []).append(plan)
            assertions.append(("A3 S3 two-node RoCE", plan))

    if not assertions:
        pytest.skip("standing hosts are reachable but expose no idle S1/S2 lease")
    for host, plans in work.items():
        _exercise_sandbox(host, tuple(plans))
    for label, plan in assertions:
        assert plan.ok, (label, plan.blockers)
        assert plan.lease_id.startswith("gang-")
        assert plan.sandbox_root.startswith("/tmp/mindie-npu-test-")
        if plan.spec.placement == "disaggregated_pd_pair":
            assert plan.prefill and plan.decode
            assert set(plan.prefill.visible_phy_ids).isdisjoint(plan.decode.visible_phy_ids)
