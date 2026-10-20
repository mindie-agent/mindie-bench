"""S0-S3 production topology gang leases and distributed invariance gates.

[Role]
Turn an S0-S3 topology contract plus real `npu-smi` inventories into an isolated
multi-die/multi-card/multi-node gang lease, and expose the distributed stress,
parallelism-invariance, SLA and L4-03 evaluation entrypoint.

[Invariants]
- A2 exposes 8 NPU x 1 chip (Phy-ID == NPU ID); A3 exposes 8 NPU x 2 chips
  (Phy-ID 0..15, siblings `(2k, 2k+1)`). A busy process table row and residual
  HBM both block allocation.
- `same_npu_dual_die` never allocates a straddling or odd pair such as `(1, 2)`;
  it reports `cross_package_degradation` instead. PD instances own disjoint
  `ASCEND_RT_VISIBLE_DEVICES`; S3 requires >=2 reachable hosts with HCCL
  interfaces and a clean `/dev/shm/hccl*`.
- Missing measurements are `not_measured`, never a zero or a pass. HCCL timeout,
  idle-rank non-participation, recurrent-state drift, KL/top-1 divergence,
  state-checksum mismatch, SLA miss, workspace overflow and guard regression all
  fail closed. `L4-03` names the workspace/guard rejection boundary.

[Timeouts & Locks]
- Parsing, planning and evaluation are pure memory work. `probe_npu_smi` is the
  only network call and is read-only with a caller-bounded SSH timeout. Lease
  planning performs no remote write and acquires no device lock.

[Side-Effect Policy]
- Never starts, stops or resets an NPU workload. A live probe only runs
  `npu-smi info`, `npu-smi info -l`, interface lookup and `/dev/shm/hccl*`
  listing. Sandbox paths are returned for the caller; this module never creates
  or deletes them.
"""

from __future__ import annotations

import shlex
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from mindie_bench.evaluation.production_topology_inventory import (
    NpuDie,
    NpuTopology,
    parse_npu_smi_info,
    parse_npu_smi_info_l,
    probe_npu_smi,
)
from mindie_bench.evaluation.production_topology_metrics import (
    L4_03,
    SCHEMA_VERSION,
    evaluate_distributed_invariance_and_sla,
)

Scale = Literal["S0", "S1", "S2", "S3"]
Placement = Literal[
    "same_npu_dual_die", "intra_node_hccs", "disaggregated_pd_pair", "multi_node_roce"
]
_SCALE_DIES = {"S0": (1, 1), "S1": (2, 4), "S2": (8, 16), "S3": (2, 32)}
_PLACEMENTS = {
    "same_npu_dual_die",
    "intra_node_hccs",
    "disaggregated_pd_pair",
    "multi_node_roce",
}


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


# --------------------------------------------------------------- topology spec
@dataclass(frozen=True)
class TopologySpec:
    scale: Scale
    die_count: int
    placement: Placement
    require_clean_shm: bool = True
    hosts: tuple[str, ...] = ()
    visible_phy_ids: tuple[int, ...] = ()
    prefill_die_count: int | None = None
    decode_die_count: int | None = None
    max_workspace_delta_bytes: int | None = None
    idle_hbm_limit_mb: int = 4096

    def __post_init__(self) -> None:
        if self.scale not in _SCALE_DIES:
            raise ValueError(f"unknown topology scale: {self.scale}")
        low, high = _SCALE_DIES[self.scale]
        if not low <= self.die_count <= high:
            raise ValueError(f"{self.scale} requires {low}..{high} dies, got {self.die_count}")
        if self.placement not in _PLACEMENTS:
            raise ValueError(f"unknown placement policy: {self.placement}")
        if isinstance(self.hosts, str):
            object.__setattr__(self, "hosts", (self.hosts,))
        if isinstance(self.visible_phy_ids, int):
            object.__setattr__(self, "visible_phy_ids", (self.visible_phy_ids,))
        object.__setattr__(self, "hosts", _unique(tuple(self.hosts)))
        ids = tuple(int(item) for item in self.visible_phy_ids)
        if any(item < 0 for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("visible Phy-IDs must be distinct non-negative integers")
        object.__setattr__(self, "visible_phy_ids", ids)
        if self.placement == "same_npu_dual_die" and self.die_count % 2:
            raise ValueError("same_npu_dual_die requires an even die count")
        if self.placement == "disaggregated_pd_pair" and self.die_count % 2:
            raise ValueError("disaggregated_pd_pair requires an even die count")
        if self.placement == "disaggregated_pd_pair":
            total = self.die_count
            prefill = total // 2 if self.prefill_die_count is None else self.prefill_die_count
            decode = total - prefill if self.decode_die_count is None else self.decode_die_count
            if prefill <= 0 or decode <= 0 or prefill + decode != total:
                raise ValueError("PD die counts must be positive and sum to die_count")
            object.__setattr__(self, "prefill_die_count", prefill)
            object.__setattr__(self, "decode_die_count", decode)
        if self.max_workspace_delta_bytes is not None and self.max_workspace_delta_bytes < 0:
            raise ValueError("max_workspace_delta_bytes must be non-negative")

    @property
    def placement_policy(self) -> str:
        return self.placement

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "die_count": self.die_count,
            "placement": self.placement,
            "require_clean_shm": self.require_clean_shm,
            "hosts": list(self.hosts),
            "visible_phy_ids": list(self.visible_phy_ids),
            "prefill_die_count": self.prefill_die_count,
            "decode_die_count": self.decode_die_count,
            "max_workspace_delta_bytes": self.max_workspace_delta_bytes,
        }


# --------------------------------------------------------------- gang lease
@dataclass(frozen=True)
class InstanceAllocation:
    host: str
    role: str
    visible_phy_ids: tuple[int, ...]
    visible_npu_ids: tuple[int, ...]
    sibling_pairs: tuple[tuple[int, int], ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def ascend_rt_visible_devices(self) -> str:
        return ",".join(str(item) for item in self.visible_phy_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "role": self.role,
            "visible_phy_ids": list(self.visible_phy_ids),
            "visible_npu_ids": list(self.visible_npu_ids),
            "sibling_pairs": [list(pair) for pair in self.sibling_pairs],
            "ascend_rt_visible_devices": self.ascend_rt_visible_devices,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class GangLeasePlan:
    spec: TopologySpec
    allocations: tuple[InstanceAllocation, ...]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    cross_package_degradation: bool = False
    lease_id: str = ""
    sandbox_root: str = ""
    hccl_interfaces: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.blockers and bool(self.allocations)

    @property
    def ready(self) -> bool:
        return self.ok

    @property
    def status(self) -> str:
        return "allocated" if self.ok else "rejected"

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def visible_phy_ids(self) -> tuple[int, ...]:
        return tuple(phy for row in self.allocations for phy in row.visible_phy_ids)

    @property
    def selected_phy_ids(self) -> tuple[int, ...]:
        return self.visible_phy_ids

    @property
    def prefill(self) -> InstanceAllocation | None:
        return next((row for row in self.allocations if row.role == "prefill"), None)

    @property
    def decode(self) -> InstanceAllocation | None:
        return next((row for row in self.allocations if row.role == "decode"), None)

    @property
    def prefill_visible_phy_ids(self) -> tuple[int, ...]:
        return self.prefill.visible_phy_ids if self.prefill else ()

    @property
    def decode_visible_phy_ids(self) -> tuple[int, ...]:
        return self.decode.visible_phy_ids if self.decode else ()

    @property
    def cleanup_command(self) -> str:
        return f"rm -rf {shlex.quote(self.sandbox_root)}" if self.sandbox_root else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "lease_id": self.lease_id,
            "status": self.status,
            "ok": self.ok,
            "success": self.ok,
            "scale": self.spec.scale,
            "placement": self.spec.placement,
            "die_count": self.spec.die_count,
            "visible_phy_ids": list(self.visible_phy_ids),
            "selected_phy_ids": list(self.visible_phy_ids),
            "sandbox_root": self.sandbox_root,
            "cleanup_command": self.cleanup_command,
            "cross_package_degradation": self.cross_package_degradation,
            "allocations": [row.to_dict() for row in self.allocations],
            "prefill": self.prefill.to_dict() if self.prefill else None,
            "decode": self.decode.to_dict() if self.decode else None,
            "hccl_interfaces": {key: dict(value) for key, value in self.hccl_interfaces.items()},
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def _snapshots(
    inventory: Any, inventories: Any, *, host: str | None, hosts: Sequence[str]
) -> list[NpuTopology]:
    selected = inventories if inventories is not None else inventory
    fallback = host or (hosts[0] if hosts else "local")
    rows: list[NpuTopology] = []
    if selected is None:
        return rows
    if isinstance(selected, NpuTopology):
        rows = [selected]
    elif isinstance(selected, str):
        rows = [parse_npu_smi_info(selected, host=fallback)]
    elif isinstance(selected, Mapping):
        for key, value in selected.items():
            if isinstance(value, NpuTopology):
                rows.append(replace(value, host=str(key)))
            elif isinstance(value, str):
                rows.append(parse_npu_smi_info(value, host=str(key)))
    elif isinstance(selected, Sequence):
        for index, value in enumerate(selected):
            host_name = hosts[index] if index < len(hosts) else fallback
            if isinstance(value, NpuTopology):
                rows.append(replace(value, host=value.host or host_name))
            elif isinstance(value, str):
                rows.append(parse_npu_smi_info(value, host=host_name))
    return [row for row in rows if row.reachable]


def _explicit_dies(
    snapshot: NpuTopology, phy_ids: Sequence[int], blockers: list[str]
) -> tuple[NpuDie, ...]:
    rows = []
    for phy in phy_ids:
        die = snapshot.die_by_phy(int(phy))
        if die is None:
            blockers.append(f"Phy-ID {phy} does not exist on {snapshot.host}")
        elif die.busy:
            blockers.append(f"Phy-ID {phy} is busy on {snapshot.host}: {die.busy_reason}")
        else:
            rows.append(die)
    return tuple(sorted(rows, key=lambda item: item.phy_id))


def _whole_packages(snapshot: NpuTopology, count: int) -> tuple[NpuDie, ...]:
    selected: list[NpuDie] = []
    for npu in snapshot.dual_die_packages:
        pair = sorted(
            (die for die in snapshot.idle_dies if die.npu_id == npu), key=lambda item: item.phy_id
        )
        if len(pair) == 2:
            selected.extend(pair)
        if len(selected) >= count:
            break
    return tuple(selected[:count])


def _straddles_packages(phy_ids: Sequence[int]) -> list[tuple[int, int]]:
    pairs = []
    ordered = sorted(int(phy) for phy in phy_ids)
    for index in range(0, len(ordered) - 1, 2):
        first, second = ordered[index], ordered[index + 1]
        if second != first + 1 or first % 2:
            pairs.append((first, second))
    return pairs


def _select_same_npu(
    spec: TopologySpec,
    snapshot: NpuTopology,
    requested: Sequence[int],
    blockers: list[str],
    warnings: list[str],
    degradation: list[bool],
) -> tuple[NpuDie, ...]:
    if not snapshot.dual_die_packages:
        blockers.append(f"{snapshot.host} exposes no dual-die NPU package (A2 is 1 Die/NPU)")
        return ()
    straddling = _straddles_packages(requested) if requested else []
    if straddling:
        degradation[0] = True
        blockers.append(
            "cross_package_degradation: requested "
            + ",".join(str(phy) for phy in requested)
            + " are not sibling Dies on one NPU package"
        )
    if requested:
        selected = _explicit_dies(snapshot, requested, blockers)
        if len(selected) != spec.die_count:
            blockers.append(f"requested {len(requested)} dies, topology requires {spec.die_count}")
    else:
        selected = _whole_packages(snapshot, spec.die_count)
        if len(selected) < spec.die_count:
            blockers.append(
                f"{snapshot.host} has {len(selected)} idle same-package dies, "
                f"needs {spec.die_count}"
            )
    if not requested:
        for pair in [selected[index : index + 2] for index in range(0, len(selected), 2)]:
            if len(pair) != 2 or (
                pair[0].phy_id % 2 == 0 and pair[0].sibling_phy_id != pair[1].phy_id
            ):
                degradation[0] = True
                blockers.append(
                    "cross_package_degradation: "
                    + ",".join(str(die.phy_id) for die in pair)
                    + " are not sibling dies on one NPU package"
                )
    if len(selected) == spec.die_count and not straddling:
        warnings.append("same_npu_dual_die uses HCCS sibling Dies inside each NPU package")
    return selected


def _select_single_host(
    spec: TopologySpec,
    snapshots: Sequence[NpuTopology],
    requested: Sequence[int],
    blockers: list[str],
) -> tuple[NpuTopology | None, tuple[NpuDie, ...]]:
    for snapshot in snapshots:
        if requested and any(snapshot.die_by_phy(phy) is None for phy in requested):
            continue
        if requested:
            selected = _explicit_dies(snapshot, requested, blockers)
        else:
            selected = _whole_packages(snapshot, spec.die_count)
            if len(selected) < spec.die_count:
                selected = tuple(sorted(snapshot.idle_dies, key=lambda die: die.phy_id))[
                    : spec.die_count
                ]
        if len(selected) == spec.die_count:
            return snapshot, selected
    blockers.append(
        f"explicit dies {list(requested)} are not free on one host"
        if requested
        else f"no single host has {spec.die_count} idle dies"
    )
    return None, ()


def _allocation(
    snapshot: NpuTopology,
    role: str,
    dies: Sequence[NpuDie],
    *,
    environment: Mapping[str, str] | None = None,
) -> InstanceAllocation:
    ordered = tuple(sorted(dies, key=lambda die: die.phy_id))
    env = {
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(die.phy_id) for die in ordered),
        "MINDIE_BENCH_HOST": snapshot.host,
        "MINDIE_BENCH_LEASE_ROLE": role,
    }
    if snapshot.hccl_if_ip:
        env["HCCL_IF_IP"] = snapshot.hccl_if_ip
    if snapshot.hccl_socket_ifname:
        env["HCCL_SOCKET_IFNAME"] = snapshot.hccl_socket_ifname
    env.update(environment or {})
    return InstanceAllocation(
        host=snapshot.host,
        role=role,
        visible_phy_ids=tuple(die.phy_id for die in ordered),
        visible_npu_ids=_unique([die.npu_id for die in ordered]),
        sibling_pairs=tuple(
            (die.phy_id, die.sibling_phy_id)
            for die in ordered
            if die.phy_id % 2 == 0 and die.sibling_phy_id in {item.phy_id for item in ordered}
        ),
        environment=env,
    )


def plan_topology_gang_lease(
    spec: TopologySpec,
    inventory: Any = None,
    *,
    inventories: Any = None,
    host: str | None = None,
    hosts: Sequence[str] | None = None,
    requested_visible_phy_ids: Sequence[int] | None = None,
    sandbox_root: str | None = None,
) -> GangLeasePlan:
    """Plan a non-overlapping gang lease; rejected plans explain every blocker."""
    host_names = tuple(hosts) if hosts is not None else spec.hosts
    snapshots = _snapshots(inventory, inventories, host=host, hosts=host_names)
    requested = tuple(
        requested_visible_phy_ids
        if requested_visible_phy_ids is not None
        else spec.visible_phy_ids
    )
    blockers: list[str] = []
    warnings: list[str] = []
    degradation = [False]
    allocations: list[InstanceAllocation] = []
    hccl_interfaces: dict[str, Mapping[str, str]] = {}
    if not snapshots:
        blockers.append("no reachable npu-smi inventory supplied")
    if spec.placement == "multi_node_roce":
        if len(snapshots) < 2:
            blockers.append("multi_node_roce requires at least two reachable hosts")
        if spec.die_count % max(len(snapshots), 1):
            blockers.append(
                f"{spec.die_count} dies do not split evenly across {len(snapshots)} hosts"
            )
        per_host = spec.die_count // max(len(snapshots), 1)
        for snapshot in snapshots:
            hccl_interfaces[snapshot.host] = {
                "HCCL_IF_IP": snapshot.hccl_if_ip or "",
                "HCCL_SOCKET_IFNAME": snapshot.hccl_socket_ifname or "",
            }
            if not snapshot.hccl_if_ip or not snapshot.hccl_socket_ifname:
                blockers.append(
                    f"{snapshot.host} has no HCCL_IF_IP/HCCL_SOCKET_IFNAME interface binding"
                )
            if spec.require_clean_shm and not snapshot.shm_clean:
                blockers.append(
                    f"{snapshot.host} has residual /dev/shm/hccl*: "
                    + ", ".join(snapshot.shm_hccl_entries)
                )
            if per_host < 1:
                continue
            selected = _explicit_dies(snapshot, requested, blockers) if requested else ()
            if len(selected) > per_host:
                selected = selected[:per_host]
            if len(selected) < per_host:
                selected = _whole_packages(snapshot, per_host)
            if len(selected) < per_host:
                selected = tuple(sorted(snapshot.idle_dies, key=lambda die: die.phy_id))[:per_host]
            if len(selected) < per_host:
                blockers.append(
                    f"{snapshot.host} has {len(selected)} idle dies, needs {per_host}"
                )
                continue
            if per_host % 2:
                warnings.append(
                    f"{snapshot.host} cross-package allocation {per_host} dies (odd package split)"
                )
                degradation[0] = True
            allocations.append(_allocation(snapshot, "worker", selected))
        return GangLeasePlan(
            spec=spec,
            allocations=tuple(allocations),
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            cross_package_degradation=degradation[0],
            lease_id=f"gang-{uuid.uuid4().hex[:16]}",
            sandbox_root=sandbox_root or f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex[:12]}",
            hccl_interfaces=hccl_interfaces,
        )
    if spec.placement == "same_npu_dual_die":
        candidates = [row for row in snapshots if row.dual_die_packages]
        soft_blockers: list[str] = []
        hard_blockers: list[str] = []
        for snapshot in candidates:
            local_blockers: list[str] = []
            local_warnings: list[str] = []
            local_degradation = [False]
            selected = _select_same_npu(
                spec, snapshot, requested, local_blockers, local_warnings, local_degradation
            )
            if local_degradation[0]:
                degradation[0] = True
                hard_blockers.extend(local_blockers)
            elif len(selected) == spec.die_count and not degradation[0]:
                warnings.extend(local_warnings)
                allocations.append(_allocation(snapshot, "worker", selected))
                break
            else:
                soft_blockers.extend(local_blockers)
        if not allocations:
            blockers.extend([*hard_blockers, *soft_blockers])
        if not candidates:
            blockers.append("same_npu_dual_die requires an A3 dual-die host")
    else:
        snapshot, selected = _select_single_host(spec, snapshots, requested, blockers)
        if snapshot is not None and len(selected) == spec.die_count:
            if spec.placement == "disaggregated_pd_pair":
                prefill_count = spec.prefill_die_count or spec.die_count // 2
                for role, group in (("prefill", selected[:prefill_count]),
                                    ("decode", selected[prefill_count:])):
                    if len(group) % 2 and snapshot.dual_die_packages:
                        degradation[0] = True
                        warnings.append(
                            f"{role} instance spans a partial NPU package: "
                            + ",".join(str(die.phy_id) for die in group)
                        )
                    allocations.append(
                        _allocation(
                            snapshot,
                            role,
                            group,
                            environment={"MINDIE_BENCH_PD_ROLE": role, "VLLM_PD_ROLE": role},
                        )
                    )
            else:
                allocations.append(_allocation(snapshot, "worker", selected))
    for snapshot in snapshots:
        hccl_interfaces[snapshot.host] = {
            "HCCL_IF_IP": snapshot.hccl_if_ip or "",
            "HCCL_SOCKET_IFNAME": snapshot.hccl_socket_ifname or "",
        }
        if (
            spec.require_clean_shm
            and not snapshot.shm_clean
            and any(row.host == snapshot.host for row in allocations)
        ):
            blockers.append(
                f"{snapshot.host} has residual /dev/shm/hccl*: "
                + ", ".join(snapshot.shm_hccl_entries)
            )
    return GangLeasePlan(
        spec=spec,
        allocations=tuple(allocations),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        cross_package_degradation=degradation[0],
        lease_id=f"gang-{uuid.uuid4().hex[:16]}",
        sandbox_root=sandbox_root or f"/tmp/mindie-npu-test-bench-{uuid.uuid4().hex[:12]}",
        hccl_interfaces=hccl_interfaces,
    )


# ------------------------------------------------------- production task specs
@dataclass(frozen=True)
class ProductionBenchmarkTask:
    root: Path
    task: Mapping[str, Any]
    topology: TopologySpec
    query: str
    conditions: Mapping[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.task["task"]["instance_id"])

    def plan(self, inventory: Any, **kwargs: Any) -> GangLeasePlan:
        return plan_topology_gang_lease(self.topology, inventory, **kwargs)


def load_production_benchmark_task(root: str | Path) -> ProductionBenchmarkTask:
    """Load a task definition before its oracle/verifier assets are authored.

    `TaskSpec.load` intentionally requires `verifier/test.sh` and a soundness
    receipt. Production topology definitions are staged first, so this loader
    validates the same `TaskSpec` contract but does not fabricate a verifier.
    """
    from mindie_bench.models import TaskSpec

    path = Path(root)
    raw = tomllib.loads((path / "task.toml").read_text())
    task = TaskSpec.model_validate(raw)
    environment = task.environment.get("l3") or next(iter(task.environment.values()))
    if environment.topology_scale is None or environment.placement is None:
        raise ValueError(f"{path} is missing topology_scale/placement")
    topology = TopologySpec(
        scale=environment.topology_scale,
        die_count=environment.die_count or environment.npu_count,
        placement=environment.placement,
        hosts=tuple(environment.hosts),
        visible_phy_ids=tuple(environment.visible_phy_ids),
        max_workspace_delta_bytes=environment.max_workspace_delta_bytes,
    )
    condition_rows = dict(task.mindie_knowledge.conditions)
    query = str(condition_rows.pop("query", "")) or " ".join(
        str(condition_rows.get(key, ""))
        for key in ("topology_scale", "placement", "model", "domain")
        if condition_rows.get(key)
    )
    instruction = path / "instruction.md"
    if not instruction.is_file() or not instruction.read_text().strip():
        raise ValueError(f"{path} is missing a non-empty instruction.md")
    return ProductionBenchmarkTask(
        root=path,
        task=raw,
        topology=topology,
        query=query,
        conditions=condition_rows,
    )


NpuSnapshot = NpuTopology
parse_npu_smi = parse_npu_smi_info
plan_gang_lease = plan_topology_gang_lease

__all__ = [
    "GangLeasePlan",
    "InstanceAllocation",
    "L4_03",
    "NpuDie",
    "NpuSnapshot",
    "NpuTopology",
    "ProductionBenchmarkTask",
    "SCHEMA_VERSION",
    "TopologySpec",
    "evaluate_distributed_invariance_and_sla",
    "load_production_benchmark_task",
    "parse_npu_smi",
    "parse_npu_smi_info",
    "parse_npu_smi_info_l",
    "plan_gang_lease",
    "plan_topology_gang_lease",
    "probe_npu_smi",
]
