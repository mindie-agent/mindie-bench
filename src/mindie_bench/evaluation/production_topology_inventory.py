"""Read-only Ascend A2/A3 `npu-smi` inventory parsing and live probing.

[Role]
Parse real A2 (`8 NPU x 1 Chip`) and A3 (`8 NPU x 2 Chips`) `npu-smi info` /
`npu-smi info -l` output into per-Die occupancy records, and provide one bounded
read-only SSH probe for Tier 2 topology evidence.

[Invariants]
- Busy means a process-table row, non-OK health, or residual HBM above the idle
  limit. A process row without a Chip column occupies every chip of that NPU.
- A3 `Phy-ID` is authoritative; A2 has no Phy-ID column and maps Phy-ID == NPU ID.
- `npu-smi info -l` chip counts must agree with parsed chip rows; mismatches stay
  visible and never turn an unknown topology into an allocatable one.

[Timeouts & Locks]
- Parsing is pure and unbounded only by input size. The live probe uses a caller
  timeout on SSH and acquires no device lock.

[Side-Effect Policy]
- The probe only runs read-only `npu-smi`, interface lookup and `/dev/shm/hccl*`
  listing. It never starts, stops, resets or writes an NPU.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

_CHIP_WITH_PHY = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+(?:\.\d+)?)\s+"
    r"(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_CHIP_NO_PHY = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+(?:\.\d+)?)\s+"
    r"(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_PARENT = re.compile(r"^\|\s*(\d+)\s+([A-Za-z0-9][^|]*?)\s*\|\s*([A-Za-z]+)\s*\|")
_PROCESS = re.compile(
    r"^\|\s*(\d+)(?:\s+(\d+))?\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|(?:\s*(\d+)\s*\|)?"
)


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class NpuDie:
    npu_id: int
    chip_id: int
    phy_id: int
    chip_name: str | None = None
    board_name: str | None = None
    health: str = "OK"
    bus_id: str | None = None
    hbm_used_mb: int = 0
    hbm_total_mb: int = 0
    aicore_pct: float = 0.0
    processes: tuple[tuple[int, str], ...] = ()
    busy: bool = False
    busy_reason: str | None = None

    @property
    def sibling_phy_id(self) -> int:
        return self.phy_id + 1 if self.phy_id % 2 == 0 else self.phy_id - 1

    @property
    def process_names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self.processes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "phy_id": self.phy_id,
            "chip_name": self.chip_name,
            "board_name": self.board_name,
            "health": self.health,
            "bus_id": self.bus_id,
            "hbm_used_mb": self.hbm_used_mb,
            "hbm_total_mb": self.hbm_total_mb,
            "aicore_pct": self.aicore_pct,
            "processes": [{"pid": pid, "name": name} for pid, name in self.processes],
            "busy": self.busy,
            "busy_reason": self.busy_reason,
        }


@dataclass(frozen=True)
class NpuTopology:
    host: str
    dies: tuple[NpuDie, ...]
    chip_counts: Mapping[int, int] = field(default_factory=dict)
    total_count: int | None = None
    reachable: bool = True
    error: str | None = None
    hccl_if_ip: str | None = None
    hccl_socket_ifname: str | None = None
    shm_hccl_entries: tuple[str, ...] = ()
    layout_mismatches: tuple[str, ...] = ()

    @property
    def die_count(self) -> int:
        return len(self.dies)

    @property
    def idle_dies(self) -> tuple[NpuDie, ...]:
        return tuple(die for die in self.dies if not die.busy)

    @property
    def busy_dies(self) -> tuple[NpuDie, ...]:
        return tuple(die for die in self.dies if die.busy)

    @property
    def idle_phy_ids(self) -> tuple[int, ...]:
        return tuple(die.phy_id for die in self.idle_dies)

    @property
    def busy_phy_ids(self) -> tuple[int, ...]:
        return tuple(die.phy_id for die in self.busy_dies)

    @property
    def dual_die_packages(self) -> tuple[int, ...]:
        by_npu: dict[int, set[int]] = {}
        for die in self.dies:
            by_npu.setdefault(die.npu_id, set()).add(die.chip_id)
        return tuple(sorted(npu for npu, chips in by_npu.items() if chips >= {0, 1}))

    @property
    def shm_clean(self) -> bool:
        return not self.shm_hccl_entries

    def die_by_phy(self, phy_id: int) -> NpuDie | None:
        return next((die for die in self.dies if die.phy_id == phy_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "reachable": self.reachable,
            "error": self.error,
            "die_count": self.die_count,
            "chip_counts": dict(self.chip_counts),
            "total_count": self.total_count,
            "hccl_if_ip": self.hccl_if_ip,
            "hccl_socket_ifname": self.hccl_socket_ifname,
            "shm_hccl_entries": list(self.shm_hccl_entries),
            "layout_mismatches": list(self.layout_mismatches),
            "dies": [die.to_dict() for die in self.dies],
        }


def parse_npu_smi_info_l(text: str) -> tuple[dict[int, int], int | None]:
    """Parse `npu-smi info -l`; the same output distinguishes A2 from A3."""
    counts: dict[int, int] = {}
    total = None
    current: int | None = None
    for line in text.splitlines():
        total_match = re.search(r"Total Count\s*:\s*(\d+)", line)
        if total_match:
            total = int(total_match.group(1))
            continue
        npu_match = re.search(r"NPU ID\s*:\s*(\d+)", line)
        if npu_match:
            current = int(npu_match.group(1))
            continue
        chip_match = re.search(r"Chip Count\s*:\s*(\d+)", line)
        if chip_match and current is not None:
            counts[current] = int(chip_match.group(1))
    return counts, total


def parse_npu_smi_info(
    text: str,
    *,
    info_l_text: str | None = None,
    host: str = "local",
    reachable: bool = True,
    error: str | None = None,
    hccl_if_ip: str | None = None,
    hccl_socket_ifname: str | None = None,
    shm_hccl_entries: Sequence[str] = (),
    idle_hbm_limit_mb: int = 4096,
) -> NpuTopology:
    """Parse real A2/A3 `npu-smi info` tables without trusting filenames."""
    chip_counts: dict[int, int] = {}
    total = None
    if info_l_text:
        chip_counts, total = parse_npu_smi_info_l(info_l_text)
    has_phy = "Phy-ID" in text
    chips: list[dict[str, Any]] = []
    parent_name: str | None = None
    parent_health = "OK"
    current_npu = 0
    saw_parent = False
    for line in text.splitlines():
        chip = _CHIP_WITH_PHY.match(line)
        if chip:
            chip_id, phy_id, bus, aicore, mem_used, mem_total, hbm_used, hbm_total = chip.groups()
        else:
            chip = _CHIP_NO_PHY.match(line)
            if chip:
                chip_id, bus, aicore, mem_used, mem_total, hbm_used, hbm_total = chip.groups()
                phy_id = str(current_npu)
            else:
                chip = None
        if chip:
            # The first chip column is Chip-ID on both A2 and A3; the NPU package
            # is the most recent parent header. On A2 the Phy-ID equals NPU-ID.
            npu = current_npu if saw_parent else int(chip_id)
            chips.append(
                {
                    "npu_id": npu,
                    "chip_id": int(chip_id),
                    "phy_id": int(phy_id) if has_phy else npu,
                    "bus_id": bus,
                    "aicore_pct": float(aicore),
                    "hbm_used_mb": int(hbm_used),
                    "hbm_total_mb": int(hbm_total),
                    "board_name": parent_name,
                    "health": parent_health,
                }
            )
            continue
        parent = _PARENT.match(line)
        if parent:
            saw_parent = True
            current_npu = int(parent.group(1))
            parent_name, parent_health = parent.group(2).strip(), parent.group(3)
    process_header = next(
        (
            index
            for index, line in enumerate(text.splitlines())
            if "Process id" in line and "Process name" in line
        ),
        None,
    )
    processes: dict[tuple[int, int | None], list[tuple[int, str]]] = {}
    if process_header is not None:
        for line in text.splitlines()[process_header + 1 :]:
            row = _PROCESS.match(line)
            if not row:
                continue
            npu_text, chip_text, pid_text, name_text, _memory = row.groups()
            key = (int(npu_text), int(chip_text) if chip_text is not None else None)
            processes.setdefault(key, []).append((int(pid_text), name_text.strip()))
    dies = []
    for row in chips:
        key = (row["npu_id"], row["chip_id"])
        fallback = processes.get((row["npu_id"], None), [])
        rows = tuple(_unique([*processes.get(key, []), *fallback]))
        reason = None
        if row["health"] != "OK":
            reason = f"health:{row['health']}"
        elif rows:
            reason = "process:" + ",".join(sorted({name for _, name in rows}))
        elif process_header is None and row["hbm_used_mb"] > idle_hbm_limit_mb:
            # Without a process table, residual HBM is the only occupancy signal.
            # With one, an explicit no-process Die stays idle even if HBM has not
            # been released yet; the allocator still reports its HBM in the snapshot.
            reason = f"hbm:{row['hbm_used_mb']}MB"
        dies.append(NpuDie(**row, processes=rows, busy=reason is not None, busy_reason=reason))
    by_npu: dict[int, int] = {}
    for die in dies:
        by_npu[die.npu_id] = by_npu.get(die.npu_id, 0) + 1
    expected = chip_counts or by_npu
    mismatches = tuple(
        f"NPU {npu}: info reports {expected.get(npu, 0)} chips, parsed {by_npu.get(npu, 0)}"
        for npu in sorted(set(expected) | set(by_npu))
        if expected.get(npu, 0) != by_npu.get(npu, 0)
    )
    return NpuTopology(
        host=host,
        dies=tuple(dies),
        chip_counts=expected,
        total_count=total,
        reachable=reachable,
        error=error,
        hccl_if_ip=hccl_if_ip or None,
        hccl_socket_ifname=hccl_socket_ifname or None,
        shm_hccl_entries=tuple(shm_hccl_entries),
        layout_mismatches=mismatches,
    )


def _probe_script() -> str:
    return r"""
printf '@@@INFO@@@\n'; npu-smi info 2>/dev/null || true
printf '@@@LAYOUT@@@\n'; npu-smi info -l 2>/dev/null || true
printf '@@@HCCL_IF_IP@@@\n'; printf '%s\n' "${HCCL_IF_IP:-}"
printf '@@@HCCL_SOCKET_IFNAME@@@\n'; printf '%s\n' "${HCCL_SOCKET_IFNAME:-}"
printf '@@@SHM@@@\n'; find /dev/shm -maxdepth 1 -name 'hccl*' -printf '%f\n' 2>/dev/null || true
printf '@@@ROUTE@@@\n'; ip -o -4 route show default 2>/dev/null | head -1 || true
printf '@@@ADDR@@@\n'; ip -o -4 addr show 2>/dev/null || true
"""


def _section(text: str, name: str) -> str:
    marker = f"@@@{name}@@@"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("@@@", start)
    return text[start:end if end >= 0 else None].strip()


def probe_npu_smi(host: str, *, timeout: float = 8.0, ssh: str = "ssh") -> NpuTopology:
    """Read-only SSH probe. Unreachable hosts return a non-allocatable snapshot."""
    command = [
        ssh, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        host, shlex.join(["bash", "-lc", _probe_script()]),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NpuTopology(host=host, dies=(), reachable=False, error=str(exc))
    if completed.returncode != 0:
        return NpuTopology(
            host=host,
            dies=(),
            reachable=False,
            error=(completed.stderr or completed.stdout).strip()[-400:],
        )
    output = completed.stdout
    if_ip = _section(output, "HCCL_IF_IP") or None
    if_name = _section(output, "HCCL_SOCKET_IFNAME") or None
    route = _section(output, "ROUTE")
    addresses = _section(output, "ADDR")
    if not if_ip or not if_name:
        iface = re.search(r"\bdev\s+(\S+)", route)
        iface_name = iface.group(1) if iface else None
        if iface_name:
            for line in addresses.splitlines():
                if f" {iface_name} " in f" {line} ":
                    address = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", line)
                    if address:
                        if_ip = if_ip or address.group(1)
                        if_name = if_name or iface_name
                        break
    snapshot = parse_npu_smi_info(
        _section(output, "INFO"),
        info_l_text=_section(output, "LAYOUT"),
        host=host,
        hccl_if_ip=if_ip,
        hccl_socket_ifname=if_name,
        shm_hccl_entries=tuple(
            line.strip() for line in _section(output, "SHM").splitlines() if line.strip()
        ),
    )
    if not snapshot.dies:
        return replace(snapshot, reachable=False, error="npu-smi returned no chip rows")
    return snapshot


__all__ = [
    "NpuDie",
    "NpuTopology",
    "parse_npu_smi_info",
    "parse_npu_smi_info_l",
    "probe_npu_smi",
]
