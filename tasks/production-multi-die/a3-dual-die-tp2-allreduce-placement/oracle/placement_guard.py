"""A3 dual-Die TP placement guard (gold reference).

[Role]
Reject a TP group whose visible Dies straddle two A3 NPU packages before an
HCCL/MC2 communicator is created. A3 exposes eight packages with two sibling
Dies each; Phy-ID `2k` and `2k+1` share one package. The production entrypoint
is `validate_rank_tp2_placement`, called from `init_ascend_model_parallel` only
for single-node A3 TP=2.

[Invariants]
- Phy-IDs are nonnegative, unique integers. A TP=2 rank slot must resolve to
  exactly two sibling Dies; `CrossPackagePlacementError` names the straddling
  case. A rank outside the visible slot range is an explicit error, never a
  silent pass.
- `ASCEND_RT_VISIBLE_DEVICES` is parsed without silently accepting empty,
  duplicated or non-integer entries. No environment variable produces a
  topology or measurement on its own.
- The returned mapping is derived from the caller's real Phy-ID list; it never
  invents a package, latency or success value.

[Timeouts & Locks]
- Pure in-memory validation; no waits, subprocesses or device locks.

[Side-Effect Policy]
- Reads the visibility environment only; never writes a file or touches an NPU.
"""

from __future__ import annotations

import os
from collections.abc import Sequence


class CrossPackagePlacementError(RuntimeError):
    """TP ranks were placed on Dies from more than one A3 NPU package."""


def package_of_phy_id(phy_id: int) -> int:
    """Return the A3 package index for one Phy-ID; siblings are 2k and 2k+1."""
    if isinstance(phy_id, bool) or not isinstance(phy_id, int) or phy_id < 0:
        raise ValueError(f"Phy-ID must be a nonnegative integer, got {phy_id!r}")
    return phy_id // 2


def _parse_visible(value: str) -> list[int]:
    if not value or not value.strip():
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES is empty")
    parsed: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"invalid Phy-ID token {token!r}")
        parsed.append(int(token))
    if not parsed:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES contains no Phy-ID")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate Phy-IDs in {value!r}")
    return parsed


def visible_phy_ids_from_env(visible: str | None = None) -> list[int] | None:
    """Return the real visible Phy-ID list, or None when the launcher did not set it."""
    raw = visible if visible is not None else os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return _parse_visible(raw)


def validate_tp_placement(device_phy_ids: Sequence[int]) -> dict[str, object]:
    """Validate one TP group and return its real package/sibling mapping."""
    ids = list(device_phy_ids)
    if not ids:
        raise ValueError("TP placement requires at least one Phy-ID")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
        raise ValueError(f"TP Phy-IDs must be nonnegative integers, got {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"TP Phy-IDs must be unique, got {ids!r}")
    by_package: dict[int, list[int]] = {}
    for phy_id in ids:
        by_package.setdefault(package_of_phy_id(phy_id), []).append(phy_id)
    for package, siblings in by_package.items():
        if len(siblings) > 2:
            raise ValueError(
                f"package {package} exposes {len(siblings)} TP Dies; A3 packages have two"
            )
    packages = tuple(sorted(by_package))
    if len(packages) != 1:
        raise CrossPackagePlacementError(
            "cross_package_degradation: Phy-IDs "
            f"{sorted(ids)} span A3 packages {list(packages)}; "
            "same-package TP requires sibling Dies 2k and 2k+1"
        )
    sibling_pairs = [
        sorted(siblings) for _, siblings in sorted(by_package.items()) if len(siblings) == 2
    ]
    return {
        "visible_phy_ids": sorted(ids),
        "device_phy_ids": sorted(ids),
        "packages": list(packages),
        "sibling_pairs": sibling_pairs,
    }


def validate_rank_tp2_placement(rank: int, visible_phy_ids: Sequence[int]) -> dict[str, object]:
    """Validate the TP=2 device pair selected by one global single-node rank.

    vLLM's rank grid puts TP on the innermost dimension, so rank N and N^1 are
    the two TP=2 peers for N even. The visible list is the real launcher list
    (for example `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`); this function slices only
    that rank's two Dies instead of rejecting a larger DP/EP visibility list.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a nonnegative integer, got {rank!r}")
    ids = list(visible_phy_ids)
    if not ids:
        raise ValueError("TP=2 placement requires a nonempty visible Phy-ID list")
    slot_base = rank - (rank % 2)
    if slot_base + 1 >= len(ids):
        raise ValueError(
            f"TP=2 rank {rank} needs visible slots {slot_base} and {slot_base + 1}, "
            f"but only {len(ids)} Phy-IDs are visible: {ids}"
        )
    pair = [ids[slot_base], ids[slot_base + 1]]
    result = validate_tp_placement(pair)
    result.update(rank=rank, tp2_slot_base=slot_base)
    return result


def enforce_visible_tp_placement(visible: str | None = None) -> dict[str, object]:
    """Validate the process visibility list before any MC2/HCCL group is built."""
    parsed = visible_phy_ids_from_env(visible)
    if parsed is None:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES is not set")
    return validate_tp_placement(parsed)
