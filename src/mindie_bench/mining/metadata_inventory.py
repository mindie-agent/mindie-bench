"""Prepare cross-source lineage candidates without opening source material.

[Role] Project the authorized inventories onto metadata, rank session
bundles and feed explicit provenance links into family-split preparation.
[Invariants] Selection/topic labels are hints, not resolved problem lineage.
Only listed inventory files are opened; transcript, diff, source and redacted
body fields are never used and referenced source paths are never followed.
Public source pilots contain one upstream source family, all Seed, and do not
assert independent Heldout tasks or global cross-source contamination clearance.
[Timeouts & Locks] Bounded local file reads and in-memory grouping; no workers,
network calls, subprocesses or locks.
[Side-Effect Policy] Writes derived metadata reports and an optional source-only
pilot manifest below the explicitly selected coldstart output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from mindie_bench.evaluation.split_manifest import SourceAsset, admit_materials, build_manifest
from mindie_bench.io import write_json

INPUTS = {
    "fleet": "collector/outputs/fleet-inventory.json",
    "offline": "collector/outputs/offline-inventory.json",
    "map03": "map03/model-0day-branch-diff-map.jsonl",
    "map03_summary": "map03/scan-summary.json",
    "map04": "map04/deshelling-map.jsonl",
    "fetcher": "fetcher/gitcode-ascend-cann-map.jsonl",
}
PRIVACY_VALIDATION = "collector/outputs/redacted-offline-privacy-validation.json"
DOMAIN = re.compile(r"(?<![a-z0-9])(npu|ascend|cann|mindie|triton|910[bc])(?![a-z0-9])", re.I)
TECHNICAL = re.compile(
    r"benchmark|ablation|performance|kernel|operator|recompil|quant|repro|debug|调试|优化", re.I
)
COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def repository_identity(value: str | None) -> str | None:
    """Normalize an explicit remote spelling without querying or resolving it."""
    if not value:
        return None
    value = value.removeprefix("git+").rstrip("/")
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        return f"{host.lower()}/{path.removesuffix('.git')}"
    parsed = urlsplit(value)
    if parsed.scheme in {"https", "http", "ssh", "git"} and parsed.hostname:
        return f"{parsed.hostname.lower()}/{parsed.path.strip('/').removesuffix('.git')}"
    if value.startswith("gitcode:"):
        return "gitcode.com/" + value[len("gitcode:") :].split("@", 1)[0]
    return "local:" + str(PurePosixPath(value))


def _record(collection: str, source_id: str, locator: str, **metadata) -> dict:
    namespace = "collector" if collection in {"fleet", "offline"} else collection
    return {
        "asset_id": "metadata-" + _hash([namespace, source_id, locator])[:24],
        "source_namespace": namespace,
        "source_collection": collection,
        "source_id": source_id,
        "source_locator": locator,
        "session_id": None,
        "sha256": None,
        "hash_lineage_scope": None,
        "size": None,
        "repo_identity": None,
        "revision": None,
        "branch": None,
        "declared_family": None,
        "related_source_ids": [],
        "split": "unassigned",
        "public_catalog": False,
        "timestamp": None,
        **metadata,
    }


def project_inventory(collection: str, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        machine, root, relpath = row["machine"], row["root_label"], row["relpath"]
        locator = f"inventory://{machine}/{root}/{relpath}"
        session = row.get("session_id")
        suffix = PurePosixPath(row["filename"]).suffix.lower()
        cache = any(
            part in {"plugins", "node_modules", ".tmp", "cache"} for part in PurePosixPath(relpath).parts
        )
        if session and not cache:
            role = (
                "native_transcript_candidate"
                if suffix in {".jsonl", ".zstd"}
                else "session_derived_artifact"
            )
        elif not cache and suffix == ".jsonl" and root == "mindie-workspaces":
            role = "unbound_runtime_log"
        else:
            role = "supporting_file"
        complete_transcript = (
            role == "native_transcript_candidate"
            and "chunks" not in PurePosixPath(relpath).parts
            and (
                row["filename"]
                in {"transcript.jsonl", "transcript_full.jsonl", "session.v4.jsonl.zstd", "wire.jsonl"}
                or row["filename"].startswith("rollout-")
                or PurePosixPath(row["filename"]).stem == session
            )
        )
        result.append(
            _record(
                collection,
                row["stable_source_id"],
                locator,
                record_type="supporting_file" if role == "supporting_file" else "session_artifact",
                artifact_role=role,
                machine=machine,
                root_label=root,
                relpath=relpath,
                filename=row["filename"],
                session_id=session,
                sha256=row.get("sha256"),
                hash_lineage_scope="complete_session_transcript" if complete_transcript else None,
                size=row.get("size"),
                timestamp=row.get("mtime"),
                timestamp_basis="source_file_mtime",
                selection_state=row.get("selection_state"),
                selection_evidence=row.get("selection_evidence", []),
                metadata_npu=row.get("metadata_npu"),
                metadata_found=row.get("metadata_found"),
                split=row.get("split", "unassigned"),
                upstream_collect=row.get("collect"),
                upstream_collect_reason=row.get("collect_reason"),
            )
        )
    return result


def project_branches(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        repo = repository_identity(row.get("repo_common_dir") or row.get("repo"))
        revision = row.get("commit_sha")
        path = row.get("path") or row.get("repo_input_path")
        locator = f"branch-metadata:{path}@{revision or 'unresolved'}"
        result.append(
            _record(
                "map03",
                row["source_id"],
                locator,
                record_type="branch",
                repo_identity=repo,
                revision=revision,
                branch=row.get("branch"),
                declared_family=row.get("task_family"),
                related_source_ids=row.get("related_source_ids", []),
                timestamp=row.get("head_commit_time") or row.get("scanned_at_utc"),
                timestamp_basis="commit_time" if row.get("head_commit_time") else "metadata_observed_at",
                repo_aliases=[value for value in (row.get("repo"), path) if value],
                family_candidates=row.get("task_family_candidates", []),
                family_confidence=row.get("task_family_confidence"),
                family_uncertain=row.get("task_family_uncertain"),
                branch_diff_available_nonempty=row.get("branch_diff_available_nonempty"),
                source_status=row.get("status"),
                split=row.get("split", "unassigned"),
            )
        )
    return result


def project_public(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        full_name = row.get("full_name")
        repo = "gitcode.com/" + full_name if full_name else repository_identity(row.get("url"))
        revision = row.get("head_commit")
        result.append(
            _record(
                "fetcher",
                row["source_id"],
                row["source_locator"],
                record_type="source_repository",
                repo_identity=repo,
                revision=revision,
                branch=row.get("branch") or row.get("default_branch"),
                public_catalog=bool(
                    repo and row.get("source") == "gitcode" and row.get("org") in {"Ascend", "cann"}
                ),
                timestamp=row.get("metadata_fetched_at") or row.get("fetched_at"),
                timestamp_basis="metadata_observed_at",
                source_status=row.get("fetch_status"),
                split=row.get("split", "unassigned"),
                domain_tags=row.get("domain_tags", []),
                upstream_content_read=row.get("content_read"),
                metadata_error_count=len(row.get("errors", [])),
            )
        )
    return result


def project_source_files(
    rows: list[dict], public_repositories: set[str], observed_at: str
) -> list[dict]:
    result = []
    for row in rows:
        repo = repository_identity(row.get("repo"))
        result.append(
            _record(
                "map04",
                row["source_id"],
                row["source_locator"],
                record_type="source_file",
                repo_identity=repo,
                revision=row.get("commit_sha"),
                declared_family=row.get("task_family"),
                public_catalog=repo in public_repositories,
                timestamp=observed_at,
                timestamp_basis="map_metadata_file_mtime_not_commit_time",
                path=row.get("path"),
                source_status=row.get("type"),
                split=row.get("split", "unassigned"),
                upstream_evidence_kind=row.get("evidence_kind"),
            )
        )
    return result


def load_metadata(root: Path) -> tuple[list[dict], dict, list[dict]]:
    """Read only the explicit input inventory paths, never their source locators."""
    inputs, provenance, errors = {}, {}, []
    for name, relative in INPUTS.items():
        path = root / relative
        raw = path.read_bytes()
        inputs[name] = (
            [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(raw)
        )
        provenance[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "metadata_file_mtime": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
    records = []
    for name in ("fleet", "offline"):
        records.extend(project_inventory(name, inputs[name]["files"]))
        errors.extend(
            {
                "kind": "upstream_inventory_error",
                "source_collection": name,
                "metadata": {
                    key: row[key] for key in ("machine", "root_label", "kind", "relpath") if key in row
                },
            }
            for row in inputs[name]["errors"]
        )
    records.extend(project_branches(inputs["map03"]))
    public = project_public(inputs["fetcher"])
    records.extend(public)
    known_public = {row["repo_identity"] for row in public if row["public_catalog"]}
    records.extend(
        project_source_files(inputs["map04"], known_public, provenance["map04"]["metadata_file_mtime"])
    )
    # A producer's opaque error text is not needed; preserve explicit counts and
    # categories without copying commands, notes or any content-bearing field.
    provenance["map03_summary"]["counts"] = inputs["map03_summary"]["counts"]
    validation_path = root / PRIVACY_VALIDATION
    if validation_path.is_file():
        raw = validation_path.read_bytes()
        validation = json.loads(raw)
        provenance["offline_privacy_validation"] = {
            "path": str(validation_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": {
                key: validation.get(key)
                for key in (
                    "scanner",
                    "scope",
                    "exit_code",
                    "finding_count",
                    "elapsed_seconds",
                    "scan_complete",
                )
            },
        }
    unique = {}
    for record in records:
        previous = unique.get(record["asset_id"])
        if previous is not None and previous != record:
            errors.append(
                {"kind": "duplicate_identity_conflicting_metadata", "asset_id": record["asset_id"]}
            )
            record = {**record, "asset_id": record["asset_id"] + "-" + _hash(record)[:12]}
        unique[record["asset_id"]] = record
    return list(unique.values()), provenance, errors


def attach_admission_status(groups: list[dict], provenance: dict) -> None:
    """An offline privacy failure follows the entire linked session/copy group."""
    validation = provenance.get("offline_privacy_validation", {}).get("metadata", {})
    privacy_passed = (
        validation.get("exit_code") == 0
        and validation.get("finding_count") == 0
        and validation.get("scan_complete") is True
    )
    for group in groups:
        offline = "offline" in group["source_collections"]
        group["body_admission_status"] = (
            "blocked_privacy_validation"
            if offline and not privacy_passed
            else "metadata_only_not_admitted"
        )
        if offline:
            group["offline_privacy_validation"] = validation or {"status": "not_available"}


def repository_hints(records: list[dict]) -> dict[str, list[dict]]:
    """Explicit repository basename in a path suggests review, never a union."""
    aliases = defaultdict(list)
    for row in records:
        if row["record_type"] != "branch":
            continue
        for value in row.get("repo_aliases", []):
            alias = PurePosixPath(value).name
            if len(alias) >= 8:
                aliases[alias.casefold()].append(row)
    result = {}
    for row in records:
        if row["record_type"] != "session_artifact":
            continue
        path = row["relpath"].casefold()
        hints = {}
        for alias, branches in aliases.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", path):
                for branch in branches:
                    hints[branch["asset_id"]] = {
                        "asset_id": branch["asset_id"],
                        "source_id": branch["source_id"],
                        "declared_family": branch["declared_family"],
                        "repo_identity": branch["repo_identity"],
                        "reason": "explicit_repository_basename_in_inventory_path_candidate_only",
                    }
        if hints:
            result[row["asset_id"]] = list(hints.values())
    return result


def metadata_priority(row: dict, hints: list[dict]) -> tuple[int, list[str]]:
    if row["record_type"] != "session_artifact":
        return 0, []
    score, reasons = 0, []
    if row["source_collection"] == "offline":
        score += 100
        reasons.append("explicitly_listed_offline_session")
    if row.get("metadata_npu") is True:
        score += 50
        reasons.append("upstream_session_metadata_npu_signal")
    tags = sorted({value.casefold() for value in DOMAIN.findall(row["relpath"])})
    if tags:
        score += 30
        reasons.append("bounded_path_domain_tags:" + ",".join(tags))
    if hints:
        score += 15
        reasons.append("repository_path_candidate_match")
    if score and TECHNICAL.search(row["filename"]):
        score += 10
        reasons.append("technical_artifact_filename")
    return score, reasons


def shortlist(
    records: list[dict], groups: list[dict], hints: dict, limit: int
) -> tuple[list[dict], int]:
    by_id = {row["asset_id"]: row for row in records}
    candidates = []
    for group in groups:
        scored = [
            (row, *metadata_priority(row, hints.get(row["asset_id"], [])))
            for row in (by_id[asset] for asset in group["asset_ids"])
        ]
        selected = [(row, score, reasons) for row, score, reasons in scored if score]
        if not selected:
            continue
        selected.sort(key=lambda item: (-item[1], item[0]["asset_id"]))
        candidates.append(
            {
                "group_id": group["group_id"],
                "priority_score": selected[0][1],
                "ranking_scope": "metadata_review_priority_only_not_verified_technical_value",
                "lineage_resolved": False,
                "split": "unassigned",
                "session_ids": group["session_ids"],
                "asset_ids": group["asset_ids"],
                "source_collections": group["source_collections"],
                "priority_reasons": sorted({reason for _, _, reasons in selected for reason in reasons}),
                "candidate_family_labels": sorted(
                    {
                        hint["declared_family"]
                        for row, _, _ in selected
                        for hint in hints.get(row["asset_id"], [])
                        if hint["declared_family"]
                    }
                ),
                "artifacts": [
                    {
                        key: row.get(key)
                        for key in (
                            "asset_id",
                            "source_locator",
                            "session_id",
                            "filename",
                            "artifact_role",
                            "size",
                            "sha256",
                            "timestamp",
                        )
                    }
                    for row, _, _ in selected
                ],
                "unresolved_reasons": group["unresolved_reasons"],
                "body_admission_status": group["body_admission_status"],
            }
        )
    candidates.sort(key=lambda row: (-row["priority_score"], row["group_id"]))
    for rank, row in enumerate(candidates[:limit], 1):
        row["rank"] = rank
    return candidates[:limit], len(candidates)


def freeze_public_pilot(records: list[dict], groups: list[dict], repo: str, output: Path) -> dict:
    selected_groups = [
        group for group in groups if group["repo_identities"] == [repo] and group["lineage_resolved"]
    ]
    if len(selected_groups) != 1:
        return {
            "status": "not_frozen",
            "repo_identity": repo,
            "reason": "exactly_one_resolved_public_source_family_required",
        }
    group = selected_groups[0]
    by_id = {row["asset_id"]: row for row in records}
    family = "public-source:" + repo
    assets = []
    for identity in group["asset_ids"]:
        row = by_id[identity]
        if not row["revision"] or not COMMIT.fullmatch(row["revision"]):
            return {"status": "not_frozen", "repo_identity": repo, "reason": "exact_revision_missing"}
        assets.append(
            SourceAsset(
                asset_id=identity,
                task_family=family,
                source_id="source-artifact:" + row["source_locator"],
                source_locator=row["source_locator"],
                source_kind="commit",
                timestamp=row["timestamp"],
                content_sha256=row["sha256"],
                lineage_resolved=True,
            )
        )
    manifest = build_manifest(assets, {family: "seed"}, seed_cutoff=datetime.now(timezone.utc))
    admission = admit_materials(manifest, [asset.asset_id for asset in assets], "seed_distillation")
    write_json(output / "pilot-public-source-manifest.json", manifest.model_dump(mode="json"))
    result = {
        "status": "frozen_source_only",
        "repo_identity": repo,
        "group_id": group["group_id"],
        "manifest_sha256": manifest.manifest_sha256,
        "assets": len(assets),
        "revisions": group["revisions"],
        "admission": admission,
        "scope": "one explicitly identified public upstream source family, all Seed; no Heldout or task-independence claim",
        "downstream_evaluation_ready": False,
        "body_distillation_started": False,
        "timestamp_scope": "metadata observation or map-file mtime, never inferred commit author date",
        "remaining_gate": "resolve cross-session/branch problem lineage and freeze the complete assignment before any heldout evaluation",
    }
    write_json(output / "pilot-public-source-admission.json", result)
    return result


def _jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))


def prepare(root: Path, *, shortlist_limit: int = 200, pilot_repo: str | None = None) -> dict:
    from mindie_bench.mining.lineage_links import build_lineage_groups

    records, provenance, errors = load_metadata(root)
    graph = build_lineage_groups(records)
    hints = repository_hints(records)
    group_by_asset = {asset: group for group in graph["groups"] for asset in group["asset_ids"]}
    attach_admission_status(graph["groups"], provenance)
    for group in graph["groups"]:
        group["repository_path_family_candidates"] = sorted(
            {
                hint["declared_family"]
                for asset in group["asset_ids"]
                for hint in hints.get(asset, [])
                if hint["declared_family"]
            }
        )
    path_edges = {}
    for asset, candidates in hints.items():
        for hint in candidates:
            group_id = group_by_asset[asset]["group_id"]
            key = group_id, hint["asset_id"]
            edge = path_edges.setdefault(
                key,
                {
                    "left": asset,
                    "right": hint["asset_id"],
                    "reason": hint["reason"],
                    "evidence": {
                        "lineage_group_id": group_id,
                        "matching_inventory_assets": 0,
                        "repo_identity": hint["repo_identity"],
                        "declared_family": hint["declared_family"],
                        "claim": "path candidate only; all individual matches remain in union inventory; no same-problem evidence or union",
                    },
                },
            )
            edge["evidence"]["matching_inventory_assets"] += 1
    graph["candidate_edges"].extend(path_edges.values())
    union = [
        {
            **row,
            "lineage_group_id": group_by_asset[row["asset_id"]]["group_id"],
            "lineage_resolved": group_by_asset[row["asset_id"]]["lineage_resolved"],
            "body_admission_status": group_by_asset[row["asset_id"]]["body_admission_status"],
            "repository_path_candidate_asset_ids": [
                hint["asset_id"] for hint in hints.get(row["asset_id"], [])
            ],
        }
        for row in records
    ]
    short, total_candidates = shortlist(records, graph["groups"], hints, shortlist_limit)
    output, map01 = root / "joint-split", root / "map01"
    output.mkdir(parents=True, exist_ok=True)
    map01.mkdir(parents=True, exist_ok=True)
    _jsonl(output / "union-inventory.jsonl", union)
    _jsonl(output / "family-candidate-inventory.jsonl", graph["groups"])
    _jsonl(output / "strong-lineage-edges.jsonl", graph["strong_edges"])
    _jsonl(output / "candidate-lineage-edges.jsonl", graph["candidate_edges"])
    _jsonl(output / "unresolved-references.jsonl", graph["unresolved_references"])
    _jsonl(
        output / "unresolved-lineage.jsonl",
        [group for group in graph["groups"] if not group["lineage_resolved"]],
    )
    _jsonl(output / "conflicts.jsonl", [*errors, *graph["conflicts"]])
    _jsonl(map01 / "transcript-high-value-shortlist.jsonl", short)
    pilot = (
        freeze_public_pilot(records, graph["groups"], pilot_repo, output)
        if pilot_repo
        else {"status": "not_requested"}
    )
    summary = {
        "schema_version": "mindie-metadata-lineage-preparation/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": provenance,
        "records": len(records),
        "records_by_collection": dict(Counter(row["source_collection"] for row in records)),
        "family_candidates": len(graph["groups"]),
        "resolved_public_source_families": sum(group["lineage_resolved"] for group in graph["groups"]),
        "unresolved_families": sum(not group["lineage_resolved"] for group in graph["groups"]),
        "strong_edges": len(graph["strong_edges"]),
        "candidate_edges": len(graph["candidate_edges"]),
        "unresolved_references": len(graph["unresolved_references"]),
        "body_admission_status_groups": dict(
            Counter(group["body_admission_status"] for group in graph["groups"])
        ),
        "conflicts_by_kind": dict(Counter(row["kind"] for row in [*errors, *graph["conflicts"]])),
        "shortlist_candidates": total_candidates,
        "shortlist_written": len(short),
        "shortlist_omitted_by_limit": total_candidates - len(short),
        "upstream_path_only_supporting_files_excluded_from_shortlist": sum(
            row["record_type"] == "supporting_file"
            and "path-npu-keyword" in row.get("selection_evidence", [])
            for row in records
        ),
        "public_source_pilot": pilot,
        "referenced_source_files_opened": 0,
        "body_fields_used": False,
        "distillation_started": False,
        "global_split_status": "unresolved_metadata_only",
        "next_step": "review unresolved family provenance using authorized metadata; assign linked session/branch/source units together before body consumers; dispatch separate contexts with fixed admission purpose",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coldstart-root", type=Path, required=True)
    parser.add_argument("--shortlist-limit", type=int, default=200)
    parser.add_argument(
        "--pilot-public-source",
        help="exact canonical catalog repo, e.g. gitcode.com/Ascend/agent-skills",
    )
    args = parser.parse_args(argv)
    if args.shortlist_limit < 1:
        parser.error("--shortlist-limit must be positive")
    summary = prepare(
        args.coldstart_root, shortlist_limit=args.shortlist_limit, pilot_repo=args.pilot_public_source
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "records",
                    "family_candidates",
                    "resolved_public_source_families",
                    "unresolved_families",
                    "shortlist_written",
                    "conflicts_by_kind",
                    "global_split_status",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
