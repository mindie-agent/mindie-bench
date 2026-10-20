from __future__ import annotations

from mindie_bench.mining.lineage_links import build_lineage_groups
from mindie_bench.mining.metadata_inventory import (
    attach_admission_status,
    metadata_priority,
    project_branches,
    project_inventory,
    project_public,
    project_source_files,
    repository_hints,
)


def inventory_row(**updates):
    return {
        "machine": "machine-a",
        "root_label": ".codex",
        "relpath": "sessions/rollout-session-a.jsonl",
        "filename": "rollout-session-a.jsonl",
        "stable_source_id": "sha256:" + "a" * 64,
        "session_id": "session-a",
        "size": 100,
        "sha256": "a" * 64,
        "mtime": "2026-10-19T12:00:00Z",
        "selection_state": "selected",
        "selection_evidence": ["path-npu-keyword"],
        "metadata_npu": None,
        "metadata_found": False,
        "split": "unassigned",
        **updates,
    }


def test_input_substring_and_plugin_cache_are_not_ranked_as_npu_sessions():
    rows = [
        inventory_row(relpath="sessions/input-profile/rollout.jsonl"),
        inventory_row(relpath=".tmp/plugins/npu/SKILL.md", filename="SKILL.md"),
        inventory_row(relpath="sessions/npu-profile/rollout.jsonl"),
    ]
    projected = project_inventory("fleet", rows)
    assert metadata_priority(projected[0], [])[0] == 0
    assert projected[1]["artifact_role"] == "supporting_file"
    assert metadata_priority(projected[1], [])[0] == 0
    assert metadata_priority(projected[2], [])[0] > 0
    assert projected[0]["selection_state"] == "selected"


def test_source_revision_and_observation_time_are_preserved_without_invented_file_hashes():
    public = project_public(
        [
            {
                "source_id": "gitcode:Ascend/agent-skills",
                "source": "gitcode",
                "org": "Ascend",
                "full_name": "Ascend/agent-skills",
                "source_locator": "gitcode:Ascend/agent-skills@" + "a" * 40,
                "head_commit": "a" * 40,
                "metadata_fetched_at": "2026-10-19T12:00:00Z",
                "content": "not an authorized metadata field",
            }
        ]
    )
    files = project_source_files(
        [
            {
                "source_id": "agent-skills:one.md",
                "repo": "git@gitcode.com:Ascend/agent-skills.git",
                "commit_sha": "b" * 40,
                "source_locator": "git+fixture@" + "b" * 40 + "#one.md",
                "task_family": "measurement",
                "content": "not an authorized metadata field",
            }
        ],
        {public[0]["repo_identity"]},
        "2026-10-19T13:00:00Z",
    )
    assert public[0]["repo_identity"] == files[0]["repo_identity"]
    assert public[0]["revision"] != files[0]["revision"]
    assert files[0]["public_catalog"] is True
    assert files[0]["sha256"] is None
    assert files[0]["timestamp_basis"] == "map_metadata_file_mtime_not_commit_time"
    assert "content" not in files[0] and "content" not in public[0]


def test_project_path_hint_is_a_review_candidate_and_does_not_assign_problem_family():
    source = project_inventory(
        "fleet",
        [
            inventory_row(
                root_label=".claude",
                relpath="projects/-Users-user-code-vllm-ascend-workspace/session-a.jsonl",
            )
        ],
    )[0]
    branch = project_branches(
        [
            {
                "source_id": "L03-one",
                "repo": "/Users/user/code/vllm-ascend-workspace",
                "repo_common_dir": "/Users/user/code/vllm-ascend-workspace/.git",
                "path": "/Users/user/code/vllm-ascend-workspace",
                "commit_sha": "a" * 40,
                "task_family": "vllm/quant",
                "branch": "feature/quant",
                "diffs": {"body": "not an authorized metadata field"},
            }
        ]
    )[0]
    hint = repository_hints([source, branch])[source["asset_id"]][0]
    assert hint["source_id"] == "L03-one"
    assert hint["reason"].endswith("candidate_only")
    assert source["declared_family"] is None
    assert source["split"] == "unassigned"
    assert "diffs" not in branch


def test_offline_privacy_failure_blocks_every_linked_copy_and_session_segment():
    offline = project_inventory("offline", [inventory_row()])
    fleet = project_inventory(
        "fleet",
        [
            inventory_row(machine="machine-b", sha256="b" * 64, stable_source_id="sha256:" + "b" * 64),
            inventory_row(
                machine="machine-c",
                session_id="independent-session",
                sha256="c" * 64,
                stable_source_id="sha256:" + "c" * 64,
            ),
        ],
    )
    groups = build_lineage_groups([*offline, *fleet])["groups"]
    attach_admission_status(
        groups,
        {
            "offline_privacy_validation": {
                "metadata": {
                    "scanner": "gitleaks",
                    "exit_code": 1,
                    "finding_count": 4,
                    "scan_complete": None,
                }
            }
        },
    )
    linked = next(group for group in groups if "session-a" in group["session_ids"])
    other = next(group for group in groups if "independent-session" in group["session_ids"])
    assert set(linked["asset_ids"]) == {offline[0]["asset_id"], fleet[0]["asset_id"]}
    assert linked["body_admission_status"] == "blocked_privacy_validation"
    assert linked["offline_privacy_validation"]["finding_count"] == 4
    assert other["body_admission_status"] == "metadata_only_not_admitted"
