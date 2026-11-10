"""[Role] Enforce and inspect the GitHub-side publication boundary.

[Invariants] New v* tags are reserved to Actions after main CI. Published tags
cannot move or disappear, releases are immutable, and main requires passing CI.
Repository administrators can change policy; ordinary release calls cannot bypass it.
[Timeouts & Locks] Bounded gh calls; repository API operations are reconciled by readback.
[Side-Effect Policy] Default is read-only. --apply changes the named repository settings.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from release_key import ENVIRONMENT, configure
from release_transport import GitHub

ACTION_APP = 15368
CHECKLIST = "Verify LF Line Endings & Pre-PR Design Principles Checklist"


def verify_entry(source, repository):
    text = (Path(source) / ".github/workflows/release.yml").read_text(encoding="utf-8")
    target = (r"\./\.github/workflows/publish-release\.yml" if repository == "mindie-agent/mindie-bench"
              else r"mindie-agent/mindie-bench/\.github/workflows/publish-release\.yml@main")
    if not re.search(r"^\s+uses:\s*" + target + r"\s*$", text, re.M):
        raise ValueError("the release entry must use the current protected-main publisher")
    if re.search(r"^\s+benchmark-ref:", text, re.M):
        raise ValueError("a caller-maintained benchmark pin would bypass controller updates")
    if not re.search(
            r"^\s+MINDIE_RELEASE_DEPLOY_KEY:\s*\$\{\{\s*secrets\.MINDIE_RELEASE_DEPLOY_KEY\s*\}\}\s*$",
            text, re.M):
        raise ValueError("the release caller must explicitly forward its environment deploy key")
    if not re.search(r"^on:\s*\n\s+workflow_dispatch:", text, re.M):
        raise ValueError("the standard release entry must start from main workflow dispatch")


def rulesets():
    conditions = {"ref_name": {"include": ["~ALL"], "exclude": []}}
    return [
        dict(name="mindie-release-create", target="tag", enforcement="active",
             conditions=conditions, rules=[{"type": "creation"}],
             bypass_actors=[dict(actor_type="DeployKey", actor_id=None,
                                 bypass_mode="always")]),
        dict(name="mindie-release-immutable-tags", target="tag", enforcement="active",
             conditions=conditions, rules=[{"type": "update"}, {"type": "deletion"}],
             bypass_actors=[]),
    ]


def verify(api, *, workflow=False):
    # GitHub does not grant Actions tokens Administration:read. Deployment
    # audits inspect that setting; the publisher verifies the actual immutable
    # public release. Runner audits never claim to have read an admin setting.
    if not workflow:
        immutable = api.api(f"{api.prefix}/immutable-releases")
        if immutable.get("enabled") is not True:
            raise ValueError("immutable releases must be enabled before publication")
    summaries = api.api(f"{api.prefix}/rulesets")
    actual = {row["name"]: api.api(f"{api.prefix}/rulesets/{row['id']}")
              for row in summaries if row["name"].startswith("mindie-release-")}
    for expected in rulesets():
        row = actual.get(expected["name"], {})
        # GitHub omits bypass identities from non-administrator responses.
        # Compare public rules here; the administrator deployment audit also
        # verifies identities and private environment settings below.
        keys = ("target", "enforcement", "conditions", "rules")
        if not workflow:
            keys += ("bypass_actors",)
        for key in keys:
            if row.get(key) != expected[key]:
                raise ValueError(f"publication policy is missing or differs: {expected['name']} {key}")
    if workflow:
        return dict(tag_creation="restricted", tag_mutation="restricted",
                    private_policy="administrator-deployment-audit",
                    immutable_releases="publish-readback-required")
    environment = api.api(f"{api.prefix}/environments/{ENVIRONMENT}")
    if environment.get("deployment_branch_policy") != {
            "protected_branches": False, "custom_branch_policies": True}:
        raise ValueError("the release key environment is not restricted by branch policy")
    policies = api.api(
        f"{api.prefix}/environments/{ENVIRONMENT}/deployment-branch-policies")["branch_policies"]
    if len(policies) != 1 or policies[0]["name"] != "main" or policies[0]["type"] != "branch":
        raise ValueError("the release deploy key must be restricted to main")
    return dict(immutable_releases=True, tag_creation="main-only-deploy-key",
                tag_mutation="denied")


def apply(api):
    configure(api)
    summaries = api.api(f"{api.prefix}/rulesets")
    actual = {row["name"]: row for row in summaries}
    for rule in rulesets():
        existing = actual.get(rule["name"])
        path = f"{api.prefix}/rulesets" + (f"/{existing['id']}" if existing else "")
        api.api(path, method="PUT" if existing else "POST", data=rule)
    api.api(f"{api.prefix}/immutable-releases", method="PUT")
    branch = api.api(f"{api.prefix}/branches/main")
    previous = (api.api(f"{api.prefix}/branches/main/protection")
                if branch.get("protected") else {})
    checks = (previous.get("required_status_checks") or {}).get("checks", [])
    checks = [row for row in checks if row["context"] not in {"release-ready", CHECKLIST}]
    checks.extend(dict(context=name, app_id=ACTION_APP) for name in ("release-ready", CHECKLIST))
    reviews = previous.get("required_pull_request_reviews") or {}
    review_settings = {key: reviews[key] for key in (
        "dismiss_stale_reviews", "require_code_owner_reviews",
        "required_approving_review_count", "require_last_push_approval") if key in reviews}
    review_settings.setdefault("required_approving_review_count", 0)
    restrictions = previous.get("restrictions")
    if restrictions:
        restrictions = {
            "users": [row["login"] for row in restrictions.get("users", [])],
            "teams": [row["slug"] for row in restrictions.get("teams", [])],
            "apps": [row["slug"] for row in restrictions.get("apps", [])],
        }
    protection = dict(
        required_status_checks=dict(strict=True, checks=checks), enforce_admins=True,
        required_pull_request_reviews=review_settings, restrictions=restrictions,
    )
    for name in ("required_linear_history", "required_conversation_resolution",
                 "allow_force_pushes", "allow_deletions", "block_creations",
                 "lock_branch", "allow_fork_syncing"):
        if name in previous:
            protection[name] = previous[name]["enabled"]
    api.api(f"{api.prefix}/branches/main/protection", method="PUT", data=protection)
    return verify(api)


def verify_ci(api, revision):
    rows = api.api(f"{api.prefix}/commits/{revision}/check-runs?per_page=100")["check_runs"]
    for name in ("release-ready", CHECKLIST):
        candidates = [row for row in rows if row["name"] == name
                      and row.get("app", {}).get("id") == ACTION_APP]
        if not candidates:
            raise ValueError(f"source has no required CI result: {name}")
        newest = max(candidates, key=lambda row: row["id"])
        if newest["status"] != "completed" or newest["conclusion"] != "success":
            raise ValueError(f"source CI did not pass: {name}")
    runs = api.api(
        f"{api.prefix}/actions/runs?head_sha={revision}&event=push&per_page=100")["workflow_runs"]
    latest = {}
    for row in runs:
        if row["head_branch"] == "main":
            if row["path"] not in latest or row["id"] > latest[row["path"]]["id"]:
                latest[row["path"]] = row
    if ".github/workflows/publication-policy.yml" not in latest:
        raise ValueError("the source has no main-branch deployed-policy verification")
    if any(row["status"] != "completed" or row["conclusion"] != "success"
                         for row in latest.values()):
        raise ValueError("all main-branch CI workflows must finish successfully before release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--ci-only", action="store_true")
    parser.add_argument("--entry")
    parser.add_argument("--workflow", action="store_true",
                        help="audit runner-readable controls; publication must read back immutability")
    args = parser.parse_args()
    api = GitHub(args.repository)
    if args.ci_only and (args.apply or not args.revision):
        parser.error("--ci-only requires --revision and cannot change policy")
    if args.workflow and (args.apply or args.ci_only):
        parser.error("--workflow is a separate read-only runner policy audit")
    result = ({"verification": "source-ci"} if args.ci_only
              else apply(api) if args.apply else verify(api, workflow=args.workflow))
    if args.revision:
        verify_ci(api, args.revision)
        result["source_ci"] = "passed"
    if args.entry:
        verify_entry(args.entry, args.repository)
        result["release_entry"] = "passed"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
