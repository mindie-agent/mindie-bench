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

from release_transport import GitHub

ACTION_APP = 15368
CHECKLIST = "Verify LF Line Endings & Pre-PR Design Principles Checklist"


def rulesets():
    conditions = {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}}
    return [
        dict(name="mindie-release-create", target="tag", enforcement="active",
             conditions=conditions, rules=[{"type": "creation"}],
             bypass_actors=[dict(actor_type="Integration", actor_id=ACTION_APP,
                                 bypass_mode="always")]),
        dict(name="mindie-release-immutable-tags", target="tag", enforcement="active",
             conditions=conditions, rules=[{"type": "update"}, {"type": "deletion"}],
             bypass_actors=[]),
    ]


def verify(api):
    immutable = api.api(f"{api.prefix}/immutable-releases")
    if immutable.get("enabled") is not True:
        raise ValueError("immutable releases must be enabled before publication")
    summaries = api.api(f"{api.prefix}/rulesets")
    actual = {row["name"]: api.api(f"{api.prefix}/rulesets/{row['id']}")
              for row in summaries if row["name"].startswith("mindie-release-")}
    for expected in rulesets():
        row = actual.get(expected["name"], {})
        for key in ("target", "enforcement", "conditions", "rules", "bypass_actors"):
            if row.get(key) != expected[key]:
                raise ValueError(f"publication policy is missing or differs: {expected['name']} {key}")
    return dict(immutable_releases=True, tag_creation="github-actions",
                tag_mutation="denied")


def apply(api):
    summaries = api.api(f"{api.prefix}/rulesets")
    actual = {row["name"]: row for row in summaries}
    for rule in rulesets():
        existing = actual.get(rule["name"])
        path = f"{api.prefix}/rulesets" + (f"/{existing['id']}" if existing else "")
        api.api(path, method="PUT" if existing else "POST", data=rule)
    api.api(f"{api.prefix}/immutable-releases", method="PUT")
    protection = dict(
        required_status_checks=dict(strict=True, contexts=["release-ready", CHECKLIST]),
        enforce_admins=True,
        required_pull_request_reviews=dict(required_approving_review_count=0),
        restrictions=None,
    )
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    api = GitHub(args.repository)
    result = apply(api) if args.apply else verify(api)
    if args.revision:
        verify_ci(api, args.revision)
        result["source_ci"] = "passed"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
