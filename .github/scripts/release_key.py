"""[Role] Bind version-tag creation to a per-repository main-only deploy key.

[Invariants] Private bytes go only to an environment secret or an owned temporary
SSH file. They never enter command arguments, reports, Git or diagnostic output.
[Timeouts & Locks] GitHub calls share the transaction budget; SSH push is bounded.
[Side-Effect Policy] --apply owns the dedicated environment/key. Publication only
creates the exact tested tag; it never force-pushes or moves an existing tag.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

ENVIRONMENT = "mindie-release"
SECRET = "MINDIE_RELEASE_DEPLOY_KEY"
TITLE = "mindie-release-main-only"


def configure(api):
    api.api(f"{api.prefix}/environments/{ENVIRONMENT}", method="PUT", data={
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}})
    policies = api.api(
        f"{api.prefix}/environments/{ENVIRONMENT}/deployment-branch-policies")["branch_policies"]
    if any(row["name"] != "main" or row["type"] != "branch" for row in policies):
        raise ValueError("the dedicated release environment allows an unexpected ref")
    if not policies:
        api.api(f"{api.prefix}/environments/{ENVIRONMENT}/deployment-branch-policies",
                method="POST", data={"name": "main", "type": "branch"})
    installed = api.api(f"{api.prefix}/keys")
    if any(not row["read_only"] and row["title"] != TITLE for row in installed):
        raise ValueError("an unrelated write deploy key would bypass the publication boundary")
    keys = [row for row in installed if row["title"] == TITLE]
    secrets = api.api(f"{api.prefix}/environments/{ENVIRONMENT}/secrets")["secrets"]
    if len(keys) == 1 and any(row["name"] == SECRET for row in secrets):
        if keys[0]["read_only"]:
            raise ValueError("the release deploy key is read-only")
        return keys[0]["id"]
    if keys:
        raise ValueError("incomplete deploy-key setup requires reconciling its environment secret")
    with tempfile.TemporaryDirectory(prefix="mindie-release-key-") as directory:
        private = Path(directory) / "id_ed25519"
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", TITLE, "-f", str(private)],
                       capture_output=True, check=True, timeout=15, **flags)
        public = private.with_suffix(".pub").read_text(encoding="ascii").strip()
        try:
            key = api.api(f"{api.prefix}/keys", method="POST",
                          data={"title": TITLE, "key": public, "read_only": False})
        except Exception:
            matches = [row for row in api.api(f"{api.prefix}/keys") if row["key"] == public]
            if len(matches) != 1:
                raise
            key = matches[0]
        api.run(["secret", "set", SECRET, "--repo", api.repository, "--env", ENVIRONMENT],
                payload=private.read_bytes())
    return key["id"]


def create_tag(api, version, revision):
    tag = "v" + version
    # Query refs separately from release drafts, which may exist without a tag.
    refs = api.api(f"{api.prefix}/git/matching-refs/tags/{tag}")
    existing = next((row for row in refs if row["ref"] == "refs/tags/" + tag), None)
    if existing:
        if (existing["object"].get("type") != "commit"
                or existing["object"].get("sha") != revision):
            raise ValueError("existing release tag differs from the tested source")
        return
    try:
        _push(api, revision, "refs/tags/" + tag, dry_run=False)
    except (RuntimeError, subprocess.TimeoutExpired):
        observed = api.api(f"{api.prefix}/git/matching-refs/tags/{tag}")
        if not any(row["ref"] == "refs/tags/" + tag
                   and row["object"].get("sha") == revision for row in observed):
            raise


def check_key(api, revision):
    """Exercise the same SSH credential without creating a tag or branch."""
    _push(api, revision, "refs/tags/vmindie-credential-check-" + revision, dry_run=True)


def _push(api, revision, reference, *, dry_run):
    secret = os.environ.get(SECRET)
    if not secret:
        raise ValueError("the main-only release deploy key is unavailable")
    with tempfile.TemporaryDirectory(prefix="mindie-release-push-") as directory:
        private, hosts = Path(directory) / "key", Path(directory) / "known_hosts"
        # gh secret set trims trailing stdin whitespace; OpenSSH requires this LF.
        private.write_text(secret.replace("\r\n", "\n").rstrip("\n") + "\n",
                           encoding="ascii", newline="\n")
        private.chmod(0o600)
        hosts.write_text("".join("github.com " + key + "\n"
                                 for key in api.api("meta")["ssh_keys"]),
                         encoding="ascii", newline="\n")
        environment = dict(os.environ)
        environment["GIT_SSH_COMMAND"] = shlex.join([
            "ssh", "-i", private.as_posix(), "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + hosts.as_posix()])
        environment.pop(SECRET, None)
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        command = ["git", "push"]
        if dry_run:
            command.append("--dry-run")
        result = subprocess.run(
            [*command, "git@github.com:" + api.repository + ".git",
             revision + ":" + reference], env=environment, capture_output=True,
            timeout=api.remaining(90), check=False, **flags)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            detail = detail.replace(secret, "<redacted release key>")
            raise RuntimeError(f"release SSH push exited {result.returncode}: {detail[-4000:]}")


if __name__ == "__main__":
    from release_transport import GitHub
    parser = argparse.ArgumentParser(description="Check the release SSH credential without writes")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    check_key(GitHub(args.repository, budget=150), args.revision)
    print(json.dumps(dict(status="checked", repository=args.repository, dry_run=True)))
