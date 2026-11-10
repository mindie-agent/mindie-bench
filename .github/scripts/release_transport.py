"""[Role] Bound GitHub release reads and writes for the publication transaction.

[Invariants] Credentials stay with gh. Missing objects require an HTTP 404;
transport failures never masquerade as absence. Asset hashes use downloaded bytes.
[Timeouts & Locks] gh calls have 30/120 second limits; the workflow serializes writers.
[Side-Effect Policy] GitHub mutations are reconciled by the transaction caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time


class GitHub:
    def __init__(self, repository, *, budget=600):
        self.repository = repository
        self.prefix = f"repos/{repository}"
        self.deadline = time.monotonic() + budget

    def remaining(self, maximum):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("release transaction budget exhausted; reconcile before retry")
        return min(maximum, remaining)

    def run(self, args, *, payload=None, output=None, timeout=30):
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(
            ["gh", *args], input=payload, stdout=output or subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=self.remaining(timeout), check=False, **flags,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            if output is None and result.stdout:
                try:
                    response = json.loads(result.stdout)
                except ValueError:
                    response = None
                if isinstance(response, dict):
                    detail += "\n" + json.dumps({
                        key: response[key] for key in ("message", "errors", "status")
                        if key in response}, ensure_ascii=True)[:2000]
            raise RuntimeError(detail)
        return result.stdout

    def api(self, path, *, method="GET", data=None):
        args = ["api", path, "--method", method]
        payload = None
        if data is not None:
            args += ["--input", "-"]
            payload = json.dumps(data).encode()
        raw = self.run(args, payload=payload)
        return json.loads(raw) if raw else None

    def release(self, tag):
        # gh's ordinary JSON error body carries the status even when stderr is
        # localized. Do not treat authentication or network errors as a 404.
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(
            ["gh", "api", f"{self.prefix}/releases/tags/{tag}"],
            capture_output=True, timeout=self.remaining(30), check=False, **flags,
        )
        if result.returncode:
            try:
                error = json.loads(result.stdout)
            except ValueError:
                error = {}
            if str(error.get("status")) != "404":
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4000:])
            # GitHub's tag endpoint omits drafts whose tag does not exist yet.
            # The authenticated release collection is the authority for them.
            for page in range(1, 11):
                rows = self.api(f"{self.prefix}/releases?per_page=100&page={page}")
                matches = [row for row in rows if row["tag_name"] == tag]
                if len(matches) > 1:
                    raise ValueError("multiple release drafts identify the same tag")
                if matches:
                    return matches[0]
                if len(rows) < 100:
                    return None
            raise ValueError("release reconciliation exceeded its page budget")
        return json.loads(result.stdout)

    def asset(self, row):
        raw = self.run(["api", f"{self.prefix}/releases/assets/{row['id']}",
                        "-H", "Accept: application/octet-stream"])
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("release evidence exceeds 2 MiB")
        return raw

    def asset_digest(self, row):
        with tempfile.TemporaryFile() as stream:
            self.run(["api", f"{self.prefix}/releases/assets/{row['id']}",
                      "-H", "Accept: application/octet-stream"], output=stream, timeout=120)
            stream.seek(0)
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def create(self, tag, revision, body):
        return self.api(f"{self.prefix}/releases", method="POST", data={
            "tag_name": tag, "target_commitish": revision, "name": tag,
            "body": body, "draft": True, "prerelease": False,
        })

    def upload(self, tag, path):
        self.run(["release", "upload", tag, str(path), "--repo", self.repository], timeout=120)

    def delete_asset(self, asset_id):
        self.api(f"{self.prefix}/releases/assets/{asset_id}", method="DELETE")

    def publish(self, release_id):
        return self.api(f"{self.prefix}/releases/{release_id}", method="PATCH",
                        data={"draft": False, "prerelease": False, "make_latest": "true"})

    def create_tag(self, version, revision):
        from release_key import create_tag
        create_tag(self, version, revision)

    def require_new_version(self, version):
        requested = tuple(map(int, version.split(".")))
        for page in range(1, 11):
            rows = self.api(f"{self.prefix}/releases?per_page=100&page={page}")
            for row in rows:
                if row["draft"] or row["prerelease"]:
                    continue
                tag = row["tag_name"].removeprefix("v")
                if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", tag):
                    if tuple(map(int, tag.split("."))) >= requested:
                        raise ValueError("a new publication must advance the highest formal version")
            if len(rows) < 100:
                return
        raise ValueError("formal version verification exceeded its page budget")
