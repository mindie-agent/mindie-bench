"""Export the exact full base tree without ancestry, alternates or a gold object."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

BASE = "9a3a8fdeb248e9fe444fe80353ea3795d1030df0"
GOLD = "8ddf72b3b697c29b8b0cd98401dd396fa319fd26"


def prepare(source_repo: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("preserve the existing source repository")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def git(repo, *args, **kwargs):
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, **kwargs
        ).stdout

    git(source_repo, "cat-file", "-e", BASE + "^{commit}")
    objects = (
        BASE.encode()
        + b"\n"
        + git(source_repo, "rev-list", "--objects", "--no-object-names", BASE + "^{tree}")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(output)], env=env, check=True, capture_output=True)
    with tempfile.TemporaryFile() as pack:
        subprocess.run(
            ["git", "-C", str(source_repo), "pack-objects", "--stdout"],
            input=objects,
            stdout=pack,
            stderr=subprocess.PIPE,
            env=env,
            check=True,
        )
        pack.seek(0)
        package_sha = hashlib.file_digest(pack, "sha256").hexdigest()
        pack.seek(0)
        subprocess.run(
            ["git", "-C", str(output), "index-pack", "--stdin"],
            stdin=pack,
            env=env,
            check=True,
            capture_output=True,
        )
    (output / "shallow").write_text(BASE + "\n")
    git(output, "update-ref", "refs/heads/source", BASE)
    git(output, "symbolic-ref", "HEAD", "refs/heads/source")
    git(output, "fsck", "--full")
    absent = subprocess.run(
        ["git", "-C", str(output), "cat-file", "-e", GOLD + "^{commit}"], env=env, capture_output=True
    )
    if absent.returncode == 0 or (output / "objects/info/alternates").exists():
        raise RuntimeError("base-only repository unexpectedly exposes gold/history")
    count = int(git(output, "rev-list", "--all", "--count").strip())
    if count != 1:
        raise RuntimeError("base-only repository has more than one commit")
    return {
        "status": "completed",
        "base_commit": BASE,
        "commit_count": count,
        "gold_object_absent": True,
        "alternates_absent": True,
        "full_tree_pack_sha256": package_sha,
        "model_calls": 0,
        "npu_tensor_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_repo.resolve(), args.output.resolve())))
