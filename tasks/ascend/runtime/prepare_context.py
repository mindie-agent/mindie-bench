"""Prepare an owned Docker build context; do not invoke Docker or contact a host."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=["a2", "a3"])
    parser.add_argument(
        "--repo", required=True, type=Path, help="existing repository with both base objects"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parent
    manifest = json.loads((recipe / "manifest.json").read_text())
    blobs = {}
    for source in manifest["sources"]:
        data = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.repo.resolve(strict=True)),
                "show",
                source["base_commit"] + ":" + source["relative_path"],
            ]
        )
        if hashlib.sha256(data).hexdigest() != source["blob_sha256"]:
            raise ValueError("historical source blob differs from the recipe manifest")
        blobs[source["family"] + ".py"] = data
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "historical").mkdir()
    for name, data in blobs.items():
        (output / "historical" / name).write_bytes(data)
    for name in ["Dockerfile", "project_runtime.py", "manifest.json"]:
        shutil.copy2(recipe / name, output / name)
    shutil.copy2(recipe.parents[2] / "scripts/ascend_probe.py", output / "ascend_probe.py")
    files = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    record = {
        "schema": "mindie-bench-ascend-build-context/1",
        "profile": args.profile,
        "base_image": manifest["profiles"][args.profile]["base_image"],
        "files": files,
        "docker_invoked": False,
        "historical_git_metadata_included": False,
        "fixed_original_bytes_included": False,
        "warning": "Build on an owned host with the exact pre-existing local base image, then use a new image identity and fresh soundness.",
    }
    (output / "context-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
