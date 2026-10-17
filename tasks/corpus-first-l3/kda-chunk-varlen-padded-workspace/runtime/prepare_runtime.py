"""Hide known runtime answer copies in a dedicated owned task container."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def project(runtime: Path, private: Path) -> dict:
    runtime = runtime.resolve(strict=True)
    if private.exists() or private.resolve().is_relative_to(runtime):
        raise ValueError("private backup must be a new path outside the foreground runtime")
    private.mkdir(parents=True)
    records = []
    paths = [
        runtime / "csrc",
        runtime / "tests",
        runtime / ".git",
        runtime / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "custom_transformer",
    ]
    for path in paths:
        if path.is_symlink():
            raise ValueError("runtime projection requires ordinary owned paths: " + str(path))
        if not path.exists():
            records.append(
                {"relative_path": path.relative_to(runtime).as_posix(), "already_absent": True}
            )
            continue
        destination = private / path.relative_to(runtime)
        destination.parent.mkdir(parents=True, exist_ok=True)
        inventory = {}
        for child in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if child.is_file() and not child.is_symlink():
                inventory[child.relative_to(runtime).as_posix()] = hashlib.sha256(
                    child.read_bytes()
                ).hexdigest()
        shutil.move(str(path), destination)
        records.append(
            {"relative_path": path.relative_to(runtime).as_posix(), "preserved_files_sha256": inventory}
        )
    result = {
        "schema": "mindie-bench-kda-runtime-projection/1",
        "status": "known_paths_hidden_pending_full_grant_inventory",
        "runtime": str(runtime),
        "private_backup": str(private),
        "paths": records,
        "remaining_compiled_boundary": (
            "Python wrapper/native extension, CANN, driver and vendor objects are pinned dependencies; "
            "this operation is not a universal compiled-answer absence proof."
        ),
        "model_calls": 0,
        "npu_tensor_calls": 0,
    }
    (private / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--private-backup", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(project(args.runtime, args.private_backup)))
