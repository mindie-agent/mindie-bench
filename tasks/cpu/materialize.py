"""Bind CPU task templates to a specific locked interpreter in a fresh revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.runtime_manifest.read_text())
    if (
        manifest.get("schema") != "mindie-bench-cpu-task-runtime/1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("a completed CPU runtime manifest is required")
    python = Path(manifest["interpreter"]).absolute()
    if any(character in str(python) for character in "'\n\r"):
        raise ValueError("unsupported interpreter path quoting")
    if sha(python.resolve(strict=True)) != manifest["python_binary_sha256"]:
        raise ValueError("CPU interpreter binary no longer matches its manifest")
    source = Path(__file__).resolve().parent
    if sha(source / "runtime/uv.lock") != manifest["uv_lock_sha256"]:
        raise ValueError("CPU runtime manifest does not match the current lock")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    bindings = []
    for task in sorted(source.glob("cpu-*")):
        if not (task / "task.toml").is_file():
            continue
        destination = output / task.name
        shutil.copytree(task, destination)
        if (destination / "soundness_receipt.json").exists():
            raise ValueError("canonical templates must not inherit another revision's proof")
        config = tomllib.loads((task / "task.toml").read_text())
        repository = (task / config["environment"]["l2"]["repo_path"]).resolve(strict=True)
        text, count = re.subn(
            r'(?m)^repo_path = "[^"]+"$',
            "repo_path = " + json.dumps(str(repository)),
            (destination / "task.toml").read_text(),
        )
        if count != 1:
            raise ValueError("CPU template requires one unambiguous source repository")
        (destination / "task.toml").write_text(text)
        for relative in ["instruction.md", "verifier/test.sh"]:
            path = destination / relative
            text = path.read_text()
            if "@CPU_PYTHON@" not in text:
                raise ValueError("CPU template has no interpreter binding")
            path.write_text(text.replace("@CPU_PYTHON@", str(python)))
        binding = {
            "schema": "mindie-bench-cpu-task-runtime-binding/1",
            "interpreter": str(python),
            "runtime_manifest_sha256": sha(args.runtime_manifest),
            "uv_lock_sha256": manifest["uv_lock_sha256"],
            "torch": manifest["observed"]["torch"],
            "device": "cpu",
            "replay": False,
        }
        (destination / "verifier/runtime-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        bindings.append({"task": task.name, **binding})
    if len(bindings) != 4:
        raise ValueError("expected the four canonical CPU task templates")
    report = {
        "schema": "mindie-bench-cpu-task-revision/1",
        "runtime_manifest_sha256": sha(args.runtime_manifest),
        "tasks": bindings,
        "proofs_inherited": False,
    }
    (output / "task-revision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
