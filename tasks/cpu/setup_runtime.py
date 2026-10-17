"""Create a new, lock-bound CPU interpreter without modifying another venv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination.exists():
        raise FileExistsError("preserve existing runtime; choose a fresh destination")
    version = subprocess.check_output(
        [
            str(args.python.resolve(strict=True)),
            "-I",
            "-c",
            "import json,sys;print(json.dumps(list(sys.version_info[:3])))",
        ],
        text=True,
    )
    if json.loads(version) != [3, 13, 12]:
        raise ValueError("this CPU recipe requires the explicitly pinned CPython 3.13.12")
    destination.mkdir(parents=True)
    recipe = Path(__file__).resolve().parent / "runtime"
    lock_sha = sha(recipe / "uv.lock")
    command = [
        args.uv,
        "sync",
        "--locked",
        "--no-dev",
        "--no-install-project",
        "--project",
        str(recipe),
        "--python",
        str(args.python.resolve(strict=True)),
    ]
    (destination / "setup-started.json").write_text(
        json.dumps(
            {
                "command": command,
                "uv_lock_sha256": lock_sha,
                "requested_python": str(args.python.resolve(strict=True)),
                "warning": "A failed/partial runtime is retained; do not silently overwrite it.",
            },
            indent=2,
        )
        + "\n"
    )
    result = subprocess.run(
        command,
        env=dict(os.environ, UV_PROJECT_ENVIRONMENT=str(destination / "venv")),
        capture_output=True,
        text=True,
        check=False,
    )
    (destination / "setup.stdout").write_text(result.stdout)
    (destination / "setup.stderr").write_text(result.stderr)
    result.check_returncode()
    if sha(recipe / "uv.lock") != lock_sha:
        raise RuntimeError("locked installation changed its lock")
    python = destination / "venv/bin/python"
    probe = """
import hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
import numpy,torch
if torch.__version__ != "2.10.0" or numpy.__version__ != "2.3.5":
    raise RuntimeError("CPU runtime version does not match its lock")
tensor=torch.arange(8,dtype=torch.float32,device="cpu")
if tensor.device.type != "cpu" or float(tensor.sum()) != 28.0:
    raise RuntimeError("real CPU tensor smoke failed")
packages={}
for item in importlib.metadata.distributions():
    record=item.read_text("RECORD")
    packages[item.metadata["Name"]]=dict(version=item.version,
        record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None)
print(json.dumps(dict(executable=sys.executable,base_prefix=sys.base_prefix,
    python=sys.version,platform=platform.platform(),machine=platform.machine(),
    device=tensor.device.type,torch=torch.__version__,numpy=numpy.__version__,
    torch_file=str(Path(torch.__file__).resolve()),packages=packages)))
"""
    checked = subprocess.run(
        [str(python), "-I", "-c", probe], capture_output=True, text=True, check=False
    )
    (destination / "probe.stdout").write_text(checked.stdout)
    (destination / "probe.stderr").write_text(checked.stderr)
    checked.check_returncode()
    observed = json.loads(checked.stdout)
    manifest = {
        "schema": "mindie-bench-cpu-task-runtime/1",
        "status": "completed",
        "interpreter": str(python),
        "resolved_python": str(python.resolve(strict=True)),
        "python_binary_sha256": sha(python.resolve(strict=True)),
        "uv_version": subprocess.check_output([args.uv, "--version"], text=True).strip(),
        "recipe_pyproject_sha256": sha(recipe / "pyproject.toml"),
        "uv_lock_sha256": lock_sha,
        "observed": observed,
        "scope": "Real CPU torch tensors; no torch_npu, NPU, HCCL or full-model claim.",
        "relocation": "Materialize a fresh task revision and re-prove soundness after any path/runtime change.",
    }
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
