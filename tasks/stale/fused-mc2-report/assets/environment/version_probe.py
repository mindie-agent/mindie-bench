"""Observe the two complete, unmodified historical environment API modules."""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
KEY = "VLLM_ASCEND_ENABLE_FUSED_MC2"


def observe(root: Path):
    versions = json.loads((root / "version-files.json").read_text())
    results = []
    previous = os.environ.get(KEY)
    try:
        for name, record in versions.items():
            path = root / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError("historical environment module identity changed")
            loader = importlib.machinery.SourceFileLoader(name, str(path))
            spec = importlib.util.spec_from_loader(name, loader)
            if spec is None:
                raise RuntimeError("cannot load complete historical environment module")
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            for value in ["0", "1"]:
                os.environ[KEY] = value
                try:
                    observed = getattr(module, KEY)
                except AttributeError:
                    observed = "unrecognized"
                results.append(
                    {
                        "version": name,
                        "source_commit": record["commit"],
                        "input": value,
                        "observed": observed,
                    }
                )
    finally:
        if previous is None:
            os.environ.pop(KEY, None)
        else:
            os.environ[KEY] = previous
    return results


if __name__ == "__main__":
    print(json.dumps(observe(Path(__file__).resolve().parent), indent=2))
