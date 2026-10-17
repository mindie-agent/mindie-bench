"""Stage exact offline inputs, including one generic compiler; no KDA binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(downloads: Path, output: Path) -> dict:
    manifest = json.loads((Path(__file__).parent / "dependencies.json").read_text())
    records = []
    for row in manifest["archives"]:
        path = downloads / row["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("offline dependency hash mismatch: " + row["filename"])
        records.append({"filename": row["filename"], "sha256": row["sha256"]})
    output.mkdir(parents=True, exist_ok=False)
    for row in records:
        shutil.copyfile(downloads / row["filename"], output / row["filename"])
    result = {
        "status": "completed",
        "archives": records,
        "compiled_kda_inputs": [],
        "compiled_generic_inputs": manifest.get("compiled_generic_inputs", []),
        "model_calls": 0,
        "npu_tensor_calls": 0,
    }
    (output / "inputs-readback.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.downloads, args.output)))
