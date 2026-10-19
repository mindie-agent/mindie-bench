"""Bind tracked real-source recipes to explicitly supplied local CPU inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from mindie_bench.mining.knowledge_to_task import compile_recipe


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    *,
    corpus: Path,
    knowledge_repo: Path,
    codex_repo: Path,
    python: Path,
    gitleaks: Path,
    output: Path,
    proof_root: Path | None = None,
    templates: Path | None = None,
) -> dict:
    templates = templates or Path(__file__).resolve().parent
    python = python.absolute()
    gitleaks = gitleaks.absolute()
    runtime = {
        "python_binary_sha256": sha(python.resolve(strict=True)),
        "gitleaks": str(gitleaks),
        "gitleaks_sha256": sha(gitleaks.resolve(strict=True)),
    }
    observed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import json,sys,yaml;print(json.dumps({'python':sys.version,'pyyaml':yaml.__version__}))",
            ],
            text=True,
            timeout=20,
        )
    )
    replacements = {
        "@KNOWLEDGE_REPO@": str(knowledge_repo.resolve(strict=True)),
        "@CODEX_REPO@": str(codex_repo.resolve(strict=True)),
    }
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for template in sorted(templates.glob("*/recipe.json")):
        data = json.loads(template.read_text())
        identity = data["task_spec"]["task"]["instance_id"]
        root = output / identity
        root.mkdir()
        for destination, record in data["assets"].items():
            source = template.parent / record["path"]
            if sha(source) != record["sha256"]:
                raise ValueError(f"tracked template asset hash mismatch: {identity}/{destination}")
        shutil.copytree(template.parent / "assets", root / "assets")
        for relative in ("environment/runtime-binding.json", "verifier/runtime-binding.json"):
            target = root / "assets" / relative
            target.write_text(json.dumps(runtime, indent=2) + "\n")
            data["assets"][relative]["sha256"] = sha(target)
        source = data["task_spec"]["environment"]["l2"]
        source["repo_path"] = replacements[source["repo_path"]]
        data["runtime"] = {
            "python": str(python),
            "python_binary_sha256": runtime["python_binary_sha256"],
        }
        if proof_root is not None:
            original = proof_root / identity / "compiled" / "soundness_receipt.json"
            proof = root / "provided-soundness.json"
            proof.write_bytes(original.read_bytes())
            data["soundness_receipt"] = {"path": proof.name, "sha256": sha(proof)}
        bound = root / "recipe.json"
        bound.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = compile_recipe(corpus, bound, root / "compiled")
        (root / "compile-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        rows.append(result)
    if not rows:
        raise ValueError("no tracked L2 recipes were found")
    result = {
        "schema": "mindie-bench-corpus-first-l2-materialization/1",
        "tasks": rows,
        "runtime": runtime,
        "runtime_observed": observed,
        "model_calls": 0,
        "soundness_executed": False,
        "proof_reuse_requested": proof_root is not None,
        "note": "Relocated or changed inputs need a new task identity and actual soundness.",
    }
    (output / "materialization.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--knowledge-repo", type=Path, required=True)
    parser.add_argument("--codex-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gitleaks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-root", type=Path)
    parser.add_argument("--templates", type=Path)
    args = parser.parse_args()
    result = materialize(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
