"""Bind the evidenced stale-configuration task to owned local source/runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

from mindie_bench.io import tree_hash
from mindie_bench.mining.knowledge_to_task import compile_recipe


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(corpus: Path, source_repo: Path, python: Path, output: Path):
    template = Path(__file__).resolve().parent / "fused-mc2-report"
    recipe = json.loads((template / "recipe.json").read_text())
    for record in recipe["assets"].values():
        if sha(template / record["path"]) != record["sha256"]:
            raise ValueError("tracked stale-task asset hash mismatch")
    python = python.absolute()
    if any(character in str(python) for character in "'\n\r"):
        raise ValueError("unsupported interpreter path quoting")
    observed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import json,sys,regex,yaml;"
                "print(json.dumps({'python':sys.executable,'python_minor':list(sys.version_info[:2]),'prefix':sys.prefix,"
                "'base_prefix':sys.base_prefix,'regex':regex.__version__,'yaml':yaml.__version__}))",
            ],
            text=True,
            timeout=20,
        )
    )
    locked = tomllib.loads((template.parent / "runtime/uv.lock").read_text())
    versions = {item["name"]: item["version"] for item in locked["package"]}
    if (
        observed["python_minor"] != [3, 13]
        or observed["regex"] != versions["regex"]
        or observed["yaml"] != versions["pyyaml"]
    ):
        raise ValueError("selected CPU runtime versions do not match this task's dependency lock")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(template / "assets", output / "assets")
    binding = {
        "python": sha(python.resolve(strict=True)),
        "regex": observed["regex"],
        "yaml": observed["yaml"],
        "uv_lock_sha256": sha(template.parent / "runtime/uv.lock"),
    }
    target = output / "assets/verifier/runtime-binding.json"
    target.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    recipe["assets"]["verifier/runtime-binding.json"]["sha256"] = sha(target)
    recipe["runtime"] = {"python": str(python), "python_binary_sha256": binding["python"]}
    recipe["task_spec"]["environment"]["l2"]["repo_path"] = str(source_repo.resolve(strict=True))
    bound = output / "recipe.json"
    bound.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    result = compile_recipe(corpus, bound, output / "compiled")
    selected = {item["package_id"] for item in recipe["knowledge_evidence"]}
    scope = output / "stale-knowledge-corpus"
    (scope / "tasks").mkdir(parents=True)
    for package_id in selected:
        shutil.copytree(corpus / "tasks" / package_id, scope / "tasks" / package_id)
    contract = corpus / "publication-contract.json"
    if contract.is_file():
        shutil.copy2(contract, scope / contract.name)
    scope_manifest = {
        "source_corpus_commit": recipe["corpus_commit"],
        "selected_package_ids": sorted(selected),
        "label": "version-scoped stale advice; not an exhaustive semantic relevance judgement",
    }
    (scope / "scope.json").write_text(json.dumps(scope_manifest, indent=2, sort_keys=True) + "\n")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Benchmark Scope",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-qm",
            "Freeze actual old advice for version-scoped stale evaluation",
        ],
    ]:
        subprocess.run(["git", "-C", str(scope), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(scope), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "schema": "mindie-bench-stale-config-materialization/1",
        "compiled": result,
        "runtime_observed": observed,
        "runtime_binding": binding,
        "runtime_read_paths": [observed["prefix"], observed["base_prefix"]],
        "knowledge_scope": {"path": str(scope), "commit": commit, "tree_sha256": tree_hash(scope)},
        "relation": "stale",
        "stale_package_ids": sorted(selected),
        "same_origin_task_hashes": [],
        "proofs_inherited": False,
        "model_calls": 0,
        "ssh_calls": 0,
        "npu_calls": 0,
    }
    (output / "materialization.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(materialize(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
