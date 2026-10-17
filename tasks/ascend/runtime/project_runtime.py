"""Project historical failing functions only in a new owned runtime build."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def functions(text: str) -> dict:
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in found:
                raise ValueError(f"duplicate top-level function: {node.name}")
            found[node.name] = node
    return found


def project_functions(original: bytes, failing: bytes, mappings: dict) -> tuple[bytes, list[dict]]:
    current_text, failing_text = original.decode(), failing.decode()
    current, historical = functions(current_text), functions(failing_text)
    lines, source_lines = current_text.splitlines(keepends=True), failing_text.splitlines(keepends=True)
    records = []
    for installed in sorted(mappings, key=lambda name: current[name].lineno, reverse=True):
        source_name = mappings[installed]
        old, source = current[installed], historical[source_name]
        start = min([old.lineno, *[node.lineno for node in old.decorator_list]]) - 1
        replacement = "".join(source_lines[source.lineno - 1 : source.end_lineno])
        source_function_sha256 = sha(replacement.encode())
        replacement = replacement.replace(f"def {source_name}(", f"def {installed}(", 1)
        normalized = ast.parse(replacement).body[0]
        normalized.name = source_name
        if ast.dump(normalized, include_attributes=False) != ast.dump(source, include_attributes=False):
            raise ValueError(f"function projection differs from historical source: {installed}")
        lines[start : old.end_lineno] = [replacement.rstrip() + "\n"]
        records.append(
            {
                "installed_name": installed,
                "source_name": source_name,
                "source_ast_sha256": sha(ast.dump(source, include_attributes=False).encode()),
                "original_ast_sha256": sha(ast.dump(old, include_attributes=False).encode()),
                "source_function_sha256": source_function_sha256,
                "projected_function_sha256": sha((replacement.rstrip() + "\n").encode()),
            }
        )
    projected = "".join(lines).encode()
    ast.parse(projected)
    return projected, records


def prepare_projection(manifest: dict, profile_name: str, runtime_root: Path, historical: Path):
    if manifest["schema"] != "mindie-bench-ascend-runtime-recipe/1":
        raise ValueError("unsupported runtime recipe")
    profile = manifest["profiles"][profile_name]
    ascend = runtime_root / "vllm-ascend"
    layernorm = ascend / "vllm_ascend/ops/layernorm.py"
    if sha(layernorm.read_bytes()) != manifest["rmsnorm_failing_sha256"]:
        raise ValueError("base image does not retain the exact failing RMSNorm projection")
    prepared = []
    for source in manifest["sources"]:
        family = source["family"]
        target = ascend / source["relative_path"]
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(ascend.resolve()):
            raise ValueError(f"target escapes the owned runtime root: {target}")
        original = target.read_bytes()
        failing = (historical / f"{family}.py").read_bytes()
        if sha(original) != profile["original_file_sha256"][family]:
            raise ValueError(f"unexpected installed original bytes: {family}")
        if sha(failing) != source["blob_sha256"]:
            raise ValueError(f"unexpected historical blob bytes: {family}")
        projected, records = project_functions(original, failing, source["function_mappings"])
        if sha(projected) != profile["projected_file_sha256"][family]:
            raise ValueError(f"projection does not reproduce the observed G bytes: {family}")
        prepared.append((target, projected, records, source, sha(original)))
    return prepared


def cleanup_candidates(runtime_root: Path) -> list[Path]:
    ascend = runtime_root / "vllm-ascend"
    candidates = set()
    for name in ["tests", "docs", "examples", "benchmark", "benchmarks", "dist"]:
        path = ascend / name
        if path.exists() or path.is_symlink():
            candidates.add(path)
    for path in runtime_root.rglob("*"):
        if path.name in {".git", "__pycache__"} or path.suffix == ".pyc":
            candidates.add(path)
        elif (
            path.is_file()
            and not path.is_symlink()
            and path.is_relative_to(ascend)
            and path.name.endswith(
                (".whl", ".zip", ".egg", ".tar", ".tar.gz", ".tgz", ".patch", ".diff")
            )
        ):
            candidates.add(path)
    return sorted(candidates, key=lambda path: len(path.parts), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=["a2", "a3"])
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="mutate only the new owned build")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    profile = manifest["profiles"][args.profile]
    if args.base_image != profile["base_image"]:
        raise ValueError("build base image does not match the selected exact local checkpoint")
    runtime_root = args.runtime_root.resolve(strict=True)
    if args.historical.resolve(strict=True).is_relative_to(runtime_root):
        raise ValueError("private historical build inputs must be outside the runtime grant")
    prepared = prepare_projection(manifest, args.profile, runtime_root, args.historical)
    removal = cleanup_candidates(runtime_root)
    result = {
        "schema": "mindie-bench-target-function-projection/1",
        "profile": args.profile,
        "base_image": args.base_image,
        "mode": "apply" if args.apply else "dry_run",
        "python": sys.version,
        "records": [],
        "cleanup": [str(path.relative_to(runtime_root)) for path in removal],
        "boundary": "Known function/file projection only; all granted roots need a fresh leakage audit.",
    }
    for target, projected, records, source, original_sha in prepared:
        result["records"].append(
            {
                "path": str(target),
                "base_commit": source["base_commit"],
                "original_file_sha256": original_sha,
                "projected_file_sha256": sha(projected),
                "function_mappings": records,
            }
        )
    if args.apply:
        for target, projected, *_ in prepared:
            target.write_bytes(projected)
        for path in removal:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        if list(runtime_root.rglob(".git")) or list(runtime_root.rglob("*.pyc")):
            raise RuntimeError("runtime history/bytecode cleanup incomplete")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
