"""Compile a corpus-grounded Ascend loader task with the existing L2 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from mindie_bench.mining.knowledge_to_task import compile_recipe

FIX = "b9bd3ee01f57039effe9ca75d53b05e1b6bf575c"
BASE = "eabfc727b58ca57f2fa65db7d0324ea5df31d7a3"
ORIGIN = "0bc94d24162470fe8ce3fc189b7eb239aa868d074ff53c78706cad0c5c608243"
ANALOGY = "8db58966cd27b59d5d669539bf2a249b5b2b3023d511deb3ca4caffd4fde2c46"
BLOCKS = [
    (ORIGIN, "b7714d2edbc19b20b01aa13281e2b63357bd33006fbbc8b77f275860edc72cf3",
     "same_origin", "这也是主线 PR #10896 删除早期 monkey patch、改成注册 Ascend model class 后的正式实现。"),
    (ANALOGY, "55e20bf40e96152930ad859ccaa62f8834ba3124be3e67b69e0ba2e6985bb6fe",
     "analogical", "K3 DSpark 为节省显存把 `embed_tokens`/`lm_head` 占位设为 `None`"),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def materialize(corpus: Path, source_repo: Path, python: Path, output: Path):
    corpus, source_repo = corpus.resolve(strict=True), source_repo.resolve(strict=True)
    python = python.absolute()
    subprocess.run(
        [str(python), "-I", "-B", "-c", "import torch;assert torch.__version__.startswith('2.10.')"],
        check=True, capture_output=True, timeout=20,
    )
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    shutil.copytree(Path(__file__).parent / "quarot-draft/assets", assets)
    (assets / "oracle").mkdir()
    git = ["git", "-C", str(source_repo)]
    patch = subprocess.check_output(
        [*git, "diff", BASE, FIX, "--", "vllm_ascend/models/__init__.py"], timeout=20,
    )
    (assets / "oracle/gold_patch.diff").write_bytes(patch)
    (assets / "oracle/replay_rules.json").write_text(
        json.dumps({"schema_version": "1.0", "enabled": False, "checks": [],
                    "pass": {"exit_code": 0}, "fail": {"exit_code": 1}}) + "\n"
    )
    evidence = []
    for package, block, relation, quote in BLOCKS:
        raw = (corpus / "tasks" / package / "blocks" / (block + ".md")).read_bytes()
        evidence.append(dict(package_id=package, block_id=block, relation=relation,
                             sha256=sha(raw), supporting_text=quote))
    source_paths = ["vllm_ascend/models/__init__.py", "vllm_ascend/models/llama_eagle3.py"]
    recipe = dict(
        schema="mindie-bench-l2-recipe/1",
        corpus_commit=subprocess.check_output(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_files_sha256={
            name: sha(subprocess.check_output([*git, "show", BASE + ":" + name]))
            for name in source_paths
        },
        assets={
            path.relative_to(assets).as_posix(): {
                "path": path.relative_to(output).as_posix(), "sha256": sha(path.read_bytes())
            }
            for path in assets.rglob("*") if path.is_file()
        },
        knowledge_evidence=evidence,
        knowledge_condition={
            "rationale": "Exclude the EAGLE3 class-registration family; retain K3 DSpark "
                         "weight-basis advice as cross-model analogy. This is one CPU "
                         "dispatch regression, not full-model NPU accuracy acceptance."
        },
        runtime={"python": str(python), "python_binary_sha256": sha(python.resolve().read_bytes())},
        task_spec={
            "schema_version": "1.0",
            "task": {"instance_id": "l2-quarot-draft-dispatch",
                     "domain_category": "quantized_model_loading", "difficulty": "medium",
                     "evidence": "candidate", "role": "eval",
                     "source": {"repository": "vllm-project/vllm-ascend", "fix_commit": FIX,
                                "validation_scope": "complete source modules, CPU tensor math; "
                                                    "fixture registry/upstream sink; no NPU or serving"}},
            "mindie_knowledge": {"corpus_id": "ascend", "knowledge_gradient": "analogical",
                                "same_origin_task_hashes": [ORIGIN],
                                "analogical_task_hashes": [ANALOGY],
                                "target_files": source_paths},
            "budget": {"max_turns": 30, "max_npu_runs": 8, "timeout_sec": 600,
                       "verifier_timeout_sec": 45},
            "environment": {"l2": {"repo_path": str(source_repo), "base_commit": BASE,
                                    "npu_count": 0, "replay_commands": []}},
            "oracle": {"probes": [{
                "name": "torch_comparator_negative",
                "command": [str(python), "-I", "-B", "{oracle}/verifier/verify.py",
                            "--negative-control"],
                "expected_returncode": 0,
                "stdout_sha256": sha(b"oracle comparator intact\n"),
            }]},
        },
    )
    recipe_path = output / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=True, indent=2) + "\n")
    return compile_recipe(corpus, recipe_path, output / "compiled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "source-repo", "python", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(materialize(**vars(parser.parse_args())), indent=2))
