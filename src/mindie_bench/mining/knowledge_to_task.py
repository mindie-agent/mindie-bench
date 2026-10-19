"""Corpus candidate mining and deterministic L2 task recipe compilation.

[Role]
1. `mine_knowledge`: scans a clean `mindie-knowledge` corpus to rank and select domain-diverse
   `candidate` task skeletons (`candidate.json`).
2. `compile_recipe`: compiles human-reviewed, SHA-256-verified source/reproducer/verifier/gold
   assets and optional `soundness_receipt.json` into an executable `l2` task directory.

[Invariants]
- Narrative mining (`mine_knowledge`) only produces `evidence="candidate"` skeletons with explicit
  `missing_acceptance` items; it never fabricates `task.toml` or claims `verified` status.
- `compile_recipe` verifies exact Git commit and blob SHA-256s for both the source repository and
  corpus evidence blocks, promoting `evidence` to `"verified"` only when a matching zero-model
  `soundness_receipt.json` (`before_reward == 0.0`, `after_reward == 1.0`) is supplied.

[Timeouts & Locks]
- Local `git` inspection commands are bounded to `30.0s`.

[Side-Effect Policy]
- Creates `output` (`exist_ok=False`) and writes `candidate.json`/`mining_manifest.json` or the
  compiled `task.toml` and task assets.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from mindie_bench.corpus import read_package
from mindie_bench.io import read_json, safe_relative, task_hash, tree_hash, write_json
from mindie_bench.models import TaskSpec

DOMAINS = {
    "graph_and_kernel_debugging": re.compile(
        r"aclgraph|graph.?capture|torchair|triton|算子|图编译", re.I
    ),
    "distributed_communication": re.compile(r"\bHCCL\b|all.?reduce|通信|多卡|tensor.?parallel", re.I),
    "numerical_accuracy": re.compile(r"\bNaN\b|BF16|FP16|cosine|精度|quantiz|量化", re.I),
    "performance": re.compile(r"profil|throughput|TPOT|性能|吞吐", re.I),
    "model_adaptation": re.compile(r"adapt|迁移|适配|upstream|新模型", re.I),
}


def mine_knowledge(corpus: Path, output: Path, *, limit: int = 12) -> list[dict]:
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    commit = subprocess.run(
        ["git", "-C", str(corpus), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if subprocess.run(
        ["git", "-C", str(corpus), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise ValueError("mine a clean, exact corpus revision")
    candidates = []
    for package in sorted((corpus / "tasks").iterdir()):
        decoded = read_package(package)
        navigation, materials = decoded.navigation, decoded.blocks
        text = navigation.read_text()
        body = "\n".join(path.read_text() for path in materials)
        categories = [name for name, pattern in DOMAINS.items() if pattern.search(text + "\n" + body)]
        if not categories:
            continue
        title = decoded.title
        score = sum(
            bool(re.search(pattern, body, re.I))
            for pattern in [
                r"traceback|error|报错|失败",
                r"pytest|unittest",
                r"\b[0-9a-f]{40}\b",
                r"修复|fixed|resolved",
                r"Ascend|NPU|CANN",
            ]
        )
        candidates.append(
            dict(
                instance_id="candidate-" + package.name,
                source_package=package.name,
                source_commit=commit,
                title=title,
                source_layout=decoded.layout,
                source_status=decoded.source_status,
                package_sha256=tree_hash(package),
                domain_candidates=categories,
                discovery_score=score,
                evidence="candidate",
                same_origin_task_hashes=[package.name],
                source_blocks=[
                    dict(
                        path=path.relative_to(corpus).as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    for path in materials
                ],
                missing_acceptance=[
                    "exact failing source revision",
                    "bounded public instruction",
                    "external verifier",
                    "gold fix",
                    "fail-before/pass-after proof",
                    "analogical/stale/compositional transfer labels",
                ],
            )
        )
    # Round-robin domains rather than choosing twelve near-duplicate symptoms.
    ordered = sorted(candidates, key=lambda c: (-c["discovery_score"], c["source_package"]))
    selected, seen = [], set()
    while len(selected) < min(limit, len(ordered)):
        progress = False
        for category in DOMAINS:
            row = next(
                (
                    c
                    for c in ordered
                    if c["source_package"] not in seen and category in c["domain_candidates"]
                ),
                None,
            )
            if row and len(selected) < limit:
                selected.append(row)
                seen.add(row["source_package"])
                progress = True
        if not progress:
            break
    output.mkdir(parents=True, exist_ok=False)
    for row in selected:
        write_json(output / row["instance_id"] / "candidate.json", row, exclusive=True)
    write_json(
        output / "mining_manifest.json",
        dict(
            schema_version="1.0",
            source_commit=commit,
            scanned_packages=len(list((corpus / "tasks").iterdir())),
            matching_packages=len(candidates),
            candidates=len(selected),
            warning="Historical narratives do not establish reproducibility or NPU acceptance.",
        ),
        exclusive=True,
    )
    return selected


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=30
    ).stdout


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _asset(root: Path, record: dict) -> bytes:
    if set(record) != {"path", "sha256"}:
        raise ValueError("recipe assets require exactly path and sha256")
    path = root / safe_relative(record["path"])
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("recipe asset escapes its owned input directory")
    raw = path.read_bytes()
    if _sha(raw) != record["sha256"]:
        raise ValueError(f"recipe asset hash mismatch: {record['path']}")
    return raw


def _toml_document(data: dict) -> str:
    lines = []

    def value(item):
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=True)
        if isinstance(item, bool):
            return str(item).lower()
        if type(item) in {int, float}:
            return json.dumps(item, allow_nan=False)
        if isinstance(item, list):
            return "[" + ", ".join(value(child) for child in item) + "]"
        raise ValueError("unsupported task TOML value")

    def is_table_array(item):
        return isinstance(item, list) and item and all(isinstance(child, dict) for child in item)

    def table(items, prefix, *, array=False):
        if prefix:
            bracket = "[[" if array else "["
            close = "]]" if array else "]"
            lines.append(bracket + ".".join(json.dumps(part) for part in prefix) + close)
        for key, item in items.items():
            if not isinstance(item, dict) and not is_table_array(item):
                name = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)
                lines.append(name + " = " + value(item))
        for key, item in items.items():
            if isinstance(item, dict):
                table(item, [*prefix, key])
            elif is_table_array(item):
                for child in item:
                    table(child, [*prefix, key], array=True)

    table(data, [])
    return "\n".join(lines) + "\n"


def compile_recipe(corpus: Path, recipe_manifest: Path, output: Path) -> dict:
    """Compile externally supplied, hashed real-source assets into an L2 task.

    No source or test is generated from a narrative and no verifier is executed.
    An optional independently obtained soundness receipt may promote only an
    exact matching task. Without it, even a complete recipe remains candidate.
    """
    manifest = read_json(recipe_manifest)
    required = {
        "schema",
        "corpus_commit",
        "task_spec",
        "source_files_sha256",
        "assets",
        "knowledge_evidence",
        "knowledge_condition",
        "runtime",
    }
    if set(manifest) - (required | {"soundness_receipt"}) or required - set(manifest):
        raise ValueError("unknown or missing recipe fields")
    if manifest["schema"] != "mindie-bench-l2-recipe/1":
        raise ValueError("unsupported L2 recipe schema")
    commit = manifest["corpus_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("recipe corpus requires an exact commit")
    if _git(corpus, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("recipe corpus commit mismatch")
    root = recipe_manifest.resolve().parent
    spec = TaskSpec.model_validate(manifest["task_spec"])
    if set(spec.environment) != {"l2"} or spec.environment["l2"].npu_count:
        raise ValueError("compile_recipe accepts only explicit CPU L2 tasks")
    if spec.task.evidence != "candidate":
        raise ValueError("recipe input is candidate; proof alone controls promotion")
    source = spec.environment["l2"]
    repo = Path(source.repo_path)
    source.repo_path = str((root / repo).resolve() if not repo.is_absolute() else repo.resolve())
    repo = Path(source.repo_path)
    if _git(repo, "rev-parse", source.base_commit + "^{commit}").decode().strip() != source.base_commit:
        raise ValueError("recipe source requires an exact available commit")
    if not manifest["source_files_sha256"]:
        raise ValueError("recipe requires original complete module hashes")
    for path, expected in manifest["source_files_sha256"].items():
        safe_relative(path)
        if _sha(_git(repo, "show", source.base_commit + ":" + path)) != expected:
            raise ValueError(f"recipe source blob hash mismatch: {path}")
    relation_ids = {"same_origin": set(), "analogical": set(), "stale": set()}
    for label in manifest["knowledge_evidence"]:
        if set(label) != {"package_id", "block_id", "sha256", "relation", "supporting_text"}:
            raise ValueError("knowledge evidence requires package/block/hash/relation/supporting_text")
        package_id = safe_relative(label["package_id"])
        if "/" in package_id or label["relation"] not in relation_ids:
            raise ValueError("invalid knowledge package or relation")
        package = corpus / "tasks" / package_id
        decoded = read_package(package)
        block = next((path for path in decoded.blocks if path.stem == label["block_id"]), None)
        if block is None or _sha(block.read_bytes()) != label["sha256"]:
            raise ValueError("knowledge evidence block/hash mismatch")
        relative = block.relative_to(corpus).as_posix()
        if _git(corpus, "show", commit + ":" + relative) != block.read_bytes():
            raise ValueError("knowledge evidence differs from its frozen Git blob")
        quote = label["supporting_text"]
        if not isinstance(quote, str) or len(quote) < 12 or quote not in block.read_text():
            raise ValueError("knowledge relation needs actual supporting body text")
        relation_ids[label["relation"]].add(package_id)
    knowledge = spec.mindie_knowledge
    if set(knowledge.same_origin_task_hashes) != relation_ids["same_origin"]:
        raise ValueError("same-origin package IDs disagree with their evidence")
    if set(knowledge.analogical_task_hashes) != relation_ids["analogical"]:
        raise ValueError("analogical package IDs disagree with their evidence")
    condition = manifest["knowledge_condition"]
    if not isinstance(condition, dict) or not condition.get("rationale"):
        raise ValueError("knowledge condition needs an explicit human-reviewed rationale")
    gradient = knowledge.knowledge_gradient
    if gradient == "near_exact" and not relation_ids["same_origin"]:
        raise ValueError("near_exact requires source-package evidence")
    if gradient == "analogical" and not relation_ids["analogical"]:
        raise ValueError("analogical requires independently identified package evidence")
    if gradient == "stale" and (
        not relation_ids["stale"]
        or not condition.get("historical_version")
        or not condition.get("current_version")
        or not condition.get("counterexample_asset")
    ):
        raise ValueError("stale requires version conditions and a current counterexample")
    if gradient == "no_hit" and (
        condition.get("scope_commit") != commit
        or "scope_package_ids" not in condition
        or condition.get("scope_complete_judgement") is not True
    ):
        raise ValueError("no_hit requires a frozen explicit, completely judged knowledge scope")
    runtime = manifest["runtime"]
    python = Path(runtime["python"])
    python = (root / python).absolute() if not python.is_absolute() else python
    if _sha(python.resolve(strict=True).read_bytes()) != runtime["python_binary_sha256"]:
        raise ValueError("recipe CPU interpreter hash mismatch")
    assets = {}
    for destination, record in manifest["assets"].items():
        destination = safe_relative(destination)
        if destination in {"task.toml", "soundness_receipt.json"} or destination.startswith("."):
            raise ValueError("reserved recipe output asset")
        assets[destination] = _asset(root, record)
    needed = {
        "instruction.md",
        "environment/symptom.log",
        "environment/reproduce.py",
        "verifier/test.sh",
        "verifier/verify.py",
        "oracle/gold_patch.diff",
        "oracle/replay_rules.json",
    }
    if needed - assets.keys():
        raise ValueError("recipe lacks public reproduction/log, external judge, gold or replay assets")
    if gradient == "stale" and condition["counterexample_asset"] not in assets:
        raise ValueError("stale counterexample is not a supplied hashed task asset")
    if manifest.get("soundness_receipt"):
        spec.task.evidence = "verified"
    output.mkdir(parents=True, exist_ok=False)
    (output / "task.toml").write_text(_toml_document(spec.model_dump(exclude_none=True)))
    for destination, raw in assets.items():
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        if destination in {"verifier/test.sh", "instruction.md"}:
            raw = raw.replace(b"@L2_PYTHON@", str(python).encode())
        path.write_bytes(raw)
    if manifest.get("soundness_receipt"):
        proof = json.loads(_asset(root, manifest["soundness_receipt"]))
        if (
            proof.get("tier") != "l2"
            or proof.get("status") != "completed"
            or proof.get("before_reward") != 0.0
            or proof.get("after_reward") != 1.0
            or proof.get("source_commit") != source.base_commit
            or proof.get("task_sha256") != task_hash(output)
            or proof.get("errors")
            or proof.get("model_calls") != 0
        ):
            raise ValueError("external soundness proof does not bind the compiled task")
        write_json(output / "soundness_receipt.json", proof, exclusive=True)
    return {
        "schema": "mindie-bench-l2-recipe-compilation/1",
        "instance_id": spec.task.instance_id,
        "evidence": spec.task.evidence,
        "task_sha256": task_hash(output),
        "recipe_sha256": _sha(recipe_manifest.read_bytes()),
        "corpus_commit": commit,
        "source_commit": source.base_commit,
        "knowledge_condition": condition,
        "source_files_sha256": manifest["source_files_sha256"],
        "model_calls": 0,
        "verifier_executed": False,
    }
