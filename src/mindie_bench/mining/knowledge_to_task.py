"""Mine navigable public Markdown packages into reviewable task candidates."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from mindie_bench.corpus import read_package
from mindie_bench.io import tree_hash, write_json

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
    tree_hash(corpus)
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
