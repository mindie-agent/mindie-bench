"""Materialize an explicit same-defect seed/eval pair; never inherit a proof."""

import argparse
import shutil
from pathlib import Path

from mindie_bench.io import task_hash, write_json
from mindie_bench.mining.knowledge_to_task import _toml_document
from mindie_bench.models import TaskSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = Path(__file__).resolve().parent / "k3-gdn-idle-dp-state"
    original = TaskSpec.load(template)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for role in ["seed", "eval"]:
        identity = "k3-idle-dp-warm-" + role
        destination = output / identity
        shutil.copytree(template, destination, ignore=shutil.ignore_patterns("soundness_receipt.json"))
        spec = original.model_copy(deep=True)
        spec.task.instance_id = identity
        spec.task.role = role
        spec.task.cluster = "kimi-k3-idle-dp-same-defect-flywheel"
        spec.task.evidence = "candidate"
        spec.task.source["lineage_scope"] = (
            "An independent native session repeats the same historical defect and verifier; "
            "this pair is near-exact reuse, not two independent defects or new domains."
        )
        spec.mindie_knowledge.knowledge_gradient = "near_exact"
        (destination / "task.toml").write_text(
            _toml_document(spec.model_dump(mode="json", exclude_none=True))
        )
        TaskSpec.load(destination)
        pairs.append(dict(role=role, task=str(destination), task_sha256=task_hash(destination)))
    write_json(
        output / "materialization.json",
        dict(
            status="candidate",
            template_task_sha256=task_hash(template),
            pairs=pairs,
            model_calls=0,
            npu_calls=0,
            proofs_inherited=False,
            scope=spec.task.source["lineage_scope"],
        ),
        exclusive=True,
    )


if __name__ == "__main__":
    main()
