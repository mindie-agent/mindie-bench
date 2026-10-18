import hashlib
import json
import sys
from pathlib import Path

import pytest

from mindie_bench.mining.knowledge_to_task import compile_recipe
from mindie_bench.models import TaskSpec
from mindie_bench.runner.soundness import check_soundness


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def recipe(case, corpus, tmp_path):
    source, commit = corpus
    root = tmp_path / "recipe"
    root.mkdir()
    block = source / "tasks/origin/materials/0001.md"
    spec = case["spec"].model_dump(exclude_none=True)
    spec["task"]["evidence"] = "candidate"
    spec["mindie_knowledge"].update(
        knowledge_gradient="near_exact",
        same_origin_task_hashes=["origin"],
        analogical_task_hashes=[],
    )
    assets = {
        "instruction.md": "Repair the actual fixture module.\n",
        "environment/symptom.log": "Observed VALUE=1; expected VALUE=2.\n",
        "environment/reproduce.py": "import module; print(module.VALUE)\n",
        "verifier/test.sh": (case["task"] / "verifier/test.sh").read_text(),
        "verifier/verify.py": "# External verifier fixture supplied independently.\n",
        "oracle/gold_patch.diff": (
            "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n"
            "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
        ),
        "oracle/replay_rules.json": (case["task"] / "oracle/replay_rules.json").read_text(),
    }
    descriptors = {}
    for name, text in assets.items():
        path = root / "assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        descriptors[name] = {"path": "assets/" + name, "sha256": sha(path.read_bytes())}
    manifest = {
        "schema": "mindie-bench-l2-recipe/1",
        "corpus_commit": commit,
        "task_spec": spec,
        "source_files_sha256": {"module.py": sha((case["repo"] / "module.py").read_bytes())},
        "assets": descriptors,
        "knowledge_evidence": [
            {
                "package_id": "origin",
                "block_id": "0001",
                "sha256": sha(block.read_bytes()),
                "relation": "same_origin",
                "supporting_text": "NPU error; pytest failed then fixed.",
            }
        ],
        "knowledge_condition": {"rationale": "Unit fixture checks mechanics, not real NPU evidence."},
        "runtime": {
            "python": sys.executable,
            "python_binary_sha256": sha(Path(sys.executable).resolve().read_bytes()),
        },
    }
    path = root / "recipe.json"
    path.write_text(json.dumps(manifest))
    return root, path, manifest, source


def save(recipe):
    recipe[1].write_text(json.dumps(recipe[2]))


def test_complete_recipe_without_proof_stays_candidate_and_preserves_source(recipe, case, tmp_path):
    output = tmp_path / "compiled"
    result = compile_recipe(recipe[3], recipe[1], output)
    spec = TaskSpec.load(output)
    assert spec.task.evidence == "candidate"
    assert spec.environment["l2"].base_commit == case["commit"]
    assert result["verifier_executed"] is False
    assert not (output / "soundness_receipt.json").exists()
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"


def test_real_source_gold_execution_and_exact_proof_can_compile_verified_copy(recipe, case, tmp_path):
    first = tmp_path / "candidate"
    compile_recipe(recipe[3], recipe[1], first)
    proof = check_soundness(
        first, case["config"], tier="l2", output=tmp_path / "actual-soundness", promote=True
    )
    assert proof["before_reward"] == 0 and proof["after_reward"] == 1
    raw = (first / "soundness_receipt.json").read_bytes()
    supplied = recipe[0] / "external-proof.json"
    supplied.write_bytes(raw)
    recipe[2]["soundness_receipt"] = {"path": supplied.name, "sha256": sha(raw)}
    save(recipe)
    copied = tmp_path / "verified"
    result = compile_recipe(recipe[3], recipe[1], copied)
    assert result["evidence"] == "verified"
    assert TaskSpec.load(copied).task.evidence == "verified"
    assert (copied / "soundness_receipt.json").read_bytes() == raw
    with pytest.raises(FileExistsError):
        compile_recipe(recipe[3], recipe[1], copied)


@pytest.mark.parametrize("kind", ["asset", "source", "corpus", "interpreter", "quote", "block"])
def test_tampered_input_is_rejected_without_inventing_a_task(recipe, tmp_path, kind):
    data = recipe[2]
    if kind == "asset":
        (recipe[0] / "assets/instruction.md").write_text("changed")
    elif kind == "source":
        data["source_files_sha256"]["module.py"] = "0" * 64
    elif kind == "corpus":
        data["corpus_commit"] = "0" * 40
    elif kind == "interpreter":
        data["runtime"]["python_binary_sha256"] = "0" * 64
    elif kind == "quote":
        data["knowledge_evidence"][0]["supporting_text"] = "No such support in the body."
    elif kind == "block":
        (recipe[3] / "tasks/origin/materials/0001.md").write_text("changed source body")
        data["knowledge_evidence"][0]["sha256"] = sha(
            (recipe[3] / "tasks/origin/materials/0001.md").read_bytes()
        )
    save(recipe)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        compile_recipe(recipe[3], recipe[1], output)
    assert not output.exists()


def test_gradient_names_cannot_replace_versioned_stale_evidence(recipe, tmp_path):
    data = recipe[2]
    data["task_spec"]["mindie_knowledge"].update(knowledge_gradient="stale", same_origin_task_hashes=[])
    data["knowledge_evidence"][0]["relation"] = "stale"
    save(recipe)
    with pytest.raises(ValueError, match="version conditions"):
        compile_recipe(recipe[3], recipe[1], tmp_path / "invalid-stale")


def test_no_hit_requires_a_frozen_explicit_scope_not_keyword_search_absence(recipe, tmp_path):
    data = recipe[2]
    data["task_spec"]["mindie_knowledge"].update(knowledge_gradient="no_hit", same_origin_task_hashes=[])
    data["knowledge_evidence"] = []
    save(recipe)
    with pytest.raises(ValueError, match="completely judged"):
        compile_recipe(recipe[3], recipe[1], tmp_path / "invalid-no-hit")
    data["knowledge_condition"].update(
        scope_commit=data["corpus_commit"], scope_package_ids=[], scope_complete_judgement=True
    )
    save(recipe)
    result = compile_recipe(recipe[3], recipe[1], tmp_path / "declared-empty-scope")
    assert result["evidence"] == "candidate"
    assert result["knowledge_condition"]["scope_package_ids"] == []


def test_unknown_or_unbound_proof_is_preserved_as_failure_not_promoted(recipe, tmp_path):
    supplied = recipe[0] / "external-proof.json"
    raw = json.dumps(
        {"tier": "l2", "status": "unknown", "before_reward": 0, "after_reward": 1, "model_calls": 0}
    ).encode()
    supplied.write_bytes(raw)
    recipe[2]["soundness_receipt"] = {"path": supplied.name, "sha256": sha(raw)}
    save(recipe)
    output = tmp_path / "failed-proof"
    with pytest.raises(ValueError, match="does not bind"):
        compile_recipe(recipe[3], recipe[1], output)
    assert output.exists()
    assert not (output / "soundness_receipt.json").exists()
    with pytest.raises(ValueError, match="no soundness receipt"):
        TaskSpec.load(output)
