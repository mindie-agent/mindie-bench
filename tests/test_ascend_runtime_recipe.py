"""Local controls for an operator recipe; no Docker, model or accelerator calls."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1] / "tasks/ascend/runtime"
SPEC = importlib.util.spec_from_file_location("ascend_projection", RECIPE / "project_runtime.py")
PROJECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROJECTOR)


def test_function_projection_preserves_registration_api_and_renames_only_target():
    original = (
        "def native_filter(x):\n    return x + 100\n\n"
        "def register_custom_ops():\n    return 'modern registration survives'\n"
    ).encode()
    failing = b"def historical_filter(x):\n    return x - 1\n"
    projected, records = PROJECTOR.project_functions(
        original, failing, {"native_filter": "historical_filter"}
    )
    namespace = {}
    exec(compile(projected, "<local-control>", "exec"), namespace)
    assert namespace["native_filter"](2) == 1
    assert namespace["register_custom_ops"]() == "modern registration survives"
    assert records[0]["source_name"] == "historical_filter"
    assert b"def historical_filter" not in projected


def test_duplicate_target_definitions_are_not_silently_selected():
    with pytest.raises(ValueError, match="duplicate top-level function"):
        PROJECTOR.project_functions(
            b"def target(): pass\ndef target(): pass\n",
            b"def target(): return 0\n",
            {"target": "target"},
        )


def prepare_local_build(tmp_path):
    runtime = tmp_path / "owned-runtime"
    ascend = runtime / "vllm-ascend"
    target = ascend / "vllm_ascend/ops/target.py"
    target.parent.mkdir(parents=True)
    original = b"def target(x):\n    return x + 100\n\ndef registration():\n    return 42\n"
    failing = b"def target(x):\n    return x - 1\n"
    target.write_bytes(original)
    layernorm = target.with_name("layernorm.py")
    layernorm.write_text("# exact local control\n")
    (ascend / "tests").mkdir()
    (ascend / "tests/answer.py").write_text("# local cleanup control\n")
    (ascend / "source.tar.gz").write_bytes(b"not a real archive; cleanup-name control")
    (runtime / "other").mkdir()
    (runtime / "other/.git").write_text("local Git metadata control")
    historical = tmp_path / "private-history-blobs"
    historical.mkdir()
    (historical / "control.py").write_bytes(failing)
    projected, _ = PROJECTOR.project_functions(original, failing, {"target": "target"})
    manifest = {
        "schema": "mindie-bench-ascend-runtime-recipe/1",
        "rmsnorm_failing_sha256": PROJECTOR.sha(layernorm.read_bytes()),
        "sources": [
            {
                "family": "control",
                "relative_path": "vllm_ascend/ops/target.py",
                "base_commit": "1" * 40,
                "blob_sha256": PROJECTOR.sha(failing),
                "function_mappings": {"target": "target"},
            }
        ],
        "profiles": {
            "a3": {
                "base_image": "sha256:" + "a" * 64,
                "original_file_sha256": {"control": PROJECTOR.sha(original)},
                "projected_file_sha256": {"control": PROJECTOR.sha(projected)},
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = [
        sys.executable,
        "-I",
        str(RECIPE / "project_runtime.py"),
        "--manifest",
        str(manifest_path),
        "--profile",
        "a3",
        "--base-image",
        manifest["profiles"]["a3"]["base_image"],
        "--runtime-root",
        str(runtime),
        "--historical",
        str(historical),
    ]
    return args, target, ascend, manifest_path


def test_dry_run_is_read_only_and_apply_cleans_owned_history(tmp_path):
    args, target, ascend, _ = prepare_local_build(tmp_path)
    before = target.read_bytes()
    dry = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)["mode"] == "dry_run"
    assert target.read_bytes() == before
    assert (ascend / "tests/answer.py").exists()
    applied = subprocess.run([*args, "--apply"], capture_output=True, text=True, check=True)
    assert json.loads(applied.stdout)["mode"] == "apply"
    assert target.read_bytes() != before
    assert b"return 42" in target.read_bytes()
    assert not (ascend / "tests").exists()
    assert not (ascend / "source.tar.gz").exists()
    assert not (ascend.parent / "other/.git").exists()


@pytest.mark.parametrize("field", ["base_image", "original_file_sha256", "projected_file_sha256"])
def test_mismatched_provenance_fails_before_writes(tmp_path, field):
    args, target, ascend, manifest_path = prepare_local_build(tmp_path)
    before = target.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    profile = manifest["profiles"]["a3"]
    if field == "base_image":
        profile[field] = "sha256:" + "b" * 64
    else:
        profile[field]["control"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run([*args, "--apply"], capture_output=True, text=True)
    assert result.returncode != 0
    assert target.read_bytes() == before
    assert (ascend / "tests/answer.py").exists()


def test_scanner_detects_fixed_archive_and_accepts_full_module_failing_bytecode(tmp_path):
    import importlib.util
    import marshal

    refs = tmp_path / "private-references"
    refs.mkdir()
    root = tmp_path / "granted-root"
    root.mkdir()
    rope = (
        "import math as torch\n"
        "def set_cos_and_sin(x):\n    return torch.cos(x)\n"
        "def update_cos_sin(x):\n    return x\n"
        "def get_cos_and_sin_slice(x):\n    return x\n"
    ).encode()
    fixed = rope.replace(b"return torch.cos(x)", b"return torch.cos(x + 1)")
    sampler = (
        b"def _apply_top_k_top_p_pytorch(x):\n    return x\n"
        b"def _apply_top_k_top_p_ascendc(x):\n    return x\n"
    )
    for name, data in [
        ("rope-base.py", rope),
        ("rope-original.py", fixed),
        ("sampler-base.py", sampler),
        ("sampler-original.py", sampler),
    ]:
        (refs / name).write_bytes(data)
    (root / "rope.pyc").write_bytes(
        importlib.util.MAGIC_NUMBER
        + bytes(12)
        + marshal.dumps(compile(rope, "<entire-base-module>", "exec", dont_inherit=True))
    )
    with zipfile.ZipFile(root / "answer.whl", "w") as archive:
        archive.writestr("hidden-copy.py", fixed)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(RECIPE / "leakage_inventory.py"),
            "--root",
            str(root),
            "--original",
            "rope",
            str(refs / "rope-original.py"),
            "--original",
            "sampler",
            str(refs / "sampler-original.py"),
            "--base",
            "rope",
            str(refs / "rope-base.py"),
            "--base",
            "sampler",
            str(refs / "sampler-base.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["known_fixed_occurrences"] > 0
    assert any("answer.whl!" in row["path"] for row in observed["exact_original_files"])
    assert observed["bytecode_targets"]
    assert all(
        row["classification"] == "exact_failing_or_unchanged_reference"
        for row in observed["bytecode_targets"]
    )
    assert observed["universal_no_leakage_claim"] is False
