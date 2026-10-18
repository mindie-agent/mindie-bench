"""Local staging controls for the real CANN task; no model, SSH or NPU calls."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from mindie_bench.io import task_hash
from mindie_bench.models import TaskSpec

TASK = Path(__file__).resolve().parents[1] / "tasks/corpus-first-l3/kda-chunk-varlen-padded-workspace"


def load(relative):
    spec = importlib.util.spec_from_file_location("kda_" + Path(relative).stem, TASK / relative)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_recipe_inspection_does_not_add_generated_files_to_task_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "TASK", tmp_path)
    (tmp_path / "recipe.py").write_text("VALUE = 4\n")
    previous = sys.dont_write_bytecode
    assert load("recipe.py").VALUE == 4
    assert sys.dont_write_bytecode == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["recipe.py"]


def fixture_inputs(tmp_path, monkeypatch):
    build = load("environment/build_kda.py")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    contents = {
        "catlass-41bf90.tar.gz": {"include/catlass.h": b"catlass source\n"},
        "makeself-release-2.5.0-patch1.tar.gz": {"release/makeself.sh": b"#!/bin/sh\n"},
        "abseil-cpp-20230802.1.tar.gz": {"absl/source.h": b"abseil source\n"},
        "protobuf-25.1.tar.gz": {"src/protobuf.h": b"protobuf source\n"},
    }
    for name, files in contents.items():
        with tarfile.open(inputs / name, "w:gz") as archive:
            for path, data in files.items():
                member = tarfile.TarInfo(path)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    generic = {
        "protobuf/include/google/protobuf/message.h": b"protobuf header\n",
        "protobuf/bin/protoc": b"#!/bin/sh\necho libprotoc25.1\n",
        "protobuf/bin/options-pinned.h": b"generic generated options\n",
        "abseil/absl/log/absl_log.h": b"patched generic header\n",
    }
    generic_manifest = {
        "compiled_generic_inputs": ["protobuf/bin/protoc"],
        "compiled_kda_inputs": [],
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in generic.items()},
        "protoc_version": "local source-staging control; no native execution",
    }
    generic["generic-manifest.json"] = json.dumps(generic_manifest).encode()
    with tarfile.open(inputs / "generic-protobuf-abseil.tar.gz", "w:gz") as archive:
        for name, data in generic.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with zipfile.ZipFile(inputs / "include.zip", "w") as archive:
        archive.writestr("include/nlohmann/json.hpp", "json header\n")
    monkeypatch.setattr(
        build,
        "INPUTS",
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.iterdir()},
    )
    source = tmp_path / "worktree"
    (source / "csrc/cmake/third_party/build").mkdir(parents=True)
    (source / "csrc/build.sh").write_text("# normal upstream build entry\n")
    (source / "csrc/cmake/third_party/build/required.patch").write_text("required upstream patch\n")
    for name in ["build", "build_out", "output"]:
        (source / "csrc" / name).mkdir()
        (source / "csrc" / name / "old_binary.so").write_bytes(b"unusable historical binary")
    return build, source, inputs


def test_fresh_source_stage_keeps_nested_build_patches_and_excludes_only_generated_roots(
    tmp_path, monkeypatch
):
    build, source, inputs = fixture_inputs(tmp_path, monkeypatch)
    result = build.build(source, inputs, tmp_path / "attempt", prepare_only=True)
    copied = Path(result["source_copy"]) / "csrc"
    assert (copied / "cmake/third_party/build/required.patch").read_text() == "required upstream patch\n"
    assert all(
        not (copied / name / "old_binary.so").exists() for name in ["build", "build_out", "output"]
    )
    cache = (copied / "build/CMakeCache.txt").read_text()
    assert "ASCEND_PROTOBUF_SHARED_INCLUDE:PATH=" in cache
    assert "ASCEND_PROTOC:PATH=" in cache
    assert (copied / "third_party/makeself/makeself.sh").is_file()
    assert (copied / "third_party/json/include/nlohmann/json.hpp").is_file()
    assert not list(copied.rglob("*.so"))
    assert result["status"] == "prepared"
    assert result["npu_tensor_calls_during_build"] == 0
    assert "--ops=chunk_kda_fwd,kda_layout_swap12" in result["command"]
    assert result["generic_dependency"]["compiled_kda_inputs"] == []


def test_changed_dependency_is_rejected_before_creating_an_output(tmp_path, monkeypatch):
    build, source, inputs = fixture_inputs(tmp_path, monkeypatch)
    (inputs / "protobuf-25.1.tar.gz").write_bytes(b"different package")
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="frozen input"):
        build.build(source, inputs, output, prepare_only=True)
    assert not output.exists()


def test_missing_generic_prefix_never_silently_falls_back_to_a_cold_build(tmp_path, monkeypatch):
    build, source, inputs = fixture_inputs(tmp_path, monkeypatch)
    (inputs / "generic-protobuf-abseil.tar.gz").unlink()
    with pytest.raises(FileNotFoundError):
        build.build(source, inputs, tmp_path / "missing", prepare_only=True)
    explicit = build.build(
        source, inputs, tmp_path / "explicit-bootstrap", prepare_only=True, cold_generic=True
    )
    assert explicit["generic_dependency"]["reuse"].startswith("none; explicitly requested")
    assert "generic-protobuf-abseil.tar.gz" not in explicit["input_archives_sha256"]
    assert not (Path(explicit["source_copy"]) / "csrc/build/CMakeCache.txt").exists()


def test_public_build_output_is_small_while_the_full_manifest_is_preserved(
    tmp_path, monkeypatch, capsys
):
    public_helper = (TASK / "environment/build_kda.py").read_text()
    assert "postWuScratchOffset" not in public_helper
    build, source, inputs = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / "public-attempt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_kda.py",
            "--source",
            str(source),
            "--inputs",
            str(inputs),
            "--output",
            str(output),
            "--prepare-only",
        ],
    )
    build.main()
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    full = json.loads((output / "build-receipt.json").read_text())
    assert len(stdout) < 2000
    assert "source_files_sha256" not in summary
    assert summary["source_file_count"] == len(full["source_files_sha256"])
    assert "cmake/third_party/build/required.patch" in full["source_files_sha256"]


def test_generic_prefix_rejects_an_extra_operator_binary(tmp_path, monkeypatch):
    build, source, inputs = fixture_inputs(tmp_path, monkeypatch)
    prefix = tmp_path / "generic"
    build.unpack_tar(inputs / "generic-protobuf-abseil.tar.gz", prefix)
    (prefix / "libcust_opapi.so").write_bytes(b"\x7fELFbad operator object")
    with pytest.raises(ValueError, match="unexpected native object"):
        build.generic_prefix(prefix)


def test_runtime_projection_preserves_original_bytes_outside_granted_runtime(tmp_path):
    projection = load("runtime/prepare_runtime.py")
    runtime = tmp_path / "runtime"
    (runtime / "csrc").mkdir(parents=True)
    answer = runtime / "csrc/fixed.cpp"
    answer.write_bytes(b"known fixed C++ source\n")
    (runtime / "vllm_ascend").mkdir()
    wrapper = runtime / "vllm_ascend/utils.py"
    wrapper.write_bytes(b"# matching ordinary wrapper stays\n")
    private = tmp_path / "backend-only"
    result = projection.project(runtime, private)
    assert not (runtime / "csrc").exists()
    assert (private / "csrc/fixed.cpp").read_bytes() == b"known fixed C++ source\n"
    assert wrapper.read_bytes() == b"# matching ordinary wrapper stays\n"
    assert result["status"] == "known_paths_hidden_pending_full_grant_inventory"
    with pytest.raises(ValueError, match="new path"):
        projection.project(runtime, private)


def test_complete_historical_verifier_and_transported_script_are_byte_bound():
    manifest = json.loads((TASK / "runtime/dependencies.json").read_text())
    whole_test = (TASK / "verifier/upstream_test.py.txt").read_bytes()
    assert hashlib.sha256(whole_test).hexdigest() == manifest["numerical_test_module"]["sha256"]
    assembled = load("runtime/assemble_verifier.py").render(TASK)
    assert (TASK / "verifier/test.sh").read_text() == assembled
    assert whole_test.decode() in assembled
    assert (TASK / "environment/build_kda.py").read_text() in assembled
    assert "PRIVATE_KDA_ASSET_2" in assembled


def test_task_does_not_inherit_operator_measurement_as_repair_soundness():
    assert not list(TASK.rglob("*.pyc")), "generated bytecode must not enter the task hash"
    spec = TaskSpec.load(TASK)
    if spec.task.evidence == "verified":
        proof = json.loads((TASK / "soundness_receipt.json").read_text())
        assert proof["status"] == "completed" and proof["task_sha256"] == task_hash(TASK)
        assert proof["before_reward"] == 0 and proof["after_reward"] == 1
        assert proof["errors"] == []
    else:
        assert spec.task.evidence == "candidate"
    assert spec.environment["l3"].npu_count == 1
    assert spec.budget.verifier_timeout_sec >= 1800
    assert len(spec.mindie_knowledge.same_origin_task_hashes) == 2
    assert spec.mindie_knowledge.analogical_task_hashes == []


def test_known_fixed_source_inside_an_archive_is_not_accepted(tmp_path):
    scanner = load("runtime/known_source_inventory.py")
    archive_path = tmp_path / "source-copy.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("renamed.cpp", b"cursor = " + scanner.FIXED_EXPRESSION + b";\n")
    result = scanner.inventory([tmp_path])
    assert result["known_source_status"] == "blocked_known_fixed_or_git"
    assert result["known_fixed"][0]["path"].endswith("source-copy.zip!renamed.cpp")
    assert result["universal_no_leakage_claim"] is False


def test_real_git_export_retains_full_tree_but_no_ancestry_or_gold(tmp_path, monkeypatch):
    recipe = load("runtime/prepare_repository.py")
    source = tmp_path / "upstream"
    source.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.email", "local-test@example.invalid")
    git("config", "user.name", "Local Control")
    tracked = source / "complete-module.cpp"
    revisions = []
    for content in ["parent", "failing original module", "fixed original module"]:
        tracked.write_text(content)
        git("add", ".")
        git("commit", "-m", content)
        revisions.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(recipe, "BASE", revisions[1])
    monkeypatch.setattr(recipe, "GOLD", revisions[2])
    output = tmp_path / "base-only.git"
    result = recipe.prepare(source, output)
    assert result["commit_count"] == 1
    assert result["gold_object_absent"]
    read = subprocess.run(
        ["git", "-C", str(output), "show", revisions[1] + ":complete-module.cpp"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert read.stdout == "failing original module"
    for forbidden in [revisions[0], revisions[2]]:
        check = subprocess.run(
            ["git", "-C", str(output), "cat-file", "-e", forbidden], capture_output=True
        )
        assert check.returncode != 0
