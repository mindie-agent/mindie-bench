import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.models import Group
from mindie_bench.runner import orchestrator
from mindie_bench.runner.profile_builder import build_profile

pytestmark = pytest.mark.skipif(
    platform.system() not in {"Darwin", "Linux"}
    or (platform.system() == "Linux" and not shutil.which("bwrap")),
    reason="requires a native sandbox; no unisolated fallback",
)


@pytest.fixture
def isolated_runtime(tmp_path):
    runtime = tmp_path / "installed-runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    purelib = runtime / Path(sysconfig.get_path("purelib")).relative_to(sys.prefix)
    for name in [
        "pydantic",
        "pydantic_core",
        "annotated_types",
        "typing_extensions",
        "typing_inspection",
        "PyYAML",
    ]:
        distribution = importlib.metadata.distribution(name)
        for entry in distribution.files:
            if ".." in entry.parts or "__pycache__" in entry.parts:
                continue
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                destination = purelib / entry
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    source = Path(orchestrator.__file__).resolve().parents[1]
    shutil.copytree(source, purelib / "mindie_bench", ignore=shutil.ignore_patterns("__pycache__"))
    return SimpleNamespace(root=runtime, purelib=purelib, python=runtime / "bin/python")


def profile_for(case):
    trial_dir = case["root"] / "preflight"
    trial_dir.mkdir()
    return trial_dir, build_profile(trial_dir, case["spec"], Group.G0, case["config"])


def test_import_failure_stops_before_workspace_remote_or_model_admission(case, monkeypatch):
    calls = []

    def failed_probe(*args, **kwargs):
        calls.append("runtime")
        raise RuntimeError("runtime helper import failed: original import diagnostic")

    def forbidden(*args, **kwargs):
        pytest.fail("runtime import failure must precede workspace, remote and model admission")

    monkeypatch.setattr(orchestrator, "probe_runtime_import", failed_probe, raising=False)
    monkeypatch.setattr(orchestrator, "WorktreeSlot", forbidden)
    monkeypatch.setattr(orchestrator, "RemoteSlot", forbidden)
    monkeypatch.setattr(orchestrator, "Gateway", forbidden)
    output = case["root"] / "run"
    output.mkdir()
    receipt = orchestrator.run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output
    )
    assert calls == ["runtime"]
    assert receipt.terminal_status.value == "infra_failed"
    assert "original import diagnostic" in receipt.failure_reason
    trial_dir = Path(receipt.artifacts["private_trial_dir"])
    assert not (trial_dir / "broker.jsonl").exists()
    assert not (trial_dir / "native_events.jsonl").exists()
    assert not (trial_dir / "workspace").exists()


def test_real_installed_helpers_import_same_digest_in_native_sandbox(
    case, isolated_runtime, monkeypatch
):
    trial_dir, profile = profile_for(case)
    monkeypatch.setattr(sys, "executable", str(isolated_runtime.python))
    grants = [isolated_runtime.root, Path(sys.base_prefix)]
    original_grants = list(grants)
    observed = orchestrator.probe_runtime_import(
        profile, grants, trial_dir, orchestrator.implementation_digest()
    )
    assert observed["benchmark_source_sha256"] == orchestrator.implementation_digest()
    assert observed["executable"] == str(isolated_runtime.python)
    assert set(observed["modules"]) == set(orchestrator.RUNTIME_HELPER_MODULES)
    assert all(
        Path(path).is_relative_to(isolated_runtime.purelib / "mindie_bench")
        for path in observed["modules"].values()
    )
    assert grants == original_grants
    assert "PYTHONPATH" not in profile.env
    assert json.loads((trial_dir / "runtime-preflight.json").read_text())["status"] == "completed"


def test_real_editable_install_without_source_grant_fails_without_fallback(
    case, isolated_runtime, monkeypatch
):
    trial_dir, profile = profile_for(case)
    hidden_source = case["root"] / "ungranted-editable-source"
    hidden_source.mkdir()
    shutil.move(str(isolated_runtime.purelib / "mindie_bench"), str(hidden_source / "mindie_bench"))
    (isolated_runtime.purelib / "editable.pth").write_text(str(hidden_source) + "\n")
    monkeypatch.setattr(sys, "executable", str(isolated_runtime.python))
    grants = [isolated_runtime.root, Path(sys.base_prefix)]
    with pytest.raises(RuntimeError, match="runtime helper import failed"):
        orchestrator.probe_runtime_import(
            profile, grants, trial_dir, orchestrator.implementation_digest()
        )
    assert "ModuleNotFoundError" in (trial_dir / "runtime-preflight.stderr").read_text()
    assert hidden_source not in grants
    assert json.loads((trial_dir / "runtime-preflight.json").read_text())["status"] == "failed"


def test_real_different_installed_source_is_rejected(case, isolated_runtime, monkeypatch):
    trial_dir, profile = profile_for(case)
    package = isolated_runtime.purelib / "mindie_bench"
    with (package / "__init__.py").open("a") as stream:
        stream.write("\n# Deliberately distinct benchmark implementation for the regression.\n")
    monkeypatch.setattr(sys, "executable", str(isolated_runtime.python))
    with pytest.raises(ValueError, match="runtime implementation digest mismatch"):
        orchestrator.probe_runtime_import(
            profile,
            [isolated_runtime.root, Path(sys.base_prefix)],
            trial_dir,
            orchestrator.implementation_digest(),
        )
    diagnostic = json.loads((trial_dir / "runtime-preflight.json").read_text())
    assert diagnostic["observed"]["benchmark_source_sha256"] != diagnostic["expected_source_sha256"]
    assert diagnostic["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("benchmark_source_sha256", 12),
        ("executable", None),
        ("executable", "/different/runtime/python"),
        ("modules", {}),
    ],
)
def test_malformed_runtime_identity_cannot_be_accepted(case, monkeypatch, field, value):
    trial_dir, profile = profile_for(case)
    expected = orchestrator.implementation_digest()
    package = Path(orchestrator.__file__).resolve().parents[1]
    observed = {
        "benchmark_source_sha256": expected,
        "executable": sys.executable,
        "package_root": str(package),
        "modules": {
            module: str(package.joinpath(*module.split(".")[1:]).with_suffix(".py"))
            for module in orchestrator.RUNTIME_HELPER_MODULES
        },
    }
    observed[field] = value
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(observed), ""),
    )
    with pytest.raises(ValueError, match="runtime .*identity"):
        orchestrator.probe_runtime_import(
            profile, orchestrator.runtime_paths(case["config"], profile), trial_dir, expected
        )
    assert json.loads((trial_dir / "runtime-preflight.json").read_text())["status"] == "failed"


def test_runtime_import_deadline_preserves_partial_diagnostics(case, monkeypatch):
    trial_dir, profile = profile_for(case)

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 20
        raise subprocess.TimeoutExpired(command, 20, output=b"partial output", stderr=b"import stuck")

    monkeypatch.setattr(orchestrator.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="runtime helper import timed out"):
        orchestrator.probe_runtime_import(
            profile,
            orchestrator.runtime_paths(case["config"], profile),
            trial_dir,
            orchestrator.implementation_digest(),
        )
    assert (trial_dir / "runtime-preflight.stdout").read_bytes() == b"partial output"
    assert (trial_dir / "runtime-preflight.stderr").read_bytes() == b"import stuck"
    assert json.loads((trial_dir / "runtime-preflight.json").read_text())["status"] == "failed"
