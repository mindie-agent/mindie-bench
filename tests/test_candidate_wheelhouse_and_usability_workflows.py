"""Orthogonal tests for candidate wheelhouse assembly, zero-Git wheelhouse upgrade, and 3-OS CI/release gates.

[Role]
  Verify deterministic candidate `.whl` substitution (`assemble_candidate_wheelhouse.py`),
  `HAR-ABI` and `ReMe` pin enforcement, real subprocess `wheelhouse.update_runtime` + `git`
  `exit 99` trap + `mindie-harness selftest --mode full` gating (`upgrade_acceptance.py`),
  `consumer_contract.validate_evidence`, and reusable/release workflow topologies.
[Invariants]
  * 100% offline deterministic Tier 0 / Tier 1 execution inside `tmp_path` sandboxes.
[Timeouts & Locks]
  * Completes well within Tier 1 (`< 15s` per test); acquires no global locks.
[Side-Effect Policy]
  * Writes only inside pytest `tmp_path` directories; zero network calls.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from mindie_bench.runner.host_platform import subprocess_window_kwargs, which_native

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


acw = _load_script("assemble_candidate_wheelhouse")
upg = _load_script("upgrade_acceptance")
contract = _load_script("consumer_contract")


def _make_wheel(
    path: Path,
    name: str,
    version: str,
    *,
    requires: tuple[str, ...] = (),
    files: dict[str, str] | None = None,
) -> Path:
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {req}\n" for req in requires)
    )
    wheel_meta = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    records: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entries = {
            **(files or {}),
            f"{dist_info}/METADATA": metadata,
            f"{dist_info}/WHEEL": wheel_meta,
        }
        for rel, content in sorted(entries.items()):
            raw = content.encode("utf-8")
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, raw)
            records.append(f"{rel},sha256={hashlib.sha256(raw).hexdigest()},{len(raw)}")
        records.append(f"{dist_info}/RECORD,,")
        rec_info = zipfile.ZipInfo(f"{dist_info}/RECORD", date_time=(1980, 1, 1, 0, 0, 0))
        rec_info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(rec_info, "\n".join(records) + "\n")
    return path


def _sibling_fixtures(base_version: str = "1.0.0"):
    revs = {
        pkg: hashlib.sha256(f"{pkg}:{base_version}".encode()).hexdigest()[:40]
        for pkg in acw.SHARED_PACKAGES
    }

    def download_fn(repo: str, version: str, dest: Path) -> None:
        pkg = acw.REPOSITORY_PACKAGES[repo]
        reqs = (
            (f"reme-ai[as] @ git+https://github.com/agentscope-ai/ReMe.git@{acw.REME_REVISION}",)
            if pkg == "mindie-knowledge"
            else ()
        )
        whl = _make_wheel(
            dest / f"{pkg.replace('-', '_')}-{version}-py3-none-any.whl",
            pkg,
            version,
            requires=reqs,
            files={f"{pkg.replace('-', '_')}/__init__.py": f"__version__ = {version!r}\n"},
        )
        sha = hashlib.sha256(whl.read_bytes()).hexdigest()
        (dest / "SHA256SUMS").write_text(f"{sha}  {whl.name}\n", encoding="utf-8", newline="\n")

    def tag_fn(repo: str, version: str) -> str:
        return revs[acw.REPOSITORY_PACKAGES[repo]]

    def latest_fn(repo: str) -> dict:
        return {"tag_name": f"v{base_version}", "draft": False, "prerelease": False}

    def reme_fn(repo: str, rev: str, dest: Path) -> Path:
        return _make_wheel(
            dest / "reme_ai-0.2.0-py3-none-any.whl",
            "reme-ai",
            "0.2.0",
            files={"reme_ai/__init__.py": "__version__ = '0.2.0'\n"},
        )

    channel = {
        "schema": "mindie-release-channel/1",
        "schema_version": "mindie-release-channel/1",
        "track": "0.1",
        "current_patch_version": "0.1.17",
        "components": {
            acw.PACKAGE_CHANNEL_KEYS[pkg]: {
                "name": acw.PACKAGE_CHANNEL_KEYS[pkg],
                "role": "shared_component",
                "repository": repo,
                "package": pkg,
                "version": base_version,
                "tag": f"v{base_version}",
                "commit": revs[pkg],
                "revision": revs[pkg],
            }
            for pkg, repo in acw.PACKAGE_REPOSITORIES.items()
        },
    }
    return revs, channel, download_fn, tag_fn, latest_fn, reme_fn


@pytest.mark.tier0
def test_assemble_wheelhouse_supports_channel_and_latest_releases_and_candidate_substitution(
    tmp_path: Path,
) -> None:
    _revs, channel, dl_fn, tag_fn, latest_fn, reme_fn = _sibling_fixtures("1.2.0")
    channel_file = tmp_path / "channel.json"
    channel_file.write_text(json.dumps(channel), encoding="utf-8", newline="\n")

    # 1. Pure formal baseline without --candidate-repo (querying latest_release_fn without --channel)
    base_stage = acw.assemble_wheelhouse(
        tmp_path / "stage-baseline",
        download_fn=dl_fn,
        tag_fn=tag_fn,
        latest_release_fn=latest_fn,
        reme_builder=reme_fn,
    )
    expected_stage_files = {
        "SHA256SUMS",
        "bundle-manifest.json",
        "mindie-wheels-bundle.tar.gz",
        "release-channel.json",
        "requirements.txt",
    }
    assert {p.name for p in base_stage.iterdir()} == expected_stage_files

    # 2. Candidate substitution with --channel and --reme-wheel (deterministic across runs)
    cand_rev = "a" * 40
    cand_whl = _make_wheel(
        tmp_path / "cand" / "mindie_harness-0.1.5-py3-none-any.whl",
        "mindie-harness",
        "0.1.5",
        requires=('packaging>=23.2; extra == "updates"', "pytest<10,>=8; extra == 'test'"),
        files={"mindie_harness/__init__.py": "__version__ = '0.1.5'\n"},
    )
    reme_whl = _make_wheel(
        tmp_path / "reme-pre" / "reme_ai-0.2.0-py3-none-any.whl",
        "reme-ai",
        "0.2.0",
        files={"reme_ai/__init__.py": "__version__ = '0.2.0'\n"},
    )
    s1 = acw.assemble_wheelhouse(
        tmp_path / "stage-cand-1",
        channel_path=channel_file,
        candidate_repo="mindie-agent/harness",
        candidate_wheel=cand_whl,
        candidate_revision=cand_rev,
        reme_wheel=reme_whl,
        download_fn=dl_fn,
        tag_fn=tag_fn,
    )
    s2 = acw.assemble_wheelhouse(
        tmp_path / "stage-cand-2",
        channel_path=channel_file,
        candidate_repo="mindie-agent/harness",
        candidate_wheel=cand_whl.parent,
        candidate_revision=cand_rev,
        reme_wheel=reme_whl.parent,
        download_fn=dl_fn,
        tag_fn=tag_fn,
    )
    for fname in expected_stage_files:
        assert (s1 / fname).read_bytes() == (s2 / fname).read_bytes()

    ch_out = json.loads((s1 / "release-channel.json").read_text(encoding="utf-8"))
    assert ch_out["schema"] == "mindie-release-channel/1"
    assert ch_out["components"]["harness"]["version"] == "0.1.5"
    assert ch_out["components"]["harness"]["revision"] == cand_rev
    assert "mindie-harness==0.1.5" in (s1 / "requirements.txt").read_text(encoding="utf-8")
    sums = contract.parse_sums((s1 / "SHA256SUMS").read_bytes())
    assert set(sums) == expected_stage_files - {"SHA256SUMS"}
    for name, sha in sums.items():
        assert hashlib.sha256((s1 / name).read_bytes()).hexdigest() == sha


@pytest.mark.tier0
def test_assemble_wheelhouse_enforces_har_abi_and_reme_pin_and_formal_releases(
    tmp_path: Path,
) -> None:
    _revs, _channel, dl_fn, tag_fn, latest_fn, reme_fn = _sibling_fixtures("1.2.0")

    # 1. HAR-ABI rejects unconditional runtime Requires-Dist on mindie-harness
    for idx, bad_req in enumerate(("pydantic>=2.10", 'packaging>=23.2; python_version >= "3.11"')):
        bad_harness = _make_wheel(
            tmp_path / f"bad-har-{idx}" / "mindie_harness-0.1.5-py3-none-any.whl",
            "mindie-harness",
            "0.1.5",
            requires=(bad_req,),
        )
        with pytest.raises(ValueError, match="HAR-ABI violation"):
            acw.assemble_wheelhouse(
                tmp_path / f"out-har-{idx}",
                candidate_repo="mindie-agent/harness",
                candidate_wheel=bad_harness,
                candidate_revision="b" * 40,
                download_fn=dl_fn,
                tag_fn=tag_fn,
                latest_release_fn=latest_fn,
                reme_builder=reme_fn,
            )

    # 2. Platform-specific wheel is rejected
    plat_whl = _make_wheel(
        tmp_path / "plat" / "mindie_sim-0.9.19-cp313-cp313-linux_x86_64.whl",
        "mindie-sim",
        "0.9.19",
    )
    with pytest.raises(ValueError, match="py3-none-any"):
        acw.assemble_wheelhouse(
            tmp_path / "out-plat",
            candidate_repo="mindie-agent/sim",
            candidate_wheel=plat_whl,
            candidate_revision="c" * 40,
            download_fn=dl_fn,
            tag_fn=tag_fn,
            latest_release_fn=latest_fn,
            reme_builder=reme_fn,
        )

    # 3. Candidate mindie-knowledge wheel with drifted or missing ReMe pin is rejected
    for idx, reqs in enumerate(
        (
            (),
            ("reme-ai>=0.2.0",),
            ("reme-ai[as] @ git+https://github.com/agentscope-ai/ReMe.git@main",),
        )
    ):
        bad_know = _make_wheel(
            tmp_path / f"bad-know-{idx}" / "mindie_knowledge-0.13.25-py3-none-any.whl",
            "mindie-knowledge",
            "0.13.25",
            requires=reqs,
        )
        with pytest.raises(ValueError, match="ReMe pin drift"):
            acw.assemble_wheelhouse(
                tmp_path / f"out-know-{idx}",
                candidate_repo="mindie-agent/knowledge",
                candidate_wheel=bad_know,
                candidate_revision="d" * 40,
                download_fn=dl_fn,
                tag_fn=tag_fn,
                latest_release_fn=latest_fn,
                reme_builder=reme_fn,
            )

    # 4. Draft/prerelease latest release is rejected when --channel is omitted
    with pytest.raises(ValueError, match="draft or prerelease"):
        acw.assemble_wheelhouse(
            tmp_path / "out-draft",
            download_fn=dl_fn,
            tag_fn=tag_fn,
            latest_release_fn=lambda _repo: {"tag_name": "v1.2.0", "draft": True, "prerelease": False},
            reme_builder=reme_fn,
        )


def _synthetic_harness_cli_source(
    *, version: str, call_git_on_update: bool = False, selftest_mode: str = "pass"
) -> dict[str, str]:
    cli_code = f"""from __future__ import annotations
import hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

FLAGS = {{"creationflags": subprocess.CREATE_NO_WINDOW}} if os.name == "nt" else {{}}

def _stage_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    return Path(uri)

def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print("subcommands: update" + ("" if {selftest_mode!r} == "absent" else " selftest"))
        return 0
    cfg_idx = args.index("--config")
    cfg_path = Path(args[cfg_idx + 1])
    data_idx = args.index("--data-dir")
    data_dir = Path(args[data_idx + 1])
    if "update" in args:
        if {call_git_on_update!r}:
            proc = subprocess.run(
                ["git", "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, **FLAGS
            )
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr or f"git exited {{proc.returncode}}\\n")
                return proc.returncode
        url_idx = args.index("--channel-base-url")
        stage = _stage_path(args[url_idx + 1])
        ch_bytes = (stage / "release-channel.json").read_bytes()
        ch = json.loads(ch_bytes.decode("utf-8"))
        gen_key = hashlib.sha256(ch_bytes).hexdigest()[:12]
        wh_dir = data_dir / f"wh-{{gen_key}}"
        wh_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(stage / "mindie-wheels-bundle.tar.gz", "r:gz") as tar:
            tar.extractall(wh_dir, filter="data")
        gen_venv = data_dir / "generations" / gen_key / ".venv"
        subprocess.run(
            ["uv", "--no-config", "venv", "--python", sys.executable, str(gen_venv)],
            stdin=subprocess.DEVNULL, capture_output=True, check=True, **FLAGS,
        )
        vpy = gen_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        whls = sorted(str(p) for p in wh_dir.glob("*.whl"))
        subprocess.run(
            ["uv", "--no-config", "pip", "install", "--python", str(vpy), "--offline", "--no-index", "--no-deps", *whls],
            stdin=subprocess.DEVNULL, capture_output=True, check=True, **FLAGS,
        )
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            json.dumps({{"python": str(vpy), "version": {version!r}, "channel": ch["current_patch_version"]}}),
            encoding="utf-8",
        )
        print(json.dumps({{"status": "active", "python": str(vpy), "version": {version!r}}}))
        return 0
    if "selftest" in args and {selftest_mode!r} != "absent":
        git_chk = subprocess.run(
            ["git", "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, **FLAGS
        )
        if git_chk.returncode != 0:
            sys.stderr.write("git failed inside selftest\\n")
            return 3
        if {selftest_mode!r} == "exit1":
            sys.stderr.write("simulated selftest crash\\n")
            return 1
        status = "passed" if {selftest_mode!r} == "pass" else "failed"
        print(json.dumps({{"schema": "mindie-selftest/1", "mode": "full", "status": status, "checks": []}}))
        return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
"""
    files = {
        "mindie_harness/__init__.py": f"__version__ = {version!r}\n",
        "mindie_harness/cli.py": cli_code,
    }
    if selftest_mode != "absent":
        files["mindie_harness/selftest.py"] = "SUPPORTED = True\n"
    return files


@pytest.mark.tier1
def test_wheelhouse_upgrade_git_trap_and_selftest_behavioral_contract(tmp_path: Path) -> None:
    uv_bin = which_native("uv")
    if not uv_bin:
        pytest.skip("uv not available on host")

    # 1. Direct git trap verification: child process resolving `git` on PATH exits 99
    trap_dir = tmp_path / "direct-trap"
    upg.write_git_trap(trap_dir)
    trap_env = {**os.environ, "PATH": f"{trap_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    trapped = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import subprocess, sys; r = subprocess.run(['git', 'status'], capture_output=True, text=True); "
            "sys.stderr.write(r.stderr); raise SystemExit(r.returncode)",
        ],
        capture_output=True,
        text=True,
        env=trap_env,
        check=False,
        timeout=10,
        **subprocess_window_kwargs(),
    )
    assert trapped.returncode == 99
    assert "git trap" in trapped.stderr

    # 2. Assemble baseline (0.1.4 without selftest) and candidate (0.1.5 with selftest) stages
    _revs, _ch, dl_fn, tag_fn, latest_fn, reme_fn = _sibling_fixtures("1.0.0")
    base_har = _make_wheel(
        tmp_path / "har-014" / "mindie_harness-0.1.4-py3-none-any.whl",
        "mindie-harness",
        "0.1.4",
        files=_synthetic_harness_cli_source(version="0.1.4", selftest_mode="absent"),
    )
    cand_har = _make_wheel(
        tmp_path / "har-015" / "mindie_harness-0.1.5-py3-none-any.whl",
        "mindie-harness",
        "0.1.5",
        files=_synthetic_harness_cli_source(version="0.1.5", selftest_mode="pass"),
    )
    base_stage = acw.assemble_wheelhouse(
        tmp_path / "wh-base",
        candidate_repo="mindie-agent/harness",
        candidate_wheel=base_har,
        candidate_revision="1" * 40,
        download_fn=dl_fn,
        tag_fn=tag_fn,
        latest_release_fn=latest_fn,
        reme_builder=reme_fn,
    )
    cand_stage = acw.assemble_wheelhouse(
        tmp_path / "wh-cand",
        candidate_repo="mindie-agent/harness",
        candidate_wheel=cand_har,
        candidate_revision="2" * 40,
        download_fn=dl_fn,
        tag_fn=tag_fn,
        latest_release_fn=latest_fn,
        reme_builder=reme_fn,
    )

    res = upg.verify_wheelhouse_upgrade_and_selftest(
        baseline_stage=base_stage,
        candidate_stage=cand_stage,
        work_dir=tmp_path / "run-pass",
        repository="mindie-agent/harness",
        offline=True,
        uv_bin=uv_bin,
    )
    assert res["wheelhouse_upgrade_verified"] is True
    assert res["selftest_verified"] is True
    assert res["selftest"]["status"] == "passed"

    # 3. Fail-closed when update attempts to call git (caught by exit 99 git trap)
    git_leaky_har = _make_wheel(
        tmp_path / "har-git-leak" / "mindie_harness-0.1.5-py3-none-any.whl",
        "mindie-harness",
        "0.1.5",
        files=_synthetic_harness_cli_source(version="0.1.5", call_git_on_update=True),
    )
    leaky_stage = acw.assemble_wheelhouse(
        tmp_path / "wh-leaky",
        candidate_repo="mindie-agent/harness",
        candidate_wheel=git_leaky_har,
        candidate_revision="3" * 40,
        download_fn=dl_fn,
        tag_fn=tag_fn,
        latest_release_fn=latest_fn,
        reme_builder=reme_fn,
    )
    with pytest.raises(RuntimeError, match="git trap"):
        upg.verify_wheelhouse_upgrade_and_selftest(
            candidate_stage=leaky_stage,
            work_dir=tmp_path / "run-leaky",
            repository="mindie-agent/harness",
            offline=True,
            uv_bin=uv_bin,
        )

    # 4. Fail-closed when selftest returns status="failed" or non-zero exit
    for mode, exc_type in (("fail_json", ValueError), ("exit1", RuntimeError)):
        bad_har = _make_wheel(
            tmp_path / f"har-{mode}" / "mindie_harness-0.1.5-py3-none-any.whl",
            "mindie-harness",
            "0.1.5",
            files=_synthetic_harness_cli_source(version="0.1.5", selftest_mode=mode),
        )
        bad_stage = acw.assemble_wheelhouse(
            tmp_path / f"wh-{mode}",
            candidate_repo="mindie-agent/harness",
            candidate_wheel=bad_har,
            candidate_revision="4" * 40,
            download_fn=dl_fn,
            tag_fn=tag_fn,
            latest_release_fn=latest_fn,
            reme_builder=reme_fn,
        )
        with pytest.raises(exc_type, match="selftest"):
            upg.verify_wheelhouse_upgrade_and_selftest(
                baseline_stage=base_stage,
                candidate_stage=bad_stage,
                work_dir=tmp_path / f"run-{mode}",
                repository="mindie-agent/harness",
                offline=True,
                uv_bin=uv_bin,
            )


@pytest.mark.tier0
def test_consumer_contract_validates_usability_and_native_mcp_tool_calls() -> None:
    rev = "e" * 40
    base_files = {
        "benchmark-results.json": json.dumps(
            {
                "schema_version": "mindie-release-benchmark/2",
                "repository": "mindie-agent/mindie-agent-codex",
                "version": "0.1.40",
                "release_gate": True,
                "blocking_reasons": [],
                "source": {"revision": rev, "dirty": False},
                "source_after": {"revision": rev, "dirty": False},
            }
        ).encode("utf-8"),
        "benchmark-results.md": b"# bench\n",
        "benchmark-SHA256SUMS": b"sums\n",
        "upgrade-acceptance.json": json.dumps(
            {
                "schema": "mindie-upgrade-acceptance/1",
                "status": "passed",
                "repository": "mindie-agent/mindie-agent-codex",
                "version": "0.1.40",
                "revision": rev,
                "scope": "all_shared_components",
                "consumer_revision": "3c478a19b5542b38520c7f98b87a07f37e305021",
                "installed_revision": rev,
            }
        ).encode("utf-8"),
        "usability-acceptance.json": json.dumps(
            {
                "schema": "mindie-usability-acceptance/1",
                "status": "passed",
                "repository": "mindie-agent/mindie-agent-codex",
                "version": "0.1.40",
                "revision": rev,
            }
        ).encode("utf-8"),
        "native-upgrade-acceptance.json": json.dumps(
            {
                "schema": "mindie-native-upgrade-acceptance/1",
                "status": "passed",
                "repository": "mindie-agent/mindie-agent-codex",
                "version": "0.1.40",
                "revision": rev,
                "scope": "native_codex_old_client",
                "baseline_plugin": "0.1.23",
                "external_components_rebound": True,
                "disabled_stop_preserved": True,
                "stale_snapshot_rejected": True,
                "native_startup_recovered": True,
                "mcp_tool_calls_verified": True,
            }
        ).encode("utf-8"),
    }

    def run_validate(overrides: dict[str, bytes]):
        files = {**base_files, **overrides}
        sums = "".join(
            f"{hashlib.sha256(raw).hexdigest()}  {name}\n" for name, raw in sorted(files.items())
        ).encode("utf-8")
        files["SHA256SUMS"] = sums
        rows = [{"name": k, "size": len(v)} for k, v in files.items()]
        return contract.validate_evidence(
            "mindie-agent/mindie-agent-codex", "0.1.40", rev, rows, lambda n: files[n]
        )

    assert "usability-acceptance.json" in run_validate({})

    # Missing mcp_tool_calls_verified fails closed
    bad_native = json.loads(base_files["native-upgrade-acceptance.json"])
    bad_native["mcp_tool_calls_verified"] = False
    with pytest.raises(ValueError, match="native old-client upgrade evidence"):
        run_validate({"native-upgrade-acceptance.json": json.dumps(bad_native).encode("utf-8")})

    # Failed usability-acceptance.json fails closed
    bad_usab = json.loads(base_files["usability-acceptance.json"])
    bad_usab["status"] = "failed"
    with pytest.raises(ValueError, match="usability gate evidence"):
        run_validate({"usability-acceptance.json": json.dumps(bad_usab).encode("utf-8")})


@pytest.mark.tier0
def test_reusable_workflows_and_release_controller_contracts() -> None:
    # 1. Version & lock synchronization at 0.4.20
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "0.4.20"
    uv_lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    bench_pkg = next(p for p in uv_lock["package"] if p["name"] == "mindie-bench")
    assert bench_pkg["version"] == "0.4.20"
    rel_wf = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert rel_wf["jobs"]["release"]["with"]["release-notes-path"] == "docs/releases/mindie-bench-0.4.20.md"

    # 2. fullstack-smoke.yml contract
    fs_wf = yaml.safe_load((ROOT / ".github/workflows/fullstack-smoke.yml").read_text(encoding="utf-8"))
    fs_inputs = fs_wf[True]["workflow_call"]["inputs"]
    assert fs_inputs["python-version"]["required"] is False
    assert fs_inputs["python-version"]["default"] == "3.13"
    fs_job = fs_wf["jobs"]["fullstack-smoke"]
    assert fs_job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    fs_runs = "\n".join(s.get("run", "") for s in fs_job["steps"])
    assert "assemble_candidate_wheelhouse.py" in fs_runs
    assert "--wheelhouse-only" in fs_runs

    # 3. verify-host-plugin.yml contract
    vh_wf = yaml.safe_load((ROOT / ".github/workflows/verify-host-plugin.yml").read_text(encoding="utf-8"))
    vh_inputs = vh_wf[True]["workflow_call"]["inputs"]
    assert vh_inputs["host"]["required"] is True
    vh_job = vh_wf["jobs"]["verify-host-plugin"]
    assert vh_job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    node_step = next(s for s in vh_job["steps"] if str(s.get("uses", "")).startswith("actions/setup-node@"))
    assert "inputs.host == 'dsh'" in node_step["if"] and "inputs.host == 'oc'" in node_step["if"]
    for step in vh_job["steps"]:
        assert "inputs.host" not in step.get("run", "")
    vh_runs = "\n".join(s.get("run", "") for s in vh_job["steps"])
    assert "assemble_candidate_wheelhouse.py" in vh_runs
    assert "mindie-bench host-accept" in vh_runs

    # 4. publish-release.yml usability-gate & @openai/codex@0.169.0 contract
    pr_wf = yaml.safe_load((ROOT / ".github/workflows/publish-release.yml").read_text(encoding="utf-8"))
    jobs = pr_wf["jobs"]
    assert "usability-gate" in jobs
    ug = jobs["usability-gate"]
    assert ug["needs"] == ["candidate"]
    assert ug["permissions"] == {"contents": "read"}
    assert ug["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    pub = jobs["publish"]
    assert pub["needs"] == ["candidate", "benchmark", "usability-gate"]
    pub_runs = "\n".join(s.get("run", "") for s in pub["steps"])
    assert "usability-acceptance.json" in pub_runs
    assert "--baseline-stage" in pub_runs and "--candidate-stage" in pub_runs
    assert "@openai/codex@0.169.0" in pub_runs
    assert pub_runs.index("usability-acceptance.json") < pub_runs.index("publish_release.py")
