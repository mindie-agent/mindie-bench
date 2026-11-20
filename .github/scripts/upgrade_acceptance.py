"""[Role] Run the released component updater and zero-Git wheelhouse upgrade gate.

[Invariants] Discovery, uv generation construction and inventory verification use
the unmodified published consumer, plus zero-Git `wheelhouse.update_runtime`
(`baseline -> candidate` with a fail-closed `git` `exit 99` trap on `PATH`) and
`mindie-harness selftest --mode full` for shared runtime packages.
[Timeouts & Locks] One 1200s workflow step encloses 210s baseline setup, 120s
discovery, 600s generation build, 300s wheelhouse cutover, and 90s full selftest.
[Side-Effect Policy] Creates only the supplied isolated acceptance directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import quote

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from build_wheels_bundle import SHARED_PACKAGES  # noqa: E402
from consumer_contract import BIRTH_COMPONENTS, load_consumer, validate_release  # noqa: E402
from release_transport import GitHub  # noqa: E402

MODULES = {
    "mindie-harness": "mindie_harness",
    "mindie-knowledge": "mindie_knowledge.loop.cli",
    "remote-dev": "remote_dev",
    "mindie-diagnostics": "mindie_diagnostics.cli",
    "mindie-debugger": "mindie_debugger.cli",
    "mindie-profiler": "mindie_profiler",
    "mindie-sim": "mindie_sim",
    "npu-top": "npu_top",
    "mindie-coordinator": "mindie_coordinator.cli",
    "mindie-bench": "mindie_bench.cli",
}
_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def run(argv, *, timeout, cwd=None, env=None):
    result = subprocess.run(
        [str(value) for value in argv], stdin=subprocess.DEVNULL, capture_output=True,
        timeout=timeout, cwd=cwd, env=env, check=False, **_FLAGS,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")[-6000:]
        raise RuntimeError(detail or f"{argv[0]} exited {result.returncode}")
    return result.stdout


def baseline_requirement(repository, entry):
    url = f"git+https://github.com/{repository}@{entry['revision']}"
    project = entry.get("project", ".")
    if project != ".":
        url += "#subdirectory=" + quote(project, safe="/")
    return f"{entry['package']} @ {url}"


def baseline_install_command(python, overrides, requirements):
    # The candidate's uv settings must not replace the published baseline pins.
    return ["uv", "--no-config", "pip", "install", "--python", python,
            "--override", overrides, *requirements]


def _with_birth_overrides(text: str, source: Path, adapter_distribution: str | None) -> str:
    extra: list[str] = []
    if adapter_distribution in BIRTH_COMPONENTS:
        extra.append(f"{adapter_distribution} @ {Path(source).resolve().as_uri()}")
    elif adapter_distribution and (Path(source) / "pyproject.toml").is_file():
        proj = tomllib.loads((Path(source) / "pyproject.toml").read_text(encoding="utf-8"))
        uv_cfg = (proj.get("tool") or {}).get("uv") or {}
        for item in uv_cfg.get("override-dependencies") or ():
            if str(item).split("@", 1)[0].strip() in BIRTH_COMPONENTS:
                extra.append(str(item))
        if not extra:
            for item in (proj.get("project") or {}).get("dependencies") or ():
                if "@" in str(item) and str(item).split("@", 1)[0].strip() in BIRTH_COMPONENTS:
                    extra.append(str(item))
        if not extra:
            for pkg, src in (uv_cfg.get("sources") or {}).items():
                if pkg in BIRTH_COMPONENTS and isinstance(src, dict) and src.get("git") and src.get("rev"):
                    extra.append(f"{pkg} @ git+{src['git']}@{src['rev']}")
    if not extra:
        return text
    parsed = tomllib.loads(text)
    current = list(((parsed.get("tool") or {}).get("uv") or {}).get("override-dependencies") or ())
    seen = {row.split("@", 1)[0].strip() for row in current}
    merged = current + [row for row in extra if row.split("@", 1)[0].strip() not in seen]
    if merged == current:
        return text
    return text.replace(
        "override-dependencies = " + json.dumps(current),
        "override-dependencies = " + json.dumps(merged),
        1,
    )


def write_git_trap(trap_dir: Path) -> Path:
    trap_dir = Path(trap_dir)
    trap_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        target = trap_dir / "git"
        target.write_text(
            "#!/bin/sh\necho 'git trap: unexpected git invocation' >&2\nexit 99\n",
            encoding="utf-8",
            newline="\n",
        )
        target.chmod(0o700)
        return target
    cmd_body = "@echo git trap: unexpected git invocation 1>&2\n@exit /b 99\n"
    for suffix in (".cmd", ".bat"):
        (trap_dir / f"git{suffix}").write_text(cmd_body, encoding="utf-8", newline="\n")
    try:
        import ensurepip

        bundled = Path(ensurepip.__file__).resolve().parent / "_bundled"
        wheel = next(bundled.glob("pip-*.whl"))
        arch = platform.machine().lower()
        stub_name = "t64-arm.exe" if arch in {"arm64", "aarch64"} else "t64.exe"
        with zipfile.ZipFile(wheel) as zf:
            stub = zf.read(f"pip/_vendor/distlib/{stub_name}")
        base_py = Path(sys.base_prefix).resolve() / "python.exe"
        if not base_py.is_file():
            base_py = Path(sys.executable).resolve()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            info = zipfile.ZipInfo("__main__.py", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(
                info,
                b"import sys; sys.stderr.write('git trap: unexpected git invocation\\n'); raise SystemExit(99)\n",
            )
        exe_path = trap_dir / "git.exe"
        exe_path.write_bytes(stub + f'#!"{base_py}" -I -X utf8\r\n'.encode("utf-8") + buf.getvalue())
        return exe_path
    except TimeoutError:
        raise
    except OSError:
        return trap_dir / "git.cmd"


def _sandbox_env(work_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for secret_key in ("GH_TOKEN", "GITHUB_TOKEN", "MINDIE_RELEASE_DEPLOY_KEY", "GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH"):
        env.pop(secret_key, None)
    for name in (
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
        "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT",
    ):
        sub = work_dir / "sandbox" / name
        sub.mkdir(parents=True, exist_ok=True)
        env[name] = str(sub)
    uv_cache = Path(
        env.get("UV_CACHE_DIR")
        or (Path(env["RUNNER_TEMP"]) / "uv-cache" if env.get("RUNNER_TEMP") else work_dir / "uv-cache")
    )
    uv_cache.mkdir(parents=True, exist_ok=True)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["MINDIE_DISABLE_FSYNC"] = "1"
    diag_cfg = work_dir / "diagnostics-disabled.json"
    diag_cfg.write_text(
        json.dumps({"schema": "mindie.diagnostics.reporting.v1", "decision": "disabled"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    env["MINDIE_DIAGNOSTICS_CONFIG"] = str(diag_cfg)
    for key in ("MINDIE_AGENT_CONFIG", "MINDIE_CHANNEL_BASE_URL", "MINDIE_GITHUB_BASE_URL"):
        env.pop(key, None)
    return env


def _bootstrap_harness_from_stage(
    stage_dir: Path, boot_dir: Path, env: dict[str, str], *, uv_bin: str = "uv"
) -> Path:
    bundle_path = Path(stage_dir) / "mindie-wheels-bundle.tar.gz"
    if not bundle_path.is_file():
        raise ValueError(f"wheelhouse stage is missing mindie-wheels-bundle.tar.gz: {stage_dir}")
    wheels_dir = boot_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    harness_whl: Path | None = None
    with tarfile.open(bundle_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name == member.name and name.startswith("mindie_harness-") and name.endswith(".whl"):
                extracted = tar.extractfile(member)
                if extracted is not None:
                    target = wheels_dir / name
                    target.write_bytes(extracted.read())
                    harness_whl = target
                    break
    if harness_whl is None:
        raise ValueError(f"mindie-harness wheel missing from {bundle_path}")
    venv_dir = boot_dir / "venv"
    run([uv_bin, "--no-config", "venv", "--python", sys.executable, venv_dir], timeout=60, env=env)
    boot_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(
        [uv_bin, "--no-config", "pip", "install", "--python", boot_py,
         "--offline", "--no-index", "--no-deps", harness_whl],
        timeout=60, env=env,
    )
    return boot_py


def _stage_component_pins(stage_dir: Path) -> dict[str, tuple[str, str]]:
    doc = json.loads((Path(stage_dir) / "release-channel.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, str]] = {}
    for key, meta in (doc.get("components") or {}).items():
        if isinstance(meta, dict):
            role = str(meta.get("role") or "")
            if role and role != "shared_component":
                continue
            pkg = str(meta.get("package") or key)
            out[pkg] = (str(meta.get("version") or ""), str(meta.get("revision") or meta.get("commit") or ""))
    return out


def _supports_selftest(candidate_python: Path, env: dict[str, str]) -> bool:
    probe = subprocess.run(
        [
            str(candidate_python), "-I", "-X", "utf8", "-c",
            "import importlib.util, json; print(json.dumps(bool(importlib.util.find_spec('mindie_harness.selftest'))))",
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=15, check=False, env=env, **_FLAGS,
    )
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        return True
    help_proc = subprocess.run(
        [str(candidate_python), "-I", "-X", "utf8", "-m", "mindie_harness.cli", "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=15, check=False, env=env, **_FLAGS,
    )
    return help_proc.returncode == 0 and "selftest" in (help_proc.stdout or "")


def verify_wheelhouse_upgrade_and_selftest(
    *,
    candidate_stage: Path,
    work_dir: Path,
    baseline_stage: Path | None = None,
    repository: str = "",
    offline: bool = False,
    uv_bin: str = "uv",
    update_timeout: float = 300.0,
) -> dict:
    candidate_stage = Path(candidate_stage).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    base_env = _sandbox_env(work_dir)
    trap_dir = work_dir / "git-trap"
    write_git_trap(trap_dir)
    trap_env = {**base_env, "PATH": f"{trap_dir}{os.pathsep}{base_env.get('PATH', '')}"}
    cfg_path = Path(base_env["MINDIE_CONFIG_HOME"]) / "codex.json"
    data_dir = Path(base_env["MINDIE_DATA_HOME"])
    offline_flags = ["--offline"] if offline else []

    baseline_python: Path | None = None
    if baseline_stage is not None:
        baseline_stage = Path(baseline_stage).resolve()
        boot_base_py = _bootstrap_harness_from_stage(
            baseline_stage, work_dir / "boot-baseline", trap_env, uv_bin=uv_bin
        )
        raw_base = run(
            [
                boot_base_py, "-I", "-X", "utf8", "-m", "mindie_harness.cli",
                "--host", "codex", "--config", cfg_path, "--data-dir", data_dir,
                "update", "--force", *offline_flags, "--channel-base-url", baseline_stage.as_uri(),
            ],
            timeout=update_timeout, env=trap_env,
        )
        base_res = json.loads(raw_base.decode("utf-8"))
        if base_res.get("status") not in {"active", "up_to_date"} or not Path(str(base_res.get("python") or "")).is_file():
            raise ValueError(f"baseline wheelhouse update_runtime failed: {base_res}")
        baseline_python = Path(base_res["python"])

    updater_py = _bootstrap_harness_from_stage(
        candidate_stage, work_dir / "boot-candidate", trap_env, uv_bin=uv_bin
    )
    raw_cand = run(
        [
            updater_py, "-I", "-X", "utf8", "-m", "mindie_harness.cli",
            "--host", "codex", "--config", cfg_path, "--data-dir", data_dir,
            "update", "--force", *offline_flags, "--channel-base-url", candidate_stage.as_uri(),
        ],
        timeout=update_timeout, env=trap_env,
    )
    cand_res = json.loads(raw_cand.decode("utf-8"))
    expected_statuses = {"active", "up_to_date"}
    if baseline_stage is not None and _stage_component_pins(baseline_stage) != _stage_component_pins(candidate_stage):
        expected_statuses = {"active"}
    if cand_res.get("status") not in expected_statuses or not Path(str(cand_res.get("python") or "")).is_file():
        raise ValueError(f"candidate wheelhouse update_runtime failed: {cand_res}")
    candidate_python = Path(cand_res["python"])

    has_selftest = _supports_selftest(candidate_python, base_env)
    if repository == "mindie-agent/harness" and not has_selftest:
        raise ValueError("candidate mindie-harness is missing required `selftest` subcommand")
    selftest_report = None
    if has_selftest:
        proc = subprocess.run(
            [
                str(candidate_python), "-I", "-X", "utf8", "-m", "mindie_harness.cli",
                "--host", "codex", "--config", str(cfg_path), "--data-dir", str(data_dir),
                "selftest", "--mode", "full", "--json",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=90, check=False, env=base_env, **_FLAGS,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mindie-harness selftest --mode full failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-4000:]}"
            )
        selftest_report = json.loads(proc.stdout)
        if selftest_report.get("status") != "passed":
            raise ValueError(f"mindie-harness selftest --mode full did not pass: {selftest_report}")
    return {
        "wheelhouse_upgrade_verified": True,
        "selftest_verified": bool(has_selftest),
        "selftest": selftest_report,
        "baseline_python": str(baseline_python) if baseline_python else None,
        "candidate_python": str(candidate_python),
        "config_path": str(cfg_path),
    }


def accept(args):
    if bool(args.baseline_stage) != bool(args.candidate_stage):
        raise ValueError("must provide both --baseline-stage and --candidate-stage, or neither")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("HOME", "USERPROFILE", "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME",
                 "MINDIE_DIAGNOSTICS_ROOT", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                 "XDG_STATE_HOME"):
        path = root / "profile" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    uv_cache = Path(
        os.environ.get("UV_CACHE_DIR")
        or (Path(os.environ["RUNNER_TEMP"]) / "uv-cache"
            if os.environ.get("RUNNER_TEMP") else root / "uv-cache")
    )
    uv_cache.mkdir(parents=True, exist_ok=True)
    os.environ["UV_CACHE_DIR"] = str(uv_cache)
    os.environ["MINDIE_DIAGNOSTICS_CONFIG"] = str(root / "diagnostics-disabled.json")
    os.environ["MINDIE_DISABLE_FSYNC"] = "1"
    (root / "diagnostics-disabled.json").write_text(json.dumps({
        "schema": "mindie.diagnostics.reporting.v1", "purpose": "tool_fault_reporting",
        "decision": "disabled", "repository": args.repository,
    }), encoding="utf-8")
    Catalog, components = load_consumer(args.consumer)
    from mindie_knowledge.component_runtime import (
        build,
        discover,
        read_inventory,
        source_identity,
    )

    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    entry = policy["repositories"][args.repository]
    package = entry["package"]
    source = Path(args.source).resolve()
    api = GitHub(args.repository)
    assets = Path(args.assets).resolve() if args.assets else None
    release = dict(tag_name="v" + args.version, draft=False, prerelease=False,
                   assets=[dict(name=path.name, size=path.stat().st_size)
                           for path in assets.iterdir() if path.is_file()
                           and not path.name.startswith(".")]) if assets else None

    class CandidateCatalog(Catalog):
        def read(self, path, *, force=False):
            if release is not None and path == f"/{api.prefix}/releases/latest":
                return release
            if release is not None and path == f"/{api.prefix}/git/ref/tags/v{args.version}":
                return {"object": {"type": "commit", "sha": args.revision}}
            return super().read(path, force=force)

        def asset(self, name, tag, filename):
            if assets is not None and components[name] == args.repository:
                return (assets / filename).read_bytes()
            return super().asset(name, tag, filename)

    current = root / "current"
    run(["uv", "--no-config", "venv", "--python", sys.executable, current], timeout=30)
    python = current / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    overrides = root / "baseline-overrides.txt"
    overrides.write_text("".join(
        baseline_requirement(repo, row) + "\n"
        for repo, row in policy["repositories"].items()
        if (row["package"] in components or row["package"] in BIRTH_COMPONENTS) and row.get("revision")),
        encoding="utf-8", newline="\n")
    native_tree = (
        entry.get("kind") == "native_tree"
        or (package not in components
            and not (source / entry.get("project", ".") / "pyproject.toml").is_file())
    )
    birth_release = entry.get("revision") is None and entry.get("prior_revision") is None
    requirements = (
        [] if (native_tree or birth_release)
        else [baseline_requirement(args.repository, entry)]
    )
    requirements.extend(
        baseline_requirement(repo, row)
        for repo, row in policy["repositories"].items()
        if row["package"] in components
        and row.get("revision")
        and row["package"] not in {"mindie-bench", package})
    run(baseline_install_command(python, overrides, requirements), timeout=180)
    installed = read_inventory(python)
    required = {
        name for name, repo in components.items()
        if name != "mindie-bench"
        and policy["repositories"].get(repo, {}).get("revision") is not None
    }
    if package in components and not birth_release:
        required.add(package)
    if not required <= installed.keys():
        raise ValueError("the released baseline omits shared components: "
                         + ", ".join(sorted(required - installed.keys())))
    catalog = CandidateCatalog(cache=root / "release-cache", budget=120)
    selected = discover(python, {}, catalog=catalog, force=True)
    if args.repository in BIRTH_COMPONENTS.values():
        if assets is not None and release is not None:
            validate_release(
                args.consumer, args.repository, args.version, args.revision,
                release, lambda filename: (assets / filename).read_bytes(),
                lambda path: (
                    {"object": {"type": "commit", "sha": args.revision}}
                    if path == f"{api.prefix}/git/ref/tags/v{args.version}"
                    else api.api(path)
                ),
            )
        else:
            pub_release = api.release("v" + args.version)
            if pub_release is None:
                raise ValueError(f"birth component release is not published: {args.repository} v{args.version}")
            pub_assets = {row["name"]: row for row in pub_release["assets"]}
            validate_release(
                args.consumer, args.repository, args.version, args.revision,
                pub_release, lambda filename: api.asset(pub_assets[filename]), api.api,
            )
    distribution = package if package not in components and not native_tree else None
    if package in components:
        if selected["runtime"][package]["revision"] != args.revision:
            raise ValueError("old updater did not select the candidate as latest")
    if native_tree:
        seed = source
    elif distribution is None:
        seed = root / "adapter-source"
        seed.mkdir(exist_ok=True)
        (seed / "pyproject.toml").write_text(
            '[project]\nname="mindie-release-upgrade-probe"\nversion="0.0.0"\n',
            encoding="utf-8", newline="\n")
    else:
        seed = source / entry.get("project", ".")
    import mindie_knowledge.component_runtime as cr

    baseline = dict(source_sha256=source_identity(seed),
                    runtime={name: row["revision"] for name, row in installed.items()})
    orig_project_text = cr.project_text
    cr.project_text = lambda src, targets, **kw: _with_birth_overrides(
        orig_project_text(src, targets, **kw), Path(src), kw.get("adapter_distribution")
    )
    try:
        artifact = build(seed, root / "generations", baseline, selected["runtime"],
                         python=python, adapter_distribution=distribution, timeout=600)
    finally:
        cr.project_text = orig_project_text
    actual = read_inventory(artifact["python"])
    if actual != selected["runtime"]:
        raise ValueError("prepared runtime differs from the old client's latest selection")
    for name in actual:
        if name in MODULES:
            run([artifact["python"], "-I", "-m", MODULES[name], "--help"],
                timeout=20, cwd=root)
    if distribution:
        metadata = json.loads(run([
            artifact["python"], "-I", "-c",
            "import importlib.metadata as m,json,sys;"
            "d=m.distribution(sys.argv[1]);"
            "print(json.dumps({'version':d.version,'direct':json.loads(d.read_text('direct_url.json'))}))",
            distribution], timeout=15))
        if metadata["version"] != args.version:
            raise ValueError("prepared adapter version differs from the candidate")
        project = tomllib.loads((seed / "pyproject.toml").read_text(encoding="utf-8"))
        if project["project"]["version"] != args.version:
            raise ValueError("candidate project version differs")
        if distribution in BIRTH_COMPONENTS:
            run([artifact["python"], "-I", "-c", f"import {MODULES[distribution]}"], timeout=15, cwd=root)
    consumer_revision = run(["git", "-C", args.consumer, "rev-parse", "HEAD"],
                            timeout=10).decode().strip()
    if consumer_revision != policy["consumer_revision"]:
        raise ValueError("upgrade evidence did not use the recorded published consumer")
    report = dict(
        schema="mindie-upgrade-acceptance/1", status="passed",
        repository=args.repository, version=args.version, revision=args.revision,
        consumer_revision=consumer_revision,
        consumer_validator_sha256=hashlib.sha256(
            (Path(args.consumer) / "mindie_knowledge/component_releases.py").read_bytes()).hexdigest(),
        scope="all_shared_components", transport="staged_candidate" if assets else "public_release",
        baseline=installed, selected=selected["runtime"], installed=actual,
        installed_revision=args.revision,
        generation=Path(artifact["generation"]).name,
    )
    if package in SHARED_PACKAGES and args.baseline_stage and args.candidate_stage:
        wh = verify_wheelhouse_upgrade_and_selftest(
            baseline_stage=Path(args.baseline_stage),
            candidate_stage=Path(args.candidate_stage),
            work_dir=root / "wheelhouse-upgrade",
            repository=args.repository,
        )
        report["wheelhouse_upgrade_verified"] = True
        report["selftest_verified"] = wh["selftest_verified"]
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    for name in ("repository", "root", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("version", "revision", "consumer", "policy", "source",
                 "assets", "baseline-stage", "candidate-stage"):
        parser.add_argument("--" + name, default=None)
    args = parser.parse_args(argv)
    if args.wheelhouse_only:
        if not args.candidate_stage:
            parser.error("--wheelhouse-only requires --candidate-stage")
        result = verify_wheelhouse_upgrade_and_selftest(
            baseline_stage=Path(args.baseline_stage) if args.baseline_stage else None,
            candidate_stage=Path(args.candidate_stage),
            work_dir=Path(args.root),
            repository=args.repository,
            offline=args.offline,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"status": "passed", **result}, sort_keys=True, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
        return
    for req in ("version", "revision", "consumer", "policy", "source"):
        if not getattr(args, req):
            parser.error(f"the following arguments are required: --{req}")
    accept(args)


if __name__ == "__main__":
    main()
