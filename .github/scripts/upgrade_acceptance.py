"""[Role] Run the released component updater against a staged or public candidate.

[Invariants] Discovery, uv generation construction and inventory verification use
the unmodified published consumer. No mocked selector, in-place install or user
profile participates. The only prospective transport is the explicitly staged release.
[Timeouts & Locks] One 1200s workflow step encloses 210s baseline setup, 120s
discovery, 600s generation build, nine 20s CLI probes and readback/cleanup.
[Side-Effect Policy] Creates only the supplied isolated acceptance directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

from consumer_contract import BIRTH_COMPONENTS, load_consumer, validate_release
from release_transport import GitHub

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


def run(argv, *, timeout, cwd=None):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run([str(value) for value in argv], capture_output=True,
                            timeout=timeout, cwd=cwd, check=False, **flags)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-6000:])
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


def accept(args):
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
        if row["package"] in components and row.get("revision")),
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
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "version", "revision", "consumer", "policy",
                 "source", "root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--assets")
    accept(parser.parse_args())


if __name__ == "__main__":
    main()
