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

from consumer_contract import load_consumer
from release_transport import GitHub

MODULES = {
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


def accept(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("HOME", "USERPROFILE", "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME",
                 "MINDIE_DIAGNOSTICS_ROOT", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                 "XDG_STATE_HOME"):
        path = root / "profile" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
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
    run(["uv", "venv", "--python", sys.executable, current], timeout=30)
    python = current / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    overrides = root / "baseline-overrides.txt"
    overrides.write_text("".join(
        f"{row['package']} @ git+https://github.com/{repo}@{row['revision']}\n"
        for repo, row in policy["repositories"].items() if row["package"] in components),
        encoding="utf-8", newline="\n")
    native_tree = entry.get("kind") == "native_tree"
    requirements = ([] if native_tree else [
        f"{package} @ git+https://github.com/{args.repository}@{entry['revision']}"])
    requirements.extend(
        f"{row['package']} @ git+https://github.com/{repo}@{row['revision']}"
        for repo, row in policy["repositories"].items()
        if row["package"] in components and row["package"] not in {"mindie-bench", package})
    run(["uv", "pip", "install", "--python", python, "--override", overrides, *requirements],
        timeout=180)
    installed = read_inventory(python)
    catalog = CandidateCatalog(cache=root / "release-cache", budget=120)
    selected = discover(python, {}, catalog=catalog, force=True)
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
    baseline = dict(source_sha256=source_identity(seed),
                    runtime={name: row["revision"] for name, row in installed.items()})
    artifact = build(seed, root / "generations", baseline, selected["runtime"],
                     python=python, adapter_distribution=distribution, timeout=600)
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
