"""Fresh profiles and physically filtered corpus views, never personal profiles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mindie_bench.corpus import read_package
from mindie_bench.io import tree_hash, write_json
from mindie_bench.models import ExperimentConfig, Group, TaskSpec


@dataclass
class Profile:
    root: Path
    env: dict[str, str]
    corpus: Path | None
    fingerprint: dict
    runtime_paths: list[Path]


def bind_mcp_transport(profile: Profile):
    # Native hosts filter the environment inherited by MCP subprocesses.
    path = profile.root / "mcp.json"
    configured = json.loads(path.read_text())
    for server in configured["mcpServers"].values():
        server["env"].update(
            PATH=profile.env["PATH"],
            MINDIE_BENCH_GATEWAY=profile.env["MINDIE_BENCH_GATEWAY"],
        )
    write_json(path, configured)
    profile.fingerprint["mcp_config"] = configured


def filtered_corpus(source: Path, destination: Path, excluded: set[str]) -> str:
    if not (source / "tasks").is_dir():
        raise ValueError("corpus must contain tasks/")
    tree_hash(source)
    destination.mkdir()
    for package in sorted((source / "tasks").iterdir()):
        if not package.is_dir():
            raise ValueError(f"unexpected corpus entry: {package.name}")
        if package.name in excluded:
            continue
        read_package(package)
        for path in package.rglob("*"):
            if path.is_file() and path.suffix != ".md":
                raise ValueError("corpus views only accept authoritative Markdown, not cached indexes")
        shutil.copytree(package, destination / "tasks" / package.name)
    (destination / "tasks").mkdir(exist_ok=True)
    contract = source / "publication-contract.json"
    if contract.is_file():
        shutil.copy2(contract, destination / contract.name)
    return tree_hash(destination)


def build_profile(
    trial_dir: Path,
    task: TaskSpec,
    group: Group,
    config: ExperimentConfig,
    *,
    corpus_override: Path | None = None,
) -> Profile:
    root = trial_dir / "profile"
    root.mkdir(mode=0o700)
    home = root / "home"
    home.mkdir()
    directories = {
        "HOME": home,
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_STATE_HOME": root / "state",
        "CODEX_HOME": root / "codex",
        "CLAUDE_CONFIG_DIR": root / "claude",
        "KIMI_HOME": root / "kimi",
        "TMPDIR": root / "tmp",
    }
    for directory in directories.values():
        directory.mkdir(exist_ok=True)
    # Do not inherit proxy credentials, GitHub tokens, SSH agents, Python import
    # paths, plugin state, cloud credentials, or the caller's native session ID.
    env = {name: str(path) for name, path in directories.items()}
    env.update(
        PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        LANG="en_US.UTF-8",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        DISABLE_AUTOUPDATER="1",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    group_config = config.groups.get(group)
    servers = group_config.mcp_servers if group_config else {}
    components = {server.component for server in servers.values()}
    if group == Group.G0 and (servers or (group_config and group_config.skills_path)):
        raise ValueError("G0 must contain no MindIE tools or skills")
    if group != Group.G0 and "remote-dev" not in components:
        raise ValueError(f"{group} requires an explicit remote-dev MCP component")
    if group == Group.G1 and ("knowledge" in components or (group_config and group_config.skills_path)):
        raise ValueError("G1 must contain no knowledge or skills")
    if group in {Group.G2, Group.G3} and "knowledge" not in components:
        raise ValueError(f"{group} requires the real knowledge MCP component")
    excluded = set()
    corpus = None
    corpus_fingerprint = None
    corpus_commit = None
    if group in {Group.G2, Group.G3}:
        corpus_id = task.mindie_knowledge.corpus_id
        if corpus_id not in config.corpora:
            raise ValueError(f"missing explicitly selected corpus: {corpus_id}")
        declared = config.corpora[corpus_id]
        source = Path(declared.path).resolve()
        actual = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if actual != declared.commit or dirty:
            raise ValueError("corpus must be clean at its exact declared commit")
        if (
            group == Group.G2
            or task.task.role == "seed"
            or task.mindie_knowledge.knowledge_gradient != "near_exact"
        ):
            excluded.update(task.mindie_knowledge.same_origin_task_hashes)
        corpus = root / "corpus"
        corpus_fingerprint = filtered_corpus(corpus_override or source, corpus, excluded)
        corpus_commit = declared.commit
        env["MINDIE_BENCH_CORPUS"] = str(corpus)
    local_bare = root / "local_knowledge.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(local_bare)], check=True, env=env, capture_output=True
    )
    env.update(
        MINDIE_BENCH_LOCAL_KNOWLEDGE_REMOTE=str(local_bare), MINDIE_BENCH_PUBLICATION_POLICY="local-only"
    )
    runtime_paths = []
    model_catalog_sha256 = None
    if config.harness.model_catalog_path:
        catalog = Path(config.harness.model_catalog_path).resolve(strict=True)
        content = catalog.read_bytes()
        parsed = json.loads(content)
        entries = parsed["models"]
        if (
            not isinstance(entries, list)
            or sum(
                isinstance(entry, dict) and entry.get("slug") == config.harness.model
                for entry in entries
            )
            != 1
        ):
            raise ValueError("native model catalog does not contain the explicitly selected model")
        model_catalog_sha256 = hashlib.sha256(content).hexdigest()
        (root / "model-catalog.json").write_bytes(content)
    mcp = {}
    for name, server in servers.items():
        # Python discovers its virtual environment from the invoked executable
        # path. Resolving that symlink here silently selects the base runtime.
        command = Path(server.command).absolute()
        command.resolve(strict=True)
        runtime_paths.append(command)
        replacements = {
            "{corpus}": str(corpus or ""),
            "{profile}": str(root),
            "{local_remote}": str(local_bare),
        }

        def expand(value: str) -> str:
            for old, new in replacements.items():
                value = value.replace(old, new)
            return value

        mcp[name] = {
            "command": str(command),
            "args": [expand(arg) for arg in server.args],
            "env": {key: expand(value) for key, value in server.env.items()},
        }
    write_json(root / "mcp.json", {"mcpServers": mcp})
    skills_fingerprint = None
    if group_config and group_config.skills_path:
        source = Path(group_config.skills_path).resolve(strict=True)
        skills_fingerprint = tree_hash(source)
        shutil.copytree(source, root / "skills")
    write_json(root / "settings.json", {"disableAllHooks": True, "autoUpdatesChannel": "stable"})
    fingerprint = {
        "group": group.value,
        "model_catalog_sha256": model_catalog_sha256,
        "skills_sha256": skills_fingerprint,
        "corpus_commit": corpus_commit,
        "corpus_view_sha256": corpus_fingerprint,
        "excluded_task_hashes": sorted(excluded),
        "components": sorted(components),
        "server_components": {name: server.component for name, server in servers.items()},
        "publication_policy": "local-only",
        "corpus_projection": "task Markdown and format contract; no feedback, docs, or cached indexes",
        "mcp_config": json.loads((root / "mcp.json").read_text()),
    }
    return Profile(root, env, corpus, fingerprint, runtime_paths)
