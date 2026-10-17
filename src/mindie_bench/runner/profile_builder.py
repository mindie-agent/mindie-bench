"""Fresh profiles and physically filtered corpus views, never personal profiles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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
    private_root: Path | None = None
    private_env: dict[str, str] = field(default_factory=dict)
    knowledge_servers: dict = field(default_factory=dict)


def capture_git_runtime(env: dict[str, str]) -> dict:
    """Resolve Apple's launcher before sandboxing, and bind local Git helpers."""

    def inspect(command):
        try:
            return subprocess.run(
                command, check=True, capture_output=True, text=True, env=env
            ).stdout.strip()
        except subprocess.CalledProcessError as error:
            error.add_note("Git discovery stderr (at most 16 KiB): " + (error.stderr or "")[:16_384])
            raise

    executable = shutil.which("git", path=env["PATH"])
    if executable is None:
        raise FileNotFoundError("Git is required for local capture publication")
    if platform.system() == "Darwin" and Path(executable).resolve() == Path("/usr/bin/git"):
        executable = inspect(["/usr/bin/xcrun", "--find", "git"])
        if Path(executable).resolve() == Path("/usr/bin/git"):
            raise ValueError("xcrun did not resolve the actual Git executable")
    binary = Path(executable).resolve(strict=True)
    helpers = Path(inspect([str(binary), "--exec-path"])).resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK) or not helpers.is_dir():
        raise ValueError("capture requires an executable Git and its actual helper directory")
    return dict(
        executable=str(binary),
        executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        exec_path=str(helpers),
    )


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
    private_root = trial_dir / "knowledge-private"
    private_root.mkdir(mode=0o700)
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
    knowledge_mode = group_config.knowledge_mode if group_config else "advisory"
    servers = group_config.mcp_servers if group_config else {}
    components = {server.component for server in servers.values()}
    if group == Group.G0 and (servers or (group_config and group_config.skills_path)):
        raise ValueError("G0 must contain no MindIE tools or skills")
    if group != Group.G0 and "remote-dev" not in components:
        raise ValueError(f"{group} requires an explicit remote-dev MCP component")
    if group == Group.G1 and ("knowledge" in components or (group_config and group_config.skills_path)):
        raise ValueError("G1 must contain no knowledge or skills")
    if group in {Group.G0, Group.G1} and knowledge_mode != "advisory":
        raise ValueError("explicit and oracle knowledge conditions require G2 or G3")
    if knowledge_mode == "explicit" and not group_config.skills_path:
        raise ValueError("explicit knowledge condition requires the installed domain skill")
    if group in {Group.G2, Group.G3} and "knowledge" not in components:
        raise ValueError(f"{group} requires the real knowledge MCP component")
    excluded = set()
    corpus = None
    corpus_fingerprint = None
    corpus_commit = None
    retained_analogies = []
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
        declared_tasks = {
            *task.mindie_knowledge.same_origin_task_hashes,
            *task.mindie_knowledge.analogical_task_hashes,
        }
        missing = sorted(
            identity for identity in declared_tasks if not (source / "tasks" / identity).is_dir()
        )
        if missing:
            raise ValueError(f"declared knowledge task IDs are absent from the source corpus: {missing}")
        if (
            group == Group.G2
            or task.task.role == "seed"
            or task.mindie_knowledge.knowledge_gradient != "near_exact"
        ):
            excluded.update(task.mindie_knowledge.same_origin_task_hashes)
        corpus = private_root / "corpus"
        corpus_fingerprint = filtered_corpus(corpus_override or source, corpus, excluded)
        retained_analogies = sorted(
            identity
            for identity in task.mindie_knowledge.analogical_task_hashes
            if (corpus / "tasks" / identity).is_dir()
        )
        if retained_analogies != sorted(task.mindie_knowledge.analogical_task_hashes):
            raise ValueError("declared analogical knowledge must remain in the filtered corpus")
        corpus_commit = declared.commit
    local_bare = private_root / "local_knowledge.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(local_bare)], check=True, env=env, capture_output=True
    )
    env["MINDIE_BENCH_PUBLICATION_POLICY"] = "local-only"
    private_env = {
        "MINDIE_BENCH_KNOWLEDGE_PRIVATE_ROOT": str(private_root),
        "MINDIE_BENCH_LOCAL_KNOWLEDGE_REMOTE": str(local_bare),
    }
    if corpus is not None:
        private_env["MINDIE_BENCH_CORPUS"] = str(corpus)
    runtime_paths = []
    git_runtime = None
    if group == Group.G3:
        git_runtime = capture_git_runtime(env)
        private_env.update(
            MINDIE_BENCH_GIT=git_runtime["executable"], GIT_EXEC_PATH=git_runtime["exec_path"]
        )
        runtime_paths.extend([Path(git_runtime["executable"]), Path(git_runtime["exec_path"])])
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
    knowledge_servers = {}
    for name, server in servers.items():
        # Python discovers its virtual environment from the invoked executable
        # path. Resolving that symlink here silently selects the base runtime.
        command = Path(server.command).absolute()
        command.resolve(strict=True)
        if server.component != "knowledge":
            runtime_paths.append(command)
        replacements = {
            "{corpus}": str(corpus or ""),
            "{profile}": str(private_root if server.component == "knowledge" else root),
            "{local_remote}": str(local_bare),
        }

        def expand(value: str) -> str:
            for old, new in replacements.items():
                value = value.replace(old, new)
            return value

        configured = {
            "command": str(command),
            "args": [expand(arg) for arg in server.args],
            "env": {key: expand(value) for key, value in server.env.items()},
        }
        if server.component == "knowledge":
            knowledge_servers[name] = configured
            mcp[name] = {
                "command": str(Path(sys.executable).absolute()),
                "args": ["-I", "-m", "mindie_bench.runner.knowledge_stdio", name],
                "env": {},
            }
        else:
            mcp[name] = configured
    write_json(root / "mcp.json", {"mcpServers": mcp})
    skills_fingerprint = None
    if group_config and group_config.skills_path:
        source = Path(group_config.skills_path).resolve(strict=True)
        skills_fingerprint = tree_hash(source)
        shutil.copytree(source, root / "skills")
    oracle_files = {}
    if knowledge_mode == "oracle":
        for relative in group_config.oracle_context_files:
            path = corpus / relative
            if not path.resolve(strict=True).is_relative_to(corpus.resolve()):
                raise ValueError("oracle material escapes the filtered corpus")
            oracle_files[relative] = path.read_text()
        oracle_text = "\n\n".join(
            f"Source: {relative}\n\n{text}" for relative, text in oracle_files.items()
        )
        if len(oracle_text.encode()) > 128 * 1024:
            raise ValueError("oracle context exceeds its explicit 128 KiB prompt budget")
        (root / "oracle-context.md").write_text(oracle_text)
    write_json(root / "settings.json", {"disableAllHooks": True, "autoUpdatesChannel": "stable"})
    fingerprint = {
        "group": group.value,
        "knowledge_mode": knowledge_mode,
        "oracle_context_sha256": (
            hashlib.sha256((root / "oracle-context.md").read_bytes()).hexdigest()
            if oracle_files
            else None
        ),
        "oracle_context_files_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in oracle_files.items()
        },
        "model_catalog_sha256": model_catalog_sha256,
        "skills_sha256": skills_fingerprint,
        "corpus_commit": corpus_commit,
        "corpus_view_sha256": corpus_fingerprint,
        "excluded_task_hashes": sorted(excluded),
        "retained_analogical_task_hashes": retained_analogies,
        "components": sorted(components),
        "server_components": {name: server.component for name, server in servers.items()},
        "publication_policy": "local-only",
        "capture_git_runtime": git_runtime,
        "corpus_projection": "task Markdown and format contract; no feedback, docs, or cached indexes",
        "mcp_config": json.loads((root / "mcp.json").read_text()),
        "knowledge_access": "parent-owned MCP stdio; no foreground corpus or state file grant",
        "knowledge_backend_config_sha256": {
            name: hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            for name, spec in knowledge_servers.items()
        },
    }
    return Profile(
        root,
        env,
        corpus,
        fingerprint,
        runtime_paths,
        private_root=private_root,
        private_env=private_env,
        knowledge_servers=knowledge_servers,
    )


def knowledge_prompt(profile: Profile) -> str:
    mode = profile.fingerprint.get("knowledge_mode", "advisory")
    if mode == "explicit":
        return (
            "Use the mindie-agent skill for this task. First use knowledge_query to find "
            "historical engineering experience relevant to this request, then use "
            "knowledge_explain to inspect relevant hits before diagnosing it. If the selected "
            "corpus has no useful hit, continue from source and runtime evidence. Historical "
            "experience is advisory; verify its applicability to the current environment.\n\n"
        )
    if mode == "oracle":
        return (
            "Controlled oracle-context condition: the following selected historical material "
            "is supplied directly to distinguish content utility from routing. Treat it as "
            "fallible reference material and verify applicability in the current environment.\n\n"
            + (profile.root / "oracle-context.md").read_text()
            + "\n\nCurrent engineering request:\n"
        )
    return ""
