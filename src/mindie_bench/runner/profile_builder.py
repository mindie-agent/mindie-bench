"""Trial profile materialization and physically filtered corpus projection.

[Role]
Builds isolated per-trial home directories (`CODEX_HOME`, `CLAUDE_CONFIG_DIR`, `KIMI_HOME`,
`DSH_HOME`), MCP configurations (`mcp.json`), and physically filtered Markdown corpus views
for `G0`..`G3` ablation groups.

[Invariants]
- Personal credentials (`GITHUB_TOKEN`, `SSH_AUTH_SOCK`, cloud keys, host session IDs) are never
  inherited into `profile.env`.
- `filtered_corpus` physically excludes `same_origin_task_hashes` (when required by group/gradient)
  and rejects non-Markdown files (such as prebuilt `.sqlite` indexes) to prevent data leakage.
- Component Python executables preserve their virtualenv path (`Path.absolute()`) without
  dereferencing symlinks to the base system interpreter.

[Timeouts & Locks]
- Headless DSH profile initialization (`--dump-config`): `120.0s` subprocess timeout.

[Side-Effect Policy]
- Writes only inside `trial_dir / "profile"` and `trial_dir / "knowledge-private"` (`0o700`),
  including a bare local Git remote (`local_knowledge.git`) with `publication_policy="local-only"`.
"""

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
from mindie_bench.io import sha256_file, tree_hash, write_json
from mindie_bench.models import ExperimentConfig, Group, TaskSpec
from mindie_bench.runner.dsh_catalog import (
    assert_catalog_policy as assert_catalog_policy,
)
from mindie_bench.runner.dsh_catalog import (
    probe_dsh_catalog as probe_dsh_catalog,
)

# The pi-ai provider rejects a custom route model that does not advertise the
# requested effort. Declare the complete supported ladder, including `off`.
DSH_REASONING_EFFORTS = {"off": None, "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
DSH_DEEPSEEK_COMPAT = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "maxTokensField": "max_tokens",
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
    "supportsStrictMode": True,
}


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
                command, check=True, capture_output=True, text=True, env=env, timeout=30
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
        executable_sha256=sha256_file(binary),
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
    from mindie_bench.runner import graded_view

    if graded_view.is_graded(source):
        return graded_view.project(source, destination, excluded)
    if not (source / "tasks").is_dir():
        raise ValueError("corpus must contain tasks/")
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
    product_experiment: bool = False,
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
    base_path = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
    if os.name == "nt":
        win_dirs = []
        git_bin = shutil.which("git")
        if git_bin:
            git_dir = Path(git_bin).resolve().parent
            win_dirs.append(str(git_dir))
            for sub in (("usr", "bin"), ("bin",)):
                candidate = git_dir.parent.joinpath(*sub)
                if candidate.is_dir():
                    win_dirs.append(str(candidate))
        sys_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
        win_dirs.extend([str(Path(sys_root) / "System32"), sys_root])
        base_path = os.pathsep.join(win_dirs)
        for win_key in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP"):
            if win_key in os.environ:
                env[win_key] = os.environ[win_key]
    env.update(
        PATH=base_path,
        LANG="en_US.UTF-8",
        # zsh uses TMPPREFIX, not TMPDIR, for heredoc and here-string files.
        TMPPREFIX=str(root / "tmp" / "zsh"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        DISABLE_AUTOUPDATER="1",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    group_config = config.groups.get(group)
    native_dsh = group_config.native_dsh_plugin if group_config else None
    native_codex = group_config.native_codex_plugin if group_config else None
    if native_dsh and native_codex:
        raise ValueError("one trial can select only its own native host plugin")
    native_plugin = native_dsh or native_codex
    native_harness = "dsh" if native_dsh else "codex"
    if native_plugin and (
        config.harness.name != native_harness or group not in {Group.G2, Group.G3, Group.G4}
    ):
        raise ValueError("the native product requires its matching knowledge-enabled host profile")
    knowledge_mode = group_config.knowledge_mode if group_config else "advisory"
    servers = group_config.mcp_servers if group_config else {}
    components = {server.component for server in servers.values()}
    if native_plugin:
        if "knowledge" in components:
            raise ValueError(
                "the native product host owns knowledge tools; do not register a duplicate MCP"
            )
        components.add("knowledge")
        if native_dsh:
            components.add("remote-dev")
    if group == Group.G0 and (servers or (group_config and group_config.skills_path)):
        raise ValueError("G0 must contain no MindIE tools or skills")
    if config.remote and group != Group.G0:
        mindie_dir = home / ".mindie"
        mindie_dir.mkdir(exist_ok=True)
        write_json(
            mindie_dir / "endpoints.json",
            {
                "default_alias": "bench-slot",
                "endpoints": {
                    "bench-slot": {
                        "host": "bench-slot",
                        "port": 22,
                    }
                },
            },
        )
    if group != Group.G0 and "remote-dev" not in components:
        raise ValueError(f"{group} requires an explicit remote-dev MCP component")
    if group == Group.G1 and ("knowledge" in components or (group_config and group_config.skills_path)):
        raise ValueError("G1 must contain no knowledge or skills")
    if group in {Group.G0, Group.G1} and knowledge_mode != "advisory":
        raise ValueError("explicit and oracle knowledge conditions require G2 or G3")
    if knowledge_mode == "explicit" and not (group_config.skills_path or native_plugin):
        raise ValueError("explicit knowledge condition requires the installed domain skill")
    if group in {Group.G2, Group.G3, Group.G4} and "knowledge" not in components:
        raise ValueError(f"{group} requires the real knowledge MCP component")
    excluded = set()
    corpus = None
    corpus_fingerprint = None
    corpus_commit = None
    retained_analogies = []
    if group in {Group.G2, Group.G3, Group.G4}:
        corpus_id = task.mindie_knowledge.corpus_id
        if corpus_id not in config.corpora:
            raise ValueError(f"missing explicitly selected corpus: {corpus_id}")
        declared = config.corpora[corpus_id]
        source = Path(declared.path).resolve()
        actual = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        if actual != declared.commit or dirty:
            raise ValueError("corpus must be clean at its exact declared commit")
        from mindie_bench.runner.graded_view import identifiers, is_graded

        graded_source = is_graded(source)
        same_origin = [
            item
            for item in task.mindie_knowledge.same_origin_task_hashes
            if not (graded_source and len(item) == 64 and all(c in "0123456789abcdef" for c in item))
        ]
        analogical = [
            item
            for item in task.mindie_knowledge.analogical_task_hashes
            if not (graded_source and len(item) == 64 and all(c in "0123456789abcdef" for c in item))
        ]
        declared_tasks = {*same_origin, *analogical}
        missing = sorted(declared_tasks - identifiers(source))
        if missing:
            raise ValueError(f"declared knowledge task IDs are absent from the source corpus: {missing}")
        if (
            group == Group.G2
            or task.task.role == "seed"
            or task.mindie_knowledge.knowledge_gradient != "near_exact"
        ):
            excluded.update(same_origin)
        corpus = private_root / "corpus"
        corpus_fingerprint = filtered_corpus(corpus_override or source, corpus, excluded)
        retained_analogies = sorted(
            identity
            for identity in analogical
            if identity in identifiers(corpus)
        )
        if retained_analogies != sorted(analogical):
            raise ValueError("declared analogical knowledge must remain in the filtered corpus")
        corpus_commit = declared.commit
    local_bare = private_root / "local_knowledge.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(local_bare)],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
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
    if group in {Group.G3, Group.G4}:
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
        resolved_cmd = command.resolve(strict=True)
        if server.component != "knowledge":
            runtime_paths.append(command)
            if (command.parent / "mindie-profiler").is_file():
                runtime_paths.extend([command.parent.parent, resolved_cmd.parent.parent])
                if (
                    group in {Group.G2, Group.G3, Group.G4}
                    and str(command.parent) not in env["PATH"].split(os.pathsep)
                ):
                    env["PATH"] = f"{command.parent}{os.pathsep}{env['PATH']}"
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
    dsh_profile = None
    native_product = None
    if native_plugin:
        from mindie_bench.runner.native_product import prepare

        domain_corpora_paths = {}
        for dom_name, dom_cid in sorted((native_plugin.domain_corpora or {}).items()):
            if dom_cid not in config.corpora:
                raise ValueError(f"missing explicitly selected domain corpus: {dom_cid}")
            dom_decl = config.corpora[dom_cid]
            dom_src = Path(dom_decl.path).resolve()
            dom_head = subprocess.run(
                ["git", "-C", str(dom_src), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            dom_dirty = subprocess.run(
                ["git", "-C", str(dom_src), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            if dom_head != dom_decl.commit or dom_dirty:
                raise ValueError("domain corpus must be clean at its exact declared commit")
            dom_dest = private_root / f"corpus-{dom_name}"
            dom_effective_src = (
                (corpus_override or dom_src)
                if dom_cid == task.mindie_knowledge.corpus_id
                else dom_src
            )
            filtered_corpus(dom_effective_src, dom_dest, excluded)
            domain_corpora_paths[dom_name] = dom_dest
        active_domain = (
            "operators"
            if (
                task.task.domain_category == "native_operator_performance"
                or task.mindie_knowledge.corpus_id == "operators"
            )
            and ("operators" in domain_corpora_paths or native_plugin.domain == "operators")
            else native_plugin.domain
        )
        native_product = prepare(
            root, env, native_plugin, corpus, domain_corpora=domain_corpora_paths or None,
            active_domain=active_domain, harness=native_harness,
        )
        runtime_paths.extend(
            [
                Path(native_plugin.package_path),
                Path(native_plugin.runtime_python).absolute().parent.parent,
                Path(native_product["initialization"]["python_base_prefix"]),
                Path(native_plugin.redactor_executable),
            ]
        )
        runtime_bin = str(Path(native_plugin.runtime_python).absolute().parent)
        if runtime_bin not in env["PATH"].split(os.pathsep):
            env["PATH"] = runtime_bin + os.pathsep + env["PATH"]
    if config.harness.name == "dsh":
        dsh_profile = build_dsh_profile(
            root, env, config, mcp, servers,
            native_product=native_product, catalog_checkpoint=product_experiment or bool(native_dsh),
        )
        runtime_root = Path(config.harness.executable).resolve(strict=True)
        node = shutil.which("node", path=env["PATH"])
        dsh_roots = [runtime_root]
        if len(runtime_root.parents) >= 2:
            dsh_roots += [runtime_root.parents[1], runtime_root.parents[1].parent]
        if node:
            dsh_roots.append(Path(node).resolve(strict=True))
        runtime_paths.extend(path for path in dsh_roots if path.exists())
    if config.harness.name == "kimi":
        from mindie_bench.runner.kimi_profile import build_kimi_profile

        build_kimi_profile(root, env, config)
        runtime_paths.append(Path(config.harness.executable).resolve(strict=True).parent)
    fingerprint = {
        "group": group.value,
        "harness": config.harness.name,
        "knowledge_mode": knowledge_mode,
        "knowledge_conditions": {
            k: v for k, v in (task.mindie_knowledge.conditions or {}).items() if k != "query"
        },
        "oracle_context_sha256": (
            hashlib.sha256((root / "oracle-context.md").read_bytes()).hexdigest()
            if oracle_files
            else None
        ),
        "oracle_context_files_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in oracle_files.items()
        },
        "model_catalog_sha256": model_catalog_sha256,
        "dsh_profile": dsh_profile,
        "native_product": native_product,
        "skills_sha256": skills_fingerprint,
        "corpus_commit": corpus_commit,
        "corpus_view_sha256": corpus_fingerprint,
        "excluded_task_hashes": sorted(excluded),
        "retained_analogical_task_hashes": retained_analogies,
        "components": sorted(components),
        "server_components": {
            **({"remote-dev": "remote-dev"} if native_dsh else {}),
            **{name: server.component for name, server in servers.items()},
        },
        "publication_policy": "local-only",
        "capture_git_runtime": git_runtime,
        "corpus_projection": "task Markdown and format contract; no feedback, docs, or cached indexes",
        "mcp_config": json.loads((root / "mcp.json").read_text()),
        "knowledge_access": (
            "native product Store contains only the filtered trial corpus; original corpus remains private"
            if native_product
            else "parent-owned MCP stdio; no foreground corpus or state file grant"
        ),
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


def build_dsh_profile(
    root: Path,
    env: dict[str, str],
    config,
    mcp: dict,
    servers: dict,
    *,
    native_product=None,
    catalog_checkpoint: bool = False,
) -> dict:
    """Create an owned headless DSH profile with the benchmark MCP/model row."""
    import yaml

    dsh_home = root / "dsh-home"
    profile_dir = dsh_home / "profiles" / "bench"
    dsh_home.mkdir(mode=0o700, exist_ok=True)
    create_env = dict(env)
    create_env.update(
        DSH_HOME=str(dsh_home),
        DSH_PROFILE="bench",
        DSH_PROFILE_DIR=str(profile_dir),
        HOME=str(root / "home"),
    )
    completed = subprocess.run(
        [
            config.harness.executable,
            "--profile",
            "bench",
            "--from-default-profile",
            "headless",
            "--dump-config",
        ],
        env=create_env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise ValueError("DSH profile initialization failed: " + (completed.stderr or "")[-2000:])
    if not (profile_dir / "package.json").is_file():
        raise ValueError("DSH profile initialization did not create package.json")
    # Every benchmark arm uses the standard native-tools host and the maximum
    # default thinking effort; content, retrieval and ablation groups differ,
    # not the execution mode.
    effort = config.harness.reasoning_effort or "max"
    is_deepseek = config.harness.api_protocol == "openai-completions" and (
        "deepseek.com" in (config.harness.provider_url or "").lower() or config.harness.model.startswith("deepseek-")
    )
    lines = [
        "- id: llm-pi-ai",
        "  name: '@deepseek-ai/dsh-llm-pi-ai'",
        "  config:",
        "    providers:",
        "      bench:",
        f"        api: {config.harness.api_protocol}",
        "        baseURL: !!js process.env.MINDIE_BENCH_GATEWAY + '/v1'",
        "        apiKeyEnv: MINDIE_BENCH_DUMMY_KEY",
        "        models:",
        f"          - id: {json.dumps(config.harness.model)}",
        "            reasoningEfforts: " + json.dumps(DSH_REASONING_EFFORTS),
        *(["            compat: " + json.dumps(DSH_DEEPSEEK_COMPAT)] if is_deepseek else []),
        "- id: agent-default-model",
        "  name: '@deepseek-ai/dsh-agent-default-model'",
        "  config:",
        "    provider: bench",
        f"    model: {json.dumps(config.harness.model)}",
        f"    reasoningEffort: {json.dumps(effort)}",
        "- id: tools",
        "  name: '@deepseek-ai/dsh-tools'",
        "  config:",
        "    mode: native",
    ]
    insert_rows = []
    if native_product:
        package = Path(native_product["package_path"])
        insert_rows += [
            "    - id: mindie-agent-host",
            "      name: " + json.dumps(str(package / "host.mjs")),
            "      config:",
            "        mcpPluginUrl: !!js process.getBuiltinModule('node:url').pathToFileURL(process.getBuiltinModule('node:module').createRequire(baseUrl).resolve('@deepseek-ai/dsh-mcp-client')).href",
            "      inject: [tools, agents, llm]",
            "    - id: mindie-agent-skill",
            "      name: '@deepseek-ai/dsh-skill-filesystem'",
            "      config:",
            "        providerName: mindie",
            "        includeDefaultRoots: false",
            "        watch: false",
            "        customSkillDirs:",
            "          - " + json.dumps(str(package / "skills")),
        ]
    if catalog_checkpoint:
        checkpoint = Path(__file__).resolve().parent / "catalog_checkpoint.mjs"
        insert_rows += [
            "    - id: bench-catalog-checkpoint",
            "      name: " + json.dumps(str(checkpoint)),
        ]
    if mcp:
        for name, server in mcp.items():
            row_id = "bench-mcp-" + "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
            insert_rows += [
                f"    - id: {row_id}",
                "      name: '@deepseek-ai/dsh-mcp-client'",
                "      config:",
                f"        serverName: {json.dumps(name)}",
                "        transport: stdio",
                f"        command: {json.dumps(str(server['command']))}",
                "        args: " + json.dumps([str(a) for a in server.get("args", [])]),
                "        env:",
                f"          PATH: {json.dumps(str(root / 'bin') + os.pathsep + env['PATH'])}",
                "          MINDIE_BENCH_GATEWAY: !!js process.env.MINDIE_BENCH_GATEWAY",
            ]
            for key, value in (server.get("env") or {}).items():
                insert_rows.append(f"          {key}: {json.dumps(str(value))}")
            insert_rows += [
                f"        cwd: {json.dumps(str(root))}",
                "        toolCallTimeoutMs: 600000",
                "        failOnStartupError: true",
            ]
    if (root / "skills").is_dir():
        insert_rows += [
            "    - id: bench-skills",
            "      name: '@deepseek-ai/dsh-skill-filesystem'",
            "      config:",
            "        providerName: bench",
            "        customSkillDirs:",
            f"          - {json.dumps(str(root / 'skills'))}",
        ]
    if insert_rows:
        lines.append("- insert:")
        lines.extend(insert_rows)
    (profile_dir / "cordis.patch.yml").write_text("\n".join(lines) + "\n")

    # Validate the generated YAML without evaluating the !!js expressions.
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node))
    yaml.load((profile_dir / "cordis.patch.yml").read_text(), Loader=Loader)
    env.update(
        DSH_HOME=str(dsh_home),
        DSH_PROFILE="bench",
        DSH_PROFILE_DIR=str(profile_dir),
        MINDIE_BENCH_DUMMY_KEY="broker-only",
    )
    checkpoint_info = None
    if catalog_checkpoint:
        checkpoint_info = {
            "row_id": "bench-catalog-checkpoint",
            "source": str(Path(__file__).resolve().parent / "catalog_checkpoint.mjs"),
            "output": str(root / "catalog-checkpoint.json"),
        }
        env.update(
            MINDIE_BENCH_CATALOG_CHECKPOINT_OUT=checkpoint_info["output"],
            MINDIE_BENCH_CATALOG_CHECKPOINT_DELAY_MS="1500",
        )
    return {
        "home": str(dsh_home),
        "profile": "bench",
        "patch_sha256": hashlib.sha256((profile_dir / "cordis.patch.yml").read_bytes()).hexdigest(),
        "catalog_checkpoint": checkpoint_info,
    }


def knowledge_prompt(profile: Profile) -> str:
    mode = profile.fingerprint.get("knowledge_mode", "advisory")
    if mode == "explicit":
        gesture = "$mindie-agent" if profile.fingerprint.get("harness") == "codex" else "/mindie-agent"
        return f"{gesture}\n\n"
    if mode == "oracle":
        return (
            "Controlled oracle-context condition: the following selected historical material "
            "is supplied directly to distinguish content utility from routing. Treat it as "
            "fallible reference material and verify applicability in the current environment.\n\n"
            + (profile.root / "oracle-context.md").read_text()
            + "\n\nCurrent engineering request:\n"
        )
    return ""
