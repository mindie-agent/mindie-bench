"""[Role] Upgrade a published Codex installation through its real native updater.

[Invariants] Native installation, component discovery/build and idle activation
are production code. Staging changes only the plugin release-list transport.
Independent component floors, external commands and explicit hook disable survive.
[Timeouts & Locks] The workflow grants 40 minutes; each install/build/native API
call keeps its own bounded production budget and the profile is exclusively owned.
[Side-Effect Policy] Writes only the supplied isolated native acceptance profile.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def invoke(argv, env, *, timeout=90, json_output=True):
    result = subprocess.run([str(value) for value in argv], env=env, capture_output=True,
                            timeout=timeout, check=False, **FLAGS)
    if result.returncode:
        raise RuntimeError(
            f"{Path(argv[0]).name} exited {result.returncode}: "
            + result.stderr.decode("utf-8", errors="replace")[-6000:])
    return json.loads(result.stdout) if json_output else result.stdout


def environment(root, host):
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "CODEX_HOME", "MINDIE_CONFIG_HOME",
                 "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT", "MINDIE_REMOTE_STATE_DIR",
                 "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(
        MINDIE_AGENT_CONFIG=str(root / "MINDIE_CONFIG_HOME/codex.json"),
        MINDIE_DIAGNOSTICS_CONFIG=str(root / "diagnostics-disabled.json"),
        MINDIE_CODEX_BIN=str(host), UV_SYSTEM_CERTS="true", GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="core.longpaths", GIT_CONFIG_VALUE_0="true",
        GIT_CONFIG_KEY_1="core.autocrlf", GIT_CONFIG_VALUE_1="false",
    )
    for name in ("CODEX_THREAD_ID", "CODEX_APP_RUNTIME_CLI_PATH"):
        env.pop(name, None)
    (root / "diagnostics-disabled.json").write_text(json.dumps({
        "schema": "mindie.diagnostics.reporting.v1", "purpose": "tool_fault_reporting",
        "decision": "disabled", "repository": "mindie-agent/mindie-agent-codex",
    }), encoding="utf-8", newline="\n")
    return env


def child(argv, env, *, python=None, timeout=1200):
    return invoke([python or sys.executable, "-I", "-X", "utf8", "-B",
                   Path(__file__).resolve(), *argv], env, timeout=timeout)


def baseline(args, env):
    invoke([args.baseline_host, "plugin", "marketplace", "add",
            "https://github.com/mindie-agent/mindie-agent-codex.git",
            "--ref", "v0.1.23", "--json"], env, timeout=180)
    installed = invoke([args.baseline_host, "plugin", "add",
                        "mindie-agent@mindie-agent", "--json"], env, timeout=180)
    source = Path(installed["installedPath"])
    manifest = json.loads((source / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    if manifest["version"] != "0.1.23":
        raise ValueError("acceptance did not install the published legacy plugin")
    data = Path(env["CODEX_HOME"]) / "plugins/data/mindie-agent-mindie-agent"
    runtime = child(["bootstrap", "--source", source, "--root", data], env, timeout=1860)
    invoke([runtime["python"], "-I", "-X", "utf8", Path(runtime["scripts"]) / "setup.py",
            "install", "--knowledge-python", runtime["python"],
            "--config", env["MINDIE_AGENT_CONFIG"], "--root", args.root / "knowledge",
            "--no-public-feed"], env, timeout=180)
    child(["seed", "--source", source, "--root", args.root,
           "--baseline-host", args.baseline_host], env, python=runtime["python"], timeout=1500)


def seed(args):
    config = Path(os.environ["MINDIE_AGENT_CONFIG"])
    sys.path.insert(0, str(args.source / "scripts"))
    from component_update import check
    from mindie_knowledge.component_runtime import read_inventory
    from native_hooks import NativeHooks
    from release_activation import _atomic
    from update_lock import update_lock

    with NativeHooks(str(args.baseline_host), dict(os.environ)) as api:
        hook = next(row for row in api.hooks() if row["eventName"] == "stop")
        _, layer, target = api.user_config()
        api.call("config/value/write", dict(
            keyPath="hooks.state." + json.dumps(hook["key"]) + ".enabled",
            value=False, mergeStrategy="replace", expectedVersion=layer["version"],
            filePath=str(target)))
    before = json.loads(config.read_text(encoding="utf-8"))
    data = Path(os.environ["CODEX_HOME"]) / "plugins/data/mindie-agent-mindie-agent"
    result = check(config, args.source, data, force=True)
    if result["status"] != "active":
        raise ValueError(f"the old client did not independently upgrade components: {result}")
    current = json.loads(config.read_text(encoding="utf-8"))
    if current["python"] == before["python"] or not current.get("component_release"):
        raise ValueError("independent component generation was not activated")
    inventory = read_inventory(current["python"])
    overrides = args.root / "external-overrides.txt"
    overrides.write_text("".join(
        f"{name} @ git+https://github.com/{row['repository']}@{row['revision']}\n"
        for name, row in inventory.items()), encoding="utf-8", newline="\n")
    external = args.root / "external-components"
    invoke(["uv", "venv", "--python", current["python"], external],
           dict(os.environ), json_output=False)
    python = external / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = [
        f"{name} @ git+https://github.com/{inventory[name]['repository']}@{inventory[name]['revision']}"
        for name in ("mindie-sim", "mindie-profiler")]
    invoke(["uv", "pip", "install", "--python", python, "--override", overrides,
            *requirements], dict(os.environ), timeout=240, json_output=False)
    with update_lock(config, exclusive=True):
        adapter = json.loads(config.read_text(encoding="utf-8"))
        commands = dict(adapter.get("component_commands", {}))
        for name, module in (("sim", "mindie_sim"), ("profiler", "mindie_profiler")):
            commands[name] = [str(python), "-I", "-X", "utf8", "-m", module]
        adapter["component_commands"] = commands
        _atomic(config, adapter)
    print(json.dumps(dict(status="seeded", independently_upgraded=inventory,
                          external_components=["sim", "profiler"], stop_enabled=False)))


def upgrade(args):
    config = Path(os.environ["MINDIE_AGENT_CONFIG"])
    before = json.loads(config.read_text(encoding="utf-8"))
    sys.path.insert(0, before["runtime_scripts"])
    if args.metadata:
        from mindie_knowledge import releases_io
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        # Prospective publication transports exactly one declared source. Native
        # selection, immutable-source checks and activation remain unmodified.
        releases_io.fetch_releases = lambda repository, **kwargs: [
            dict(id=1, tag_name="v" + args.version, draft=False, prerelease=False,
                 metadata=metadata)]
    from harness_update import check
    result = check(config, force=True)
    if result["status"] != "active":
        raise ValueError(f"legacy native upgrade did not activate: {result}")
    print(json.dumps(result))


def verify(args):
    config = Path(os.environ["MINDIE_AGENT_CONFIG"])
    current = json.loads(config.read_text(encoding="utf-8"))
    sys.path.insert(0, current["runtime_scripts"])
    from mindie_knowledge.component_bindings import managed_commands
    from mindie_knowledge.component_runtime import read_inventory
    from native_hooks import NativeHooks

    source = Path(current["runtime_scripts"]).parent
    manifest = json.loads((source / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    if manifest["version"] != args.version:
        raise ValueError("the activated native plugin is not the candidate")
    for name, python, _module in managed_commands(current).values():
        if str(python) != str(current["python"]):
            raise ValueError(f"external component was not rebound into the verified generation: {name}")
    with NativeHooks(str(args.target_host), dict(os.environ)) as api:
        hooks = api.hooks()
        if len(hooks) != 5 or next(row for row in hooks if row["eventName"] == "stop")["enabled"]:
            raise ValueError("native hook definitions or explicit disable changed")
        thread = api.call("thread/start", dict(
            cwd=str(args.root), ephemeral=True, approvalPolicy="never"))["thread"]["id"]
        rows = api.call("mcpServerStatus/list", dict(threadId=thread))["data"]
        remote = next(row for row in rows if "mindie-remote-dev" in row["name"])
        tools = next(row for row in rows if "mindie-tools" in row["name"])
        if ("remote_probe" not in remote["tools"] or "diagnostics_status" not in tools["tools"]
                or any("knowledge_" in name for row in rows for name in row["tools"])):
            raise ValueError("fresh native MCP discovery differs from the public contract")
    print(json.dumps(dict(installed=read_inventory(current["python"]),
                          disabled_stop_preserved=True, external_components_rebound=True,
                          native_plugin=args.version, remote_tools=len(remote["tools"]),
                          component_tools=len(tools["tools"]))))


def accept(args):
    args.root.mkdir(parents=True, exist_ok=True)
    env = environment(args.root, args.baseline_host)
    baseline(args, env)
    config = Path(env["MINDIE_AGENT_CONFIG"])
    before = json.loads(config.read_text(encoding="utf-8"))
    baseline_inventory = child(["inventory", "--root", args.root], env, python=before["python"])
    env["CODEX_APP_RUNTIME_CLI_PATH"] = str(args.target_host)
    argv = ["upgrade", "--root", args.root, "--version", args.version]
    if args.metadata:
        argv.extend(["--metadata", args.metadata])
    child(argv, env, python=before["python"], timeout=2100)
    current = json.loads(config.read_text(encoding="utf-8"))
    report = child(["verify", "--root", args.root, "--version", args.version,
                    "--target-host", args.target_host], env, python=current["python"], timeout=180)
    from semantic_version import Version
    if any(name not in report["installed"] or
           Version(report["installed"][name]["version"]) < Version(row["version"])
           for name, row in baseline_inventory.items()):
        raise ValueError("native update lost an independently installed component floor")
    report.update(
        schema="mindie-native-upgrade-acceptance/1", status="passed",
        repository="mindie-agent/mindie-agent-codex", version=args.version,
        revision=args.revision, baseline_plugin="0.1.23", baseline=baseline_inventory,
        scope="native_codex_old_client",
        transport="staged_candidate" if args.metadata else "public_release",
    )
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("accept", "bootstrap", "seed", "upgrade", "verify", "inventory"))
    parser.add_argument("--root", type=Path, required=True)
    for name in ("source", "baseline-host", "target-host", "metadata", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--version")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        sys.path.insert(0, str(args.source / "scripts"))
        from bootstrap import ensure_runtime
        python, scripts, release = ensure_runtime(args.source, args.root)
        print(json.dumps(dict(python=str(python), scripts=str(scripts), release=str(release))))
    elif args.mode == "inventory":
        config = json.loads(Path(os.environ["MINDIE_AGENT_CONFIG"]).read_text(encoding="utf-8"))
        from mindie_knowledge.component_runtime import read_inventory
        print(json.dumps(read_inventory(config["python"])))
    else:
        {"accept": accept, "seed": seed, "upgrade": upgrade, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
