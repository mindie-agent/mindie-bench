"""[Role] Own one native DSH SDK summary host across post-trial C6 capture.

[Invariants] Initialize/shutdown never creates a session or sends a prompt.
The installed plugin owns the summary protocol; all inference uses the trial
Gateway. No tools, instructions, skills, web or extra MCP servers are enabled.
[Timeouts & Locks] Profile creation 120s; initialize 30s + endpoint 10s;
shutdown 10s + process exit 10s + SIGTERM 2s + SIGKILL 5s in the owned session group.
[Side-Effect Policy] Creates an isolated SDK profile and one authenticated
loopback endpoint. Only its selected port is admitted by the native sandbox.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import queue
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import yaml

from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.termination import terminate_process_group


def _rows(text):
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node))
    rows = yaml.load(text, Loader=Loader)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("DSH effective configuration must be plugin rows")
    return rows


def _forbidden(name):
    return name.startswith(
        ("tool-", "skill", "web", "mcp-", "subagent", "ptc-", "workflow-")
    ) or name in {"agent-instructions", "session-title-llm"}


def prepare(profile, harness, plugin):
    """Prepare the installed SDK profile without using the personal DSH home."""
    plugin = Path(plugin).resolve(strict=True)
    root = profile.root / "dsh-summary"
    root.mkdir()
    home, state = root / "home", root / "state"
    home.mkdir()
    state.mkdir()
    selected = home / "profiles" / "bench-summary"
    env = dict(
        profile.env,
        DSH_HOME=str(home),
        DSH_PROFILE="bench-summary",
        DSH_PROFILE_DIR=str(selected),
        MINDIE_BENCH_GATEWAY="http://127.0.0.1:9",
        MINDIE_BENCH_DUMMY_KEY="broker-only",
        MINDIE_DSH_STATE_DIR=str(state),
    )
    command = [harness.executable, "--profile", "bench-summary"]
    initialized = subprocess.run(
        [*command, "--from-default-profile", "sdk", "--dump-config"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    rows = _rows(initialized.stdout)
    patch = [{"id": row["id"], "disabled": True} for row in rows if _forbidden(row["id"])]
    patch.extend(
        [
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        "bench": {
                            "api": harness.api_protocol,
                            "baseURL": "__BENCH_GATEWAY__",
                            "apiKeyEnv": "MINDIE_BENCH_DUMMY_KEY",
                            "models": [{"id": harness.model}],
                        }
                    }
                },
            },
            {"id": "agent-default-model", "config": {"provider": "bench", "model": harness.model}},
            {
                "insert": [
                    {
                        "id": "bench-summary-host",
                        "name": str(plugin),
                        "config": {
                            "endpointFile": str(state / "summary-endpoint.json"),
                            "listenPort": "__BENCH_SUMMARY_PORT__",
                        },
                    }
                ]
            },
        ]
    )
    text = (
        yaml.safe_dump(patch, sort_keys=False)
        .replace("__BENCH_GATEWAY__", "!!js process.env.MINDIE_BENCH_GATEWAY + '/v1'")
        .replace("__BENCH_SUMMARY_PORT__", "!!js Number(process.env.MINDIE_BENCH_SUMMARY_PORT)")
    )
    (selected / "cordis.patch.yml").write_text(text)
    effective = subprocess.run(
        [*command, "--dump-config"],
        cwd=root,
        env=dict(env, MINDIE_BENCH_SUMMARY_PORT="0"),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    active = [row["id"] for row in _rows(effective.stdout) if not row.get("disabled")]
    if any(_forbidden(name) for name in active):
        raise ValueError("summary-only SDK profile still enables context or tools")
    descriptor = dict(
        root=str(root),
        home=str(home),
        state=str(state),
        profile=str(selected),
        plugin=str(plugin),
        plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        plugin_sources={
            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in (
                plugin,
                plugin.parent.parent / "host3.mjs",
                plugin.parent / "knowledge_tools.mjs",
                plugin.parent.parent / "paths.cjs",
                plugin.parent.parent / "package.json",
            )
        },
        model=harness.model,
        executable=harness.executable,
        active_plugin_ids=active,
        effective_config_sha256=hashlib.sha256(effective.stdout.encode()).hexdigest(),
    )
    profile.fingerprint["dsh_summary_profile"] = descriptor
    return descriptor


class SummaryHost:
    """A bounded owner of native JSON-RPC initialization and its one endpoint."""

    def __init__(self, profile, *, read_paths, gateway_port=None, gateway_socket=None):
        self.profile, self.read_paths = profile, read_paths
        self.gateway_port, self.gateway_socket = gateway_port, gateway_socket
        self.frames = queue.Queue()
        self.process = self.errors = None
        self.receipt = {"session_prompt_sent": False, "model_calls_during_initialization": 0}

    def _request(self, identifier, method, params=None, *, timeout=30):
        frame = dict(jsonrpc="2.0", id=identifier, method=method)
        if params is not None:
            frame["params"] = params
        self.process.stdin.write(json.dumps(frame) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self.frames.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty as error:
                raise TimeoutError(f"DSH SDK {method} deadline expired") from error
            if isinstance(item, Exception):
                raise item
            if item.get("method", "").startswith("session."):
                raise ValueError("summary host unexpectedly created a native session")
            if item.get("id") == identifier:
                if "error" in item:
                    raise ValueError(f"DSH SDK {method} rejected: {item['error']}")
                return item
        raise TimeoutError(f"DSH SDK {method} deadline expired")

    def __enter__(self):
        if shutil.which("sandbox-exec") is None and shutil.which("bwrap") is None:
            raise ValueError(
                "DSH G3 summary host requires an available OS sandbox backend "
                "(sandbox-exec or bwrap)"
            )
        selected = self.profile.fingerprint["dsh_summary_profile"]
        for source, expected in selected["plugin_sources"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError("native summary source changed after profile preparation: " + source)
        root, state = Path(selected["root"]), Path(selected["state"])
        with socket.socket() as allocation:
            allocation.bind(("127.0.0.1", 0))
            self.port = allocation.getsockname()[1]
        self.env = dict(
            self.profile.env,
            DSH_HOME=selected["home"],
            DSH_PROFILE="bench-summary",
            DSH_PROFILE_DIR=selected["profile"],
            MINDIE_DSH_STATE_DIR=str(state),
            MINDIE_BENCH_SUMMARY_PORT=str(self.port),
        )
        command = native_command(
            [selected["executable"], "--profile", "bench-summary"],
            read=[*self.read_paths, Path(selected["plugin"]).parent.parent],
            write=[self.profile.root],
            cwd=root,
            policy_path=root / "sdk.sb",
            gateway_port=self.gateway_port,
            gateway_socket=self.gateway_socket,
            loopback_listener_port=self.port,
        )
        self.errors = (root / "sdk.stderr").open("w")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=root,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.errors,
                text=True,
                start_new_session=True,
            )

            def read():
                try:
                    for line in self.process.stdout:
                        self.frames.put(json.loads(line))
                    self.frames.put(RuntimeError("DSH SDK stdout closed"))
                except TimeoutError:
                    raise
                except (ValueError, OSError) as error:
                    self.frames.put(error)

            self.reader = threading.Thread(target=read, daemon=True)
            self.reader.start()
            response = self._request(
                1, "initialize", dict(cwd=str(root), provider="bench", model=selected["model"])
            )
            if (
                response.get("result", {}).get("serverInfo", {}).get("name")
                != "deepseek-harness-sdk-runtime"
            ):
                raise ValueError("unexpected native SDK initialization response")
            endpoint = state / "summary-endpoint.json"
            deadline = time.monotonic() + 10
            while not endpoint.exists() and self.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            actual = json.loads(endpoint.read_text())
            if (actual.get("mode"), actual.get("provider"), actual.get("model"), actual.get("url")) != (
                "summary-only",
                "bench",
                selected["model"],
                f"http://127.0.0.1:{self.port}/summary",
            ):
                raise ValueError("native summary endpoint differs from the owned route")
            self.receipt.update(
                initialize=response,
                endpoint={key: actual[key] for key in ("schema", "mode", "provider", "model", "pid")},
                loopback_port=self.port,
            )
            return self
        except BaseException as error:
            self._close(error)
            raise

    def _close(self, primary=None):
        try:
            if self.process is not None and self.process.poll() is None:
                response = self._request(2, "shutdown", timeout=10)
                self.process.stdin.close()
                code = self.process.wait(timeout=10)
                if code != 0 or response.get("result") != {}:
                    raise RuntimeError("native SDK shutdown failed")
                self.receipt.update(shutdown=response, exit_code=code)
            if self.process is not None:
                endpoint = (
                    Path(self.profile.fingerprint["dsh_summary_profile"]["state"])
                    / "summary-endpoint.json"
                )
                if endpoint.exists():
                    raise RuntimeError("native summary endpoint survived SDK shutdown")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"summary host cleanup also failed: {type(error).__name__}: {error}")
        finally:
            if self.process is not None and self.process.poll() is None:
                try:
                    terminate_process_group(self.process, term_wait=2, kill_wait=5)
                except TimeoutError as cleanup:
                    self.errors.write(f"summary host termination failed: {cleanup}\n")
            if self.errors is not None:
                self.errors.close()

    def __exit__(self, kind, error, traceback):
        self._close(error)


def optional_host(profile, **kwargs):
    if "dsh_summary_profile" in profile.fingerprint:
        return SummaryHost(profile, **kwargs)
    return contextlib.nullcontext(None)
