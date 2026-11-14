"""Preflight MCP server readiness probe and deterministic process-group teardown.

[Role]
Verifies that every configured MCP server (`remote-dev`, `knowledge`, etc.) initializes
and exposes its required tool schema over stdio before the foreground agent is launched.

[Invariants]
- Every probed server must answer `initialize` and `tools/list` without pagination (`nextCursor`)
  and include all declared `required_tools` (or default component capabilities).
- Primary catalog/protocol failures are never masked by subsequent shutdown errors; shutdown
  exceptions are attached via `BaseException.add_note`.

[Timeouts & Locks]
- Per-request RPC read budget: `45.0s`.
- Graceful EOF drain (`close_probe`): `_STEP_TIMEOUT_SECONDS` (default `15.0s`), followed by
  `SIGTERM` -> `2.0s` -> `SIGKILL` -> `5.0s` on the owned process group.

[Side-Effect Policy]
- Runs each MCP probe in an isolated session (`start_new_session=True`) bound to an `owner.py`
  lifetime pipe (`owner_fd`) so no orphan MCP processes survive preflight before foreground starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from mindie_bench import __version__
from mindie_bench.io import digest
from mindie_bench.runner import graded_view
from mindie_bench.runner.host_platform import PipeReader, subprocess_window_kwargs
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.termination import spawn_owned_popen, terminate_process_group

_STEP_TIMEOUT_SECONDS = float(os.environ.get("MINDIE_BENCH_MCP_PROBE_TIMEOUT_SECONDS", "15.0"))


def close_probe(process, timeout: float | None = None, *, reader: PipeReader | None = None):
    """EOF lets a stdio relay close its independently owned backend session."""
    failure: BaseException | None = None
    budget = _STEP_TIMEOUT_SECONDS if timeout is None else timeout

    def record_cleanup(exc: BaseException, label: str) -> None:
        nonlocal failure
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"MCP preflight {label} also failed: {exc}")

    try:
        if reader is None:
            process.communicate(timeout=budget)
        else:
            deadline = time.monotonic() + budget
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, budget)
                timed_out, chunk = reader.read_chunk(remaining)
                if timed_out:
                    raise subprocess.TimeoutExpired(process.args, budget)
                if not chunk:
                    break
            process.wait(timeout=max(0.05, deadline - time.monotonic()))
        if process.returncode:
            raise RuntimeError(f"MCP preflight exited with code {process.returncode}")
    except BaseException as error:
        failure = error
    finally:
        try:
            terminate_process_group(process, term_wait=2, kill_wait=5)
        except BaseException as cleanup:
            record_cleanup(cleanup, "forced cleanup")
        try:
            if reader is None:
                process.communicate(timeout=5)
            else:
                process.wait(timeout=5)
        except BaseException as cleanup:
            record_cleanup(cleanup, "forced cleanup")
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except BaseException as cleanup:
                    record_cleanup(cleanup, "pipe close")
    if failure is not None:
        raise failure


def probe_servers(profile, group_config, read_paths, trial_dir, gateway):
    observed = {}
    if not group_config:
        return observed
    configured = json.loads((profile.root / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    for name, spec in group_config.mcp_servers.items():
        server = configured[name]
        command = native_command(
            [server["command"], *server["args"]],
            read=read_paths,
            write=[profile.root],
            cwd=profile.root,
            policy_path=trial_dir / f"mcp-{name}.sb",
            gateway_port=gateway.port,
            gateway_socket=gateway.unix_socket,
        )
        close_owner = lambda: None  # noqa: E731
        process = None
        reader = None
        failure = None
        with (trial_dir / f"mcp-{name}.stderr").open("xb") as stderr:
            try:
                process, close_owner = spawn_owned_popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    env=dict(profile.env, **server["env"]),
                    cwd=profile.root,
                    bufsize=0,
                    **subprocess_window_kwargs(),
                )
                reader = PipeReader(process.stdout)
                buffer = b""

                def send(value):
                    process.stdin.write((json.dumps(value) + "\n").encode())
                    process.stdin.flush()

                def response(identity):
                    nonlocal buffer
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        while b"\n" not in buffer:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(f"required MCP component {name} did not answer")
                            timed_out, chunk = reader.read_chunk(remaining)
                            if timed_out:
                                raise TimeoutError(f"required MCP component {name} did not answer")
                            if not chunk:
                                raise RuntimeError(f"required MCP component {name} exited")
                            buffer += chunk
                            if len(buffer) > 4 * 1024 * 1024:
                                raise ValueError("MCP initialization output exceeds bound")
                        line, buffer = buffer.split(b"\n", 1)
                        message = json.loads(line)
                        if message.get("id") == identity:
                            if "error" in message:
                                raise RuntimeError(f"required MCP component {name}: {message['error']}")
                            return message["result"]
                    raise TimeoutError(f"required MCP component {name} timed out")

                send(
                    dict(
                        jsonrpc="2.0",
                        id=1,
                        method="initialize",
                        params=dict(
                            protocolVersion="2024-11-05",
                            capabilities={},
                            clientInfo=dict(name="mindie-bench-preflight", version=__version__),
                        ),
                    )
                )
                initialized = response(1)
                send(dict(jsonrpc="2.0", method="notifications/initialized"))
                send(dict(jsonrpc="2.0", id=2, method="tools/list", params={}))
                catalog = response(2)
                if catalog.get("nextCursor"):
                    raise ValueError(
                        "required MCP catalog is paginated; configure the exact component adapter"
                    )
                tools = {tool["name"]: tool for tool in catalog["tools"]}
                required = set(spec.required_tools)
                if not required:
                    aliases = {
                        "remote-dev": {"remote_bash", "bash"},
                        "knowledge": {"knowledge_query", "query"},
                    }.get(spec.component, set(tools))
                    if not (aliases & tools.keys()):
                        raise ValueError(
                            f"MCP server {name} lacks its declared {spec.component} capability"
                        )
                if not required <= tools.keys():
                    raise ValueError(
                        f"MCP server {name} lacks required tools: {sorted(required - tools.keys())}"
                    )
                observed[name] = dict(
                    server_info=initialized.get("serverInfo"),
                    tool_names=sorted(tools),
                    tool_schema_sha256=digest(catalog),
                )
                if (
                    spec.component == "knowledge"
                    and profile.corpus is not None
                    and graded_view.is_graded(profile.corpus)
                ):
                    query_tool = next(tool for tool in ("knowledge_query", "query") if tool in tools)
                    send(
                        dict(
                            jsonrpc="2.0",
                            id=3,
                            method="tools/call",
                            params=dict(
                                name=query_tool,
                                arguments=dict(
                                    query="benchmark readiness",
                                    conditions=profile.fingerprint["knowledge_conditions"],
                                    limit=1,
                                ),
                            ),
                        )
                    )
                    result = response(3)
                    if result.get("isError"):
                        raise ValueError(
                            f"graded knowledge query failed during zero-model preflight: {result}"
                        )
                    observed[name]["read_probe"] = dict(
                        tool=query_tool, response_sha256=digest(result), status="completed"
                    )
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    if process:
                        close_probe(process, reader=reader)
                except BaseException as cleanup:
                    if failure is None:
                        raise
                    failure.add_note(f"MCP preflight shutdown also failed: {cleanup}")
                finally:
                    close_owner()
    return observed
