"""Fail visibly when a required component cannot initialize or expose its tools."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time

from mindie_bench.io import digest
from mindie_bench.runner.sandbox import native_command


def probe_servers(profile, group_config, read_paths, trial_dir, gateway):
    observed = {}
    if not group_config:
        return observed
    configured = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]
    for name, spec in group_config.mcp_servers.items():
        server = configured[name]
        command = native_command(
            [server["command"], *server["args"]],
            read=read_paths,
            write=[profile.root],
            cwd=profile.root,
            policy_path=trial_dir / f"mcp-{name}.sb",
        )
        read_fd, owner_fd = os.pipe()
        process = None
        with (trial_dir / f"mcp-{name}.stderr").open("xb") as stderr:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-m", "mindie_bench.runner.owner", str(read_fd), *command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    env=dict(profile.env, **server["env"]),
                    cwd=profile.root,
                    start_new_session=True,
                    pass_fds=(read_fd,),
                    bufsize=0,
                )
                os.close(read_fd)
                read_fd = -1
                buffer = b""

                def send(value):
                    process.stdin.write((json.dumps(value) + "\n").encode())
                    process.stdin.flush()

                def response(identity):
                    nonlocal buffer
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        while b"\n" not in buffer:
                            remaining = deadline - time.monotonic()
                            if (
                                remaining <= 0
                                or not select.select([process.stdout], [], [], remaining)[0]
                            ):
                                raise TimeoutError(f"required MCP component {name} did not answer")
                            chunk = os.read(process.stdout.fileno(), 65536)
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
                            clientInfo=dict(name="mindie-bench-preflight", version="0.1.0"),
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
            finally:
                if process:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                    process.stdin.close()
                    process.stdout.close()
                os.close(owner_fd)
                if read_fd >= 0:
                    os.close(read_fd)
    return observed
