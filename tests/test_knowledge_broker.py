import json
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mindie_bench.models import CorpusConfig, Group, GroupConfig, MCPServer
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.orchestrator import runtime_paths
from mindie_bench.runner.profile_builder import bind_mcp_transport, build_profile
from mindie_bench.runner.sandbox import native_command

SERVER = """
import json,sys
from pathlib import Path
for line in sys.stdin:
    request=json.loads(line)
    if "id" not in request:
        continue
    method=request["method"]
    if method=="initialize":
        result={"protocolVersion":"2024-11-05","serverInfo":{"name":"transport-fixture","version":"1"}}
    elif method=="tools/list":
        result={"tools":[{"name":"knowledge_query"},{"name":"knowledge_explain"}]}
    else:
        if request["params"]["arguments"].get("fail"):
            result={"isError":True,"content":[{"type":"text","text":"fixture error remains visible"}]}
        else:
            result={"structuredContent":{"text":Path(sys.argv[1]).read_text()},"isError":False}
    print(json.dumps({"jsonrpc":"2.0","id":request["id"],"result":result}),flush=True)
"""


def setup_profile(case, corpus):
    source, commit = corpus
    server = case["root"] / "fixture-mcp.py"
    server.write_text(SERVER)
    config = case["config"].model_copy(deep=True)
    config.runtime_read_paths = [str(server)]
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.groups[Group.G2] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(
                command=sys.executable,
                args=["-I", str(server), "{corpus}/tasks/analogy/materials/0001.md"],
                env={"STATE": "{profile}/knowledge-store"},
                component="knowledge",
            ),
        }
    )
    trial = case["root"] / "trial"
    trial.mkdir()
    profile = build_profile(trial, case["spec"], Group.G2, config)
    return config, trial, profile


def start_front(profile, paths, trial, gateway, code=None, args=()):
    spec = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]["knowledge"]
    base = [sys.executable, "-I", "-c", code, *args] if code else [spec["command"], *spec["args"]]
    command = native_command(
        base,
        read=paths,
        write=[profile.root],
        cwd=profile.root,
        policy_path=trial / ("client.sb" if not code else "direct-read.sb"),
        gateway_port=gateway.port,
    )
    return subprocess.Popen(
        command,
        env=profile.env,
        cwd=profile.root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def request(process, identity, method, params=None):
    process.stdin.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": identity,
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        ).encode()
    )
    process.stdin.flush()
    data = b""
    deadline = time.monotonic() + 10
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "MCP transport fixture timed out"
        assert select.select([process.stdout], [], [], remaining)[0]
        chunk = os.read(process.stdout.fileno(), 65536)
        assert chunk, process.stderr.read().decode()
        data += chunk
    return json.loads(data)


def stop_front(process):
    if not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=10)
        stderr = process.stderr.read().decode()
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_profile_does_not_disclose_or_grant_private_knowledge(case, corpus):
    _, _, profile = setup_profile(case, corpus)
    assert not profile.corpus.is_relative_to(profile.root)
    assert profile.corpus == profile.private_root / "corpus"
    assert (profile.private_root / "local_knowledge.git/HEAD").is_file()
    public = (profile.root / "mcp.json").read_text() + json.dumps(profile.env)
    assert str(profile.corpus) not in public
    assert str(profile.private_root) not in public
    assert "MINDIE_BENCH_CORPUS" not in profile.env
    assert "knowledge-store" not in public
    assert str(profile.private_root) in json.dumps(profile.knowledge_servers)
    assert profile.fingerprint["knowledge_backend_config_sha256"]["knowledge"]
    assert profile.fingerprint["retained_analogical_task_hashes"] == ["analogy"]
    assert Path(sys.executable).absolute() not in profile.runtime_paths


@pytest.mark.parametrize("field", ["same_origin_task_hashes", "analogical_task_hashes"])
def test_declared_knowledge_labels_must_exist_in_the_actual_source(case, corpus, field):
    config, _, _ = setup_profile(case, corpus)
    setattr(case["spec"].mindie_knowledge, field, ["absent-task"])
    trial = case["root"] / "missing-label-trial"
    trial.mkdir()
    with pytest.raises(ValueError, match="absent from the source corpus"):
        build_profile(trial, case["spec"], Group.G2, config)


def test_declared_analogy_must_survive_the_actual_warm_projection(case, corpus):
    from mindie_bench.runner.profile_builder import filtered_corpus

    config, _, _ = setup_profile(case, corpus)
    source, _ = corpus
    warm = case["root"] / "warm-with-missing-analogy"
    filtered_corpus(source, warm, {"analogy"})
    trial = case["root"] / "missing-warm-analogy"
    trial.mkdir()
    with pytest.raises(ValueError, match="must remain in the filtered corpus"):
        build_profile(trial, case["spec"], Group.G2, config, corpus_override=warm)


def test_actual_os_boundary_denies_direct_corpus_and_state_but_relays_mcp(case, corpus):
    # This fixture verifies real OS/process/HTTP transport; real-component semantic
    # acceptance is performed separately with the locked component runtime.
    config, trial, profile = setup_profile(case, corpus)
    state = profile.private_root / "knowledge-store"
    state.mkdir()
    (state / "index.sqlite").write_text("private cache contents")
    material = profile.corpus / "tasks/analogy/materials/0001.md"
    expected = material.read_text()
    alias = profile.root / "knowledge-alias"
    alias.symlink_to(profile.private_root, target_is_directory=True)
    paths = runtime_paths(config, profile)
    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, paths)
        profile.env["MINDIE_BENCH_GATEWAY"] = gateway.url
        bind_mcp_transport(profile)
        code = (
            "from pathlib import Path\nimport sys\n"
            "import os\n"
            "for path in sys.argv[1:]:\n"
            " try: Path(path).read_bytes()\n"
            " except (PermissionError,FileNotFoundError): pass\n"
            " else: raise SystemExit('PRIVATE_KNOWLEDGE_VISIBLE')\n"
            " try: os.link(path,'owned-hardlink')\n"
            " except OSError: pass\n"
            " else:\n"
            "  try: Path('owned-hardlink').read_bytes()\n"
            "  except (PermissionError,FileNotFoundError): pass\n"
            "  else: raise SystemExit('PRIVATE_KNOWLEDGE_HARDLINK_VISIBLE')\n"
            " Path('owned-hardlink').unlink(missing_ok=True)\n"
            "Path('owned-public-file').write_text('allowed')\nprint('blocked')\n"
        )
        direct = start_front(
            profile,
            paths,
            trial,
            gateway,
            code,
            [
                str(material),
                str(state / "index.sqlite"),
                str(alias / "corpus/tasks/analogy/materials/0001.md"),
            ],
        )
        stdout, stderr = direct.communicate(timeout=10)
        assert direct.returncode == 0, stderr.decode()
        assert stdout.strip() == b"blocked"
        front = start_front(profile, paths, trial, gateway)
        try:
            assert request(front, 1, "initialize")["result"]["serverInfo"]["name"] == "transport-fixture"
            result = request(
                front,
                2,
                "tools/call",
                {
                    "name": "knowledge_explain",
                    "arguments": {"ref": "fixture-ref"},
                },
            )
            assert result["result"]["structuredContent"]["text"] == expected
            failed = request(
                front,
                3,
                "tools/call",
                {
                    "name": "knowledge_query",
                    "arguments": {"fail": True},
                },
            )
            assert failed["result"]["isError"] is True
        finally:
            stop_front(front)
        assert gateway.infrastructure_error is None
    events = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
    ends = [event for event in events if event["kind"] == "knowledge_tool_end"]
    assert [row["is_error"] for row in ends] == [False, True]
    assert all(row["elapsed_sec"] >= 0 for row in ends)
    assert all(len(row["response_sha256"]) == 64 for row in ends)
    assert all(s.closed and s.process.poll() is not None for s in gateway.knowledge.sessions.values())
    assert not (profile.root / "knowledge-store").exists()


def test_unknown_backend_exit_is_visible_and_does_not_fall_back(case, corpus):
    config, trial, profile = setup_profile(case, corpus)
    script = config.runtime_read_paths[0]
    Path(script).write_text("raise SystemExit(7)\n")
    with pytest.raises(RuntimeError, match="code 7"):
        with Gateway(
            config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
        ) as gateway:
            gateway.attach_knowledge(profile, runtime_paths(config, profile))
            opened = gateway.knowledge.request({"action": "open", "server": "knowledge"})
            identity = opened["session_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = gateway.knowledge.request(
                    {
                        "action": "exchange",
                        "session_id": identity,
                    }
                )
                if result["exit_code"] is not None:
                    assert result["exit_code"] == 7
                    break
            else:
                pytest.fail("broken backend did not exit")
    assert gateway.infrastructure_error
    assert all(s.closed and s.process.poll() is not None for s in gateway.knowledge.sessions.values())


def test_parent_cleanup_drain_does_not_claim_a_late_response_reached_the_front(case, corpus):
    config, trial, profile = setup_profile(case, corpus)
    script = Path(config.runtime_read_paths[0])
    script.write_text(
        SERVER.replace(
            "    print(json.dumps(",
            "    import time; time.sleep(0.2)\n    print(json.dumps(",
        )
    )
    import base64

    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, runtime_paths(config, profile))
        identity = gateway.knowledge.request({"action": "open", "server": "knowledge"})["session_id"]
        result = gateway.knowledge.request(
            {
                "action": "exchange",
                "session_id": identity,
                "stdin_b64": base64.b64encode(
                    b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
                ).decode(),
            }
        )
        assert not result["stdout_b64"]
    rows = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
    drained = [row for row in rows if row["kind"] == "knowledge_delivery_failed"]
    assert drained and all(row["stage"] == "parent_cleanup_drain" for row in drained)
    assert all(row["stdout_bytes"] > 0 and row["external_effects"] == "completed" for row in drained)
    assert all(s.closed and s.process.poll() is not None for s in gateway.knowledge.sessions.values())


@pytest.mark.parametrize("pure_poll", [True, False])
def test_closed_client_distinguishes_an_empty_poll_from_a_required_mcp_response(
    case, tmp_path, pure_poll
):
    admitted, release = threading.Event(), threading.Event()

    class TransportFixture:
        def request(self, payload):
            admitted.set()
            assert release.wait(timeout=5)
            return {"stdout_b64": "" if pure_poll else "e30K", "exit_code": None}

        def close(self):
            pass

    journal = tmp_path / "broker.jsonl"
    with Gateway(case["config"].harness, case["spec"], case["task"], case["repo"], journal) as gateway:
        gateway.knowledge = TransportFixture()
        body = json.dumps(
            {
                "action": "exchange",
                "session_id": "fixture",
                "stdin_b64": "" if pure_poll else "e30K",
            }
        ).encode()
        connection = socket.create_connection(("127.0.0.1", gateway.port))
        connection.sendall(
            b"POST /bench/knowledge-stdio HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        assert admitted.wait(timeout=5)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.close()
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rows = (
                [json.loads(line) for line in journal.read_text().splitlines()]
                if journal.exists()
                else []
            )
            failed = [row for row in rows if row["kind"] == "knowledge_delivery_failed"]
            if failed:
                assert failed[0]["external_effects"] == ("none" if pure_poll else "completed")
                break
            time.sleep(0.01)
        else:
            pytest.fail("lost MCP delivery did not retain a receipt")
        assert bool(gateway.infrastructure_error) is not pure_poll
