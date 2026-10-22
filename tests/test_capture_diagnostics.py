import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mindie_bench.models import CorpusConfig, Group, GroupConfig, HarnessConfig, MCPServer, Tokens
from mindie_bench.runner import orchestrator
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.process import ProcessResult
from mindie_bench.runner.usage import UsageMeter


def test_budget_refusal_journals_zero_new_admission_with_settled_first_request(case, monkeypatch):
    accepted = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            accepted.append(self.rfile.read(int(self.headers["Content-Length"])))
            raw = b'data: {"type":"response.completed","response":{"usage":{"input_tokens":12,"output_tokens":3}}}\n\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_BROKER_KEY", "zero-paid-mock")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="mock",
        expected_version="test",
        api_key_env="TEST_BROKER_KEY",
        provider_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    case["spec"].budget.max_turns = 1
    journal = case["root"] / "broker.jsonl"
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            first = json.dumps({"model": "mock", "input": "first"}).encode()
            request = urllib.request.Request(
                gateway.url + "/v1/responses", data=first, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                response.read()
            second = json.dumps({"model": "mock", "input": "a distinct second request"}).encode()
            request = urllib.request.Request(
                gateway.url + "/v1/responses", data=second, headers={"Content-Type": "application/json"}
            )
            with pytest.raises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(request, timeout=5)
            assert failure.value.code == 429
            gateway.finish_inference()
            assert gateway.model_calls == len(accepted) == 1
            assert gateway.stop_reason == "max_turns"
            assert gateway.metered_tokens().complete
            assert not gateway.unknown
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        rejected = [event for event in events if event["kind"] == "model_admission_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "max_turns"
        assert rejected[0]["http_status"] == 429
        assert rejected[0]["request_sha256"] == hashlib.sha256(second).hexdigest()
        assert rejected[0]["admitted_model_calls"] == rejected[0]["limit"] == 1
        assert "input" not in rejected[0]
        assert sum(event["kind"] == "model_start" for event in events) == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize(
    "capture_errors",
    [[], [{"stage": "agent", "type": "ValueError", "message": "original native event diagnostic"}]],
)
def test_capture_interruption_preserves_process_state_and_separate_settled_usage(
    case, corpus, monkeypatch, capture_errors
):
    source, commit = corpus
    config = case["config"]
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.groups[Group.G3] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge"),
        },
        capture_command=["/usr/bin/true"],
        capture_max_model_calls=1,
    )
    monkeypatch.setattr(orchestrator, "probe_runtime_import", lambda *args: {})
    monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)

    def process(command, *, gateway, output, consume, cwd, **kwargs):
        if output.name == "native_events.jsonl":
            (cwd / "module.py").write_text("VALUE = 2\n")
            consume({"type": "fixture.trajectory", "trajectory": {"native_completed": True}}, 0, 0)
            return ProcessResult(returncode=0, elapsed=0.1, started=True)
        gateway.model_calls = 1
        meter = UsageMeter()
        meter.tokens = Tokens(input_uncached=12, output=3, complete=True)
        gateway.usage_meters[1] = meter
        return ProcessResult(
            returncode=None, elapsed=2.7, stop_reason="max_turns", errors=capture_errors, started=True
        )

    monkeypatch.setattr(orchestrator, "stream_process", process)
    output = case["root"] / "run"
    output.mkdir()
    receipt = orchestrator.run_trial(
        case["task"], config, tier="l2", group=Group.G3, repetition=1, run_root=output
    )
    assert receipt.terminal_status.value == "completed_pass"
    assert receipt.auxiliary_metrics["business_terminal_status"] == "completed_pass"
    assert receipt.auxiliary_metrics["capture"]["status"] == "failed"
    assert any(error["stage"] == "capture" for error in receipt.errors)
    assert receipt.auxiliary_metrics["capture_provider_tokens"]["complete"]
    assert receipt.auxiliary_metrics["capture_model_calls"] == 1
    expected = {
        "returncode": None,
        "stop_reason": "max_turns",
        "errors": capture_errors,
        "elapsed_sec": 2.7,
        "started": True,
    }
    assert receipt.auxiliary_metrics["capture_process"] == expected
    saved = json.loads((Path(receipt.artifacts["private_trial_dir"]) / "trial_receipt.json").read_text())
    assert saved["auxiliary_metrics"]["capture_process"] == expected
    assert not receipt.primary_metrics["tokens"]["complete"]
    assert receipt.primary_metrics["tokens"]["indexer_small_model_total"] == 15


def test_corpus_rejection_findings_survive_without_inventing_capture_token_receipts(
    case, corpus, monkeypatch
):
    source, commit = corpus
    config = case["config"]
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.groups[Group.G3] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge"),
        },
        capture_command=["/usr/bin/true"],
    )
    monkeypatch.setattr(orchestrator, "probe_runtime_import", lambda *args: {})
    monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)
    report = {"status": "rejected", "findings": [{"code": "unconditional_global_ban"}]}

    def process(command, *, output, consume, cwd, **kwargs):
        if output.name == "native_events.jsonl":
            (cwd / "module.py").write_text("VALUE = 2\n")
            consume({"type": "fixture.trajectory", "trajectory": {"native_completed": True}}, 0, 0)
            return ProcessResult(returncode=0, elapsed=0.1, started=True)
        consume({"status": "failed", "corpus_lint": report}, 0, 0)
        return ProcessResult(returncode=1, elapsed=0.1, started=True)

    monkeypatch.setattr(orchestrator, "stream_process", process)
    output = case["root"] / "rejected"
    output.mkdir()
    receipt = orchestrator.run_trial(
        case["task"], config, tier="l2", group=Group.G3, repetition=1, run_root=output
    )
    assert receipt.terminal_status.value == "completed_pass"
    assert receipt.primary_metrics["reward"] == 1
    assert receipt.auxiliary_metrics["capture"]["corpus_lint"] == report
    assert receipt.auxiliary_metrics["capture"]["status"] == "failed"
    assert receipt.auxiliary_metrics["business_terminal_status"] == "completed_pass"
    assert not any(error["stage"] == "scoring" for error in receipt.errors)
