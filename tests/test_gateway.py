import json
import urllib.error
import urllib.request

import pytest

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway


def post(url, path, payload):
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=5)


def test_no_pr_issue_or_arbitrary_http_egress(case):
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        for path in ["/repos/mindie-agent/knowledge/pulls", "/repos/x/y/issues", "/v1/uploads"]:
            with pytest.raises(urllib.error.HTTPError) as failure:
                post(gateway.url, path, {"body": "should never leave this machine"})
            assert failure.value.code == 403
        assert gateway.model_calls == 0


def test_replay_rule_paths_never_leave_parent_and_have_hard_budget(case):
    case["spec"].budget.max_npu_runs = 1
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        with post(gateway.url, "/bench/replay", {"argv": ["pytest", "-q"]}) as response:
            payload = json.load(response)
            assert payload["exit_code"] == 1
            assert "checks" not in payload
        with post(gateway.url, "/bench/replay", {"argv": ["pytest"]}) as response:
            assert json.load(response)["exit_code"] == 124
        assert gateway.stop_reason == "max_npu_runs"
        assert gateway.replay_calls == 1


def test_replay_symlink_escape_rejected(case):
    (case["repo"] / "module.py").unlink()
    (case["repo"] / "module.py").symlink_to(case["task"] / "oracle/replay_rules.json")
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/bench/replay", {"argv": ["pytest"]})
        assert failure.value.code == 500


def test_unknown_paid_request_not_retried_and_key_stays_parent(case, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_KEY", "synthetic-parent-only-key")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="test",
        expected_version="test",
        api_key_env="BENCH_TEST_KEY",
        provider_url="https://example.invalid/v1",
    )
    attempts = []

    class BrokenProvider:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, body, headers):
            attempts.append(headers["Authorization"])
            raise TimeoutError("synthetic lost provider receipt")

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", BrokenProvider)
    with Gateway(harness, case["spec"], case["task"], case["repo"], case["root"] / "journal") as gateway:
        try:
            post(gateway.url, "/v1/responses", {"model": "test", "input": "sample"})
        except (OSError, Exception):
            pass
        assert gateway.unknown
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/v1/responses", {"model": "test", "input": "sample"})
        assert failure.value.code == 409
        assert len(attempts) == 1
        assert gateway.model_calls == 1
    assert "synthetic-parent-only-key" not in (case["root"] / "journal").read_text()


def test_agent_cannot_change_the_controlled_model(case, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_KEY", "synthetic-key")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="pinned-model",
        expected_version="test",
        api_key_env="BENCH_TEST_KEY",
        provider_url="https://example.invalid/v1",
    )
    with Gateway(harness, case["spec"], case["task"], case["repo"], case["root"] / "journal") as gateway:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/v1/responses", {"model": "different-model", "input": "sample"})
        assert failure.value.code == 403
        assert gateway.model_calls == 0
        assert gateway.infrastructure_error
