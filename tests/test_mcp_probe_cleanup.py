import json

import pytest
from test_knowledge_broker import setup_profile

from mindie_bench.models import Group, GroupConfig
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.mcp_probe import probe_servers
from mindie_bench.runner.orchestrator import runtime_paths
from mindie_bench.runner.profile_builder import bind_mcp_transport


def test_preflight_closes_the_parent_backend_before_foreground_admission(case, corpus):
    config, trial, profile = setup_profile(case, corpus)
    paths = runtime_paths(config, profile)
    only_knowledge = GroupConfig(
        mcp_servers={"knowledge": config.groups[Group.G2].mcp_servers["knowledge"]}
    )
    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, paths)
        profile.env["MINDIE_BENCH_GATEWAY"] = gateway.url
        bind_mcp_transport(profile)
        observed = probe_servers(profile, only_knowledge, paths, trial, gateway)
        assert observed["knowledge"]["tool_names"] == ["knowledge_explain", "knowledge_query"]
        assert gateway.knowledge.sessions
        assert all(
            session.closed and session.process.poll() == 0
            for session in gateway.knowledge.sessions.values()
        )
        events = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
        closes = [row for row in events if row["kind"] == "knowledge_session_close"]
        assert len(closes) == 1
        assert closes[0]["returncode"] == 0
        assert closes[0]["pending_requests"] == 0
        assert closes[0]["outcome"] == "completed"
        assert gateway.infrastructure_error is None


def test_original_catalog_failure_survives_a_separate_backend_shutdown_failure(case, corpus):
    config, trial, profile = setup_profile(case, corpus)
    script = case["root"] / "fixture-mcp.py"
    script.write_text(
        script.read_text().replace('[{"name":"knowledge_query"},{"name":"knowledge_explain"}]', "[]")
        + "\nraise SystemExit(7)\n"
    )
    only_knowledge = GroupConfig(
        mcp_servers={"knowledge": config.groups[Group.G2].mcp_servers["knowledge"]}
    )
    paths = runtime_paths(config, profile)
    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, paths)
        profile.env["MINDIE_BENCH_GATEWAY"] = gateway.url
        bind_mcp_transport(profile)
        with pytest.raises(ValueError, match="lacks its declared knowledge capability") as original:
            probe_servers(profile, only_knowledge, paths, trial, gateway)
        assert any("shutdown also failed" in note for note in original.value.__notes__)
        assert all(
            session.closed and session.process.poll() == 7
            for session in gateway.knowledge.sessions.values()
        )
