import asyncio
import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from mindie_bench.integrations.graded_consumer import GradedConsumer
from mindie_bench.runner.graded_view import identifiers
from mindie_bench.runner.profile_builder import filtered_corpus


def corpus(tmp_path):
    knowledge = pytest.importorskip("mindie_knowledge.graded.units")
    contract = pytest.importorskip("mindie_knowledge.publication_contract")
    source = tmp_path / "source"
    source.mkdir()
    (source / "publication-contract.json").write_text(
        contract.render_contract(contract.make_contract("npu", "a" * 40, schema=contract.GRADED_SCHEMA))
    )
    for name, parents in (("unit-base", []), ("unit-child", ["unit-base"]), ("unit-independent", [])):
        unit = knowledge.GradedUnit.from_parts(
            dict(
                schema="mindie-graded-unit/1",
                id=name,
                tier="L2",
                category="sop",
                evidence_strength="single_observation",
                domain_path="serving/queue",
                title="Queue cancellation ownership",
                inherits_from=parents,
                conditions={"source_revision": "b" * 40},
                triggers={"error_signatures": ["queue_owned"], "keywords": ["cancellation", "queue"]},
                provenance={
                    "source_type": "ecosystem_repo",
                    "source_locator": "example/repo@" + "b" * 40 + ":queue.py",
                    "source_digest": "c" * 64,
                    "evidence_refs": ["queue-source"],
                    "supersedes": [],
                },
            ),
            f"Check queue ownership before cancellation for {name}; the observed request was still queued.",
        )
        path = source / f"serving/queue/{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit.render())
    return source


def test_projection_removes_excluded_parent_descendants_and_non_authoritative_docs(tmp_path):
    source = corpus(tmp_path)
    (source / "docs").mkdir()
    (source / "docs" / "hidden-answer.md").write_text("excluded heldout answer")
    (source / "index").mkdir()
    (source / "index" / "routing-manifest.json").write_text(json.dumps({"ids": ["unit-base"]}))
    destination = tmp_path / "view"
    filtered_corpus(source, destination, {"unit-base"})
    assert identifiers(destination) == {"unit-independent"}
    assert sorted(
        p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
    ) == ["publication-contract.json", "serving/queue/unit-independent.md"]


def test_graded_consumer_preserves_conditions_and_exact_current_read(tmp_path):
    source = corpus(tmp_path)
    consumer = GradedConsumer(source, tmp_path / "state")
    try:
        assert consumer.search("queue cancellation")["results"] == []
        assert (
            consumer.search("queue cancellation", conditions={"source_revision": "d" * 40})["results"]
            == []
        )
        hits = consumer.search("queue cancellation", conditions={"source_revision": "b" * 40})["results"]
        assert hits
        read = consumer.explain(hits[0]["ref"])
        assert read["revision"] == hits[0]["revision"]
        assert read["ref"] == hits[0]["ref"]
        assert "queue ownership" in read["content"]
    finally:
        consumer.close()


def test_changed_graded_body_replaces_revision_without_substituting_old_reads(tmp_path):
    source = corpus(tmp_path)
    state = tmp_path / "state"
    path = source / "serving/queue/unit-independent.md"
    with closing(GradedConsumer(source, state)) as consumer:
        hits = consumer.search(
            "queue cancellation", conditions={"source_revision": "b" * 40}
        )["results"]
        previous = next(hit for hit in hits if hit["id"] == "unit-independent")
        original = consumer.explain(previous["ref"])
        path.write_text(path.read_text() + "A later reviewed correction.\n")
        assert consumer.explain(previous["ref"])["content"] == original["content"]

    with closing(GradedConsumer(source, state)) as consumer:
        hits = consumer.search(
            "queue cancellation", conditions={"source_revision": "b" * 40}
        )["results"]
        current = next(hit for hit in hits if hit["id"] == "unit-independent")
        assert current["revision"] == previous["revision"] + 1
        with pytest.raises(ValueError, match="replaced"):
            consumer.explain(previous["ref"])
        assert "later reviewed correction" in consumer.explain(current["ref"])["content"]


def test_real_mcp_serializes_graded_conditions_and_exact_read(tmp_path, monkeypatch):
    source = corpus(tmp_path)
    FastMCP = pytest.importorskip("mcp.server.fastmcp").FastMCP

    from mindie_bench.integrations import first_party

    async def scenario(server):
        _, unknown = await server.call_tool("knowledge_query", {"query": "queue cancellation"})
        assert unknown["result"]["results"] == []
        _, matched = await server.call_tool(
            "knowledge_query",
            {"query": "queue cancellation", "conditions": {"source_revision": "b" * 40}},
        )
        hit = matched["result"]["results"][0]
        contents = await server.call_tool("knowledge_explain", {"ref": hit["ref"]})
        exact = json.loads(contents[0].text)
        assert exact["revision"] == hit["revision"]
        assert exact["ref"] == hit["ref"]
        assert "queue ownership" in exact["content"]

    def drive(server, transport):
        assert transport == "stdio"
        asyncio.run(scenario(server))

    # Keep the actual MCP input/output validation and C5 Store; only replace
    # the blocking transport loop with deterministic requests to that server.
    monkeypatch.setattr(FastMCP, "run", drive)
    first_party.serve(SimpleNamespace(corpus=str(source), state=str(tmp_path / "state")))
