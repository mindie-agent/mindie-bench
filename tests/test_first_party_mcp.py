import asyncio
import json
import sqlite3
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from mindie_bench.integrations import first_party


def install_mcp(monkeypatch, scenario):
    module = ModuleType("mcp.server.fastmcp")

    class FakeMCP:
        def __init__(self, name):
            self.tools = {}

        def tool(self):
            def register(function):
                self.tools[function.__name__] = function
                return function

            return register

        def run(self, transport):
            assert transport == "stdio"
            asyncio.run(scenario(self.tools))

    module.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp", ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", module)


def test_mcp_owns_store_on_one_thread_outside_server_event_loop(tmp_path, monkeypatch):
    events = []
    result = [dict(ref="exact-block", related=[], related_count=0, related_next=None)]

    class ThreadBoundStore:
        def __init__(self):
            self.db = sqlite3.connect(":memory:")
            self.runner = asyncio.Runner()
            events.append(("create", threading.get_ident()))

        def search(self, **request):
            self.db.execute("SELECT 1").fetchone()

            async def retrieve():
                return result

            events.append(("search", threading.get_ident()))
            return self.runner.run(retrieve())

        def close(self):
            self.db.close()
            self.runner.close()
            events.append(("close", threading.get_ident()))

    def consume(*args):
        return ThreadBoundStore()

    def read(store, ref):
        store.db.execute("SELECT 1").fetchone()
        events.append(("explain", threading.get_ident()))
        assert ref == "exact-block"
        return {"text": "authoritative body"}

    async def scenario(tools):
        # Concurrent MCP requests must serialize on the same component owner.
        pages = await asyncio.gather(
            tools["knowledge_query"](query="NPU"),
            tools["knowledge_query"](query=None, continuation="cursor"),
        )
        assert pages == [result, result]
        assert await tools["knowledge_explain"](ref=pages[0][0]["ref"]) == {"text": "authoritative body"}

    install_mcp(monkeypatch, scenario)
    monkeypatch.setattr(first_party, "consumer", consume)
    monkeypatch.setattr(first_party, "explain", read)
    first_party.serve(SimpleNamespace(corpus=str(tmp_path), state=str(tmp_path)))
    assert [name for name, _ in events] == ["create", "search", "search", "explain", "close"]
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != threading.get_ident()
    observations = [
        json.loads(line) for line in (tmp_path / "retrieval-observations.jsonl").read_text().splitlines()
    ]
    assert [entry["kind"] for entry in observations] == ["query", "query", "explain"]
    assert observations[1]["continuation"] == "cursor"
    assert observations[0]["result"] == result


def test_mcp_preserves_query_failure_and_close_failure(tmp_path, monkeypatch):
    owner = []

    class BrokenStore:
        def search(self, **request):
            owner.append(threading.get_ident())
            raise ValueError("required index is corrupt")

        def close(self):
            owner.append(threading.get_ident())
            raise RuntimeError("close also failed")

    async def scenario(tools):
        await tools["knowledge_query"](query="NPU")

    install_mcp(monkeypatch, scenario)
    monkeypatch.setattr(first_party, "consumer", lambda *args: BrokenStore())
    with pytest.raises(ValueError, match="required index is corrupt") as caught:
        first_party.serve(SimpleNamespace(corpus=str(tmp_path), state=str(tmp_path)))
    assert owner[0] == owner[1]
    assert any("close also failed" in note for note in caught.value.__notes__)
    observation = json.loads((tmp_path / "retrieval-observations.jsonl").read_text())
    assert observation["error"] == {"type": "ValueError", "message": "required index is corrupt"}
    assert "result" not in observation


def test_consumer_releases_partly_initialized_store_without_hiding_install_error(tmp_path, monkeypatch):
    closed = []
    module = ModuleType("mindie_knowledge.materials.store")

    class BrokenInstall:
        def __init__(self, *args):
            pass

        def install_packages(self, *args, **kwargs):
            raise ValueError("package identity is invalid")

        def close(self):
            closed.append(True)
            raise RuntimeError("partial store close failed")

    module.MaterialStore = BrokenInstall
    monkeypatch.setitem(sys.modules, "mindie_knowledge", ModuleType("mindie_knowledge"))
    monkeypatch.setitem(
        sys.modules, "mindie_knowledge.materials", ModuleType("mindie_knowledge.materials")
    )
    monkeypatch.setitem(sys.modules, "mindie_knowledge.materials.store", module)
    with pytest.raises(ValueError, match="package identity is invalid") as caught:
        first_party.consumer(tmp_path, tmp_path, "revision")
    assert closed == [True]
    assert any("partial store close failed" in note for note in caught.value.__notes__)
