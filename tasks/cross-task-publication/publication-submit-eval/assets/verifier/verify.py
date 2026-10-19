# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from contextlib import closing

from mindie_knowledge.loop import settings
from mindie_knowledge.loop.engine import Engine
from mindie_knowledge.loop.export import build_batch
from mindie_knowledge.loop.store import Store

assert Path(importlib.import_module(Store.__module__).__file__).resolve().is_relative_to(WORKSPACE)


def check():
    for failure in (False, True):
        with tempfile.TemporaryDirectory(prefix="l2-submit-cleanup-") as directory:
            root = Path(directory)
            shared = settings.write(
                root / "sharing.json",
                enabled=True,
                repository="fixture-owner/fixture-corpus",
                project_roots=[root],
                idle_seconds=300,
            )
            with closing(Store(root / "data", "test")) as store:
                store.create_draft(
                    kind="experience",
                    title="Public method",
                    summary="Fixture.",
                    content="A source-bound local protocol check.",
                    generation=shared.generation,
                )
                batch_id = build_batch(store, settings=shared)[0]
                engine = Engine(store, settings_path=shared.path)
                sent = []
                receipt = {
                    "status": "submitted",
                    "head_sha": "b" * 40,
                    "pr_url": "https://example.invalid/pull/2",
                }

                def submit(*args, **kwargs):
                    sent.append(batch_id)
                    return dict(receipt)

                engine.community = {"submit_batch": submit}
                if failure:

                    def compact(*args, **kwargs):
                        raise OSError("fixture local cleanup after accepted publication")

                    store.compact_confirmed = compact
                try:
                    result = engine.publisher.submit(store.batch(batch_id))
                except OSError as exc:
                    observed = getattr(exc, "mindie_publication_result", None)
                    print(json.dumps({"cleanup_failed": True, "returned_result": observed}))
                    assert failure and observed == receipt, (
                        "cleanup fault lost its known business result"
                    )
                else:
                    print(json.dumps({"cleanup_failed": False, "returned_result": result}))
                    assert not failure and result == receipt, "submit must return its original receipt"
                assert store.batch(batch_id)["status"] == "submitted"
                assert sent == [batch_id], "one fixture transport operation only"


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
