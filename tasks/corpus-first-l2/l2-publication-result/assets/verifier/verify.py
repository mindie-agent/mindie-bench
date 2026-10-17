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
from mindie_knowledge.loop.export import ExportQuarantinedError
from mindie_knowledge.loop.store import Store

assert Path(importlib.import_module(Store.__module__).__file__).resolve().is_relative_to(WORKSPACE)


def check():
    for fault in ("receipt", "cleanup", "unknown", "none"):
        with tempfile.TemporaryDirectory(prefix="l2-publication-") as directory:
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
                    title="Unsafe endpoint",
                    summary="Fixture.",
                    content="host=benchmark-node7",
                    generation=shared.generation,
                )
                store.create_draft(
                    kind="experience",
                    title="Public method",
                    summary="Fixture.",
                    content="The CPU protocol trial remains bounded.",
                    generation=shared.generation,
                )
                engine = Engine(store, settings_path=shared.path)
                sent = []

                def submit(batch, *args, **kwargs):
                    sent.append(batch["batch_id"])
                    if fault == "unknown":
                        raise OSError("fixture remote acknowledgement unavailable")
                    return {
                        "status": "submitted",
                        "head_sha": "a" * 40,
                        "pr_url": "https://example.invalid/pull/1",
                    }

                engine.community = {"submit_batch": submit}
                if fault == "receipt":
                    mark = store.mark_batch

                    def mark_batch(batch_id, status, **kwargs):
                        if status == "submitted":
                            raise OSError("fixture receipt disk failure")
                        return mark(batch_id, status, **kwargs)

                    store.mark_batch = mark_batch
                if fault == "cleanup":

                    def compact(*args, **kwargs):
                        raise OSError("fixture owned cleanup failure")

                    store.compact_confirmed = compact
                try:
                    engine.publisher.flush()
                except ExportQuarantinedError as exc:
                    result = exc.publication
                    print(
                        json.dumps(
                            {
                                "fault": fault,
                                "publication": result,
                                "cause": type(exc.__cause__).__name__,
                            }
                        )
                    )
                    assert isinstance(result, dict), (
                        "returned business result was lost after local failure"
                    )
                    assert result["status"] == ("unknown" if fault == "unknown" else "submitted"), (
                        "returned remote publication status must survive a separate local fault"
                    )
                    if fault != "unknown":
                        assert result["head_sha"] == "a" * 40
                    if fault in {"receipt", "cleanup"}:
                        assert isinstance(exc.__cause__, OSError), "local failure must remain visible"
                else:
                    raise AssertionError("unsafe draft rejection must remain visible")
                engine.publisher.flush()
                assert len(sent) == 1, "a later idle tick must not replay the original write"


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
