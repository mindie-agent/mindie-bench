"""Judge partial export against real historical Store and export behavior."""

# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import importlib
import json
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge.loop import export, settings
from mindie_knowledge.loop.store import Store

assert Path(importlib.import_module(Store.__module__).__file__).resolve().is_relative_to(WORKSPACE)


def check():
    for content_set in ("mixed", "all-unsafe", "all-public"):
        with tempfile.TemporaryDirectory(prefix="publication-export-judge-") as directory:
            root = Path(directory)
            shared = settings.write(
                root / "sharing.json",
                enabled=True,
                repository="fixture-owner/fixture-corpus",
                project_roots=[root],
                idle_seconds=300,
            )
            with closing(Store(root / "data", "test")) as store:
                rejected, kept = [], []
                for unsafe in {
                    "mixed": (True, False),
                    "all-unsafe": (True, True),
                    "all-public": (False, False),
                }[content_set]:
                    document = store.create_draft(
                        kind="experience",
                        title="Export observation",
                        summary="Fixture.",
                        content=(
                            "Private endpoint: " + "10.3." + "5.7"
                            if unsafe
                            else "A bounded public CPU protocol observation."
                        ),
                        generation=shared.generation,
                    )
                    (rejected if unsafe else kept).append(document["entry_id"])
                caught = None
                try:
                    built = export.build_batch(store, settings=shared)
                except export.ExportContentError as error:
                    caught = error
                    built = getattr(error, "built", None)
                if rejected:
                    assert caught is not None, "content rejection was invisible to the caller"
                    rejections = getattr(caught, "rejections", [])
                    assert {row["entry_id"] for row in rejections} == set(rejected)
                    assert getattr(caught, "publication", None) is None
                    assert set(store.quarantined_entries()) == set(rejected)
                else:
                    assert caught is None, "valid content must build normally"
                if kept:
                    assert built is not None, "rejected content discarded unrelated valid staging"
                    batch_id, _, _, entry_ids, _ = built
                    assert set(entry_ids) == set(kept)
                    assert store.batch(batch_id)["status"] == "pending"
                else:
                    assert built is None, "all-rejected export must not create a batch"
                assert export.build_batch(store, settings=shared) is None


try:
    check()
except AssertionError as error:
    print("Business contract failed: " + str(error), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
