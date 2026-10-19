# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
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

with tempfile.TemporaryDirectory(prefix="l2-public-cleanup-") as directory:
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
            title="Public",
            summary="Fixture.",
            content="Bounded CPU protocol.",
            generation=shared.generation,
        )
        batch_id = build_batch(store, settings=shared)[0]
        engine = Engine(store, settings_path=shared.path)
        engine.community = {
            "submit_batch": lambda *args, **kwargs: {
                "status": "submitted",
                "head_sha": "b" * 40,
                "pr_url": "https://example.invalid/pull/2",
            }
        }

        def compact(*args, **kwargs):
            raise OSError("fixture cleanup failed after confirmed submission")

        store.compact_confirmed = compact
        try:
            engine.publisher.submit(store.batch(batch_id))
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "database_status": store.batch(batch_id)["status"],
                        "exception_business_result": getattr(exc, "mindie_publication_result", None),
                    }
                )
            )
