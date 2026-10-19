"""Show the historical partial-export failure without a remote publication."""

# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge.loop import settings
from mindie_knowledge.loop.export import build_batch
from mindie_knowledge.loop.store import Store

with tempfile.TemporaryDirectory(prefix="publication-export-reproduction-") as directory:
    root = Path(directory)
    shared = settings.write(
        root / "sharing.json",
        enabled=True,
        repository="fixture-owner/fixture-corpus",
        project_roots=[root],
        idle_seconds=300,
    )
    with closing(Store(root / "data", "test")) as store:
        for title, body in (
            ("Rejected observation", "host=benchmark-node7"),
            ("Public observation", "A local CPU publication contract."),
        ):
            store.create_draft(
                kind="experience",
                title=title,
                summary="Fixture.",
                content=body,
                generation=shared.generation,
            )
        try:
            result = build_batch(store, settings=shared)
        except ValueError as error:
            print(
                json.dumps(
                    {
                        "caller_error": type(error).__name__,
                        "built_batch_retained": getattr(error, "built", None) is not None,
                        "quarantined_count": len(store.quarantined_entries()),
                    }
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "caller_error": None,
                        "built_batch_retained": result is not None,
                        "quarantined_count": len(store.quarantined_entries()),
                    }
                )
            )
