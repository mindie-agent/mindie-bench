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
from mindie_knowledge.loop.export import ExportQuarantinedError
from mindie_knowledge.loop.store import Store

with tempfile.TemporaryDirectory(prefix="l2-public-reproduction-") as directory:
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
            ("Unsafe", "host=benchmark-node7"),
            ("Public", "A bounded CPU protocol test."),
        ):
            store.create_draft(
                kind="experience",
                title=title,
                summary="Fixture.",
                content=body,
                generation=shared.generation,
            )
        engine = Engine(store, settings_path=shared.path)
        engine.community = {
            "submit_batch": lambda *args, **kwargs: {
                "status": "submitted",
                "head_sha": "a" * 40,
                "pr_url": "https://example.invalid/pull/1",
            }
        }
        original = store.mark_batch

        def mark(batch_id, status, **kwargs):
            if status == "submitted":
                raise OSError("fixture local recording failure after transport returned")
            return original(batch_id, status, **kwargs)

        store.mark_batch = mark
        try:
            engine.publisher.flush()
        except ExportQuarantinedError as exc:
            print(
                json.dumps(
                    {
                        "returned_transport_status": "submitted",
                        "visible_publication": exc.publication,
                        "local_failure": type(exc.__cause__).__name__,
                    }
                )
            )
