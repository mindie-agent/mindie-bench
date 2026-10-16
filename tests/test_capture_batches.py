import sqlite3
from types import SimpleNamespace

import pytest

from mindie_bench.integrations.first_party import finish_indexes


def engine():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE transcript_tasks(entry_id, summary_status, summary_detail)")
    db.executemany("INSERT INTO transcript_tasks VALUES (?, 'pending', '')", [("one",), ("two",)])
    return SimpleNamespace(
        store=SimpleNamespace(db=db),
        index_queue=SimpleNamespace(has_due=lambda: True),
    )


def test_native_batches_are_drained_instead_of_accepting_only_the_first():
    selected = engine()
    calls = []

    def summarize(current):
        identity = ["one", "two"][len(calls)]
        calls.append(identity)
        current.store.db.execute(
            "UPDATE transcript_tasks SET summary_status='complete' WHERE entry_id=?", (identity,)
        )

    try:
        rows = finish_indexes(selected, summarize)
        assert calls == ["one", "two"]
        assert {row["summary_status"] for row in rows} == {"complete"}
    finally:
        selected.store.db.close()


@pytest.mark.parametrize("status", ["failed", "outcome_unknown", "cancelled"])
def test_failed_or_unknown_indexing_never_advances_another_paid_batch(status):
    selected = engine()
    calls = []

    def summarize(current):
        calls.append("one")
        current.store.db.execute(
            "UPDATE transcript_tasks SET summary_status=? WHERE entry_id='one'", (status,)
        )

    try:
        with pytest.raises(RuntimeError, match=status):
            finish_indexes(selected, summarize)
        assert calls == ["one"]
    finally:
        selected.store.db.close()
