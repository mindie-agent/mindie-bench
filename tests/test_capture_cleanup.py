from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mindie_bench.integrations import first_party
from mindie_bench.models import UnknownExternalEffects


def capture_finalizer():
    """Exercise real cleanup without component or model startup."""

    def exercise(engine, store, operation):
        primary = None
        try:
            return operation()
        except BaseException as error:
            primary = error
            raise
        finally:
            first_party.close_capture_resources(
                engine,
                store,
                SimpleNamespace(measure=lambda stage, **kwargs: nullcontext()),
                primary=primary,
            )

    return exercise


def resources(first=None, second=None):
    called = []

    def cleanup(name, error):
        called.append(name)
        if error is not None:
            raise error

    return (
        SimpleNamespace(operations=SimpleNamespace(shutdown=lambda: cleanup("shutdown", first))),
        SimpleNamespace(close=lambda: cleanup("close", second)),
        called,
    )


def test_capture_unknown_survives_both_cleanup_errors_and_both_are_attempted():
    engine, store, called = resources(RuntimeError("shutdown failed"), OSError("close failed"))
    original = UnknownExternalEffects("original paid attempt outcome unknown")

    def operation():
        raise original

    with pytest.raises(UnknownExternalEffects) as caught:
        capture_finalizer()(engine, store, operation)
    assert caught.value is original
    assert called == ["shutdown", "close"]
    assert "shutdown failed" in original.__notes__[0]
    assert "close failed" in original.__notes__[1]


def test_completed_capture_returns_cleanup_failures_instead_of_raising():
    original = OSError("first cleanup failed")
    engine, store, called = resources(original, RuntimeError("second cleanup failed"))
    failures = first_party.close_capture_resources(
        engine,
        store,
        SimpleNamespace(measure=lambda stage, **kwargs: nullcontext()),
        primary=None,
    )
    assert [type(error) for _, error in failures] == [OSError, RuntimeError]
    assert called == ["shutdown", "close"]


def test_completed_capture_with_caller_error_still_attaches_cleanup_notes():
    engine, store, called = resources(None, OSError("close failed"))
    original = UnknownExternalEffects("caller's prior failure")
    failures = first_party.close_capture_resources(
        engine,
        store,
        SimpleNamespace(measure=lambda stage, **kwargs: nullcontext()),
        primary=original,
    )
    assert failures == []
    assert "close failed" in original.__notes__[0]
    assert called == ["shutdown", "close"]


def test_capture_returns_business_result_after_both_cleanups_succeed():
    engine, store, called = resources()
    assert capture_finalizer()(engine, store, lambda: "completed") == "completed"
    assert called == ["shutdown", "close"]


def test_stage_journal_failure_does_not_skip_required_cleanup(tmp_path, monkeypatch):
    from mindie_bench.evaluation import timing

    engine, store, called = resources()
    recorder = timing.StageRecorder(tmp_path / "stages.json")
    monkeypatch.setattr(
        timing, "write_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
    )
    original = UnknownExternalEffects("original uncertain request")
    with pytest.raises(UnknownExternalEffects) as caught:
        try:
            raise original
        finally:
            first_party.close_capture_resources(engine, store, recorder, primary=original)
    assert caught.value is original
    assert called == ["shutdown", "close"]
    assert sum("disk full" in note for note in original.__notes__) == 2
