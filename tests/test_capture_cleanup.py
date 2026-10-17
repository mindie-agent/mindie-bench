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


def test_capture_with_no_primary_raises_first_cleanup_error_and_retains_second():
    original = OSError("first cleanup failed")
    engine, store, called = resources(original, RuntimeError("second cleanup failed"))
    with pytest.raises(OSError) as caught:
        capture_finalizer()(engine, store, lambda: "business completed")
    assert caught.value is original
    assert called == ["shutdown", "close"]
    assert "second cleanup failed" in original.__notes__[0]


def test_successful_capture_does_not_hide_cleanup_error_in_callers_handled_exception():
    engine, store, called = resources(None, OSError("close failed"))
    try:
        raise UnknownExternalEffects("caller's prior failure")
    except UnknownExternalEffects:
        with pytest.raises(OSError, match="close failed"):
            capture_finalizer()(engine, store, lambda: "business completed")
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
