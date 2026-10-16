import ast
import inspect
from types import SimpleNamespace

import pytest

from mindie_bench.integrations import first_party
from mindie_bench.models import UnknownExternalEffects


def capture_finalizer():
    """Exercise the actual cleanup block without component or model startup."""
    capture = ast.parse(inspect.getsource(first_party.capture)).body[0]
    finalizer = next(node.finalbody for node in capture.body if isinstance(node, ast.Try))
    function = ast.parse(
        "def exercise(engine, store, operation):\n    try:\n        return operation()\n    finally:\n"
        "        pass\n"
    ).body[0]
    function.body[0].finalbody = finalizer
    namespace = {"sys": first_party.sys}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), __file__, "exec"
        ),
        namespace,
    )
    return namespace["exercise"]


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
