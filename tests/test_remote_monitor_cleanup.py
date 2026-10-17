import io
from types import SimpleNamespace

import pytest

from mindie_bench.runner.remote_slot import RemoteSlot


def slot(case):
    return RemoteSlot(
        SimpleNamespace(slots_root="/owned/slots"),
        case["spec"].environment["l2"],
        "owned-monitor",
        [8],
        1,
        artifact_dir=case["root"],
    )


def test_monitor_exit_cleanup_frame_cannot_be_consumed_as_a_business_ack(case, monkeypatch):
    subject = slot(case)
    subject.process = SimpleNamespace(stdin=io.BytesIO())
    cleanup = {"cleanup": {"errors": [], "probe": {}}}
    monkeypatch.setattr(subject, "_send_frame", lambda *args: None)
    monkeypatch.setattr(subject, "_read", lambda timeout: cleanup)
    with pytest.raises(RuntimeError, match="cleanup"):
        subject.execute("interrupted-command")
    assert subject.cleanup_receipt == cleanup
    assert subject.operation_unknown


def test_close_still_finalizes_component_lease_when_local_group_is_already_gone(case, monkeypatch):
    import os

    subject = slot(case)
    subject.process = SimpleNamespace(
        pid=12345, stdin=io.BytesIO(), poll=lambda: 0, wait=lambda timeout: 0
    )
    finished = []
    subject.component_runtime = SimpleNamespace(
        finish=lambda **kwargs: finished.append(kwargs["cleanup_confirmed"]),
        stop_monitor=lambda: None,
    )
    monkeypatch.setattr(subject, "_read", lambda timeout: (_ for _ in ()).throw(RuntimeError("EOF")))
    monkeypatch.setattr(
        os, "killpg", lambda *args: (_ for _ in ()).throw(PermissionError("already reaped group"))
    )
    with pytest.raises(RuntimeError, match="cleanup"):
        subject.close()
    assert finished == [False]
