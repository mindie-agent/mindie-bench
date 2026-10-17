"""Explicit in-process checks, not a claim to detect arbitrary Python attacks."""

import json


def check_comparators(torch):
    before, wrong = torch.tensor([1.0]), torch.tensor([7.0])
    if torch.allclose(before, wrong):
        raise RuntimeError("torch.allclose accepted a known incorrect answer")
    try:
        torch.testing.assert_close(before, wrong)
    except AssertionError:
        pass
    else:
        raise RuntimeError("torch.testing.assert_close accepted a known incorrect answer")


def reject_oracle_tampering(error, context="actual judge comparator control"):
    print(
        json.dumps(
            {
                "reward": 0.0,
                "oracle_status": "oracle_tampered",
                "failure_reason": f"oracle_tampered: {context}: {type(error).__name__}: {error}",
                "warnings": [
                    "An explicit integrity control failed in the actual judge process. "
                    "These controls do not detect arbitrary in-memory Python attacks."
                ],
            }
        ),
        flush=True,
    )
    raise SystemExit(1) from error


def enforce_comparator_control(torch):
    try:
        check_comparators(torch)
    except Exception as error:
        reject_oracle_tampering(error)
