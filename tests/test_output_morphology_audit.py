"""Tests for output morphology audit and PD disaggregation transfer gate."""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pytest

from mindie_bench.evaluation.output_morphology_audit import (
    audit,
    classify_morphology,
    compare_pd_outputs,
    main,
)


def test_tag_and_correctness_summary():
    records = [
        {"id": "a", "text": "<think>reason</think>answer", "correct": True},
        {"id": "b", "text": "unfinished", "correct": False},
        {"id": "c", "text": "</think>answer</think>\ufffd"},
    ]
    summary = audit(records, expect_think_close=True)["summary"]
    assert summary["samples"] == 3
    assert summary["closing_tag_zero"] == 1
    assert summary["closing_tag_one"] == 1
    assert summary["closing_tag_multiple"] == 1
    assert summary["missing_expected_closing_tag"] == 1
    assert summary["correct_known"] == 2
    assert summary["correct_true"] == 1
    assert summary["correct_false"] == 1
    assert summary["correct_unknown"] == 1
    assert summary["replacement_character_samples"] == 1


def test_missing_tag_requires_opt_in():
    record = [{"id": "a", "text": "plain answer"}]
    assert audit(record)["summary"]["missing_expected_closing_tag"] == 0
    assert audit(record, expect_think_close=True)["summary"]["missing_expected_closing_tag"] == 1


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"finish_reason": "length"}, "explicit_length_limit"),
        ({"output_tokens": 8, "max_output_tokens": 8}, "suspected_length_limit"),
        ({"output_tokens": 2, "max_output_tokens": 8}, "unknown_length_status"),
        ({"finish_reason": "stop"}, "no_length_limit_reported"),
        ({"finish_reason": "error"}, "unknown_length_status"),
    ],
)
def test_length_evidence_is_not_invented(fields, expected):
    sample = audit([{"id": "a", "text": "answer", **fields}])["samples"][0]
    assert sample["length_status"] == expected


def test_repetition_crossing_window_and_unscanned_middle_limit():
    def _noise(seed: int, length: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choices(string.ascii_letters + string.digits, k=length))

    crossed = _noise(42, 8190) + "abcdefghijklmnop" * 3 + _noise(99, 800)
    sample = audit([{"id": "a", "text": crossed}])["samples"][0]
    assert sample["repetition_candidate"]["start_character"] == 8190
    assert sample["repetition_characters_scanned"] == len(crossed)

    middle_only = _noise(43, 17000) + "abcdefghijklmnop" * 4 + _noise(44, 17000)
    sample_mid = audit([{"id": "b", "text": middle_only}])["samples"][0]
    assert sample_mid["repetition_candidate"] is None
    assert sample_mid["repetition_characters_scanned"] == 16384


def test_morphology_flags_and_pd_transfer_comparison():
    assert "empty" in classify_morphology("   ")
    assert "replacement_char" in classify_morphology("hello \ufffd world")
    assert "repeated_char_run" in classify_morphology("a" * 20)
    assert "repeated_ngram" in classify_morphology("0123456789abcdef" * 3)
    assert "punctuation_run" in classify_morphology("Answer!!!!!!!!!!")
    assert "immediate_eos" in classify_morphology(".", output_tokens=1, finish_reason="stop")

    baseline = [
        {
            "id": "req-1",
            "text": "The capital of France is Paris.",
            "output_tokens": 8,
            "finish_reason": "stop",
        }
    ]
    healthy_pd = [
        {
            "id": "req-1",
            "text": "The capital of France is Paris.",
            "output_tokens": 8,
            "finish_reason": "stop",
        }
    ]
    passed = compare_pd_outputs(
        baseline,
        healthy_pd,
        transferred_kv_bytes=65536,
        recomputed_locally=False,
    )
    assert passed["status"] == "passed"
    assert passed["reasons"] == []
    assert passed["samples"][0]["exact_match"] is True

    # HTTP 200 with identical text must still fail if D fell back to local recompute
    recompute_fallback = compare_pd_outputs(
        baseline,
        healthy_pd,
        transferred_kv_bytes=65536,
        recomputed_locally=True,
    )
    assert recompute_fallback["status"] == "failed"
    assert "pd_transfer_fell_back_to_local_recompute" in recompute_fallback["reasons"]

    # Missing transfer byte telemetry stays inconclusive rather than passing silently
    missing_bytes = compare_pd_outputs(
        baseline,
        healthy_pd,
        transferred_kv_bytes=None,
        recomputed_locally=False,
    )
    assert missing_bytes["status"] == "inconclusive"
    assert "transferred_kv_bytes_not_recorded" in missing_bytes["reasons"]

    # Corrupted PD output with repeated n-gram / punctuation loop fails with offset
    corrupted_pd = [
        {
            "id": "req-1",
            "text": "The capital of " + ("France is Paris! " * 3) + "!!!!!!!!!!",
            "output_tokens": 32,
            "finish_reason": "length",
        }
    ]
    failed = compare_pd_outputs(
        baseline,
        corrupted_pd,
        transferred_kv_bytes=65536,
        recomputed_locally=False,
    )
    assert failed["status"] == "failed"
    assert failed["samples"][0]["first_diff_offset"] == len("The capital of France is Paris")
    assert set(failed["samples"][0]["candidate_only_flags"]) == {
        "repeated_ngram",
        "punctuation_run",
    }


def test_cli_omits_generated_text_and_rejects_invalid_input(tmp_path: Path, capsys):
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text(
        json.dumps({"id": "case-1", "text": "PRIVATE-GENERATED-CONTENT"}) + "\n",
        encoding="utf-8",
    )
    rc = main([str(valid_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "PRIVATE-GENERATED-CONTENT" not in captured.out
    assert json.loads(captured.out)["samples"][0]["id"] == "case-1"

    invalid_cases = [
        "",
        "{PRIVATE-MALFORMED-CONTENT}",
        json.dumps({"id": "a"}),
        json.dumps({"id": "a", "text": "PRIVATE", "output_tokens": True}),
        json.dumps({"id": "a", "text": "PRIVATE", "max_output_tokens": 0}),
        json.dumps({"id": "a", "text": "PRIVATE", "correct": "true"}),
        "\n".join([json.dumps({"id": "a", "text": "PRIVATE"})] * 2),
    ]
    for index, payload in enumerate(invalid_cases):
        bad_path = tmp_path / f"bad-{index}.jsonl"
        bad_path.write_text(payload, encoding="utf-8")
        rc = main([str(bad_path)])
        captured = capsys.readouterr()
        assert rc == 2
        assert captured.out == ""
        assert "PRIVATE" not in captured.err
