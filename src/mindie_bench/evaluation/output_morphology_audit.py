"""Deterministic output morphology audit and PD disaggregation transfer gate.

[Role] Inspect generated output records for structural truncation, repetition loops,
tag imbalance, replacement characters, and silent PD local-recompute fallbacks
without echoing private prompt or generated text.
[Invariants] Length-limit status is derived strictly from explicit finish_reason or
token budget saturation; missing fields stay unknown. PD transfer acceptance fails
whenever local prefill recomputation occurred or transferred_kv_bytes is zero.
[Timeouts & Locks] Pure in-memory regex and token-counter analysis over bounded
8192-character prefix/suffix windows; zero network, subprocess, or lock waits.
[Side-Effect Policy] Read-only over supplied records or JSONL files; never echoes
raw generated text in validation errors or audit summaries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Three adjacent copies of a 16..128-character span over bounded 8192-char windows.
REPEATED_SPAN = re.compile(r"(.{16,128}?)\1{2,}", re.DOTALL)
REPEATED_CHAR_RUN = re.compile(r"(.)\1{15,}", re.DOTALL)
PUNCTUATION_RUN = re.compile(r"([^\w\s])\1{7,}")
WINDOW = 8192


def repeated_span(text: str) -> dict[str, int] | None:
    windows = [(0, text[:WINDOW])]
    if len(text) > WINDOW:
        offset = len(text) - WINDOW
        windows.append((offset, text[offset:]))
    for offset, part in windows:
        match = REPEATED_SPAN.search(part)
        if match:
            return {
                "start_character": offset + match.start(),
                "span_characters": len(match.group(1)),
                "adjacent_copies": len(match.group(0)) // len(match.group(1)),
            }
    return None


def classify_morphology(
    text: str,
    *,
    output_tokens: int | None = None,
    finish_reason: str | None = None,
) -> list[str]:
    """Return deterministic structural anomaly flags without exposing raw text."""
    flags: list[str] = []
    stripped = text.strip()
    if not stripped:
        flags.append("empty")
    if "\ufffd" in text:
        flags.append("replacement_char")
    if REPEATED_CHAR_RUN.search(text[:WINDOW]) or (
        len(text) > WINDOW and REPEATED_CHAR_RUN.search(text[-WINDOW:])
    ):
        flags.append("repeated_char_run")
    if repeated_span(text) is not None:
        flags.append("repeated_ngram")
    if PUNCTUATION_RUN.search(text[:WINDOW]) or (
        len(text) > WINDOW and PUNCTUATION_RUN.search(text[-WINDOW:])
    ):
        flags.append("punctuation_run")
    if (
        finish_reason in ("stop", "eos_token", "end_turn")
        and output_tokens is not None
        and output_tokens <= 1
        and len(stripped) <= 2
    ):
        flags.append("immediate_eos")
    return flags


def validate(record: Any, line_number: int, seen: set[str]) -> None:
    def invalid(reason: str) -> None:
        # Never echo malformed data, generated text, or credentials from input.
        raise ValueError(f"line {line_number}: {reason}")

    if not isinstance(record, dict):
        invalid("expected a JSON object")
    if not isinstance(record.get("id"), str) or not record["id"]:
        invalid("id must be a nonempty string")
    if record["id"] in seen:
        invalid("duplicate id; assign unique IDs to response attempts")
    if not isinstance(record.get("text"), str):
        invalid("text must be a generated-output string")
    for key in ("output_tokens", "max_output_tokens"):
        value = record.get(key)
        if value is not None and (type(value) is not int or value < 0):
            invalid(f"{key} must be a nonnegative integer or null")
    if record.get("max_output_tokens") == 0:
        invalid("max_output_tokens must be positive when supplied")
    if record.get("finish_reason") is not None and not isinstance(record["finish_reason"], str):
        invalid("finish_reason must be a string or null")
    if record.get("correct") is not None and type(record["correct"]) is not bool:
        invalid("correct must be boolean or null")
    seen.add(record["id"])


def audit(records: list[dict[str, Any]], expect_think_close: bool = False) -> dict[str, Any]:
    totals = Counter(
        {
            key: 0
            for key in (
                "samples",
                "closing_tag_zero",
                "closing_tag_one",
                "closing_tag_multiple",
                "missing_expected_closing_tag",
                "explicit_length_limit",
                "suspected_length_limit",
                "unknown_length_status",
                "no_length_limit_reported",
                "replacement_character_samples",
                "repetition_candidates",
                "correct_known",
                "correct_true",
                "correct_false",
                "correct_unknown",
            )
        }
    )
    samples = []
    for index, record in enumerate(records, 1):
        text = record["text"]
        closing = text.count("</think>")
        reason = record.get("finish_reason")
        count, limit = record.get("output_tokens"), record.get("max_output_tokens")
        at_budget = count is not None and limit is not None and count >= limit
        if reason in ("length", "max_tokens", "max_output_tokens"):
            length_status = "explicit_length_limit"
        elif at_budget:
            length_status = "suspected_length_limit"
        elif reason in ("stop", "eos_token", "end_turn", "tool_calls", "function_call"):
            length_status = "no_length_limit_reported"
        else:
            length_status = "unknown_length_status"
        repeated = repeated_span(text)
        morphology_flags = classify_morphology(
            text, output_tokens=count, finish_reason=reason
        )
        correct = record.get("correct")
        totals["samples"] += 1
        totals[
            "closing_tag_zero"
            if closing == 0
            else "closing_tag_one"
            if closing == 1
            else "closing_tag_multiple"
        ] += 1
        totals["missing_expected_closing_tag"] += int(expect_think_close and closing == 0)
        totals[length_status] += 1
        totals["replacement_character_samples"] += int("\ufffd" in text)
        totals["repetition_candidates"] += int(repeated is not None)
        totals["correct_unknown" if correct is None else "correct_known"] += 1
        if correct is not None:
            totals["correct_true" if correct else "correct_false"] += 1
        samples.append(
            {
                "id": record.get("id", f"sample-{index}"),
                "characters": len(text),
                "output_tokens": count,
                "max_output_tokens": limit,
                "finish_reason": reason,
                "opening_tags": text.count("<think>"),
                "closing_tags": closing,
                "length_status": length_status,
                "replacement_characters": text.count("\ufffd"),
                "repetition_candidate": repeated,
                "repetition_characters_scanned": min(len(text), 2 * WINDOW),
                "morphology_flags": morphology_flags,
                "correct": correct,
            }
        )
    return {
        "schema_version": 1,
        "expect_think_close": expect_think_close,
        "scope": (
            "Heuristic candidate selection; no model tokenization or accuracy verdict. "
            "Text is omitted; IDs and metadata remain visible. Repetition checks only "
            "the first/last 8192 characters for three adjacent 16..128-character spans."
        ),
        "summary": dict(totals),
        "samples": samples,
    }


def compare_pd_outputs(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    transferred_kv_bytes: int | None = None,
    recomputed_locally: bool = False,
) -> dict[str, Any]:
    """Compare colocated baseline records against PD disaggregated candidate records."""
    baseline_seen: set[str] = set()
    candidate_seen: set[str] = set()
    for idx, rec in enumerate(baseline_records, 1):
        validate(rec, idx, baseline_seen)
    for idx, rec in enumerate(candidate_records, 1):
        validate(rec, idx, candidate_seen)

    baseline_by_id = {rec["id"]: rec for rec in baseline_records}
    candidate_by_id = {rec["id"]: rec for rec in candidate_records}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate record ID sets must match")

    reasons: list[str] = []
    if recomputed_locally:
        reasons.append("pd_transfer_fell_back_to_local_recompute")
    if transferred_kv_bytes is None:
        reasons.append("transferred_kv_bytes_not_recorded")
    elif transferred_kv_bytes <= 0:
        reasons.append("transferred_kv_bytes_nonpositive")

    per_sample: list[dict[str, Any]] = []
    for sample_id in sorted(baseline_by_id):
        base = baseline_by_id[sample_id]
        cand = candidate_by_id[sample_id]
        base_text, cand_text = base["text"], cand["text"]
        base_flags = classify_morphology(
            base_text,
            output_tokens=base.get("output_tokens"),
            finish_reason=base.get("finish_reason"),
        )
        cand_flags = classify_morphology(
            cand_text,
            output_tokens=cand.get("output_tokens"),
            finish_reason=cand.get("finish_reason"),
        )
        new_flags = [flag for flag in cand_flags if flag not in base_flags]
        first_diff = None
        if base_text != cand_text:
            limit = min(len(base_text), len(cand_text))
            first_diff = next(
                (i for i in range(limit) if base_text[i] != cand_text[i]),
                limit,
            )
        if new_flags:
            reasons.append(f"candidate_morphology_regression:{sample_id}:{','.join(new_flags)}")
        per_sample.append(
            {
                "id": sample_id,
                "exact_match": base_text == cand_text,
                "first_diff_offset": first_diff,
                "baseline_flags": base_flags,
                "candidate_flags": cand_flags,
                "candidate_only_flags": new_flags,
            }
        )

    status = (
        "failed"
        if any(
            r.startswith(("pd_transfer_", "transferred_kv_bytes_nonpositive", "candidate_morphology_"))
            for r in reasons
        )
        else "inconclusive"
        if "transferred_kv_bytes_not_recorded" in reasons
        else "passed"
    )
    return {
        "schema_version": "mindie-pd-output-audit/1",
        "status": status,
        "recomputed_locally": recomputed_locally,
        "transferred_kv_bytes": transferred_kv_bytes,
        "reasons": reasons,
        "samples": per_sample,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Normalized JSONL, one complete response attempt per line",
    )
    parser.add_argument(
        "--expect-think-close",
        action="store_true",
        help="Mark zero closing tags only when the output contract requires one",
    )
    args = parser.parse_args(argv)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with args.input.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"line {line_number}: invalid JSON") from None
                validate(record, line_number, seen)
                records.append(record)
        if not records:
            raise ValueError("input contains no response records")
    except TimeoutError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        print(f"audit_outputs: {error}", file=sys.stderr)
        return 2
    json.dump(audit(records, args.expect_think_close), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
