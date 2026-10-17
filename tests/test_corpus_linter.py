import copy

import pytest

from mindie_bench.evaluation.corpus_linter import CorpusLintRejected, audit_packages

CHAIN = """### assistant
## Root cause
The padded tile addresses exceed the packed-token scratch allocation.
## Verification evidence
The independent reference comparison passes the held-out sequence lengths.
## Applicability
Verified for CANN 9.1 and this operator ABI; retest other compiler versions.
"""


def package(body=CHAIN, navigation="# Workspace allocation\n"):
    return [dict(task_id="new-seed", files={"PACKAGE.md": navigation, "materials/0001.md": body})]


def test_real_advice_is_add_only_and_relative_measurements_remain_allowed():
    value = package(CHAIN + "\nObserved improvement: 1.18x, roofline utilization 72%.\n")
    before = copy.deepcopy(value)
    assert audit_packages(value)["status"] == "passed"
    assert value == before


@pytest.mark.parametrize(
    "text",
    ["在昇腾上绝对不要使用 ACLGraph。", "Never use graph capture on Ascend."],
)
def test_global_prohibitions_block_consumption_without_filtering(text):
    value = package(CHAIN + "\n" + text)
    report = audit_packages(value)
    assert "unconditional_global_ban" in {row["code"] for row in report["findings"]}
    assert text in value[0]["files"]["materials/0001.md"]
    with pytest.raises(CorpusLintRejected, match="before publication"):
        raise CorpusLintRejected(report)


def test_versioned_warning_requires_a_reproduction_action():
    bounded = "\nWhen using CANN 8.0, never use this workaround before you verify the reference output."
    assert audit_packages(package(CHAIN + bounded))["status"] == "passed"
    report = audit_packages(package(CHAIN + "\nWhen using CANN 8.0, never use this workaround."))
    assert report["status"] == "rejected"


@pytest.mark.parametrize("claim", ["Latency is 3.5 ms.", "Measured throughput: 900 GB/s.", "耗时为 7 毫秒。"])
def test_machine_specific_absolute_results_require_normalization(claim):
    report = audit_packages(package(CHAIN + "\n" + claim))
    assert "absolute_performance_claim" in {row["code"] for row in report["findings"]}


def test_operational_poll_deadline_is_not_an_absolute_performance_claim():
    assert audit_packages(package(CHAIN + "\nPolling timeout: 10000 ms.\n"))["status"] == "passed"


def test_user_supplied_causal_template_cannot_certify_the_assistants_advice():
    report = audit_packages(package(CHAIN.replace("### assistant", "### user")))
    assert "missing_causal_chain" in {row["code"] for row in report["findings"]}


def test_index_budget_counts_visible_index_and_rejects_51_lines():
    assert audit_packages(package(navigation="\n".join(["entry"] * 50)))["status"] == "passed"
    report = audit_packages(package(navigation="\n".join(["entry"] * 51)))
    assert "index_line_budget" in {row["code"] for row in report["findings"]}


def test_yaml_metadata_does_not_replace_or_inflate_the_human_index():
    navigation = "---\n" + "metadata: value\n" * 55 + "---\n# A short navigation\n"
    assert audit_packages(package(navigation=navigation))["status"] == "passed"


def test_empty_and_malformed_exports_never_count_as_qualified_experience():
    assert audit_packages([])["status"] == "rejected"
    assert audit_packages([dict(task_id="bad", files={"../escape.md": "text"})])["status"] == "rejected"


def test_natural_engineering_summary_passes_without_rigid_headings():
    natural = (
        "### assistant\n"
        "Fixed the RMSNorm fusion pass in vllm_ascend by registering the quantized pattern when "
        "quant_description is present, and verified that all unit tests pass.\n"
    )
    assert audit_packages(package(body=natural))["status"] == "passed"
    incomplete = "### assistant\nI fixed the bug in the file.\n"
    report = audit_packages(package(body=incomplete))
    assert "missing_causal_chain" in {row["code"] for row in report["findings"]}
