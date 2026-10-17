import hashlib
import json
import sys
from types import ModuleType

import pytest

from mindie_bench.evaluation.retrieval import latency_summary, load_labels
from mindie_bench.io import read_json


def write_case(tmp_path):
    corpus = tmp_path / "corpus"
    task = corpus / "tasks/example"
    (task / "blocks").mkdir(parents=True)
    (task / "index.md").write_text("navigation")
    block = task / "blocks/target.md"
    block.write_text("canonical body")
    row = dict(
        sample_id="paraphrase",
        query="Why does an explicit rotary width allocate a full-width cache?",
        relevant_task_ids=["example"],
        relevant_block_ids=["example:target"],
        source_commit="a" * 40,
        label_scope="positive_only_target_recovery",
        source_files_sha256={
            str(path.relative_to(corpus)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [task / "index.md", block]
        },
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(row) + "\n")
    return corpus, labels, row


def package_reader(root):
    from types import SimpleNamespace

    return SimpleNamespace(navigation=root / "index.md")


def test_retrieval_labels_bind_actual_relevance_files(tmp_path, monkeypatch):
    from mindie_bench.evaluation import retrieval

    corpus, labels, row = write_case(tmp_path)
    monkeypatch.setattr(retrieval, "read_package", package_reader)
    assert load_labels(labels, corpus, "a" * 40) == [row]
    (corpus / "tasks/example/blocks/target.md").write_text("new content")
    with pytest.raises(ValueError, match="evidence changed"):
        load_labels(labels, corpus, "a" * 40)


def test_missing_target_provenance_is_not_treated_as_valid_labels(tmp_path, monkeypatch):
    from mindie_bench.evaluation import retrieval

    corpus, labels, row = write_case(tmp_path)
    monkeypatch.setattr(retrieval, "read_package", package_reader)
    row["source_files_sha256"].pop("tasks/example/blocks/target.md")
    labels.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="exact target blocks"):
        load_labels(labels, corpus, "a" * 40)


def test_relevance_labels_cannot_escape_the_selected_corpus(tmp_path, monkeypatch):
    from mindie_bench.evaluation import retrieval

    corpus, labels, row = write_case(tmp_path)
    monkeypatch.setattr(retrieval, "read_package", package_reader)
    row["source_files_sha256"]["../private"] = "b" * 64
    labels.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="confined relative"):
        load_labels(labels, corpus, "a" * 40)


def test_different_sample_ids_cannot_inflate_the_number_of_independent_queries(tmp_path, monkeypatch):
    from mindie_bench.evaluation import retrieval

    corpus, labels, row = write_case(tmp_path)
    monkeypatch.setattr(retrieval, "read_package", package_reader)
    duplicate = dict(row, sample_id="other-label", query="  " + row["query"].upper() + "   ")
    labels.write_text(json.dumps(row) + "\n" + json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="independent retrieval"):
        load_labels(labels, corpus, "a" * 40)


def test_tiny_latency_samples_do_not_report_p95():
    small = latency_summary([0.004, 0.009])
    assert small["samples"] == 2
    assert small["p50_sec"] == pytest.approx(0.0065)
    assert small["p95_sec"] is None
    assert latency_summary(list(range(20)))["p95_sec"] == pytest.approx(18.05)


def runtime(tmp_path, monkeypatch, *, query_error=None, close_error=None, wrong_body=False):
    from mindie_bench.evaluation import retrieval
    from mindie_bench.integrations import first_party

    corpus, labels, row = write_case(tmp_path)
    body = corpus / "tasks/example/blocks/target.md"
    body.write_text("---\nschema: fixture\n---\n\ncanonical body")
    row["source_files_sha256"]["tasks/example/blocks/target.md"] = hashlib.sha256(
        body.read_bytes()
    ).hexdigest()
    labels.write_text(json.dumps(row) + "\n")
    component = tmp_path / "component"
    component.mkdir()
    (component / "__init__.py").write_text("# fixture component\n")
    module = ModuleType("mindie_knowledge")
    module.__file__ = str(component / "__init__.py")
    monkeypatch.setitem(sys.modules, "mindie_knowledge", module)
    monkeypatch.setattr(retrieval, "read_package", package_reader)
    monkeypatch.setattr(
        retrieval.subprocess,
        "check_output",
        lambda args, **kw: "a" * 40 + "\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(retrieval, "_selected_identity", lambda hit: ("example", "target"))
    closed = []

    class Reader:
        def refresh_index(self):
            return {"complete": True}

        def search(self, **kwargs):
            if query_error is not None:
                raise query_error
            return [{"ref": "example:target"}]

        def close(self):
            closed.append(True)
            if close_error is not None:
                raise close_error

    monkeypatch.setattr(first_party, "consumer", lambda *a: Reader())
    monkeypatch.setattr(
        first_party,
        "explain",
        lambda *a: {"content": "wrong body" if wrong_body else "canonical body"},
    )
    return corpus, labels, closed


def test_component_runner_measures_calls_without_inflating_independent_queries(tmp_path, monkeypatch):
    from mindie_bench.evaluation.retrieval import evaluate_retrieval

    corpus, labels, closed = runtime(tmp_path, monkeypatch)
    result = evaluate_retrieval(corpus, labels, tmp_path / "run", repetitions=3)
    assert result["distinct_queries"] == 1
    assert result["measured_query_calls"] == 3
    assert result["task_metrics"]["recall_at_k"] == 1
    assert result["block_metrics"]["reciprocal_rank"] == 1
    assert result["query_latency"]["p95_sec"] is None
    assert result["token_reduction"] is None
    assert closed == [True]


def test_component_body_mismatch_fails_instead_of_publishing_a_score(tmp_path, monkeypatch):
    from mindie_bench.evaluation.retrieval import evaluate_retrieval

    corpus, labels, closed = runtime(tmp_path, monkeypatch, wrong_body=True)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="canonical Markdown"):
        evaluate_retrieval(corpus, labels, output)
    assert closed == [True]
    assert read_json(output / "failure.json")["phase"] == "explain"
    assert not (output / "summary.json").exists()


def test_component_original_failure_and_cleanup_failure_are_both_durable(tmp_path, monkeypatch):
    from mindie_bench.evaluation.retrieval import evaluate_retrieval

    original = ValueError("required query failed")
    corpus, labels, closed = runtime(
        tmp_path, monkeypatch, query_error=original, close_error=OSError("close failed")
    )
    output = tmp_path / "run"
    with pytest.raises(ValueError) as caught:
        evaluate_retrieval(corpus, labels, output)
    assert caught.value is original
    failure = read_json(output / "failure.json")
    assert failure["error"]["message"] == "required query failed"
    assert failure["cleanup_error"]["message"] == "close failed"
    assert closed == [True]


def test_component_successful_queries_do_not_hide_required_close_failure(tmp_path, monkeypatch):
    from mindie_bench.evaluation.retrieval import evaluate_retrieval

    corpus, labels, closed = runtime(tmp_path, monkeypatch, close_error=OSError("close failed"))
    output = tmp_path / "run"
    with pytest.raises(OSError, match="close failed"):
        evaluate_retrieval(corpus, labels, output)
    assert read_json(output / "failure.json")["phase"] == "close"
    assert not (output / "summary.json").exists()
    assert closed == [True]
