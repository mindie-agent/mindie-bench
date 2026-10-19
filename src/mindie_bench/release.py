"""Execute and aggregate small repository-owned release benchmarks.

[Role] Bind real workloads and raw samples to an exact release candidate.
[Invariants] Unknown external writes are never retried; dirty or missing evidence
cannot pass a release gate. Measurements do not imply causal productivity gains.
[Timeouts & Locks] Case timeout + 5 seconds for local process cleanup; Git: 10s.
[Side-Effect Policy] Runs explicitly declared commands; writes local reports only.
Only allowlisted numeric payloads enter public assets. Logs remain under private/.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mindie_bench.io import read_json, write_json
from mindie_bench.release_contract import (
    BenchmarkReport,
    Measurement,
    ReleaseManifest,
    ReleaseSuite,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()


def _identity(repo: Path) -> dict:
    return {
        "revision": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=all")),
    }


def _execute(command, *, repo, env, destination, timeout):
    """Keep child output private and bound cleanup without parsing arbitrary log text."""
    started = time.perf_counter()
    with (destination / "stdout.log").open("wb") as out, (
        destination / "stderr.log"
    ).open("wb") as err:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=out, stderr=err,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
        finally:
            # A command can leave descendants after its leader exits.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=err, stderr=err, timeout=5, check=True,
                )
            process.wait(timeout=5)
    return returncode, timed_out, time.perf_counter() - started


def _run_case(case, *, repo, python, private):
    attempts = []
    expected_metrics = None
    for repetition in range(case.repetitions):
        destination = private / case.id / str(repetition + 1)
        destination.mkdir(parents=True)
        measurement = destination / "measurement.json"
        env = dict(os.environ, MINDIE_BENCH_RESULT_PATH=str(measurement))
        command = [argument.replace("{python}", python) for argument in case.command]
        attempt = {"repetition": repetition + 1}
        try:
            code, timed_out, elapsed = _execute(
                command, repo=repo, env=env, destination=destination,
                timeout=case.timeout_seconds,
            )
            attempt.update(returncode=code, process_seconds=elapsed)
            if timed_out:
                attempt.update(
                    status="unknown" if case.side_effects == "external_write" else "fail",
                    reason_code="execution_timeout",
                )
            elif not measurement.is_file():
                attempt.update(status="fail", reason_code="missing_measurement")
            else:
                payload = Measurement.model_validate(read_json(measurement))
                attempt.update(payload.model_dump())
                attempt["evidence_sha256"] = hashlib.sha256(measurement.read_bytes()).hexdigest()
                signature = {(metric.name, metric.unit) for metric in payload.metrics}
                if code != 0 and payload.status == "pass":
                    attempt.update(status="fail", reason_code="nonzero_exit")
                elif payload.status == "pass" and case.kind != "functional" and not signature:
                    attempt.update(status="fail", reason_code="missing_metrics")
                elif expected_metrics is not None and signature != expected_metrics:
                    attempt.update(status="fail", reason_code="measurement_contract_changed")
                if attempt["status"] == "pass":
                    expected_metrics = signature
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # The public reason is stable; potentially sensitive diagnostics stay local.
            (destination / "runner-error.log").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            attempt.update(
                status="unknown" if case.side_effects == "external_write" else "fail",
                reason_code="execution_or_measurement_error",
            )
        attempts.append(attempt)
        if attempt["status"] != "pass":
            break
    metrics = []
    for name, unit in sorted(expected_metrics or []):
        observed = [
            metric for attempt in attempts if attempt["status"] == "pass"
            for metric in attempt.get("metrics", []) if metric["name"] == name
        ]
        if observed:
            values = [metric["value"] for metric in observed]
            metrics.append({
                "name": name, "unit": unit, "samples": values,
                "iterations": [metric["iterations"] for metric in observed],
                "median": statistics.median(values), "mean": statistics.mean(values),
                "min": min(values), "max": max(values),
            })
    return {
        "id": case.id, "kind": case.kind, "required": case.required,
        "repetitions_planned": case.repetitions, "repetitions_completed": len(attempts),
        "status": attempts[-1]["status"], "attempts": attempts, "metrics": metrics,
    }


def _markdown(report):
    lines = [
        "# Release benchmark results", "",
        f"Release gate: **{'pass' if report['release_gate'] else 'blocked'}**", "",
        "Functional checks, microbenchmarks and E2E measurements have separate scope. "
        "Command wall time includes process startup. No causal productivity or NPU "
        "performance claim follows from functional checks or CPU microbenchmarks.", "",
    ]
    if "repository" in report:
        source = report["source"]
        lines += [
            f"Repository: `{report['repository']}`; version: `{report['version']}`",
            f"Revision: `{source['revision']}`; dirty: `{source['dirty']}`",
            f"Suite: `{report['suite']}`; environment: "
            f"`{report['environment']['platform']}/{report['environment']['machine']}`",
            "",
            "| Case | Kind | Status | Repetitions | Metric | Median | Unit |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for case in report["cases"]:
            for metric in case["metrics"] or [None]:
                lines.append(
                    f"| {case['id']} | {case['kind']} | {case['status']} | "
                    f"{case['repetitions_completed']}/{case['repetitions_planned']} | "
                    f"{metric['name'] if metric else '-'} | "
                    f"{format(metric['median'], '.6g') if metric else '-'} | "
                    f"{metric['unit'] if metric else '-'} |"
                )
            for attempt in case["attempts"]:
                if attempt.get("reason_code"):
                    lines.append(
                        f"\nCase `{case['id']}` repetition {attempt['repetition']}: "
                        f"`{attempt['reason_code']}`.\n"
                    )
    else:
        lines += [
            f"Product version: `{report['product_version']}`", "",
            "| Repository | Version | Revision | Platform | Gate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in report["reports"]:
            lines.append(
                f"| {item['repository']} | {item['version']} | "
                f"{item['source']['revision']} | {item['environment']['platform']} | "
                f"{'pass' if item['release_gate'] else 'blocked'} |"
            )
    if report["blocking_reasons"]:
        lines += ["", "Blocking reasons:", ""]
        lines += [f"- `{reason}`" for reason in report["blocking_reasons"]]
    return "\n".join(lines) + "\n"


def _write_assets(output, report):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "benchmark-results.json", report)
    (output / "benchmark-results.md").write_text(_markdown(report), encoding="utf-8")
    (output / "benchmark-SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ["benchmark-results.json", "benchmark-results.md"]
        ), encoding="utf-8",
    )
    return report


def run_release_suite(*, repo, repository, version, suite, output, python=sys.executable):
    repo, suite, output = repo.resolve(), suite.resolve(), output.resolve()
    contract = ReleaseSuite.model_validate(read_json(suite))
    source = _identity(repo)
    workload_python = subprocess.run(
        [python, "--version"], cwd=repo, capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    started = _now()
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir(mode=0o700)
    cases = [
        _run_case(case, repo=repo, python=python, private=private)
        for case in contract.cases
    ]
    after = _identity(repo)
    blockers = [
        f"required_case:{case['id']}:{case['status']}"
        for case in cases if case["required"] and case["status"] != "pass"
    ]
    if source["dirty"] or after["dirty"]:
        blockers.append("dirty_checkout")
    if source["revision"] != after["revision"]:
        blockers.append("source_changed_during_run")
    report = {
        "schema_version": "mindie-release-benchmark/1",
        "repository": repository, "version": version, "source": source,
        "source_after": after, "suite": contract.suite,
        "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "tool": {"name": "mindie-bench", "version": importlib.metadata.version("mindie-bench")},
        "environment": {
            "platform": sys.platform, "machine": platform.machine(),
            "os_release": platform.release(), "runner_python": platform.python_version(),
            "workload_python": workload_python, "cpu_count": os.cpu_count(),
        },
        "started_at": started, "completed_at": _now(), "cases": cases,
        "release_gate": not blockers, "blocking_reasons": blockers,
    }
    return _write_assets(output, BenchmarkReport.model_validate(report).model_dump())


def collect_release(*, manifest, inputs, output):
    contract = ReleaseManifest.model_validate(read_json(manifest))
    reports = [
        BenchmarkReport.model_validate(read_json(path / "benchmark-results.json")).model_dump()
        for path in inputs
    ]
    blockers = []
    expected = {target.repository: target for target in contract.repositories}
    seen = set()
    for report in reports:
        name, host = report["repository"], report["environment"]["platform"]
        key = name, host
        if key in seen:
            blockers.append(f"duplicate_report:{name}:{host}")
        seen.add(key)
        target = expected.get(name)
        if target is None:
            blockers.append(f"unexpected_repository:{name}")
            continue
        if report["schema_version"] != "mindie-release-benchmark/1":
            blockers.append(f"unsupported_schema:{name}")
        if report["source"]["revision"] != target.revision or report["version"] != target.version:
            blockers.append(f"stale_source_or_version:{name}")
        if report["source"]["dirty"] or not report["release_gate"]:
            blockers.append(f"component_gate_blocked:{name}:{host}")
        cases = {case["id"]: case for case in report["cases"]}
        for required in target.required_cases:
            case = cases.get(required)
            if case is None or case["status"] != "pass":
                blockers.append(f"required_case_missing_or_failed:{name}:{host}:{required}")
    for target in contract.repositories:
        for host in target.platforms:
            if (target.repository, host) not in seen:
                blockers.append(f"missing_report:{target.repository}:{host}")
    return _write_assets(output.resolve(), {
        "schema_version": "mindie-release-benchmark-collection/1",
        "product_version": contract.product_version, "generated_at": _now(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "reports": reports, "release_gate": not blockers, "blocking_reasons": blockers,
    })
