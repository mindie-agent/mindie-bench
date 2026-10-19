"""Measure failure-to-report delivery, redaction, deduplication and recovery.

[Role] Drive the unchanged production logger, outbox and reporter against local HTTP.
[Invariants] The receiver is a transport fixture, never a claim of a real GitHub Issue.
[Timeouts & Locks] Reporter cycles use ten seconds; real retry backoff is bounded to 90s.
[Side-Effect Policy] Only owned files, subprocesses and loopback HTTP are used.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .value_common import source_revisions, write_json

REPOSITORY = "mindie-value-fixture/local-receiver"


class Receiver:
    def __init__(self, reject_first_read):
        self.reject_first_read = reject_first_read
        self.issues = []
        self.requests = []
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, body):
                blob = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                state.requests.append({"method": "GET", "at": time.monotonic()})
                if state.reject_first_read:
                    state.reject_first_read = False
                    self.send(429, {"message": "controlled read rate limit"})
                else:
                    self.send(200, state.issues)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state.requests.append({"method": "POST", "at": time.monotonic(), "payload": body})
                issue = {"number": len(state.issues) + 1,
                         "html_url": f"https://github.com/{REPOSITORY}/issues/{len(state.issues) + 1}",
                         **body}
                state.issues.append(issue)
                self.send(201, issue)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


def one(output: str, repetition: int, exercise_retry: bool = True):
    import importlib.metadata
    import sys

    from mindie_diagnostics.integration import record_failure
    from mindie_diagnostics.outbox import Outbox
    from mindie_diagnostics.reporter import GitHub, ingest, publish_one
    from mindie_knowledge.consent_store import activate_reporting, record_choice

    root = Path(output) / f"diagnostics-{repetition}"
    root.mkdir(parents=True)
    policy = root / "diagnostics.json"
    consent = root / "mindie-consent.json"
    events = root / "events"
    previous = {name: os.environ.get(name) for name in (
        "MINDIE_DIAGNOSTICS_CONFIG", "MINDIE_DIAGNOSTICS_ROOT", "MINDIE_VALUE_RECEIVER_URL"
    )}
    os.environ["MINDIE_DIAGNOSTICS_CONFIG"] = str(policy)
    os.environ["MINDIE_DIAGNOSTICS_ROOT"] = str(events)
    bridge = root / "local-gh"
    helper = Path(__file__).with_name("value_http_bridge.py")
    bridge.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(helper))} \"$@\"\n")
    bridge.chmod(0o700)
    secret = "ghp_" + "valuefixture0123456789" * 2
    missing = root / "private-workspace" / secret / "weights.bin"
    baseline = []
    recorded = []
    distribution = importlib.metadata.distribution("remote-dev")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    started = time.monotonic()
    try:
        record_choice(consent, "contribute")
        activation = activate_reporting(
            consent, config=policy, roots=[str(events)], repository=REPOSITORY, ensure=False
        )
        assert activation["status"] == "configured"
        for _ in range(3):
            try:
                missing.read_bytes()
            except FileNotFoundError as error:
                baseline.append(traceback.format_exc())
                recorded.append(record_failure(
                    "remote-dev", "remote.read", stage="tool_call",
                    category="internal_exception", exception=error, root=events,
                    version=distribution.version,
                    revision=origin.get("vcs_info", {}).get("commit_id"),
                ))
        assert all(row["recorded"] and not row["logging_failed"] for row in recorded)
        (root / "baseline-raw.log").write_text("\n".join(baseline))
        queue = Outbox(root / "outbox.sqlite3")
        ingested = ingest(events, queue, config=policy)
        initial_rows = queue.rows()
        assert len(initial_rows) == 1 and initial_rows[0]["occurrences"] == 3
        with Receiver(exercise_retry) as receiver:
            os.environ["MINDIE_VALUE_RECEIVER_URL"] = (
                f"http://127.0.0.1:{receiver.server.server_port}"
            )
            client = GitHub(REPOSITORY, executable=str(bridge), timeout=10)
            first = publish_one(queue, client)
            retry_wait_sec = 0.0
            if exercise_retry:
                assert first["status"] == "retry" and not receiver.issues
                due = queue.rows()[0]["next_attempt"]
                retry_wait_sec = max(0.0, due - time.time())
                assert retry_wait_sec <= 90
                # This uses the real production backoff and wall clock, not an advanced test clock.
                threading.Event().wait(retry_wait_sec + 0.05)
                delivered = publish_one(queue, client)
            else:
                delivered = first
            assert delivered["status"] == "published"
            delivery_sec = time.monotonic() - started
            assert len(receiver.issues) == 1
            body = receiver.issues[0]["body"]
            assert secret not in body and str(root) not in body
            assert "FileNotFoundError" in body and "remote.read" in body
            assert "package_revision" in body and "tool_call" in body
            # A fresh reader must reconcile the existing remote marker rather than POST again.
            independent = Outbox(root / "independent-outbox.sqlite3")
            ingest(events, independent, config=policy)
            reconciled = publish_one(independent, client)
            assert reconciled["status"] == "reconciled"
            withdrawn = Outbox(root / "withdrawn-outbox.sqlite3")
            ingest(events, withdrawn, config=policy)
            record_choice(consent, "disabled")
            requests_before_off = len(receiver.requests)
            off = publish_one(withdrawn, client)
            assert off["status"] == "withdrawn"
            assert len(receiver.requests) == requests_before_off
            assert sum(row["method"] == "POST" for row in receiver.requests) == 1
            raw = {"activation": activation, "recorded": recorded, "ingest": ingested,
                   "initial_occurrences": initial_rows[0]["occurrences"],
                   "first_publish": first, "delivery": delivered, "reconcile": reconciled,
                   "withdrawn": off, "receiver_requests": receiver.requests,
                   "receiver_issues": receiver.issues, "final_outbox": queue.rows(),
                   "receiver_scope": "loopback transport fixture; no public Issue exists"}
            write_json(root / "raw.json", raw)
            (root / "delivered-report.md").write_text(body)
            return {
                "repetition": repetition, "status": "pass",
                "baseline": {"failure_occurrences": 3, "raw_tracebacks": len(baseline),
                             "contains_synthetic_secret": secret in "\n".join(baseline),
                             "automatic_delivery": False},
                "candidate": {"failure_occurrences": 3, "queue_items": 1, "http_posts": 1,
                              "delivered_reports": 1, "reconciled_without_duplicate": True,
                              "private_fixture_removed": True, "useful_fault_fields_retained": True,
                              "withdrawal_prevented_requests": True,
                              "production_retry_wait_sec": retry_wait_sec,
                              "failure_to_delivery_sec": delivery_sec},
                "raw": str(root / "raw.json"),
                "report": str(root / "delivered-report.md"),
            }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(output: Path, repetitions: int, *, exercise_retry=True):
    import concurrent.futures
    import multiprocessing

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=repetitions, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(one, str(output), number, exercise_retry)
                   for number in range(1, repetitions + 1)]
        measurements = [future.result(timeout=110) for future in futures]
    return {
        "id": "diagnostics-fault-delivery", "component": "diagnostics", "status": "pass",
        "scope": "real logging, linked consent, ingestion, outbox and reporter with local HTTP",
        "baseline": "Three real exception tracebacks retained as ordinary local raw logs.",
        "candidate": "One sanitized issue payload, remote-marker deduplication and read-failure recovery.",
        "repetitions": repetitions, "measurements": measurements,
        "limitations": [
            "The HTTP receiver implements a bounded GitHub-compatible fixture, not GitHub availability.",
            "GitHub.request and subprocess parsing are production code; only its executable transport is local.",
            "The controlled fault is FileNotFoundError; reporting does not repair the failing operation.",
            "No human triage-time or model productivity claim is made.",
            "The separate public Issue 243 native canary is not rerun or counted here.",
        ],
        "source_revisions": source_revisions(),
    }
