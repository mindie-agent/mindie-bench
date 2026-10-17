"""Durable stage observations without changing the outcome of the measured work."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from mindie_bench.io import write_json


class StageRecorder:
    def __init__(self, path: Path, *, clock=time.perf_counter):
        self.path = path
        self.clock = clock
        self.origin = clock()
        self.stages: list[dict] = []

    def save(self):
        write_json(self.path, dict(schema="mindie-bench-stage-observations/1", stages=self.stages))

    @contextmanager
    def measure(self, name: str, *, record_start: bool = True):
        start = self.clock()
        stage = dict(name=name, started_sec=start - self.origin, status="started")
        self.stages.append(stage)
        if record_start:
            self.save()
        primary = None
        try:
            yield
        except BaseException as exc:
            primary = exc
            stage.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
            raise
        else:
            stage["status"] = "completed"
        finally:
            end = self.clock()
            stage.update(ended_sec=end - self.origin, elapsed_sec=end - start)
            try:
                self.save()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"stage observation write also failed: {type(error).__name__}: {error}")
