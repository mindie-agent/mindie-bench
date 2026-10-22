# [Role] Defines the deterministic synthetic fixture harness driver.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Explicitly synthetic integration driver; cannot produce live-NPU claims."""

from mindie_bench.models import Trajectory


class FixtureDriver:
    name = "fixture"

    def __init__(self):
        self.result = Trajectory()

    def command(self, profile, config):
        return [config.executable]

    def consume(self, event, elapsed, generation):
        if event.get("type") == "fixture.trajectory":
            self.result = Trajectory.model_validate(event["trajectory"])
        else:
            raise ValueError("invalid fixture protocol")

    def trajectory(self):
        return self.result
