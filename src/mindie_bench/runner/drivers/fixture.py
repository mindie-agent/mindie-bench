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
