# Imports follow explicit owned source selection.
# ruff: noqa: E402
"""Exercise the real reporting module; no server, network or NPU is started."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd()))

from tests.e2e.nightly.multi_node.external_dp.scripts import utils
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import ServerCommand

configured = ServerCommand(
    cmd=["vllm", "serve", "fixture-model", "--additional-config", json.dumps({"enable_fused_mc2": 1})],
    env={"VLLM_ASCEND_ENABLE_FUSED_MC2": "0"},
    display_cmd="local reporting fixture, not executed",
)
legacy_only = ServerCommand(
    cmd=["vllm", "serve", "fixture-model"],
    env={"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"},
    display_cmd="local reporting fixture, not executed",
)
result = {
    "current_config_features": utils._extract_features([configured]),
    "ignored_legacy_env_features": utils._extract_features([legacy_only]),
}
print(json.dumps(result, indent=2))
assert "fused_mc2" in result["current_config_features"], "active current configuration was omitted"
assert "fused_mc2" not in result["ignored_legacy_env_features"], (
    "ignored legacy input was reported active"
)
