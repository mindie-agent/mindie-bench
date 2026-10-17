# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

sys.path.insert(0, str(WORKSPACE / "plugins/mindie-agent/scripts"))
import summary_provider as provider

with tempfile.TemporaryDirectory(prefix="l2-public-provider-") as directory:
    root = Path(directory)
    home = root / "parent"
    home.mkdir()
    record = {
        "slug": "fixture/small-model",
        "display_name": "CPU fixture",
        "supported_reasoning_levels": [],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "support_verbosity": True,
        "truncation_policy": {"mode": "tokens", "limit": 4096},
        "experimental_supported_tools": [],
        "context_window": 272000,
    }
    catalog = home / "models.json"
    catalog.write_text(json.dumps({"models": [record]}))
    (home / "config.toml").write_text(
        "model_catalog_json=" + json.dumps(str(catalog)) + "\n"
        'model_provider="fixture"\n[model_providers.fixture]\n'
        'name="CPU fixture"\nbase_url="http://127.0.0.1:1/v1"\n'
    )
    os.environ["CODEX_HOME"] = str(home)
    selected = provider.selected("small-model")
    (root / "worker").mkdir()
    environment = provider.environment(selected, root / "worker")
    import tomllib

    projected = tomllib.loads((Path(environment["CODEX_HOME"]) / "config.toml").read_text())
    print(
        json.dumps(
            {"selected_model": selected["model"], "catalog_projected": "model_catalog_json" in projected}
        )
    )
