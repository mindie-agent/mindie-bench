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

assert Path(provider.__file__).resolve().is_relative_to(WORKSPACE)


def check():
    with tempfile.TemporaryDirectory(prefix="l2-provider-") as directory:
        root = Path(directory)
        parent = root / "parent"
        parent.mkdir()
        record = {
            "slug": "fixture/small-model",
            "display_name": "CPU protocol fixture",
            "supported_reasoning_levels": [],
            "shell_type": "unified_exec",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 1,
            "support_verbosity": True,
            "truncation_policy": {"mode": "tokens", "limit": 4096},
            "experimental_supported_tools": [],
            "context_window": 272000,
            "base_instructions": "Fixture native metadata, no inference.",
        }
        catalog = parent / "models.json"

        def configure(records):
            catalog.write_text(json.dumps({"models": records}))
            (parent / "config.toml").write_text(
                "model_catalog_json=" + json.dumps(str(catalog)) + "\n"
                'model_provider="fixture"\n[model_providers.fixture]\n'
                'name="CPU fixture"\nbase_url="http://127.0.0.1:1/v1"\n'
                'experimental_bearer_token="fixture-placeholder"\n'
            )

        os.environ["CODEX_HOME"] = str(parent)
        configure([{"slug": "other"}, record])
        selected = provider.selected("small-model")
        (root / "worker").mkdir()
        projected = provider.environment(selected, root / "worker")
        import tomllib

        config = tomllib.loads((Path(projected["CODEX_HOME"]) / "config.toml").read_text())
        print(
            json.dumps(
                {
                    "selected_model": selected["model"],
                    "catalog_projected": "model_catalog_json" in config,
                }
            )
        )
        assert selected["model"] == record["slug"]
        assert "model_catalog_json" in config, "selected model catalog was lost during projection"
        assert json.loads(Path(config["model_catalog_json"]).read_text()) == {"models": [record]}
        first = selected["route_sha256"]
        configure([dict(record, context_window=280000)])
        assert provider.selected("small-model")["route_sha256"] != first
        for records in (
            [record, dict(record, slug="second/small-model")],
            [{"slug": "fixture/small-model"}],
        ):
            configure(records)
            try:
                provider.selected("small-model")
            except provider.ProviderConfigurationError:
                pass
            else:
                raise AssertionError("ambiguous/incomplete catalog must fail visibly")


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
