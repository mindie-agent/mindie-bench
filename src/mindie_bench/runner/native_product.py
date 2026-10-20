"""[Role] Mount the original DSH product host and own its native knowledge service.
[Invariants] Product code is imported unchanged from the selected bundle. The
isolated Store receives only the already filtered corpus, and contribution is
explicitly disabled through its native authority. Model traffic uses Gateway.
[Timeouts & Locks] Preparation180s; service readiness30s; stop5s plus kill5s.
[Side-Effect Policy] Writes trial profile/state and starts its one native service.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

from mindie_bench.io import tree_hash, write_json
from mindie_bench.runner.sandbox import native_command

HELPER = Path(__file__).resolve().parents[1] / "integrations" / "product_native.py"


def runtime_identity(python: str, *, env: dict[str, str] | None = None) -> dict:
    result = subprocess.run(
        [python, "-I", str(HELPER)],
        input=json.dumps({"operation": "identities"}),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        env=env,
    )
    return json.loads(result.stdout)


def prepare(root, env, selected, corpus):
    native = root / "native-product"
    config_home, data_home = native / "config", native / "data"
    config_home.mkdir(parents=True)
    (data_home / "state").mkdir(parents=True)
    (data_home / "update").mkdir()
    package = Path(selected.package_path).resolve(strict=True)
    python = str(Path(selected.runtime_python).absolute())
    engine_path = config_home / "engine.json"
    community_path = config_home / "mindie-community.json"
    consent = config_home / "mindie-consent.json"
    engine = {
        "root": str(data_home),
        "domain": selected.domain,
        "admission_path": str(data_home / "admission.sqlite3"),
        "capture_mode": "public-transcript",
        "transcript_adapter": str(package / "scripts" / "dsh_transcript.py"),
        "redactor_executable": selected.redactor_executable,
        "summary_command": [python, str(package / "scripts" / "dsh_organizer.py")],
        "community_config": str(community_path),
        "service_launch_context": {"paths": {}, "environment": ["MINDIE_DIAGNOSTICS_CONFIG"]},
        "feeds": [],
    }
    write_json(engine_path, engine)
    write_json(
        community_path,
        {
            "schema": "mindie-community-config/1",
            "enabled": False,
            "repository": None,
            "project_roots": [],
            "branch": "main",
            "idle_seconds": 300,
            "visibility": "public",
            "consent_config": str(consent),
        },
    )
    config_path = config_home / "config.json"
    write_json(
        config_path,
        {
            "python": python,
            "venv_dir": str(Path(python).parent.parent),
            "engine_config": str(engine_path),
            "state_dir": str(data_home / "state"),
        },
    )
    write_json(config_home / "update.json", {"enabled": False})
    env.update(
        MINDIE_DSH_CONFIG=str(config_path),
        MINDIE_DSH_ENGINE_CONFIG=str(engine_path),
        MINDIE_DSH_CONFIG_HOME=str(config_home),
        MINDIE_DSH_DATA_HOME=str(data_home),
        MINDIE_DSH_STATE_DIR=str(data_home / "state"),
        MINDIE_DIAGNOSTICS_CONFIG=str(config_home / "diagnostics.json"),
    )
    error_path = native / "initialization.stderr"
    with error_path.open("x") as errors:
        completed = subprocess.run(
            [python, "-I", str(HELPER)],
            input=json.dumps(
                {
                    "operation": "initialize",
                    "corpus": str(corpus),
                    "corpus_sha256": tree_hash(corpus),
                    "root": str(data_home),
                    "domain": selected.domain,
                    "consent": str(consent),
                }
            ),
            env=env,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            timeout=180,
        )
    if completed.returncode:
        raise RuntimeError("native product preparation failed: " + error_path.read_text()[-3000:])
    initialized = json.loads(completed.stdout)
    write_json(native / "initialization.json", initialized)
    return {
        "package_path": str(package),
        "package_sha256": tree_hash(package),
        "package_version": json.loads((package / "package.json").read_text())["version"],
        "host_entry_sha256": hashlib.sha256((package / "host.mjs").read_bytes()).hexdigest(),
        "runtime_python": python,
        "engine_config": str(engine_path),
        "domain": selected.domain,
        "connection_path": str(data_home / selected.domain / "connection.json"),
        "contribution_enabled": False,
        "contribution_default_source": str(package / "scripts" / "mindie_sidecar.py"),
        "contribution_default_source_sha256": hashlib.sha256(
            (package / "scripts" / "mindie_sidecar.py").read_bytes()
        ).hexdigest(),
        "initialization": initialized,
    }


class NativeProductService:
    def __init__(self, profile, read_paths, trial_dir, gateway):
        self.profile = profile
        self.read_paths = read_paths
        self.trial_dir = trial_dir
        self.gateway = gateway
        self.info = profile.fingerprint["native_product"]
        self.process = None
        self.port = None

    def __enter__(self):
        info = self.info
        command = native_command(
            [
                info["runtime_python"],
                "-I",
                "-m",
                "mindie_knowledge.loop.cli",
                "serve",
                "--config",
                info["engine_config"],
            ],
            read=self.read_paths,
            write=[self.profile.root],
            cwd=self.profile.root,
            policy_path=self.trial_dir / "native-knowledge-service.sb",
            gateway_port=self.gateway.port,
            gateway_socket=self.gateway.unix_socket,
            ephemeral_loopback_listener=True,
        )
        self.stderr = (self.trial_dir / "native-knowledge-service.stderr").open("xb")
        self.process = subprocess.Popen(
            command,
            cwd=self.profile.root,
            env=self.profile.env,
            stdout=self.stderr,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        connection = Path(info["connection_path"])
        deadline = time.monotonic() + 30
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("native knowledge service exited before readiness")
                if connection.is_file():
                    state = json.loads(connection.read_text())
                    self.port = urlparse(state["url"]).port
                    self.info["service"] = {
                        "pid": self.process.pid,
                        "port": self.port,
                        "connection_sha256": hashlib.sha256(connection.read_bytes()).hexdigest(),
                        "status": "ready",
                    }
                    return self
                time.sleep(0.02)
            raise TimeoutError("native knowledge service did not become ready in30s")
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.process and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        finally:
            self.stderr.close()

    def __exit__(self, *_exc):
        self.close()


def optional_service(profile, read_paths, trial_dir, gateway):
    if profile.fingerprint.get("native_product"):
        return NativeProductService(profile, read_paths, trial_dir, gateway)
    return nullcontext(None)
