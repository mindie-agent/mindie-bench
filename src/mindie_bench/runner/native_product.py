"""[Role] Mount the selected native product and own its knowledge service.
[Invariants] Product code is imported unchanged from the selected bundle. The
isolated Store receives only the already filtered corpus, and contribution is
explicitly disabled through its native authority. Model traffic uses Gateway.
[Timeouts & Locks] Preparation180s; service readiness30s; stop5s plus kill5s.
[Side-Effect Policy] Writes trial profile/state and starts its one native service.
"""

from __future__ import annotations

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


def _initialize_domain(python: str, env: dict[str, str], corpus: Path, data_home: Path, domain: str, consent: Path, error_path: Path) -> dict:
    with error_path.open("x") as errors:
        completed = subprocess.run(
            [python, "-I", str(HELPER)],
            input=json.dumps(
                {
                    "operation": "initialize",
                    "corpus": str(corpus),
                    "corpus_sha256": tree_hash(corpus),
                    "root": str(data_home),
                    "domain": domain,
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
    return json.loads(completed.stdout)


def prepare(root, env, selected, corpus, *, domain_corpora=None, active_domain=None, harness="dsh"):
    native = root / "native-product"
    config_home, data_home = native / "config", native / "data"
    config_home.mkdir(parents=True)
    (data_home / "state").mkdir(parents=True)
    (data_home / "update").mkdir()
    package = Path(selected.package_path).resolve(strict=True)
    python = str(Path(selected.runtime_python).absolute())
    default_domain = active_domain or selected.domain
    default_corpus = (domain_corpora or {}).get(default_domain, corpus)
    engine_path = config_home / "engine.json"
    community_path = config_home / "mindie-community.json"
    consent = config_home / "mindie-consent.json"
    def _engine_payload(domain_root: Path, domain_name: str) -> dict:
        return {
            "root": str(domain_root),
            "domain": domain_name,
            "admission_path": str(domain_root / "admission.sqlite3"),
            "capture_mode": "public-transcript",
            "transcript_adapter": str(package / "scripts" / f"{harness}_transcript.py"),
            "redactor_executable": selected.redactor_executable,
            "summary_command": [
                python, str(package / "scripts" / (
                    "dsh_organizer.py" if harness == "dsh" else "agent_worker.py"
                )),
            ],
            "community_config": str(community_path),
            "service_launch_context": {"paths": {}, "environment": ["MINDIE_DIAGNOSTICS_CONFIG"]},
            "feeds": [],
        }
    write_json(engine_path, _engine_payload(data_home, default_domain))
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
    extra_domains = {}
    domain_configs = {}
    if domain_corpora:
        domain_configs[default_domain] = str(engine_path)
        for dom_name, dom_corpus in sorted(domain_corpora.items()):
            if dom_name == default_domain:
                continue
            dom_data = native / f"data-{dom_name}"
            (dom_data / "state").mkdir(parents=True)
            dom_engine = config_home / f"engine-{dom_name}.json"
            write_json(dom_engine, _engine_payload(dom_data, dom_name))
            domain_configs[dom_name] = str(dom_engine)
            extra_domains[dom_name] = {
                "corpus": dom_corpus,
                "data_home": dom_data,
                "engine_config": str(dom_engine),
                "connection_path": str(dom_data / dom_name / "connection.json"),
            }
    config_path = config_home / "config.json"
    config_payload = {
        "python": python,
        "venv_dir": str(Path(python).parent.parent),
        "engine_config": str(engine_path),
        "state_dir": str(data_home / "state"),
    }
    if domain_configs:
        config_payload["domain_configs"] = domain_configs
    if harness == "codex":
        config_payload.update(
            runtime_scripts=str(package / "scripts"),
            admission_path=str(data_home / "admission.sqlite3"),
            community_config=str(community_path),
            component_commands={
                name: [str(Path(python).parent / executable)]
                for name, executable in (
                    ("profiler", "mindie-profiler"), ("sim", "mindie-sim"),
                    ("coordinator", "mindie-coordinator"),
                )
                if (Path(python).parent / executable).is_file()
            },
        )
    write_json(config_path, config_payload)
    write_json(config_home / "update.json", {"enabled": False})
    env["MINDIE_DIAGNOSTICS_CONFIG"] = str(config_home / "diagnostics.json")
    if harness == "dsh":
        env.update(
            MINDIE_DSH_CONFIG=str(config_path),
            MINDIE_DSH_ENGINE_CONFIG=str(engine_path),
            MINDIE_DSH_CONFIG_HOME=str(config_home),
            MINDIE_DSH_DATA_HOME=str(data_home),
            MINDIE_DSH_STATE_DIR=str(data_home / "state"),
        )
    else:
        env.update(
            MINDIE_AGENT_CONFIG=str(config_path),
            PLUGIN_ROOT=str(package),
        )
    initialized = _initialize_domain(
        python, env, default_corpus, data_home, default_domain, consent, native / "initialization.stderr"
    )
    write_json(native / "initialization.json", initialized)
    extra_initialized = {}
    for dom_name, dom_info in extra_domains.items():
        dom_init = _initialize_domain(
            python,
            env,
            dom_info["corpus"],
            dom_info["data_home"],
            dom_name,
            consent,
            native / f"initialization-{dom_name}.stderr",
        )
        write_json(native / f"initialization-{dom_name}.json", dom_init)
        extra_initialized[dom_name] = {
            "engine_config": dom_info["engine_config"],
            "connection_path": dom_info["connection_path"],
            "initialization": dom_init,
        }
    result = {
        "package_path": str(package),
        "harness": harness,
        "package_version": json.loads((package / (
            "package.json" if harness == "dsh" else ".codex-plugin/plugin.json"
        )).read_text())["version"],
        "package_sha256": tree_hash(package),
        "runtime_python": python,
        "engine_config": str(engine_path),
        "domain": default_domain,
        "connection_path": str(data_home / default_domain / "connection.json"),
        "contribution_enabled": False,
        "contribution_default_source": str(package / "scripts" / (
            "mindie_sidecar.py" if harness == "dsh" else "bridge.py"
        )),
        "initialization": initialized,
    }
    if extra_initialized:
        result["extra_domains"] = extra_initialized
    if harness == "codex":
        # The original release entry checks this actual interpreter binding;
        # this source-profile receipt makes no release-validation claim.
        write_json(native / "runtime.json", {
            "python": python,
            "components": initialized["components"],
            "source_profile": True,
        })
        result["prompt_hook"] = [
            python, "-I", str(package / "scripts" / "entry.py"),
            "--release", str(native), "user-prompt-submit",
        ]
    return result


class NativeProductService:
    def __init__(self, profile, read_paths, trial_dir, gateway):
        self.profile = profile
        self.read_paths = read_paths
        self.trial_dir = trial_dir
        self.gateway = gateway
        self.info = profile.fingerprint["native_product"]
        self.process = None
        self.port = None
        self.ports = ()
        self._extra_processes = []
        self._stderr_files = []

    def _start_one(self, engine_config: str, connection_path: str, label: str) -> tuple[subprocess.Popen, int]:
        info = self.info
        command = native_command(
            [
                info["runtime_python"],
                "-I",
                "-m",
                "mindie_knowledge.loop.cli",
                "serve",
                "--config",
                engine_config,
            ],
            read=self.read_paths,
            write=[self.profile.root],
            cwd=self.profile.root,
            policy_path=self.trial_dir / f"native-knowledge-service{label}.sb",
            gateway_port=self.gateway.port,
            gateway_socket=self.gateway.unix_socket,
            ephemeral_loopback_listener=True,
        )
        stderr_file = (self.trial_dir / f"native-knowledge-service{label}.stderr").open("xb")
        self._stderr_files.append(stderr_file)
        proc = subprocess.Popen(
            command,
            cwd=self.profile.root,
            env=self.profile.env,
            stdout=stderr_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            connection = Path(connection_path)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("native knowledge service exited before readiness")
                if connection.is_file():
                    state = json.loads(connection.read_text())
                    return proc, urlparse(state["url"]).port
                time.sleep(0.02)
            raise TimeoutError("native knowledge service did not become ready in30s")
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
            raise

    def __enter__(self):
        info = self.info
        try:
            self.process, self.port = self._start_one(info["engine_config"], info["connection_path"], "")
            ports = [self.port]
            self.info["service"] = {
                "pid": self.process.pid,
                "port": self.port,
                "status": "ready",
            }
            for dom_name, dom_info in sorted((info.get("extra_domains") or {}).items()):
                proc, dom_port = self._start_one(
                    dom_info["engine_config"], dom_info["connection_path"], f"-{dom_name}"
                )
                self._extra_processes.append(proc)
                ports.append(dom_port)
                dom_info["service"] = {
                    "pid": proc.pid,
                    "port": dom_port,
                    "status": "ready",
                }
            self.ports = tuple(ports)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            for proc in [self.process, *self._extra_processes]:
                if proc and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
        finally:
            for stderr_file in self._stderr_files:
                stderr_file.close()

    def __exit__(self, *_exc):
        self.close()


def optional_service(profile, read_paths, trial_dir, gateway):
    if profile.fingerprint.get("native_product"):
        return NativeProductService(profile, read_paths, trial_dir, gateway)
    return nullcontext(None)
