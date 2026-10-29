"""Four-harness (`codex`, `claude-code`, `kimi`, `dsh`) G1-G4 and Kimi capture contracts.

[Role]
Verifies that all 4 native harnesses (`codex`, `claude-code`, `kimi`, `dsh`) support
`G1`, `G2`, `G3`, and `G4` profile materialization, driver command generation,
native capture binding (`native_capture_payload`, `first_party`, `graded_capture`),
`QualificationGateRunner` (`G0`-`G3`), and `G4` RSI reuse-to-explore transitions.

[Invariants]
- Every harness isolates its native home directory (`CODEX_HOME`, `CLAUDE_CONFIG_DIR`,
  `KIMI_CODE_HOME`, `DSH_HOME`) inside the per-trial profile root.
- `G3` and `G4` enable native Stop capture hooks in `codex` and `claude-code`,
  bind owned `wire.jsonl` and `state.json` in `kimi`, and bind `.jsonl.zst` in `dsh`.
- `G4` RSI trajectories record `reuse -> explore` transitions and campaign qualification
  gates (`G0`-`G3`) without silent fallbacks.

[Timeouts & Locks]
- Pure Tier 0 and Tier 1 local contract tests (`< 15s` total); no external network waits.

[Side-Effect Policy]
- Writes only inside pytest `tmp_path`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from mindie_bench.integrations import first_party, graded_capture
from mindie_bench.models import (
    CorpusConfig,
    ExperimentConfig,
    Group,
    GroupConfig,
    HarnessConfig,
    MCPServer,
    TerminalStatus,
    ToolCall,
    Trajectory,
)
from mindie_bench.runner import orchestrator, verifier
from mindie_bench.runner.drivers import driver_for
from mindie_bench.runner.orchestrator import native_capture_payload, run_matrix
from mindie_bench.runner.profile_builder import build_profile
from mindie_bench.runner.qualification_gate import (
    AbbaSample,
    BoundaryPair,
    CommandReceipt,
    G0Spec,
    G1Spec,
    G2Spec,
    G3Spec,
    GateCommand,
    GateState,
    GuardWorkload,
    ModuleRun,
    ModuleTopology,
    NumericalQualification,
    QualificationGateRunner,
    frozen_baseline_digest,
)
from mindie_bench.runner.rsi_campaign import run_rsi_gate_campaign
from mindie_bench.scorer.target_access import trajectory_metrics

KIMI_SCRIPTS = Path(__file__).resolve().parents[2] / "mindie-agent-kimi" / "scripts"


def _write_model_catalog(path: Path, slug: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": slug,
                        "max_context_size": 262144,
                        "capabilities": ["thinking", "tool_use"],
                        "support_efforts": ["low", "medium", "high"],
                        "default_effort": "medium",
                    }
                ]
            }
        )
    )
    return path


def _make_harness_executable(tmp_path: Path, harness_name: str) -> Path:
    bin_dir = tmp_path / f"bin-{harness_name}"
    bin_dir.mkdir(exist_ok=True)
    if harness_name == "dsh":
        script = (
            f"#!{sys.executable} -I\n"
            "import json, os, pathlib, sys\n"
            "if '--version' in sys.argv:\n"
            "    print('dsh-fixture 1.0')\n"
            "    raise SystemExit(0)\n"
            "if '--dump-config' in sys.argv:\n"
            "    pdir = pathlib.Path(os.environ['DSH_PROFILE_DIR'])\n"
            "    pdir.mkdir(parents=True, exist_ok=True)\n"
            "    (pdir / 'package.json').write_text(json.dumps({'name': 'bench'}))\n"
            "    print(json.dumps({'status': 'ok'}))\n"
            "    raise SystemExit(0)\n"
        )
    else:
        script = (
            f"#!{sys.executable} -I\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            f"    print('{harness_name}-fixture 1.0')\n"
            "    raise SystemExit(0)\n"
        )
    if os.name == "nt":
        py_script = bin_dir / f"{harness_name}_fixture.py"
        py_script.write_text(script)
        exe = bin_dir / f"{harness_name}.cmd"
        exe.write_text(f'@"{sys.executable}" -I "{py_script}" %*\n')
        return exe
    exe = bin_dir / harness_name
    exe.write_text(script)
    exe.chmod(0o700)
    return exe


@pytest.mark.parametrize("harness_name", ["codex", "claude-code", "kimi", "dsh"])
@pytest.mark.parametrize("group", [Group.G1, Group.G2, Group.G3, Group.G4])
def test_all_four_harnesses_build_profile_and_driver_command_across_g1_to_g4(
    tmp_path, case, corpus, harness_name, group
):
    corpus_root, corpus_commit = corpus
    exe = _make_harness_executable(tmp_path, harness_name)
    model_slug = f"{harness_name}-model"
    catalog = _write_model_catalog(tmp_path / f"catalog-{harness_name}.json", model_slug)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    (skills_dir / "SKILL.md").write_text("# MindIE Skill\n")

    remote_server = MCPServer(
        component="remote-dev",
        command=sys.executable,
        args=["-I", "-c", "pass"],
    )
    knowledge_server = MCPServer(
        component="knowledge",
        command=sys.executable,
        args=["-I", "-c", "pass", "{corpus}", "{profile}", "{local_remote}"],
    )
    group_configs = {
        Group.G1: GroupConfig(mcp_servers={"remote": remote_server}),
        Group.G2: GroupConfig(
            mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
            skills_path=str(skills_dir),
        ),
        Group.G3: GroupConfig(
            mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
            skills_path=str(skills_dir),
            capture_command=[sys.executable, "-I", "-c", "pass"],
        ),
        Group.G4: GroupConfig(
            mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
            skills_path=str(skills_dir),
            capture_command=[sys.executable, "-I", "-c", "pass"],
        ),
    }
    config = ExperimentConfig(
        schema_version="1.0",
        backend="native",
        harness=HarnessConfig(
            name=harness_name,
            executable=str(exe),
            model=model_slug,
            expected_version=f"{harness_name}-fixture 1.0",
            reasoning_effort="medium",
            provider_url="http://127.0.0.1:18080/v1",
            api_key_env="MINDIE_BENCH_TEST_KEY",
            model_catalog_path=str(catalog) if harness_name in {"codex", "kimi"} else None,
        ),
        corpora={"fixture": CorpusConfig(path=str(corpus_root), commit=corpus_commit)},
        groups=group_configs,
    )
    trial_dir = tmp_path / f"trial-{harness_name}-{group.value}"
    trial_dir.mkdir()
    profile = build_profile(trial_dir, case["spec"], group, config)
    profile.env["MINDIE_BENCH_GATEWAY"] = "http://127.0.0.1:18080"

    assert profile.fingerprint["group"] == group.value
    assert profile.fingerprint["harness"] == harness_name
    if group == Group.G1:
        assert profile.fingerprint["components"] == ["remote-dev"]
        assert profile.corpus is None
    else:
        assert profile.fingerprint["components"] == ["knowledge", "remote-dev"]
        assert profile.corpus is not None and profile.corpus.is_dir()
    if group in {Group.G3, Group.G4}:
        assert "MINDIE_BENCH_GIT" in profile.private_env
        assert "GIT_EXEC_PATH" in profile.private_env

    driver = driver_for(harness_name)
    argv = driver.command(profile, config.harness)
    assert argv

    if harness_name == "codex":
        hooks_path = Path(profile.env["CODEX_HOME"]) / "hooks.json"
        if group in {Group.G3, Group.G4}:
            assert hooks_path.is_file()
            assert "--dangerously-bypass-hook-trust" in argv
        else:
            assert not hooks_path.exists()
    elif harness_name == "claude-code":
        settings = json.loads((profile.root / "settings.json").read_text())
        if group in {Group.G3, Group.G4}:
            assert settings["disableAllHooks"] is False
            assert "Stop" in settings["hooks"]
            assert "--no-session-persistence" not in argv
        else:
            assert settings["disableAllHooks"] is True
            assert "--no-session-persistence" in argv
    elif harness_name == "kimi":
        kimi_cfg = tomllib.loads((Path(profile.env["KIMI_CODE_HOME"]) / "config.toml").read_text())
        assert kimi_cfg["providers"]["benchmark"]["base_url"] == "http://127.0.0.1:18080/v1"
        assert kimi_cfg["providers"]["managed:kimi-code"]["base_url"] == "http://127.0.0.1:18080/v1"
        assert kimi_cfg["models"]["bench"]["model"] == model_slug
        assert kimi_cfg["models"]["kimi-code/k3"]["model"] == model_slug
    elif harness_name == "dsh":
        assert Path(profile.env["DSH_HOME"]).is_dir()
        assert argv == [str(exe), "--profile", "bench", "--json"]


def _create_kimi_session(profile_root: Path, workspace: Path, session_id: str) -> Path:
    kimi_home = profile_root / "kimi"
    session_dir = kimi_home / "sessions" / "proj_hash" / session_id
    wire_dir = session_dir / "agents" / "main"
    wire_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "cwd": str(workspace),
                "created_at": 1710000000.0,
            }
        ),
        encoding="utf-8",
    )
    wire = wire_dir / "wire.jsonl"
    records = [
        {"type": "metadata", "protocol_version": "1.3"},
        {
            "type": "context.append_message",
            "time": 1710000001.0,
            "message": {
                "role": "user",
                "origin": {"kind": "user"},
                "content": [{"type": "text", "text": "Fix the Ascend dtype mismatch in module.py"}],
            },
        },
        {
            "type": "context.append_loop_event",
            "time": 1710000002.0,
            "event": {
                "type": "content.part",
                "part": {"type": "think", "think": "private reasoning must never be exported"},
            },
        },
        {
            "type": "context.append_loop_event",
            "time": 1710000005.0,
            "event": {
                "type": "content.part",
                "part": {
                    "type": "text",
                    "text": "Updated module.py to use float16 on Ascend NPU and verified pytest.",
                },
            },
        },
    ]
    wire.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return wire


def test_native_capture_payload_and_adapters_across_all_four_harnesses(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "profile"
    for sub in (
        "codex/sessions/2026",
        "claude/projects/owned",
        "kimi",
        "dsh-home/sessions/proj1/session_dsh_1",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    env = {
        "CODEX_HOME": str(root / "codex"),
        "CLAUDE_CONFIG_DIR": str(root / "claude"),
        "KIMI_CODE_HOME": str(root / "kimi"),
        "DSH_HOME": str(root / "dsh-home"),
    }
    profile = SimpleNamespace(root=root, env=env)

    # 1. Codex
    codex_transcript = root / "codex/sessions/2026/rollout-session_codex_1.jsonl"
    codex_transcript.write_text("{}\n")
    codex_payload = native_capture_payload(
        "codex", Trajectory(native_session_id="session_codex_1", native_completed=True), profile, workspace
    )
    assert codex_payload["harness"] == "codex"
    assert Path(codex_payload["transcript_path"]) == codex_transcript.resolve()

    # 2. Claude Code
    claude_transcript = root / "claude/projects/owned/session_claude_1.jsonl"
    claude_transcript.write_text("{}\n")
    (root / "native-stop.json").write_text(
        json.dumps(
            {
                "hook_event_name": "Stop",
                "session_id": "session_claude_1",
                "prompt_id": "prompt_1",
                "transcript_path": str(claude_transcript),
                "cwd": str(workspace),
            }
        )
    )
    claude_payload = native_capture_payload(
        "claude-code", Trajectory(native_session_id="session_claude_1", native_completed=True), profile, workspace
    )
    assert claude_payload["harness"] == "claude-code"
    assert "turn_id" not in claude_payload
    (root / "native-stop.json").unlink()

    # 3. Kimi
    kimi_wire = _create_kimi_session(root, workspace, "session_kimi_1")
    kimi_payload = native_capture_payload(
        "kimi",
        Trajectory(native_session_id="session_kimi_1", native_completed=True),
        profile,
        workspace,
    )
    assert kimi_payload["harness"] == "kimi"
    assert kimi_payload["session_id"] == "session_kimi_1"
    assert kimi_payload["turn_id"] == "session_kimi_1"
    assert Path(kimi_payload["transcript_path"]) == kimi_wire.resolve()

    if KIMI_SCRIPTS.is_dir():
        kimi_bin = _make_harness_executable(tmp_path, "kimi")
        monkeypatch.setenv("KIMI_CODE_HOME", str(root / "kimi"))
        args = SimpleNamespace(kimi_scripts=str(KIMI_SCRIPTS), kimi_bin=str(kimi_bin))
        event, source, adapter, scripts, worker = first_party.native_capture_inputs(
            args, kimi_payload, root
        )
        assert event["session_id"] == "session_kimi_1"
        assert source["project_root"] == str(workspace.resolve())
        assert worker == "organizer.py"
        material = adapter.read_material(kimi_wire, 0, session_id="session_kimi_1")
        assert material["status"] == "ok"
        assert "Updated module.py to use float16" in material["text"]
        assert "private reasoning must never be exported" not in material["text"]

        graded_text, graded_binding = graded_capture.select_transcript(kimi_payload, KIMI_SCRIPTS)
        assert "Updated module.py to use float16" in graded_text
        assert "private reasoning must never be exported" not in graded_text
        assert graded_binding["session_id"] == "session_kimi_1"
        assert graded_binding["artifact_unchanged"] is True

    # Verify Kimi security checks reject cross-workspace state.json
    state_file = kimi_wire.parents[2] / "state.json"
    state_file.write_text(json.dumps({"session_id": "session_kimi_1", "cwd": str(tmp_path)}))
    with pytest.raises(ValueError, match="differs from the foreground workspace"):
        native_capture_payload(
            "kimi", Trajectory(native_session_id="session_kimi_1", native_completed=True), profile, workspace
        )

    # 4. DSH
    dsh_artifact = root / "dsh-home/sessions/proj1/session_dsh_1/session.v4.jsonl.zstd"
    dsh_lines = [
        json.dumps(
            {
                "type": "session",
                "version": 4,
                "id": "session_dsh_1",
                "cwd": str(workspace.resolve()),
                "timestamp": "2026-10-23T00:00:00Z",
            }
        ),
        json.dumps(
            {
                "type": "message",
                "id": "m1",
                "message": {"role": "user", "content": [{"type": "text", "text": "Fix module.py"}]},
            }
        ),
        json.dumps(
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Applied fix."}],
                },
            }
        ),
    ]
    dsh_artifact.write_bytes(
        zstandard.ZstdCompressor().compress(("\n".join(dsh_lines) + "\n").encode("utf-8"))
    )
    dsh_descriptor = {
        "harness": "dsh",
        "completed": True,
        "session_id": "session_dsh_1",
        "turn_id": "1",
        "cwd": str(workspace.resolve()),
    }
    dsh_payload = native_capture_payload(
        "dsh",
        Trajectory(native_session_id="session_dsh_1", native_completed=True),
        profile,
        workspace,
        native_descriptor=dsh_descriptor,
    )
    assert dsh_payload["harness"] == "dsh"
    assert dsh_payload["session_id"] == "session_dsh_1"


def test_g1_through_g4_trial_and_matrix_execution_with_rsi_and_qualification_gates(
    tmp_path, case, corpus, monkeypatch
):
    if os.name == "nt":
        monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)
        monkeypatch.setattr(verifier, "native_command", lambda command, **kwargs: command)
    corpus_root, corpus_commit = corpus
    seed_root = tmp_path / "task-seed"
    eval_root = tmp_path / "task-eval"
    shutil.copytree(case["task"], seed_root)
    shutil.copytree(case["task"], eval_root)

    for root, instance_id, role in [(seed_root, "task-seed", "seed"), (eval_root, "task-eval", "eval")]:
        text = (root / "task.toml").read_text()
        text = text.replace(
            'instance_id = "synthetic-dtype"',
            f'instance_id = "{instance_id}"\nrole = "{role}"\ncluster = "cluster-a"',
        )
        (root / "task.toml").write_text(text)

    capture_script = tmp_path / "capture_adapter.py"
    capture_script.write_text(
        "import hashlib, json, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        "dest = pathlib.Path(payload['destination'])\n"
        "pkg = dest / 'tasks' / f\"captured-{payload['instance_id']}\"\n"
        "(pkg / 'materials').mkdir(parents=True, exist_ok=True)\n"
        "(pkg / 'PACKAGE.md').write_text('# Captured experience\\n')\n"
        "(pkg / 'materials/0001.md').write_text('Captured trace for cluster-a.\\n')\n"
        "digest = hashlib.sha256(b'captured').hexdigest()\n"
        "print(json.dumps({\n"
        "    'status': 'completed',\n"
        "    'tokens': {'total': 12, 'complete': True},\n"
        "    'corpus_lint': {'status': 'passed', 'findings': []},\n"
        "    'packages': [{'task_id': pkg.name, 'package_hash': digest}],\n"
        "    'publication': {'kind': 'local-git', 'revision': digest[:12]},\n"
        "    'consumer': {'revision': digest[:12], 'reads': [{'ref': pkg.name}]},\n"
        "}))\n"
    )
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "SKILL.md").write_text("# MindIE Skill\n")

    remote_server = MCPServer(
        component="remote-dev",
        command=sys.executable,
        args=["-I", "-c", "pass"],
    )
    knowledge_server = MCPServer(
        component="knowledge",
        command=sys.executable,
        args=[
            "-I",
            "-m",
            "mindie_bench.integrations.first_party",
            "knowledge-mcp",
            "--corpus",
            "{corpus}",
            "--state",
            "{profile}/knowledge-state",
        ],
    )
    config = case["config"].model_copy(
        update={
            "corpora": {"fixture": CorpusConfig(path=str(corpus_root), commit=corpus_commit)},
            "runtime_read_paths": [str(capture_script)],
            "groups": {
                Group.G1: GroupConfig(mcp_servers={"remote": remote_server}),
                Group.G2: GroupConfig(
                    mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
                    skills_path=str(skills_dir),
                ),
                Group.G3: GroupConfig(
                    mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
                    skills_path=str(skills_dir),
                    capture_command=[sys.executable, "-I", str(capture_script)],
                ),
                Group.G4: GroupConfig(
                    mcp_servers={"remote": remote_server, "knowledge": knowledge_server},
                    skills_path=str(skills_dir),
                    capture_command=[sys.executable, "-I", str(capture_script)],
                ),
            },
        }
    )

    output = tmp_path / "matrix-g1-g4"
    receipts = run_matrix(
        [seed_root, eval_root],
        config,
        tier="l2",
        groups=[Group.G1, Group.G2, Group.G3, Group.G4],
        repetitions=1,
        concurrency=2,
        output=output,
    )
    assert len(receipts) == 8
    assert all(r.terminal_status == TerminalStatus.PASS for r in receipts)
    by_group = {
        g.value: [r for r in receipts if r.ablation_group == g]
        for g in [Group.G1, Group.G2, Group.G3, Group.G4]
    }
    for g_name in ("G1", "G2", "G3", "G4"):
        assert len(by_group[g_name]) == 2
    for g_name in ("G3", "G4"):
        seed_receipt = next(r for r in by_group[g_name] if r.instance_id == "task-seed")
        assert seed_receipt.auxiliary_metrics["capture"]["status"] == "completed"
        assert "captured_corpus" in seed_receipt.artifacts

    summary = json.loads((output / "summary.json").read_text())
    assert {"l2/synthetic/G1", "l2/synthetic/G2", "l2/synthetic/G3", "l2/synthetic/G4"} <= set(summary)
    assert summary["l2/synthetic/G3"]["seed_setup_attempts"] == 1
    assert summary["l2/synthetic/G4"]["seed_setup_attempts"] == 1
    accounting = json.loads((output / "matrix_accounting.json").read_text())
    assert accounting["all_cells_accounted_for"] is True
    assert accounting["all_planned_cells_passed"] is True
    assert accounting["missing_count"] == 0

    # Verify G4 RSI reuse -> explore trajectory observations
    rsi_traj = Trajectory(
        native_completed=True,
        total_turns=2,
        calls=[
            ToolCall(
                id="k1",
                name="knowledge_query",
                component="knowledge",
                arguments={"query": "Ascend ACLGraph", "mode": "reuse"},
                turn=1,
                started_sec=0.1,
                ended_sec=0.2,
                is_error=False,
            ),
            ToolCall(
                id="k2",
                name="knowledge_query",
                component="knowledge",
                arguments={
                    "query": "Ascend ACLGraph",
                    "mode": "explore",
                    "goal": "Reduce dispatch latency",
                    "bottleneck_summary": {"stage": "dispatch", "share": 0.62},
                    "ledger_entries": [{"candidate_id": "cand-0", "verdict": "discard"}],
                    "native_subcontext": True,
                },
                turn=2,
                started_sec=0.3,
                ended_sec=0.5,
                is_error=False,
            ),
        ],
    )
    _, aux = trajectory_metrics(rsi_traj, case["spec"], config.harness, 1.0, tier="l2")
    rsi_obs = aux["rsi_exploration"]
    assert rsi_obs["reuse_calls_count"] == 1
    assert rsi_obs["explore_calls_count"] == 1
    assert rsi_obs["mode_sequence"] == ["reuse", "explore"]
    assert rsi_obs["transitioned_reuse_to_explore"] is True
    assert rsi_obs["explore_calls"][0]["native_subcontext"] is True


@pytest.mark.parametrize("harness_name", ["codex", "claude-code", "kimi", "dsh"])
def test_qualification_gates_g0_to_g3_and_rsi_campaign_across_all_four_harnesses(
    tmp_path, harness_name
):
    build = tmp_path / f"build-{harness_name}"
    build.mkdir()
    (build / "flashmla.o").write_bytes(b"\x7fELF native aicore kernel\n")
    (build / "flashmla_aicpu.json").write_text(
        json.dumps(
            {
                "FlashMLA": {
                    "opInfo": {
                        "engine": "DNN_VM_AICPU",
                        "kernelSo": "libtransformer_aicpu_kernels.so",
                        "functionName": "RunCpuKernel",
                    }
                }
            }
        )
    )
    manifest = tmp_path / f"operators-{harness_name}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "mindie-native-operator-manifest/1",
                "operators": [
                    {
                        "name": "FlashMLA",
                        "aicore": f"{build.name}/flashmla.o",
                        "aicpu": f"{build.name}/flashmla_aicpu.json",
                    }
                ],
            }
        )
    )
    generated = tmp_path / f"config-{harness_name}.json"
    generated.write_text(json.dumps({"op": "FlashMLA", "tiling": {"block": 32}}))
    lib_path = tmp_path / f"libmindie_{harness_name}.so"
    lib_path.write_bytes(b"\x7fELF-synthetic-custom-op")

    def _executor(cmd: GateCommand) -> CommandReceipt:
        if cmd.argv[:2] == ("ldd", "-r"):
            return CommandReceipt(cmd.argv, 0, "", "")
        if "native-gate-opp" in cmd.argv:
            return CommandReceipt(cmd.argv, 0, "G1_BUILD_OK\n", "")
        return CommandReceipt(cmd.argv, 0, "\n".join(cmd.expect_stdout) + "\n", "")

    gate_runner = QualificationGateRunner(executor=_executor)
    g0_spec = G0Spec(
        contract_tests=(
            GateCommand(
                ("python3", "-c", "print('CPU_CONTRACT_OK')"),
                timeout_sec=5,
                expect_stdout=("CPU_CONTRACT_OK",),
            ),
        ),
        manifests=(str(manifest),),
        generated_json=(str(generated),),
        ldd_targets=(str(lib_path),),
    )
    g0 = gate_runner.verify_g0_artifact_and_contract(g0_spec)
    assert g0.state is GateState.COMPLETED

    g1_spec = G1Spec(
        source_root=str(tmp_path),
        jobs=2,
        build_sentinel=("G1_BUILD_OK",),
        eager_qualification=GateCommand(
            ("python3", "-c", "print('EAGER_SENTINEL')"),
            timeout_sec=5,
            expect_stdout=("EAGER_SENTINEL",),
        ),
        aclgraph_qualification=GateCommand(
            ("python3", "-c", "print('ACLGRAPH_SENTINEL')"),
            timeout_sec=5,
            expect_stdout=("ACLGRAPH_SENTINEL",),
        ),
    )
    g1 = gate_runner.build_and_verify_g1_incremental_gate(g1_spec)
    assert g1.state is GateState.COMPLETED

    samples = [
        AbbaSample("baseline", "warmup", 1, 200, 1, 1280.0, True),
        AbbaSample("candidate", "warmup", 1, 100, 1, 1280.0, True),
    ]
    for r in (1, 2, 3):
        samples.extend(
            [
                AbbaSample("baseline", "measured", r, 400_000_000, 1, 1280.0, True),
                AbbaSample("candidate", "measured", r, 200_000_000, 1, 1280.0, True),
                AbbaSample("candidate", "measured", r, 200_000_000, 1, 1280.0, True),
                AbbaSample("baseline", "measured", r, 400_000_000, 1, 1280.0, True),
            ]
        )
    g2_base = G2Spec(
        baseline_id=f"frozen-baseline-{harness_name}",
        candidate_id=f"cand-{harness_name}",
        frozen_baseline_sha256="",
        samples=tuple(samples),
        rounds=3,
        warmup_rounds=1,
        numerical=NumericalQualification(
            baseline_passed=True,
            candidate_passed=True,
            sentinel_ok=True,
            max_abs_error=1e-4,
            tolerance=1e-3,
        ),
        min_sustained_window_sec=0.001,
        sustained_clock_mhz=(1250.0, 1300.0),
    )
    g2_spec = G2Spec(**{**g2_base.__dict__, "frozen_baseline_sha256": frozen_baseline_digest(g2_base)})
    g2 = gate_runner.evaluate_g2_sustained_abba(g2_spec)
    assert g2.state is GateState.COMPLETED

    g3_spec = G3Spec(
        module_id=f"mlapo-flashmla-{harness_name}",
        topology=ModuleTopology(tp=2, die_count=2, observed_ranks=(0, 1)),
        boundary_pairs=(
            BoundaryPair(
                producer="MLAPO",
                consumer="FlashMLA",
                boundary="quantized_qkv",
                logical_identity="token:role:qkv_fp8",
                observed_identity="token:role:qkv_fp8",
                equivalent=True,
                ownership_preserved=True,
                numerical_passed=True,
                executed_calls=3,
            ),
        ),
        paired_runs=(
            ModuleRun("baseline", "run-before", "a" * 64, True, True, 2),
            ModuleRun("candidate", "run-after", "a" * 64, True, True, 2),
        ),
        guard_workloads=(
            GuardWorkload(
                "decode-concurrency-1",
                True,
                True,
                baseline_ns=1_000,
                candidate_ns=980,
                max_regression_ratio=1.02,
            ),
        ),
    )
    g3 = gate_runner.evaluate_g3_subgraph_module_gate(g3_spec)
    assert g3.state is GateState.COMPLETED

    campaign = run_rsi_gate_campaign(
        units=[{"id": "u1", "layer": "L1", "title": "Preserve ACLGraph stream capture"}],
        goal=f"Optimize custom op for {harness_name}",
        conditions={"harness": harness_name, "soc_version": "Ascend910B"},
        candidate={
            "candidate_id": f"cand-{harness_name}",
            "mutation_summary": "Vectorize custom kernel load",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.25,
            "ablation_retained": True,
            "t_baseline_us": 100.0,
            "t_cand_us": 80.0,
        },
        gate_runner=gate_runner,
        gate_specs={"G0": g0_spec, "G1": g1_spec, "G2": g2_spec, "G3": g3_spec},
        hidden_input_cases=[
            {"case_id": "hidden-1", "ref_ok": True, "opt_ok": True, "parity_ok": True, "speedup_ratio": 1.2}
        ],
    )
    assert campaign["campaign_passed"] is True
    assert campaign["qualification_ok"] is True
    assert campaign["verdict"] == "supported"

