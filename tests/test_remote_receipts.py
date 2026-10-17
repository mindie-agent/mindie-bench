import base64
import io
import json
from types import SimpleNamespace

import pytest

from mindie_bench.models import RemoteConfig
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.soundness import check_soundness


def remote_slot(case):
    config = RemoteConfig(
        host="fixture",
        container="fixture",
        repo_path="/fixture",
        devices=[0],
        probe_command=["probe"],
    )
    environment = (
        case["spec"].environment["l2"].model_copy(update={"npu_model": "Ascend910", "npu_count": 1})
    )
    slot = RemoteSlot(config, environment, "owned-trial", [0], 1, artifact_dir=case["root"])
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    return slot


def test_remote_lost_acknowledgement_forbids_more_commands(case, monkeypatch):
    slot = remote_slot(case)

    def lost(timeout):
        raise TimeoutError("lost acknowledgement")

    monkeypatch.setattr(slot, "_send_frame", lambda frame, deadline: slot.process.stdin.write(frame))
    monkeypatch.setattr(slot, "_read", lost)
    with pytest.raises(TimeoutError):
        slot.execute("write once")
    with pytest.raises(RuntimeError, match="prior remote operation is unknown"):
        slot.execute("write another command")
    sent = slot.process.stdin.getvalue().splitlines()
    assert len(sent) == 1
    assert json.loads(sent[0])["command"] == "write once"


def test_late_business_receipt_is_not_confused_with_cleanup(case, monkeypatch):
    slot = remote_slot(case)
    probe = {
        "0": {"used_mb": 0, "pids": [], "model": "Ascend910"},
        "_environment": dict.fromkeys(
            ["image_digest", "cann", "torch_npu", "driver", "python", "kernel"], "test"
        ),
    }
    late = {"exit_code": 0, "stdout_b64": "", "stderr_b64": ""}
    cleanup = {
        "cleanup": {"probe": probe, "errors": [], "files_modified": ["remote-production.py"]},
        "remote_artifact": "/owned.final.json",
    }
    frames = iter([late, cleanup])
    monkeypatch.setattr(slot, "_read", lambda timeout: next(frames))
    slot.close()
    assert slot.cleanup_receipt == cleanup
    assert json.loads((case["root"] / "remote_late_results.jsonl").read_text()) == late
    assert slot.postflight["0"]["pids"] == []
    assert slot.modified_files == ["remote-production.py"]


def test_remote_inventory_excludes_unchanged_public_inputs_and_keeps_their_deletion(case, monkeypatch):
    slot = remote_slot(case)
    original = base64.b64encode(b"public input\n").decode()
    slot.public_files = {"unchanged.py": original, "deleted.py": original, "edited.py": original}
    slot.cleanup_receipt = {
        "cleanup": {
            "probe": {},
            "errors": [],
            "files_modified": ["production.py", "unchanged.py", "edited.py"],
            "untracked": {
                "unchanged.py": original,
                "edited.py": base64.b64encode(b"edited input\n").decode(),
            },
        }
    }
    monkeypatch.setattr(slot, "validate_probe", lambda value: value)
    slot.close()
    assert slot.modified_files == ["deleted.py", "edited.py", "production.py"]


def test_soundness_releases_worker_when_prepare_fails(case, monkeypatch):
    manifest = case["task"] / "task.toml"
    manifest.write_text(
        manifest.read_text()
        + f"\n[environment.l3]\nrepo_path = {json.dumps(str(case['repo']))}\n"
        + f'base_commit = "{case["commit"]}"\nnpu_model = "Ascend910"\nnpu_count = 1\n'
    )
    (case["task"] / "oracle/gold_patch.diff").write_text("not reached")
    case["config"].remote = RemoteConfig(
        host="fixture", container="fixture", repo_path="/fixture", devices=[0], probe_command=["probe"]
    )
    closed = []

    class OccupiedWorker:
        postflight = None

        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            raise RuntimeError("selected device is occupied")

        def close(self):
            closed.append(True)

    monkeypatch.setattr("mindie_bench.runner.soundness.RemoteSlot", OccupiedWorker)
    with pytest.raises(RuntimeError, match="soundness failed"):
        check_soundness(
            case["task"], case["config"], tier="l3", output=case["root"] / "failed-soundness"
        )
    assert closed == [True]
    assert not (case["task"] / "soundness_receipt.json").exists()


def install_soundness_worker(case, monkeypatch, results):
    manifest = case["task"] / "task.toml"
    manifest.write_text(
        manifest.read_text()
        + f"\n[environment.l3]\nrepo_path = {json.dumps(str(case['repo']))}\n"
        + f'base_commit = "{case["commit"]}"\nnpu_model = "Ascend910"\nnpu_count = 1\n'
    )
    (case["task"] / "oracle/gold_patch.diff").write_text("external gold patch")
    case["config"].remote = RemoteConfig(
        host="fixture", container="fixture", repo_path="/fixture", devices=[0], probe_command=["probe"]
    )
    closed = []
    frames = iter(results)

    class SoundnessWorker:
        preflight = {"0": {"pids": []}}
        postflight = {"0": {"pids": []}}

        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            return self

        def upload_public(self, path):
            pass

        def verify(self, script):
            return next(frames)

        def execute(self, command, stdin):
            assert command == "git apply --no-index -"
            assert base64.b64decode(stdin) == b"external gold patch"
            return {"exit_code": 0}

        def close(self):
            closed.append(True)

    monkeypatch.setattr("mindie_bench.runner.soundness.RemoteSlot", SoundnessWorker)
    return closed


def judge_frame(returncode, stdout, stderr):
    return {
        "exit_code": returncode,
        "stdout_b64": base64.b64encode(stdout).decode(),
        "stderr_b64": base64.b64encode(stderr).decode(),
    }


def test_l3_soundness_preserves_both_judges_and_numeric_verdicts(case, monkeypatch):
    before = judge_frame(1, b'case failed\n{"reward":0.0}\n', b"base regression\n")
    after = judge_frame(
        0,
        b'30 NPU cases\n{"reward":1.0,"max_abs_error":0.000125,"cosine_similarity":0.9999}\n',
        b"driver diagnostic\n",
    )
    closed = install_soundness_worker(case, monkeypatch, [before, after])
    output = case["root"] / "soundness-evidence"
    proof = check_soundness(case["task"], case["config"], tier="l3", output=output)
    assert proof["status"] == "completed"
    assert closed == [True]
    for stage, frame in [("before", before), ("after", after)]:
        assert json.loads((output / f"verifier_{stage}.remote.json").read_text()) == frame
        assert (output / f"verifier_{stage}.stdout").read_bytes() == base64.b64decode(
            frame["stdout_b64"]
        )
        assert (output / f"verifier_{stage}.stderr").read_bytes() == base64.b64decode(
            frame["stderr_b64"]
        )
        verdict = json.loads((output / f"verifier_{stage}.verdict.json").read_text())
        assert verdict["returncode"] == frame["exit_code"]
        assert verdict["result"]["reward"] == (0.0 if stage == "before" else 1.0)
    verdict = json.loads((output / "verifier_after.verdict.json").read_text())
    assert verdict["result"]["max_abs_error"] == 0.000125
    assert verdict["result"]["cosine_similarity"] == 0.9999


def test_l3_soundness_preserves_invalid_judge_and_original_failure(case, monkeypatch):
    frame = judge_frame(2, b"broken environment\n", b"\xffraw driver failure\n")
    closed = install_soundness_worker(case, monkeypatch, [frame])
    output = case["root"] / "invalid-soundness"
    with pytest.raises(RuntimeError, match="soundness failed"):
        check_soundness(case["task"], case["config"], tier="l3", output=output)
    assert closed == [True]
    assert json.loads((output / "verifier_before.remote.json").read_text()) == frame
    assert (output / "verifier_before.stdout").read_bytes() == b"broken environment\n"
    assert (output / "verifier_before.stderr").read_bytes() == b"\xffraw driver failure\n"
    attempt = json.loads((output / "soundness_attempt.json").read_text())
    assert attempt["status"] == "failed"
    assert attempt["before_reward"] is None
    assert attempt["errors"][0]["type"] == "ValueError"
    assert "nonzero verifier without structured reward" in attempt["errors"][0]["message"]
    assert not (output / "verifier_before.verdict.json").exists()
    assert not (case["task"] / "soundness_receipt.json").exists()
