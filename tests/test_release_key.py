"""Exercise transferred SSH keys and publication effects with native local tools."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))
import release_key

FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
REAL_RUN = subprocess.run
REVISION = "a" * 40
GIT = Path(shutil.which("git")).resolve()
BUNDLED_KEYGEN = GIT.parents[1] / "usr/bin/ssh-keygen.exe"
KEYGEN = str(BUNDLED_KEYGEN) if BUNDLED_KEYGEN.is_file() else "ssh-keygen"


@pytest.fixture
def transferred_key(tmp_path, monkeypatch):
    for variable in ("USERPROFILE", "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME"):
        root = tmp_path / variable
        root.mkdir()
        monkeypatch.setenv(variable, str(root))
    private = tmp_path / "original"
    REAL_RUN(
        [KEYGEN, "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
        check=True, capture_output=True, timeout=15, **FLAGS,
    )
    secret = private.read_text(encoding="ascii").rstrip()
    monkeypatch.setenv(release_key.SECRET, secret)
    return private.with_suffix(".pub").read_text(encoding="ascii").strip()


class Repository:
    repository = "mindie-agent/release-key-fixture"
    prefix = "repos/" + repository

    def __init__(self, public, refs=None):
        self.public = public
        self.refs = refs or (lambda: [])

    def remaining(self, maximum):
        return maximum

    def api(self, path):
        if path == "meta":
            return {"ssh_keys": [self.public]}
        assert path.startswith(self.prefix + "/git/matching-refs/tags/")
        return self.refs()


def test_gh_stdin_trimmed_key_is_accepted_by_the_real_ssh_parser(
    transferred_key, monkeypatch,
):
    public = transferred_key

    def parse_key(argv, **kwargs):
        assert argv[:2] == ["git", "push"]
        command = shlex.split(kwargs["env"]["GIT_SSH_COMMAND"])
        private = command[command.index("-i") + 1]
        return REAL_RUN(
            [KEYGEN, "-y", "-f", private], check=False, capture_output=True,
            timeout=10, **FLAGS,
        )

    monkeypatch.setattr(release_key.subprocess, "run", parse_key)
    release_key.create_tag(Repository(public), "1.2.3", REVISION)


def test_ssh_rejection_keeps_the_actual_transport_error(transferred_key, monkeypatch):
    public = transferred_key

    def reject(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 128, b"", b"git@github.com: Permission denied (publickey).\n",
        )

    monkeypatch.setattr(release_key.subprocess, "run", reject)
    with pytest.raises(RuntimeError, match=r"Permission denied \(publickey\)"):
        release_key.create_tag(Repository(public), "1.2.3", REVISION)


def test_credential_probe_does_not_create_a_ref_before_the_real_push(
    transferred_key, tmp_path, monkeypatch,
):
    public = transferred_key
    source, remote = tmp_path / "source", tmp_path / "remote.git"
    source.mkdir()
    for argv in (
        ["git", "init", str(source)],
        ["git", "init", "--bare", str(remote)],
        ["git", "-C", str(source), "config", "user.name", "Release fixture"],
        ["git", "-C", str(source), "config", "user.email", "release@example.invalid"],
        ["git", "-C", str(source), "commit", "--allow-empty", "-m", "source"],
    ):
        REAL_RUN(argv, check=True, capture_output=True, timeout=10, **FLAGS)
    revision = REAL_RUN(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, timeout=10, **FLAGS,
    ).stdout.decode().strip()
    monkeypatch.chdir(source)
    pushes = []

    def local_transport(argv, **kwargs):
        pushes.append(list(argv))
        local = [str(remote) if value.startswith("git@github.com:") else value for value in argv]
        return REAL_RUN(local, **kwargs)

    def refs():
        result = REAL_RUN(
            ["git", "--git-dir", str(remote), "show-ref", "--tags"],
            check=False, capture_output=True, timeout=10, **FLAGS,
        )
        return [
            {"ref": line.split()[1], "object": {"type": "commit", "sha": line.split()[0]}}
            for line in result.stdout.decode().splitlines()
        ]

    monkeypatch.setattr(release_key.subprocess, "run", local_transport)
    api = Repository(public, refs)
    release_key.check_key(api, revision)
    assert refs() == []
    release_key.create_tag(api, "1.2.3", revision)
    assert refs() == [
        {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": revision}},
    ]
    assert "--dry-run" in pushes[0]
    assert len(pushes) == 2
