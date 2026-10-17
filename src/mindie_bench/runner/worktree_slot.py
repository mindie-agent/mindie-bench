"""Owned local worktrees; source repositories and their dirty files are preserved."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from mindie_bench.io import is_ignored_cache_path, tree_hash
from mindie_bench.models import SourceEnvironment
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS, diff_paths, project_source


class WorktreeSlot:
    def __init__(self, environment: SourceEnvironment, destination: Path, task_root: Path):
        self.repo = Path(environment.repo_path).resolve(strict=True)
        self.environment = environment
        # Git interprets relative worktree destinations from the source repo,
        # while the caller interprets them from its cwd. Freeze one identity.
        self.destination = destination.resolve()
        self.task_root = task_root.resolve(strict=True)
        self.created = False
        self.public_files = {}

    def prepare(self) -> Path:
        actual = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", self.environment.base_commit + "^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != self.environment.base_commit:
            raise ValueError("base revision did not resolve to exact declared commit")
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "add", "--detach", str(self.destination), actual],
            check=True,
            capture_output=True,
        )
        self.created = True
        # Git records the symlink-resolved worktree path in its admin metadata;
        # resolve here too so destinations reached through symlinked prefixes
        # (for example /tmp on macOS) still match during reattach_metadata.
        self.destination = self.destination.resolve(strict=True)
        # The agent gets a plain snapshot. Hiding .git closes both the oracle
        # history leak and the worktree metadata pointer back to the source repo.
        (self.destination / ".git").unlink()
        project_source(self.destination)
        public = self.task_root / "environment"
        if public.exists():
            tree_hash(public)
            for item in public.iterdir():
                if is_ignored_cache_path(Path(item.name)):
                    continue
                target = self.destination / item.name
                if target.exists() or target.is_symlink():
                    raise ValueError(f"public task asset overwrites tracked source: {item.name}")
                if item.is_dir():
                    shutil.copytree(
                        item,
                        target,
                        ignore=lambda d, names: {
                            n for n in names if is_ignored_cache_path(Path(d, n).relative_to(public))
                        },
                    )
                else:
                    shutil.copy2(item, target)
            self.public_files = {
                path.relative_to(public).as_posix(): hashlib.sha256(path.read_bytes()).digest()
                for path in public.rglob("*")
                if path.is_file() and not is_ignored_cache_path(path.relative_to(public))
            }
        for reserved in ["oracle", "verifier", ".git", *HARNESS_CONFIG_PATHS]:
            if (self.destination / reserved).exists():
                raise ValueError(f"source contains reserved evaluation/configuration path: {reserved}")
        return self.destination

    def reattach_metadata(self):
        admin = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        admin_path = Path(admin)
        if not admin_path.is_absolute():
            admin_path = self.repo / admin_path
        matches = []
        for entry in (admin_path / "worktrees").iterdir():
            try:
                target = (entry / "gitdir").read_text().strip()
            except FileNotFoundError:
                continue
            if target == str(self.destination / ".git"):
                matches.append(entry)
        if len(matches) != 1:
            raise RuntimeError("cannot identify owned worktree metadata for cleanup")
        gitfile = self.destination / ".git"
        if gitfile.is_symlink() or gitfile.is_file():
            gitfile.unlink()
        if gitfile.exists():
            raise RuntimeError("agent-created .git directory retained for inspection")
        gitfile.write_text(f"gitdir: {matches[0]}\n")

    def snapshot_patch(self, output: Path) -> list[str]:
        self.reattach_metadata()
        try:
            command = ["git", "-c", "core.fsmonitor=false", "-C", str(self.destination)]
            patch = subprocess.run(
                [*command, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", *diff_paths()],
                check=True,
                capture_output=True,
            ).stdout
            (output / "final_patch.diff").write_bytes(patch)
            changed = subprocess.run(
                [
                    *command,
                    "diff",
                    "--name-only",
                    "--no-ext-diff",
                    "--no-textconv",
                    "HEAD",
                    *diff_paths(),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            untracked = subprocess.run(
                [*command, "ls-files", "--others", "--exclude-standard", "-z", *diff_paths()],
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
            modified = set(
                changed
                + [
                    os.fsdecode(name)
                    for name in untracked
                    if name and not is_ignored_cache_path(Path(os.fsdecode(name)))
                ]
            )
            for name, original in self.public_files.items():
                path = self.destination / name
                if (
                    not path.is_symlink()
                    and path.is_file()
                    and hashlib.sha256(path.read_bytes()).digest() == original
                ):
                    modified.discard(name)
                else:
                    modified.add(name)
            return sorted(modified)
        finally:
            (self.destination / ".git").unlink()

    def cleanup(self):
        if self.created:
            self.reattach_metadata()
            subprocess.run(
                ["git", "-C", str(self.repo), "worktree", "remove", "--force", str(self.destination)],
                check=True,
                capture_output=True,
            )
