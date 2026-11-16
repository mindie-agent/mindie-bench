"""Owned local worktrees; source repositories and their dirty files are preserved.

[Role] Prepare and remove isolated source snapshots for local trials.
[Invariants] Worktree metadata edits within one runner serialize per common Git
directory; task execution and independent repositories remain concurrent.
[Timeouts & Locks] Git metadata commands have 30 seconds; lock acquisition has
35 seconds. An optional phase deadline caps every Git and metadata-lock wait.
`cleanup()` reserves `CLEANUP_GRACE_S` split into `WORKTREE_REMOVE_BUDGET_S` for
reattach/`git worktree remove` and `DISCARD_GRACE_S` for the pure-Python fallback,
even when the trial budget already expired. Process-local locks do not coordinate
independent runner processes.
[Side-Effect Policy] Remove only the owned worktree, the exact admin directory
recorded from its `.git` pointer, and its own patch output; never sweep or mutate a
sibling slot's `<repo>/.git/worktrees/<other>` entry; expose Git failures to callers.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from mindie_bench.io import is_ignored_cache_path
from mindie_bench.models import SourceEnvironment
from mindie_bench.runner.host_platform import (
    is_wsl_interop_path,
    rmtree_force,
    subprocess_window_kwargs,
    which_native,
)
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS, diff_paths, project_source

CLEANUP_GRACE_S = 15.0
WORKTREE_REMOVE_BUDGET_S = 10.0
DISCARD_GRACE_S = CLEANUP_GRACE_S - WORKTREE_REMOVE_BUDGET_S

_METADATA_LOCKS = {}
_LOCK_REGISTRY = threading.Lock()


class WorktreeSlot:
    CLEANUP_GRACE_S = CLEANUP_GRACE_S
    WORKTREE_REMOVE_BUDGET_S = WORKTREE_REMOVE_BUDGET_S
    DISCARD_GRACE_S = DISCARD_GRACE_S

    def __init__(
        self, environment: SourceEnvironment, destination: Path, task_root: Path,
        *, deadline: float | None = None,
    ):
        self.deadline = deadline
        self.repo = Path(environment.repo_path).resolve(strict=True)
        self.environment = environment
        # Git interprets relative worktree destinations from the source repo,
        # while the caller interprets them from its cwd. Freeze one identity.
        self.destination = destination.resolve()
        self.task_root = task_root.resolve(strict=True)
        self.created = False
        # Exact `<repo>/.git/worktrees/<name>` recorded from this slot's `.git`
        # pointer before the pointer is hidden from the agent. Fallback cleanup
        # removes only this entry, never a sibling slot's in-flight registration.
        self._admin_dir: Path | None = None
        self.public_files = {}
        self.git_exe = which_native("git") or "git"
        admin = subprocess.run(
            [self.git_exe, "-C", str(self.repo), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.remaining_timeout(30),
            **subprocess_window_kwargs(),
        ).stdout.strip()
        self.admin_path = (self.repo / admin).resolve()
        with _LOCK_REGISTRY:
            self.metadata_lock = _METADATA_LOCKS.setdefault(self.admin_path, threading.RLock())

    @contextmanager
    def metadata_edit(self):
        if not self.metadata_lock.acquire(timeout=self.remaining_timeout(35)):
            raise TimeoutError("timed out waiting for this repository's worktree metadata")
        try:
            yield
        finally:
            self.metadata_lock.release()

    def remaining_timeout(self, limit: float) -> float:
        """Share one phase deadline across consecutive subprocesses and locks."""
        remaining = limit if self.deadline is None else min(limit, self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("worktree phase deadline exhausted")
        return remaining

    def worktree_command(self, *arguments):
        extra = (
            ["-c", "core.autocrlf=false", "-c", "core.fileMode=false", "-c", "core.longpaths=true"]
            if os.name == "nt" or is_wsl_interop_path(self.repo) or is_wsl_interop_path(self.destination)
            else []
        )
        try:
            subprocess.run(
                [
                    self.git_exe,
                    "-C",
                    str(self.repo),
                    *extra,
                    "worktree",
                    *arguments,
                ],
                check=True,
                capture_output=True,
                timeout=self.remaining_timeout(30),
                **subprocess_window_kwargs(),
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git worktree {arguments[0]} failed: {exc.stderr.decode('utf-8', errors='replace').strip()}"
            ) from exc

    def _capture_admin_dir(self) -> Path:
        """Record this slot's exact Git admin directory while `.git` still exists."""
        gitfile = self.destination / ".git"
        if not gitfile.is_file() or gitfile.is_symlink():
            raise RuntimeError("owned worktree lacks Git's admin pointer")
        pointer = gitfile.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            raise RuntimeError("owned worktree has a malformed Git admin pointer")
        admin = Path(pointer.split(":", 1)[1].strip())
        if not admin.is_absolute():
            admin = self.destination / admin
        self._admin_dir = admin.resolve()
        return self._admin_dir

    def prepare(self) -> Path:
        actual = subprocess.run(
            [self.git_exe, "-C", str(self.repo), "rev-parse", self.environment.base_commit + "^{commit}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.remaining_timeout(60),
            **subprocess_window_kwargs(),
        ).stdout.strip()
        if actual != self.environment.base_commit:
            raise ValueError("base revision did not resolve to exact declared commit")
        # Git scans sibling admin entries while adding/removing worktrees. A
        # concurrent edit can expose an entry before its commondir is present.
        owned_destination = not self.destination.exists()
        try:
            with self.metadata_edit():
                self.worktree_command("add", "--detach", str(self.destination), actual)
        except (TimeoutError, subprocess.TimeoutExpired):
            # Git can register and populate the owned tree before its waiter
            # times out. Retain cleanup ownership even without a return code.
            self.created = owned_destination and self.destination.exists()
            if self.created:
                try:
                    self._capture_admin_dir()
                except Exception:
                    self._admin_dir = None
            raise
        self.created = True
        # Git records the symlink-resolved worktree path in its admin metadata;
        # resolve here too so destinations reached through symlinked prefixes
        # (for example /tmp on macOS) still match during reattach_metadata.
        self.destination = self.destination.resolve(strict=True)
        # Record the exact admin entry while Git's pointer is still present.
        self._capture_admin_dir()
        # The agent gets a plain snapshot. Hiding .git closes both the oracle
        # history leak and the worktree metadata pointer back to the source repo.
        (self.destination / ".git").unlink()
        project_source(self.destination)
        public = self.task_root / "environment"
        if public.exists():
            for item in public.iterdir():
                self.remaining_timeout(30)
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

    def _matching_admin_entries(self, *, bounded: bool = True):
        expected_git = os.path.normcase(str((self.destination / ".git").resolve()))
        worktrees = self.admin_path / "worktrees"
        if not worktrees.is_dir():
            return []
        matches = []
        for entry in worktrees.iterdir():
            if bounded:
                self.remaining_timeout(30)
            try:
                target = (entry / "gitdir").read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                # Other Git processes do not participate in this runner's lock.
                continue
            if os.path.normcase(str(Path(target).resolve())) == expected_git:
                matches.append(entry)
        return matches

    def reattach_metadata(self):
        with self.metadata_edit():
            matches = self._matching_admin_entries()
            if len(matches) != 1:
                raise RuntimeError("cannot identify owned worktree metadata for cleanup")
            self._admin_dir = matches[0].resolve()
            gitfile = self.destination / ".git"
            if gitfile.is_symlink() or gitfile.is_file():
                gitfile.unlink()
            if gitfile.exists():
                raise RuntimeError("agent-created .git directory retained for inspection")
            gitfile.write_text(f"gitdir: {matches[0].as_posix()}\n", encoding="utf-8")

    def snapshot_patch(self, output: Path) -> list[str]:
        self.reattach_metadata()
        try:
            filemode_cfg = (
                ["-c", "core.fileMode=false"]
                if os.name == "nt" or is_wsl_interop_path(self.repo) or is_wsl_interop_path(self.destination)
                else []
            )
            command = [
                self.git_exe,
                "-c",
                "core.fsmonitor=false",
                *filemode_cfg,
                "-c",
                "core.quotepath=false",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.longpaths=true",
                "-C",
                str(self.destination),
            ]
            patch = subprocess.run(
                [*command, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", *diff_paths()],
                check=True,
                capture_output=True,
                timeout=self.remaining_timeout(60),
                **subprocess_window_kwargs(),
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
                encoding="utf-8",
                timeout=self.remaining_timeout(60),
                **subprocess_window_kwargs(),
            ).stdout.splitlines()
            untracked = subprocess.run(
                [*command, "ls-files", "--others", "--exclude-standard", "-z", *diff_paths()],
                check=True,
                capture_output=True,
                timeout=self.remaining_timeout(60),
                **subprocess_window_kwargs(),
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

    def _remove_agent_git(self):
        gitfile = self.destination / ".git"
        if gitfile.is_symlink():
            gitfile.unlink()
        elif gitfile.is_dir():
            # The agent may have replaced Git's pointer file with a real repository.
            rmtree_force(gitfile)

    def _discard_worktree(self):
        """Remove the owned tree and only this slot's exact Git admin entry.

        This path is intentionally pure local filesystem work: `git worktree
        remove` may have already exhausted its sub-budget, so no Git subprocess
        may run here. A repo-wide sweep of stale registrations is never safe
        because sibling active slots hide their `.git` files during trials.
        """
        rmtree_force(self.destination)
        admin = self._admin_dir
        if admin is None:
            matches = self._matching_admin_entries(bounded=False)
            if len(matches) != 1:
                # Without one exact match, leaving metadata is safer than
                # deleting a concurrent sibling's registration.
                return
            admin = matches[0].resolve()
        worktrees = (self.admin_path / "worktrees").resolve()
        if admin.parent.name != "worktrees" or admin.parent != worktrees:
            raise RuntimeError("refusing to remove worktree metadata outside this slot's admin entry")
        rmtree_force(admin)

    def cleanup(self):
        if not self.created:
            return
        original_deadline = self.deadline
        # A trial that exhausted its phase deadline still owns this worktree and
        # its `<repo>/.git/worktrees/<name>` entry. Reserve one bounded grace:
        # 10s for reattach/`git worktree remove`, then 5s for pure-Python discard.
        cleanup_deadline = time.monotonic() + self.CLEANUP_GRACE_S
        self.deadline = (
            max(original_deadline, cleanup_deadline)
            if original_deadline is not None
            else cleanup_deadline
        )
        try:
            with self.metadata_edit():
                self._remove_agent_git()
                self.deadline = min(
                    self.deadline, time.monotonic() + self.WORKTREE_REMOVE_BUDGET_S
                )
                try:
                    self.reattach_metadata()
                    self.worktree_command("remove", "--force", str(self.destination))
                except Exception as exc:
                    # The remove sub-budget is already spent; discard gets the
                    # remaining cleanup grace and needs no Git subprocess.
                    self.deadline = cleanup_deadline
                    self._discard_worktree()
                    if self.destination.exists():
                        raise RuntimeError(f"owned worktree cleanup failed: {exc}") from exc
        finally:
            self.deadline = original_deadline
