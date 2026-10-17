"""File identity and explicit negative controls for an externally isolated judge.

The caller must freeze before foreground admission and keep the manifest outside
the agent's grants. Hashes cannot detect arbitrary in-memory Python attacks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mindie_bench.io import digest, is_ignored_cache_path
from mindie_bench.models import OracleContract, VerifierResult


class OracleTampered(RuntimeError):
    """A protected file or an explicit judge negative control changed."""


@dataclass(frozen=True)
class FrozenOracle:
    root: Path
    protected_paths: tuple[str, ...]
    files: tuple[tuple[str, str], ...]
    contract_sha256: str

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema_version": "mindie-bench-oracle-freezer/1",
            "protected_paths": list(self.protected_paths),
            "files": dict(self.files),
            "contract_sha256": self.contract_sha256,
            "boundary": "externally read-only files plus separately executed explicit probes",
        }


def _inventory(root: Path, paths: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    files = {}
    for relative in paths:
        path = root / relative
        if not path.exists():
            raise OracleTampered(f"protected oracle path missing: {relative}")
        for ancestor in [path, *path.parents]:
            if ancestor == root.parent:
                break
            if ancestor.is_symlink():
                raise OracleTampered(f"protected oracle path is a symlink: {relative}")
        entries = [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]
        for entry in entries:
            rel_entry = entry.relative_to(root)
            if entry != path and is_ignored_cache_path(rel_entry):
                continue
            if entry.is_symlink():
                raise OracleTampered(f"protected oracle symlink: {rel_entry}")
            if entry.is_file():
                files[rel_entry.as_posix()] = hashlib.sha256(
                    entry.read_bytes()
                ).hexdigest()
    return tuple(sorted(files.items()))


def freeze_oracle(root: Path, contract: OracleContract | None = None) -> FrozenOracle:
    contract = contract or OracleContract()
    root = root.resolve(strict=True)
    paths = tuple(
        sorted(
            {
                "verifier",
                *(["oracle"] if (root / "oracle").exists() else []),
                *(["task.toml"] if (root / "task.toml").exists() else []),
                *contract.protected_paths,
            }
        )
    )
    return FrozenOracle(root, paths, _inventory(root, paths), digest(contract.model_dump(mode="json")))


def check_oracle(manifest: FrozenOracle, contract: OracleContract | None = None) -> dict:
    if contract is not None and digest(contract.model_dump(mode="json")) != manifest.contract_sha256:
        raise OracleTampered("oracle contract changed after admission")
    observed = _inventory(manifest.root, manifest.protected_paths)
    if observed != manifest.files:
        before, after = dict(manifest.files), dict(observed)
        changed = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        raise OracleTampered(f"protected oracle SHA-256 changed: {', '.join(changed)}")
    return {"status": "intact", "manifest_sha256": manifest.sha256, "file_count": len(observed)}


def check_probe(name: str, returncode: int, stdout: bytes, contract) -> dict:
    actual = hashlib.sha256(stdout).hexdigest()
    if returncode != contract.expected_returncode or actual != contract.stdout_sha256:
        raise OracleTampered(
            f"oracle negative control {name!r} failed: exit={returncode}, stdout_sha256={actual}"
        )
    return {"name": name, "returncode": returncode, "stdout_sha256": actual, "status": "intact"}


def tampered_verdict(error: BaseException) -> VerifierResult:
    return VerifierResult(
        reward=0.0,
        oracle_status="oracle_tampered",
        failure_reason=f"oracle_tampered: {error}",
        warnings=[
            "Oracle integrity failed; this result cannot earn reward. "
            "File checks and explicit probes do not detect arbitrary in-memory Python attacks."
        ],
    )
