"""[Role] Assemble a deterministic zero-Git wheelhouse stage with optional candidate substitution.
[Invariants] Delegates archive/manifest construction to `build_wheels_bundle.build_bundle`, enforces
`HAR-ABI` (`-py3-none-any.whl` and zero unconditional runtime `Requires-Dist`) on `mindie-harness`,
enforces the exact `ReMe` pin on `mindie-knowledge` (including candidate wheels), and emits
`mindie-wheels-bundle.tar.gz`, `release-channel.json`, `requirements.txt`, `bundle-manifest.json`,
and `SHA256SUMS` consumable directly by both `install.py` and `wheelhouse.update_runtime`.
[Timeouts & Locks] GitHub API reads 60s; release downloads and ReMe builds 120-180s; no locks.
[Side-Effect Policy] Writes only `--output-stage` and temporary directories cleaned up on exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import build_wheels_bundle as bwb  # noqa: E402

MAX_BUNDLE_BYTES = bwb.MAX_BUNDLE_BYTES
REME_REPO_URL = bwb.REME_REPO_URL
REME_REVISION = bwb.REME_REVISION
SHARED_PACKAGES = bwb.SHARED_PACKAGES
_require = bwb._require
PACKAGE_REPOSITORIES = {
    "mindie-harness": "mindie-agent/harness", "mindie-knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev", "mindie-diagnostics": "mindie-agent/diagnostics",
    "mindie-debugger": "mindie-agent/debugger", "mindie-profiler": "mindie-agent/profiler",
    "mindie-sim": "mindie-agent/sim", "npu-top": "mindie-agent/npu-top",
    "mindie-coordinator": "mindie-agent/coordinator",
}
REPOSITORY_PACKAGES = {repo: pkg for pkg, repo in PACKAGE_REPOSITORIES.items()}
PACKAGE_CHANNEL_KEYS = {pkg: repo.split("/", 1)[1] for pkg, repo in PACKAGE_REPOSITORIES.items()}


def fetch_latest_formal_release(repository: str) -> dict:
    payload = json.loads(bwb._run(["gh", "api", f"repos/{repository}/releases/latest"], 60))
    _require(
        isinstance(payload, dict) and not payload.get("draft") and not payload.get("prerelease"),
        f"{repository}: latest release is not a formal non-draft, non-prerelease release",
    )
    return payload


def _stage_single_wheel(source: Path, dest: Path, label: str) -> Path:
    source = Path(source)
    dest.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        _require(source.name.endswith("-py3-none-any.whl"), f"{label}: expected -py3-none-any.whl")
        target = dest / source.name
        shutil.copyfile(source, target)
        return target
    _require(source.is_dir(), f"{label}: wheel path does not exist: {source}")
    wheels = sorted(source.glob("*.whl"))
    _require(
        len(wheels) == 1 and wheels[0].name.endswith("-py3-none-any.whl"),
        f"{label}: expected exactly one pure-Python wheel in {source}",
    )
    target = dest / wheels[0].name
    shutil.copyfile(wheels[0], target)
    return target


def _verify_special_wheel_contracts(
    package: str, metadata, *, reme_repo: str, reme_revision: str
) -> None:
    requires = list(metadata.get_all("Requires-Dist") or [])
    if package == "mindie-harness":
        unconditional = [req for req in requires if not re.search(r";.*?\bextra\s*==", req)]
        _require(
            not unconditional,
            f"HAR-ABI violation: mindie-harness declares runtime Requires-Dist: {unconditional}",
        )
    if package == "mindie-knowledge":
        expected_repo = reme_repo.removesuffix(".git").rstrip("/")
        reme_reqs = [
            req for req in requires
            if re.match(r"reme[-_.]ai\b", req, re.IGNORECASE) and not re.search(r";.*?\bextra\s*==", req)
        ]
        _require(bool(reme_reqs), "ReMe pin drift: mindie-knowledge is missing reme-ai dependency")
        for req in reme_reqs:
            match = bwb._REME.search(req)
            _require(
                match is not None
                and match.group(1).rstrip("/") == expected_repo
                and match.group(2) == reme_revision,
                f"ReMe pin drift: {req}",
            )


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp.{os.getpid()}"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except TimeoutError:
            raise
        except OSError:
            pass


def _resolve_baseline_components(
    channel_path: Path | None, *, candidate_package: str | None, latest_release_fn, tag_fn
) -> tuple[str, dict | None, dict[str, dict[str, str]]]:
    if channel_path is not None:
        raw_doc = json.loads(Path(channel_path).read_text(encoding="utf-8"))
        product_version, components = bwb.load_shared_components(channel_path)
        return product_version, raw_doc, components
    components: dict[str, dict[str, str]] = {}
    for package in SHARED_PACKAGES:
        repository = PACKAGE_REPOSITORIES[package]
        if package == candidate_package:
            components[package] = {"repository": repository, "version": "", "tag": "", "revision": ""}
            continue
        info = latest_release_fn(repository)
        _require(isinstance(info, dict) and not info.get("draft") and not info.get("prerelease"),
                 f"{repository}: release is draft or prerelease")
        raw_tag = str(info.get("tag_name") or "")
        version = info.get("version") or (raw_tag[1:] if raw_tag.startswith("v") else "")
        _require(isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version),
                 f"{repository}: invalid formal release version {version!r}")
        revision = info.get("revision") or tag_fn(repository, version)
        _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
                 f"{repository}: invalid formal release revision {revision!r}")
        components[package] = dict(repository=repository, version=version, tag=f"v{version}",
                                   revision=revision)
    return "0.1.0", None, components


def assemble_wheelhouse(
    output_stage: Path, *, channel_path: Path | None = None, candidate_repo: str | None = None,
    candidate_wheel: Path | None = None, candidate_revision: str | None = None,
    reme_wheel: Path | None = None, reme_repo: str = REME_REPO_URL,
    reme_revision: str = REME_REVISION, download_fn=None, tag_fn=None,
    latest_release_fn=None, reme_builder=None,
) -> Path:
    _require(bool(candidate_repo) == bool(candidate_wheel),
             "must specify both --candidate-repo and --candidate-wheel, or neither")
    if candidate_revision is not None:
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", candidate_revision)),
                 "--candidate-revision must be a 40-hex commit SHA")
    download_fn = download_fn or bwb.download_component_release
    raw_tag_fn = tag_fn or bwb.verify_tag_revision
    tag_cache: dict[tuple[str, str], str] = {}

    def cached_tag_fn(repo: str, ver: str) -> str:
        if (repo, ver) not in tag_cache:
            tag_cache[(repo, ver)] = raw_tag_fn(repo, ver)
        return tag_cache[(repo, ver)]

    latest_release_fn = latest_release_fn or fetch_latest_formal_release
    reme_builder = reme_builder or bwb.build_reme_wheel
    preliminary_pkg = REPOSITORY_PACKAGES.get(candidate_repo or "")
    if channel_path is None and candidate_repo is not None:
        _require(preliminary_pkg is not None, f"unsupported candidate repository: {candidate_repo}")
    product_version, base_doc, components = _resolve_baseline_components(
        channel_path, candidate_package=preliminary_pkg,
        latest_release_fn=latest_release_fn, tag_fn=cached_tag_fn,
    )
    candidate_package = preliminary_pkg or (
        next((pkg for pkg, row in components.items() if row["repository"] == candidate_repo), None)
        if candidate_repo else None
    )
    if candidate_repo is not None:
        _require(candidate_package is not None, f"unsupported candidate repository: {candidate_repo}")

    cand_info: tuple[str, bytes, str, str] | None = None
    if candidate_package is not None:
        assert candidate_wheel is not None and candidate_repo is not None
        with tempfile.TemporaryDirectory(prefix="mindie-cand-check-") as chk_tmp:
            chk_dir = Path(chk_tmp)
            _stage_single_wheel(Path(candidate_wheel), chk_dir, candidate_package)
            whl, meta = bwb._wheel(chk_dir, candidate_package)
            ver = str(meta.get("Version") or "").strip()
            _require(bool(re.fullmatch(r"\d+\.\d+\.\d+", ver)), f"{whl.name}: invalid SemVer {ver!r}")
            parts = whl.name.split("-")
            _require(
                len(parts) >= 5
                and bwb._normalized(parts[0]) == bwb._normalized(candidate_package)
                and parts[1] == ver,
                f"{whl.name}: wheel filename does not match {candidate_package}=={ver}",
            )
            _verify_special_wheel_contracts(
                candidate_package, meta, reme_repo=reme_repo, reme_revision=reme_revision
            )
            data = whl.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            env_sha = os.environ.get("GITHUB_SHA", "")
            rev = candidate_revision or (env_sha if re.fullmatch(r"[0-9a-f]{40}", env_sha) else sha[:40])
            cand_info = (whl.name, data, sha, rev)
            components[candidate_package] = {
                "repository": candidate_repo, "version": ver, "tag": f"v{ver}", "revision": rev,
            }

    channel_doc = json.loads(json.dumps(base_doc)) if isinstance(base_doc, dict) else {
        "schema": "mindie-release-channel/1", "schema_version": "mindie-release-channel/1",
        "track": ".".join(product_version.split(".")[:2]),
        "current_patch_version": product_version, "independent_update_allowed": True, "components": {},
    }
    channel_doc["schema"] = "mindie-release-channel/1"
    channel_doc.setdefault("schema_version", "mindie-release-channel/1")
    channel_doc["current_patch_version"] = product_version
    comp_map = dict(channel_doc.get("components") or {})
    existing_keys = {str(v.get("package") or k): k for k, v in comp_map.items() if isinstance(v, dict)}
    for pkg, row in components.items():
        key = existing_keys.get(pkg, PACKAGE_CHANNEL_KEYS[pkg])
        comp_map[key] = {
            **dict(comp_map.get(key) or {}), "name": key, "role": "shared_component",
            "repository": row["repository"], "package": pkg, "version": row["version"],
            "tag": f"v{row['version']}", "commit": row["revision"], "revision": row["revision"],
            "independent_update_allowed": True,
        }
    channel_doc["components"] = comp_map

    output_stage = Path(output_stage)
    output_stage.mkdir(parents=True, exist_ok=True)
    channel_out = output_stage / "release-channel.json"
    channel_bytes = (json.dumps(channel_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(channel_out, channel_bytes)

    def stage_download(repo: str, ver: str, dest: Path) -> None:
        if cand_info is not None and repo == candidate_repo:
            whl_name, whl_bytes, whl_sha, _ = cand_info
            (dest / whl_name).write_bytes(whl_bytes)
            (dest / "SHA256SUMS").write_text(f"{whl_sha}  {whl_name}\n", encoding="utf-8", newline="\n")
            return
        download_fn(repo, ver, dest)
        pkg = REPOSITORY_PACKAGES.get(repo) or next(
            p for p, r in components.items() if r["repository"] == repo
        )
        bwb._verified_component(pkg, components[pkg], dest, cached_tag_fn, reme_repo, reme_revision)
        _, meta = bwb._wheel(dest, pkg, ver)
        _verify_special_wheel_contracts(pkg, meta, reme_repo=reme_repo, reme_revision=reme_revision)

    def stage_tag(repo: str, ver: str) -> str:
        if cand_info is not None and repo == candidate_repo:
            return cand_info[3]
        return cached_tag_fn(repo, ver)

    def stage_reme(repo: str, rev: str, dest: Path) -> Path:
        if reme_wheel is not None:
            return _stage_single_wheel(Path(reme_wheel), dest, "reme-ai")
        return reme_builder(repo, rev, dest)

    bundle_out = output_stage / "mindie-wheels-bundle.tar.gz"
    bwb.build_bundle(
        channel_out, bundle_out, reme_repo=reme_repo, reme_revision=reme_revision,
        download_fn=stage_download, tag_fn=stage_tag, reme_builder=stage_reme,
    )
    _require(0 < bundle_out.stat().st_size < MAX_BUNDLE_BYTES, "wheel bundle size is outside bound")
    with tarfile.open(bundle_out, "r:gz") as tar:
        for member_name in ("requirements.txt", "bundle-manifest.json"):
            extracted = tar.extractfile(member_name)
            _require(extracted is not None, f"bundle is missing {member_name}")
            _atomic_write_bytes(output_stage / member_name, extracted.read())
    stage_files = (
        "bundle-manifest.json", "mindie-wheels-bundle.tar.gz",
        "release-channel.json", "requirements.txt",
    )
    sums = "".join(
        f"{hashlib.sha256((output_stage / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(stage_files)
    ).encode("utf-8")
    _atomic_write_bytes(output_stage / "SHA256SUMS", sums)
    bwb._sha256sums(output_stage / "SHA256SUMS")
    return output_stage


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stage", type=Path, required=True)
    parser.add_argument("--channel", dest="channel_path", type=Path, default=None)
    for flag, kw in (
        ("--candidate-repo", {"default": None}), ("--candidate-wheel", {"type": Path, "default": None}),
        ("--candidate-revision", {"default": None}), ("--reme-wheel", {"type": Path, "default": None}),
        ("--reme-repo", {"default": REME_REPO_URL}), ("--reme-revision", {"default": REME_REVISION}),
    ):
        parser.add_argument(flag, **kw)
    assemble_wheelhouse(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
