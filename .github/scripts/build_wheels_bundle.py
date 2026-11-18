"""[Role] Build a deterministic, zero-Git prebuilt wheel bundle for product releases.
[Invariants] Each shared component wheel comes from its formal release tag and matches the
channel revision, SHA256SUMS, normalized METADATA name/version, and the knowledge wheel's
ReMe pin. Archive members are flat, sorted, root-owned, mode 0644, mtime 0, and the bundle
stays below 16 MiB.
[Timeouts & Locks] GitHub reads 60s; downloads and ReMe builds 120-180s; no local locks.
[Side-Effect Policy] Reads channel/GitHub; writes only the bundle and an atomic sibling
temp file that is always removed.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

SHARED_PACKAGES = ("mindie-harness", "mindie-knowledge", "remote-dev", "mindie-diagnostics",
                  "mindie-debugger", "mindie-profiler", "mindie-sim", "npu-top", "mindie-coordinator")
REME_REPO_URL = "https://github.com/agentscope-ai/ReMe.git"
REME_REVISION = "4c54c2b650038eff2a5d0d77aaa61b3e836f1b18"
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
ZIP_SAFE_EPOCH = "315532800"  # 1980-01-01T00:00:00Z for deterministic ZIP/wheel timestamps
_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
_REME = re.compile(
    r"reme-ai(?:\[[^\]]+\])?\s*@\s*git\+(https://github\.com/[\w.-]+/[\w.-]+?)(?:\.git)?@([0-9a-f]{40})(?=$|\s|;)"
)
def _run(argv, timeout, *, env=None):
    return subprocess.run([str(value) for value in argv], check=True, capture_output=True, text=True, timeout=timeout, env=env, **_FLAGS).stdout
def _require(ok, message):
    if not ok:
        raise ValueError(message)
def _normalized(name):
    return re.sub(r"[-_.]+", "-", name or "").lower()
def _wheel(dest, package, version=None):
    wheels = sorted(dest.glob("*.whl"))
    _require(len(wheels) == 1 and wheels[0].name.endswith("-py3-none-any.whl"),
             f"expected one pure-Python wheel in {dest}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
        _require(len(names) == 1, f"{wheel.name}: expected exactly one dist-info METADATA")
        metadata = Parser().parsestr(archive.read(names[0]).decode("utf-8"))
    _require(_normalized(metadata.get("Name")) == _normalized(package)
             and (version is None or metadata.get("Version") == version),
             f"{wheel.name}: metadata mismatch")
    return wheel, metadata
def _sha256sums(path):
    entries = {}
    for line in filter(None, path.read_text(encoding="utf-8").splitlines()):
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        _require(match is not None and match[2] not in entries, f"invalid SHA256SUMS: {line!r}")
        entries[match[2]] = match[1].lower()
    return entries
def load_shared_components(channel_path):
    document = json.loads(Path(channel_path).read_text(encoding="utf-8"))
    _require(document.get("schema") == "mindie-release-channel/1", "unsupported channel schema")
    product_version = str(document.get("current_patch_version") or "")
    _require(re.fullmatch(r"\d+\.\d+\.\d+", product_version), "no formal patch version")
    selected = {}
    for key, item in (document.get("components") or {}).items():
        package = str(item.get("package") or key)
        if package not in SHARED_PACKAGES:
            continue
        _require(package not in selected, f"duplicate shared component: {package}")
        repository, version = item.get("repository"), item.get("version")
        revision = item.get("revision") or item.get("commit")
        valid = (isinstance(repository, str)
                 and re.fullmatch(r"mindie-agent/[A-Za-z0-9][A-Za-z0-9._-]*", repository)
                 and isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version)
                 and item.get("tag") == f"v{version}"
                 and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
                 and (not (item.get("revision") and item.get("commit"))
                      or item["revision"] == item["commit"]))
        _require(valid, f"{package}: invalid channel entry")
        selected[package] = dict(repository=repository, version=version, tag=item["tag"],
                                 revision=revision)
    missing = sorted(set(SHARED_PACKAGES) - set(selected))
    _require(not missing, f"release channel is missing shared components: {missing}")
    return product_version, {name: selected[name] for name in SHARED_PACKAGES}
def verify_tag_revision(repository, version):
    obj = json.loads(_run(["gh", "api", f"repos/{repository}/git/ref/tags/v{version}"], 60))
    obj = obj["object"]
    for _ in range(8):
        if obj["type"] == "commit":
            return obj["sha"]
        obj = (json.loads(_run(["gh", "api", f"repos/{repository}/git/tags/{obj['sha']}"], 60))["object"]
               if obj["type"] == "tag" else obj)
    raise ValueError(f"{repository}: v{version} does not resolve to a commit")
def download_component_release(repository, version, dest):
    _run(["gh", "release", "download", f"v{version}", "--repo", repository,
          "--pattern", "*-py3-none-any.whl", "--pattern", "SHA256SUMS", "--dir", str(dest)], 120)
def build_reme_wheel(repo_url, revision, dest):
    dest.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    with tempfile.TemporaryDirectory(prefix="mindie-reme-") as temporary:
        home, source = Path(temporary) / "home", Path(temporary) / "src"
        home.mkdir()
        env.update(HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home),
                   GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0",
                   SOURCE_DATE_EPOCH=ZIP_SAFE_EPOCH)
        for cmd, limit in ((["git", "init", "-q", source], 60),
                           (["git", "-C", source, "remote", "add", "origin", repo_url], 60),
                           (["git", "-C", source, "fetch", "--depth", "1", "origin", revision], 180),
                           (["git", "-C", source, "checkout", "-q", "FETCH_HEAD"], 60)):
            _run(cmd, limit, env=env)
        head = _run(["git", "-C", source, "rev-parse", "HEAD"], 60, env=env).strip()
        _require(head == revision, f"ReMe checkout {head} differs from requested {revision}")
        try:
            _run(["uv", "build", "--wheel", "--out-dir", dest, source], 180, env=env)
        except FileNotFoundError:
            _run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", dest, source],
                 180, env=env)
    wheel, metadata = _wheel(dest, "reme-ai")
    _require((metadata.get("Version") or "").strip(), "ReMe wheel version is empty")
    return wheel
def _verified_component(package, expected, dest, tag_fn, reme_repo, reme_revision):
    actual = tag_fn(expected["repository"], expected["version"])
    _require(actual == expected["revision"], f"{package}: tag revision differs from channel")
    sums = _sha256sums(dest / "SHA256SUMS")
    wheel, metadata = _wheel(dest, package, expected["version"])
    _require(wheel.name in sums, f"{package}: wheel is absent from SHA256SUMS")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    _require(digest == sums[wheel.name], f"{package}: wheel SHA-256 mismatch")
    expected_repo = reme_repo.removesuffix(".git").rstrip("/")
    for req in metadata.get_all("Requires-Dist") or []:
        if re.match(r"reme[-_.]ai\b", req, re.IGNORECASE) and "@" in req:
            match = _REME.search(req)
            _require(match is not None and match.group(1).rstrip("/") == expected_repo
                     and match.group(2) == reme_revision, "ReMe pin drift")
    return wheel, digest
def build_bundle(channel_path, output, *, reme_repo=REME_REPO_URL, reme_revision=REME_REVISION,
                 download_fn=None, tag_fn=None, reme_builder=None):
    download_fn, tag_fn = download_fn or download_component_release, tag_fn or verify_tag_revision
    reme_builder = reme_builder or build_reme_wheel
    product_version, components = load_shared_components(channel_path)
    output, staged, rows = Path(output), {}, []
    with tempfile.TemporaryDirectory(prefix="mindie-wheels-") as temporary:
        for package, expected in components.items():
            dest = Path(temporary) / package
            dest.mkdir()
            download_fn(expected["repository"], expected["version"], dest)
            wheel, digest = _verified_component(
                package, expected, dest, tag_fn, reme_repo, reme_revision)
            staged[wheel.name] = wheel.read_bytes()
            rows.append(dict(filename=wheel.name, package=package, sha256=digest,
                             version=expected["version"], revision=expected["revision"],
                             repository=expected["repository"]))
        reme_dest = Path(temporary) / "reme-ai"
        reme_dest.mkdir()
        built = reme_builder(reme_repo, reme_revision, reme_dest)
        reme_wheel, reme_metadata = _wheel(reme_dest, "reme-ai")
        _require(Path(built).resolve() == reme_wheel.resolve(), "ReMe builder returned another wheel")
        reme_version = (reme_metadata.get("Version") or "").strip()
        _require(reme_version, "ReMe wheel metadata is invalid")
        reme_digest = hashlib.sha256(reme_wheel.read_bytes()).hexdigest()
        staged[reme_wheel.name] = reme_wheel.read_bytes()
        rows.append(dict(filename=reme_wheel.name, package="reme-ai", version=reme_version,
                         sha256=reme_digest, repository=reme_repo, revision=reme_revision))
        _require(len({row["filename"] for row in rows}) == 10, "wheel filenames are not distinct")
        by_package = {row["package"]: row for row in rows}
        requirements = sorted([f"{name}=={components[name]['version']}" for name in SHARED_PACKAGES]
                              + [f"reme-ai[as]=={reme_version}"])
        staged["requirements.txt"] = ("\n".join(requirements) + "\n").encode("utf-8")
        manifest = {"schema_version": "mindie-wheels-bundle/1",
                    "product_version": product_version,
                    "components": {name: dict(**expected, wheel=by_package[name]["filename"],
                                              sha256=by_package[name]["sha256"])
                                   for name, expected in components.items()},
                    "extra_wheels": {"reme-ai": dict(
                        repository=reme_repo, revision=reme_revision, version=reme_version,
                        wheel=reme_wheel.name, sha256=reme_digest)},
                    "wheels": sorted(rows, key=lambda row: row["filename"])}
        staged["bundle-manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        staged["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(staged.items())
        ).encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output.parent / f".{output.name}.tmp.{os.getpid()}"
        try:
            with temporary_path.open("wb") as raw, gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0) as gz, tarfile.open(
                    mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(staged):
                    data = staged[name]
                    info = tarfile.TarInfo(name=name)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    info.uid, info.gid, info.uname, info.gname = 0, 0, "", ""
                    info.type = tarfile.REGTYPE
                    tar.addfile(info, io.BytesIO(data))
            _require(0 < temporary_path.stat().st_size < MAX_BUNDLE_BYTES,
                     "wheel bundle size is outside the accepted bound")
            os.replace(temporary_path, output)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass
    return output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", dest="channel_path", type=Path, required=True)
    for flag, kw in (("--output", {"type": Path, "required": True}),
                     ("--reme-repo", {"default": REME_REPO_URL}),
                     ("--reme-revision", {"default": REME_REVISION})):
        parser.add_argument(flag, **kw)
    build_bundle(**vars(parser.parse_args(argv)))
if __name__ == "__main__":
    main()
