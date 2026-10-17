"""Build the current complete CANN source tree into a fresh, owned OPP prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

INPUTS = {
    "generic-protobuf-abseil.tar.gz": (
        "13c73902399a1c3524ad8acd5f5386ee48b6c4a2a6a19774ed287c728d5142ea"
    ),
    "catlass-41bf90.tar.gz": "5f2705722d814365b5bfbf95651e4502eab170a1beb875bf7a90d5d5d8cb39f3",
    "makeself-release-2.5.0-patch1.tar.gz": (
        "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a"
    ),
    "abseil-cpp-20230802.1.tar.gz": ("987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed"),
    "protobuf-25.1.tar.gz": "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
    "include.zip": "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
}
BUILD_COMMAND = [
    "bash",
    "csrc/build.sh",
    "--pkg",
    "--ops=chunk_kda_fwd,kda_layout_swap12",
    "--soc=ascend910_93",
    "-j8",
]
GENERATED = {"build", "build_out", "output"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs(root: Path, *, cold_generic: bool = False) -> dict[str, str]:
    result = {}
    for name, expected in INPUTS.items():
        if cold_generic and name == "generic-protobuf-abseil.tar.gz":
            continue
        path = root / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"dependency bytes differ from the frozen input: {name}")
        result[name] = actual
    return result


def source_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in GENERATED:
            continue
        if path.is_symlink():
            raise ValueError(f"source symlink is outside this task's source contract: {relative}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path)
    if not (root / "build.sh").is_file():
        raise ValueError("run from a complete source worktree containing csrc/build.sh")
    return result


def unpack_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")


def generic_prefix(root: Path) -> dict:
    manifest = json.loads((root / "generic-manifest.json").read_text())
    if (
        manifest.get("compiled_generic_inputs") != ["protobuf/bin/protoc"]
        or manifest.get("compiled_kda_inputs") != []
    ):
        raise ValueError("generic dependency manifest has an unexpected compiled input")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("generic dependency contains a symlink")
        if path.is_file() and path.name != "generic-manifest.json":
            name = path.relative_to(root).as_posix()
            value = path.read_bytes()
            compiled = (
                path.suffix.lower() in {".so", ".a", ".o", ".run", ".pyc"}
                or value.startswith(b"\x7fELF")
                or value.startswith(b"!<arch>")
            )
            if compiled and name != "protobuf/bin/protoc":
                raise ValueError("generic dependency includes an unexpected native object: " + name)
            if b"chunk_kda_fwd" in value:
                raise ValueError("generic dependency includes target operator material: " + name)
            files[name] = hashlib.sha256(value).hexdigest()
    if files != manifest["files_sha256"]:
        raise ValueError("generic dependency file identity differs from its frozen manifest")
    for relative in [
        "protobuf/include/google/protobuf/message.h",
        "protobuf/bin/protoc",
        "protobuf/bin/options-pinned.h",
        "abseil/absl/log/absl_log.h",
    ]:
        if relative not in files:
            raise ValueError("generic dependency is incomplete: " + relative)
    return manifest


def prepare(source: Path, inputs: Path, output: Path, *, cold_generic: bool = False) -> dict:
    dependencies = checked_inputs(inputs, cold_generic=cold_generic)
    files = source_files(source / "csrc")
    output.mkdir(parents=True, exist_ok=False)
    copied = output / "source" / "csrc"

    def ignore(directory, names):
        return GENERATED & set(names) if Path(directory) == source / "csrc" else set()

    shutil.copytree(source / "csrc", copied, ignore=ignore)
    third_party = copied / "third_party"
    third_party.mkdir(exist_ok=True)
    for name in ["catlass", "makeself", "json", "abseil-cpp", "ascend_protobuf", "pkg", "generic-build"]:
        path = third_party / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    unpack_tar(inputs / "catlass-41bf90.tar.gz", third_party / "catlass")
    generic_record = {
        "reuse": "none; explicitly requested original upstream cold generic source build",
        "compiled_kda_inputs": [],
    }
    if not cold_generic:
        generic = third_party / "generic-build"
        unpack_tar(inputs / "generic-protobuf-abseil.tar.gz", generic)
        generic_manifest = generic_prefix(generic)
        shutil.copytree(generic / "abseil", third_party / "abseil-cpp")
        cache = copied / "build"
        cache.mkdir()
        (cache / "CMakeCache.txt").write_text(
            "ASCEND_PROTOBUF_SHARED_INCLUDE:PATH=" + str(generic / "protobuf" / "include") + "\n"
            "ASCEND_PROTOC:PATH=" + str(generic / "protobuf" / "bin") + "\n"
        )
        generic_record = {
            "manifest_sha256": sha256(generic / "generic-manifest.json"),
            "protoc_sha256": generic_manifest["files_sha256"]["protobuf/bin/protoc"],
            "protoc_version": generic_manifest["protoc_version"],
            "reuse": "upstream ASCEND_PROTOBUF_SHARED_INCLUDE/ASCEND_PROTOC native CMake path",
            "compiled_kda_inputs": [],
        }
    makeself = third_party / "makeself"
    unpack_tar(inputs / "makeself-release-2.5.0-patch1.tar.gz", makeself)
    if not (makeself / "makeself.sh").is_file():
        children = list(makeself.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("unexpected makeself release layout")
        nested = children[0]
        for child in list(nested.iterdir()):
            child.rename(makeself / child.name)
        nested.rmdir()
    package_dir = third_party / "pkg"
    package_dir.mkdir()
    for name in ["abseil-cpp-20230802.1.tar.gz", "protobuf-25.1.tar.gz"]:
        shutil.copyfile(inputs / name, package_dir / name)
    json_dir = third_party / "json"
    json_dir.mkdir()
    with zipfile.ZipFile(inputs / "include.zip") as archive:
        for member in archive.infolist():
            target = json_dir / member.filename
            if not target.resolve().is_relative_to(json_dir.resolve()):
                raise ValueError("JSON dependency archive contains an escaping path")
        archive.extractall(json_dir)
    return {
        "schema": "mindie-bench-kda-source-build/1",
        "status": "prepared",
        "source_files_sha256": files,
        "input_archives_sha256": dependencies,
        "generic_dependency": generic_record,
        "command": BUILD_COMMAND,
        "source_copy": str(output / "source"),
        "opp_prefix": str(output / "opp" / "vendors" / "custom_transformer"),
        "model_calls": 0,
        "npu_tensor_calls_during_build": 0,
        "started": time.time(),
    }


def build(
    source: Path, inputs: Path, output: Path, *, prepare_only: bool = False, cold_generic: bool = False
) -> dict:
    record = prepare(source.resolve(), inputs.resolve(), output.resolve(), cold_generic=cold_generic)
    receipt = output / "build-receipt.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if prepare_only:
        return record
    source_copy = Path(record["source_copy"])
    env = dict(os.environ, MAX_JOBS="8", CMAKE_BUILD_PARALLEL_LEVEL="8", CCACHE_PROGRAM="false")
    with (output / "build.stdout").open("xb") as stdout:
        with (output / "build.stderr").open("xb") as stderr:
            compiled = subprocess.run(
                BUILD_COMMAND, cwd=source_copy, env=env, stdout=stdout, stderr=stderr
            )
    record.update(build_exit_code=compiled.returncode, finished_build=time.time())
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if compiled.returncode:
        sys.stdout.buffer.write((output / "build.stdout").read_bytes())
        sys.stderr.buffer.write((output / "build.stderr").read_bytes())
        raise RuntimeError(f"normal CANN source build failed; see {output / 'build.stderr'}")
    packages = list((source_copy / "csrc" / "build_out").glob("*.run"))
    if len(packages) != 1:
        raise RuntimeError("normal CANN source build did not produce exactly one package")
    package = packages[0]
    command = ["bash", str(package), "--quiet", "--nox11", "--install-path=" + str(output / "opp")]
    with (output / "install.stdout").open("xb") as stdout:
        with (output / "install.stderr").open("xb") as stderr:
            installed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    record.update(
        package_sha256=sha256(package),
        install_command=command,
        install_exit_code=installed.returncode,
    )
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if installed.returncode:
        sys.stdout.buffer.write((output / "install.stdout").read_bytes())
        sys.stderr.buffer.write((output / "install.stderr").read_bytes())
        raise RuntimeError(f"normal OPP installation failed; see {output / 'install.stderr'}")
    library = Path(record["opp_prefix"]) / "op_api" / "lib" / "libcust_opapi.so"
    record.update(
        status="completed",
        op_api_sha256=sha256(library),
        finished=time.time(),
        source_unchanged=source_files(source / "csrc") == record["source_files_sha256"],
    )
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if not record["source_unchanged"]:
        raise RuntimeError("input source changed while the copied source was building")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(os.environ.get("MINDIE_KDA_INPUTS", "/bench/kda-runtime-inputs")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--cold-generic",
        action="store_true",
        help="explicit operator bootstrap: compile generic Protobuf from the five source archives",
    )
    args = parser.parse_args()
    record = build(
        args.source,
        args.inputs,
        args.output,
        prepare_only=args.prepare_only,
        cold_generic=args.cold_generic,
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "receipt_path": str(args.output.resolve() / "build-receipt.json"),
                "source_file_count": len(record["source_files_sha256"]),
                "opp_prefix": record["opp_prefix"],
                "package_sha256": record.get("package_sha256"),
                "elapsed_sec": record["finished"] - record["started"] if "finished" in record else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
