"""Freeze only already source-built generic Protobuf/Abseil dependency material."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

RULES = {
    "csrc/cmake/third_party/ascend_protobuf.cmake": "8617004a8b657c2fa9adff4cb433153e6bf3fb245e4b0e5e4c4ad26875a84f6a",
    "csrc/cmake/third_party/abseil-cpp.cmake": "ae1cae2f2e9890690e47c59c792ed8357da7c2e518d7f6dd178675e7c4dc96b1",
    "csrc/cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch": "210033396c1a3551f4e24dd5d49b5b4588c53b450ab05fef27d31ae389d94c4f",
    "csrc/cmake/third_party/build/modules/patch/protobuf-hide_absl_symbols.patch": "ff1419dfcb83e9c9478f63e2ece972c0362b8ecda34535b8d2628cdaac81be92",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source: Path, output: Path) -> dict:
    for relative, expected in RULES.items():
        if sha(source / relative) != expected:
            raise ValueError("generic source rule changed: " + relative)
    csrc = source / "csrc"
    built = (
        csrc
        / "build/ascend_protobuf_build_transformer-prefix/src/ascend_protobuf_build_transformer-build"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(csrc / "third_party/ascend_protobuf/src", output / "protobuf/include")
    shutil.copytree(csrc / "third_party/abseil-cpp/absl", output / "abseil/absl")
    (output / "protobuf/bin").mkdir()
    for name in ["protoc", "options-pinned.h"]:
        shutil.copyfile(built / name, output / "protobuf/bin" / name)
    (output / "protobuf/bin/protoc").chmod(0o755)
    for key, origin in [
        ("protobuf", csrc / "third_party/ascend_protobuf/LICENSE"),
        ("abseil", csrc / "third_party/abseil-cpp/LICENSE"),
    ]:
        shutil.copyfile(origin, output / key / "LICENSE")
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("generic dependency contains a symlink: " + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        data = path.read_bytes()
        if (
            path.suffix.lower() in {".so", ".a", ".o", ".run", ".pyc"}
            or data.startswith(b"!<arch>")
            or data.startswith(b"\x7fELF")
        ) and relative != "protobuf/bin/protoc":
            raise ValueError("unexpected compiled dependency: " + relative)
        if b"chunk_kda_fwd" in data or b"postWuScratchOffset" in data:
            raise ValueError("target operator material entered generic prefix: " + relative)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    protoc = output / "protobuf/bin/protoc"
    version = subprocess.run([str(protoc), "--version"], check=True, capture_output=True, text=True)
    ldd = subprocess.run(["ldd", str(protoc)], check=True, capture_output=True, text=True)
    manifest = {
        "schema": "mindie-bench-kda-generic-build-input/1",
        "protobuf_source_package": "protobuf-25.1.tar.gz",
        "abseil_source_package": "abseil-cpp-20230802.1.tar.gz",
        "source_rule_sha256": RULES,
        "files_sha256": hashes,
        "compiled_generic_inputs": ["protobuf/bin/protoc"],
        "compiled_kda_inputs": [],
        "protoc_version": version.stdout.strip(),
        "protoc_ldd": ldd.stdout,
        "source_base_commit": "9a3a8fdeb248e9fe444fe80353ea3795d1030df0",
        "model_calls": 0,
        "npu_tensor_calls": 0,
        "reuse_scope": "only normal previously source-built patched generic protobuf/protoc and patched Abseil headers; no operator objects or OPP",
    }
    (output / "generic-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.with_suffix(".tar.gz")
    with archive.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        name = path.relative_to(output).as_posix()
                        info = tarfile.TarInfo(name)
                        info.size = path.stat().st_size
                        info.mode = 0o755 if name == "protobuf/bin/protoc" else 0o644
                        with path.open("rb") as content:
                            tar.addfile(info, content)
    return {
        "status": "completed",
        "prefix": str(output),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha(archive),
        "manifest_sha256": sha(output / "generic-manifest.json"),
        "files": len(hashes),
        "protoc_sha256": sha(protoc),
        "protoc_version": version.stdout.strip(),
        "model_calls": 0,
        "npu_tensor_calls": 0,
        "compiled_kda_inputs": [],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.source, a.output)))
