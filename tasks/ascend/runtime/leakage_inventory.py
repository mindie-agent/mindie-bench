"""Read-only inventory of known fixed Python functions across actual runtime grants.

Requires original core bytes and exact failing source bytes in a private control
directory. This script never imports target modules, extracts archives to disk,
contacts a remote host, or changes a scanned file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import marshal
import os
import stat
import sys
import tarfile
import types
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
TARGETS = {
    "rope": {"set_cos_and_sin", "update_cos_sin", "get_cos_and_sin_slice"},
    "sampler": {
        "_apply_top_k_top_p_pytorch",
        "_apply_top_k_top_p_ascendc",
        "_apply_top_k_top_p_torch_npu",
    },
}
ALIASES = {"_apply_top_k_top_p_torch_npu": "_apply_top_k_top_p_ascendc"}
ARCHIVES = (".whl", ".zip", ".egg", ".pyz", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")


def canonical_name(name):
    return ALIASES.get(name, name)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def ast_sha(node):
    node = ast.parse(ast.unparse(node)).body[0]
    node.name = canonical_name(node.name)
    return sha(ast.dump(node, include_attributes=False).encode())


def code_value(value):
    if isinstance(value, types.CodeType):
        return code_fingerprint(value)
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (str, int, float, bool, complex)) or value is None:
        return repr(value)
    if isinstance(value, tuple):
        return [code_value(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(repr(item) for item in value)
    return repr(value)


def code_fingerprint(code):
    payload = dict(
        name=canonical_name(code.co_name),
        code=code.co_code.hex(),
        consts=[code_value(value) for value in code.co_consts],
        names=code.co_names,
        varnames=code.co_varnames,
        freevars=code.co_freevars,
        cellvars=code.co_cellvars,
        flags=code.co_flags & 0xFF,
        argcount=code.co_argcount,
        posonlyargcount=code.co_posonlyargcount,
        kwonlyargcount=code.co_kwonlyargcount,
    )
    return sha(json.dumps(payload, sort_keys=True).encode())


def code_functions(code):
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield value
            yield from code_functions(value)


def source_functions(data, label):
    parsed = ast.parse(data, filename=label)
    return [
        node
        for node in ast.walk(parsed)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set().union(*TARGETS.values())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument(
        "--original", nargs=2, action="append", metavar=("FAMILY", "PRIVATE_PATH"), required=True
    )
    parser.add_argument(
        "--base", nargs=2, action="append", metavar=("FAMILY", "PRIVATE_PATH"), required=True
    )
    parser.add_argument("--max-source-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-archive-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-archive-expanded-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    roots = [path.absolute() for path in args.root]
    real_roots = [path.resolve() for path in roots]
    originals = {family: Path(path) for family, path in args.original}
    bases = {family: Path(path) for family, path in args.base}
    if set(originals) != set(TARGETS) or set(bases) != set(TARGETS):
        raise ValueError("provide exactly rope and sampler original/base private reference files")
    for path in [*originals.values(), *bases.values()]:
        if any(path.resolve().is_relative_to(root) for root in real_roots):
            raise ValueError(f"reference material is inside an agent runtime grant: {path}")
    reference_rows, exact_files, function_refs = [], {}, {}
    for family in TARGETS:
        fixed = originals[family].read_bytes()
        failing = bases[family].read_bytes()
        original_nodes = {
            canonical_name(node.name): node for node in source_functions(fixed, str(originals[family]))
        }
        base_nodes = {
            canonical_name(node.name): node for node in source_functions(failing, str(bases[family]))
        }
        original_full_code = {
            canonical_name(code.co_name): code
            for code in code_functions(compile(fixed, "<original-module>", "exec", dont_inherit=True))
            if canonical_name(code.co_name) in base_nodes
        }
        base_full_code = {
            canonical_name(code.co_name): code
            for code in code_functions(compile(failing, "<base-module>", "exec", dont_inherit=True))
            if canonical_name(code.co_name) in base_nodes
        }
        if not set(base_nodes) <= set(original_nodes):
            raise ValueError(f"missing core reference target functions: {family}")
        exact_files.setdefault(len(fixed), {})[sha(fixed)] = family
        ref = dict(
            family=family,
            original_private_path=str(originals[family]),
            original_file_sha256=sha(fixed),
            base_private_path=str(bases[family]),
            base_file_sha256=sha(failing),
            functions=[],
        )
        for name, node in base_nodes.items():
            original = original_nodes[name]
            original_ast, base_ast = ast_sha(original), ast_sha(node)
            original_code = next(
                value
                for value in code_functions(
                    compile(ast.Module(body=[original], type_ignores=[]), "<reference>", "exec")
                )
                if value.co_name == original.name
            )
            base_code = next(
                value
                for value in code_functions(
                    compile(ast.Module(body=[node], type_ignores=[]), "<reference>", "exec")
                )
                if value.co_name == node.name
            )
            record = dict(
                family=family,
                name=name,
                original_ast_sha256=original_ast,
                base_ast_sha256=base_ast,
                changed_from_failing=original_ast != base_ast,
                original_bytecode_sha256=code_fingerprint(original_code),
                base_bytecode_sha256=code_fingerprint(base_code),
                original_full_module_bytecode_sha256=code_fingerprint(original_full_code[name]),
                base_full_module_bytecode_sha256=code_fingerprint(base_full_code[name]),
            )
            function_refs[name] = record
            ref["functions"].append(record)
        reference_rows.append(ref)

    result = dict(
        schema="mindie-bench-known-source-leakage-inventory/1",
        runtime_roots=[str(path) for path in roots],
        canonical_roots=[str(path) for path in real_roots],
        references=reference_rows,
        python=sys.version,
        bytecode_magic=importlib.util.MAGIC_NUMBER.hex(),
        limits=dict(
            source_bytes=args.max_source_bytes,
            archive_bytes=args.max_archive_bytes,
            archive_expanded_bytes=args.max_archive_expanded_bytes,
        ),
        counts=dict(
            directories=0,
            files=0,
            source_files_parsed=0,
            bytecode_files_parsed=0,
            archive_files_reviewed=0,
            symlinks=0,
            nonregular=0,
        ),
        exact_original_files=[],
        target_functions=[],
        bytecode_targets=[],
        symlinks=[],
        git_metadata=[],
        archive_reviews=[],
        review_required=[],
        errors=[],
        compiled_extensions=[],
        scopes=[
            "Known exact core files, named target-function ASTs, same-interpreter Python bytecode and bounded source archives.",
            "Compiled native libraries are inventoried by filename/size; this is not a universal native-code decompilation proof.",
            "External symlink targets are traversed for inventory but actual namespace readability must be checked separately.",
            "Different or unparseable target code, unread archives and filesystem errors withhold a clean inventory claim.",
        ],
    )
    visited_dirs, visited_files = set(), set()
    inventory_digest = hashlib.sha256()
    needles = [name.encode() for names in TARGETS.values() for name in names]

    def require(path, reason, **details):
        result["review_required"].append(dict(path=path, reason=reason, **details))

    def inspect_bytes(data, display, suffix):
        digest = sha(data)
        family = exact_files.get(len(data), {}).get(digest)
        if family:
            result["exact_original_files"].append(dict(path=display, sha256=digest, family=family))
        if suffix == ".py":
            if not any(needle in data for needle in needles):
                return
            try:
                nodes = source_functions(data, display)
            except (SyntaxError, UnicodeError, ValueError) as error:
                require(display, "target-bearing source could not be parsed", error=repr(error))
                return
            result["counts"]["source_files_parsed"] += 1
            for node in nodes:
                name = canonical_name(node.name)
                reference = function_refs[name]
                fingerprint = ast_sha(node)
                if fingerprint == reference["base_ast_sha256"]:
                    classification = "exact_failing_or_unchanged_reference"
                elif (
                    reference["changed_from_failing"] and fingerprint == reference["original_ast_sha256"]
                ):
                    classification = "known_original_fixed_function"
                else:
                    classification = "different_target_function_requires_review"
                    require(display, classification, function=name, ast_sha256=fingerprint)
                result["target_functions"].append(
                    dict(
                        path=display,
                        function=node.name,
                        normalized_name=name,
                        lineno=node.lineno,
                        ast_sha256=fingerprint,
                        classification=classification,
                        file_sha256=digest,
                    )
                )
        elif suffix == ".pyc":
            if not any(needle in data for needle in needles):
                return
            if data[:4] != importlib.util.MAGIC_NUMBER:
                require(
                    display, "bytecode interpreter magic differs; not inspected", magic=data[:4].hex()
                )
                return
            try:
                module = marshal.loads(data[16:])
                if not isinstance(module, types.CodeType):
                    raise ValueError("pyc payload is not a code object")
                functions = list(code_functions(module))
            except Exception as error:
                require(display, "bytecode could not be inspected", error=repr(error))
                return
            result["counts"]["bytecode_files_parsed"] += 1
            for code in functions:
                name = canonical_name(code.co_name)
                if name not in function_refs:
                    continue
                reference = function_refs[name]
                fingerprint = code_fingerprint(code)
                if fingerprint in {
                    reference["base_bytecode_sha256"],
                    reference["base_full_module_bytecode_sha256"],
                }:
                    classification = "exact_failing_or_unchanged_reference"
                elif reference["changed_from_failing"] and fingerprint in {
                    reference["original_bytecode_sha256"],
                    reference["original_full_module_bytecode_sha256"],
                }:
                    classification = "known_original_fixed_function"
                else:
                    classification = "different_target_bytecode_requires_review"
                    require(display, classification, function=name, bytecode_sha256=fingerprint)
                result["bytecode_targets"].append(
                    dict(
                        path=display,
                        function=code.co_name,
                        normalized_name=name,
                        bytecode_sha256=fingerprint,
                        classification=classification,
                    )
                )
        elif suffix in {".diff", ".patch"} and any(needle in data for needle in needles):
            require(
                display, "patch/diff mentions target function; inspect potential solution", sha256=digest
            )

    def inspect_archive(path, display, size):
        review = dict(
            path=display, bytes=size, members_checked=0, source_bytes_checked=0, status="completed"
        )
        result["archive_reviews"].append(review)
        if size > args.max_archive_bytes:
            review["status"] = "unreviewed_size_limit"
            require(display, "archive exceeds compressed-size review limit", bytes=size)
            return
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            review["sha256"] = digest
            is_zip = path.name.lower().endswith((".whl", ".zip", ".egg", ".pyz"))
            archive = zipfile.ZipFile(path) if is_zip else tarfile.open(path, "r:*")
            total = 0
            with archive:
                entries = archive.infolist() if is_zip else archive
                for member in entries:
                    name = member.filename if is_zip else member.name
                    member_size = member.file_size if is_zip else member.size
                    regular = not member.is_dir() if is_zip else member.isfile()
                    if not regular:
                        if not is_zip and (member.issym() or member.islnk()):
                            require(
                                f"{display}!{name}",
                                "archive link requires target review",
                                target=member.linkname,
                            )
                        continue
                    if name.lower().endswith(ARCHIVES):
                        require(
                            f"{display}!{name}",
                            "nested archive requires separate review",
                            bytes=member_size,
                        )
                        continue
                    if not name.lower().endswith((".py", ".pyc", ".patch", ".diff")):
                        continue
                    total += member_size
                    if member_size > args.max_source_bytes or total > args.max_archive_expanded_bytes:
                        review["status"] = "incomplete_expanded_size_limit"
                        require(f"{display}!{name}", "archive source review limit", bytes=member_size)
                        break
                    stream = archive.open(member) if is_zip else archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"archive source member unreadable: {name}")
                    with stream:
                        data = stream.read(args.max_source_bytes + 1)
                    if len(data) != member_size:
                        raise ValueError(f"archive source member size mismatch: {name}")
                    inspect_bytes(data, f"{display}!{name}", Path(name).suffix.lower())
                    review["members_checked"] += 1
                    review["source_bytes_checked"] += len(data)
            result["counts"]["archive_files_reviewed"] += 1
        except Exception as error:
            review["status"] = "failed"
            result["errors"].append(dict(path=display, operation="inspect_archive", error=repr(error)))

    def walk(path):
        display = str(path)
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                result["counts"]["symlinks"] += 1
                target = path.resolve(strict=True)
                result["symlinks"].append(
                    dict(
                        path=display,
                        target=str(target),
                        outside_declared_roots=not any(
                            target.is_relative_to(root) for root in real_roots
                        ),
                    )
                )
                walk(target)
                return
            identity = (info.st_dev, info.st_ino)
            if stat.S_ISDIR(info.st_mode):
                if identity in visited_dirs:
                    return
                visited_dirs.add(identity)
                result["counts"]["directories"] += 1
                if path.name == ".git" or (path.name == "objects" and path.parent.name == ".git"):
                    result["git_metadata"].append(display)
                with os.scandir(path) as children:
                    for child in sorted(children, key=lambda item: item.name):
                        walk(Path(child.path))
                return
            if not stat.S_ISREG(info.st_mode):
                result["counts"]["nonregular"] += 1
                return
            if identity in visited_files:
                return
            visited_files.add(identity)
            result["counts"]["files"] += 1
            inventory_digest.update(json.dumps((display, info.st_size, info.st_mtime_ns)).encode())
            lower = path.name.lower()
            if path.name == ".git":
                result["git_metadata"].append(display)
            if lower.endswith(ARCHIVES):
                inspect_archive(path, display, info.st_size)
                return
            suffix = path.suffix.lower()
            if suffix in {".so", ".a", ".o"} or ".so." in lower:
                result["compiled_extensions"].append(dict(path=display, bytes=info.st_size))
            if suffix in {".py", ".pyc", ".patch", ".diff"} or info.st_size in exact_files:
                if info.st_size > args.max_source_bytes:
                    require(display, "source/bytecode exceeds review limit", bytes=info.st_size)
                    return
                data = path.read_bytes()
                if len(data) != info.st_size:
                    require(
                        display,
                        "file changed while being read",
                        before_bytes=info.st_size,
                        after_bytes=len(data),
                    )
                inspect_bytes(data, display, suffix)
        except Exception as error:
            result["errors"].append(dict(path=display, operation="walk/read", error=repr(error)))

    for root in roots:
        walk(root)
    result["walk_inventory_sha256"] = inventory_digest.hexdigest()
    known_fixed = [
        row
        for key in ("target_functions", "bytecode_targets")
        for row in result[key]
        if row["classification"] == "known_original_fixed_function"
    ]
    result["known_fixed_occurrences"] = len(known_fixed)
    result["known_source_inventory_status"] = (
        "known_fixed_source_or_history_exposed"
        if known_fixed or result["exact_original_files"] or result["git_metadata"]
        else "incomplete_requires_review"
        if result["review_required"] or result["errors"]
        else "no_known_fixed_source_found_within_recorded_scope"
    )
    result["universal_no_leakage_claim"] = False
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
