"""Target-access recognition for shell and embedded Python commands.

[Role]
Statically parses bounded shell pipelines and literal Python `-c` payloads into
read/edit path observations without executing the target command.

[Invariants]
- Only literal paths and recognized wrappers are resolved; dynamic globs, heredoc
  boundaries and oversized payloads remain explicit unresolved reasons.
- Recursion depth and command size are bounded; quoted data payloads are never
  reinterpreted as shell syntax.

[Timeouts & Locks]
- Pure in-memory `shlex`/`ast` parsing with depth `<= 12` and input `<= 256 KiB`;
  no subprocess, network I/O or locks.

[Side-Effect Policy]
- Pure functions returning path observations; no filesystem or process mutation.
"""

from __future__ import annotations

import ast
import posixpath
import re
import shlex


def _python_accesses(code: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Follow literal path aliases and executed syntax, never strings containing code."""

    class LiteralPath(str):
        pass

    accesses, unresolved, names = [], [], {}
    path_factories = {"Path", "PurePath"}
    local_callables = set()
    if len(code) > 262_144:
        return [], ["python_static_parser_limit"]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return [], ["python_syntax_unresolved"]

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return names.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 64:
                unresolved.append("python_literal_sequence_limit")
                return None
            if all(isinstance(part, ast.Constant) and isinstance(part.value, str) for part in node.elts):
                return tuple(part.value for part in node.elts)
        if isinstance(node, ast.Subscript):
            sequence = value(node.value)
            index = node.slice
            sign = 1
            if isinstance(index, ast.UnaryOp) and isinstance(index.op, (ast.USub, ast.UAdd)):
                sign = -1 if isinstance(index.op, ast.USub) else 1
                index = index.operand
            if isinstance(sequence, tuple):
                if isinstance(index, ast.Constant) and type(index.value) is int:
                    position = sign * index.value
                    if -len(sequence) <= position < len(sequence):
                        return sequence[position]
                unresolved.append("python_literal_sequence_index_unresolved")
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id in path_factories
            and node.func.id not in local_callables
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "Path"
        ):
            parts = [value(arg) for arg in node.args]
            return (
                LiteralPath(posixpath.join(*parts))
                if parts and all(isinstance(p, str) for p in parts)
                else None
            )
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            if isinstance(left, str) and isinstance(right, str):
                if isinstance(node.op, ast.Div) and isinstance(left, LiteralPath):
                    return LiteralPath(posixpath.join(left, right))
                if isinstance(node.op, ast.Add):
                    return left + right
        return None

    def forget_sequence(sequence):
        for name in [name for name, known in names.items() if known is sequence]:
            names.pop(name)
        unresolved.append("python_literal_sequence_invalidated")

    def escape(node):
        """An untracked receiver or argument may mutate any alias of a list."""
        known = value(node)
        if isinstance(known, tuple):
            forget_sequence(known)
        elif not isinstance(known, str):
            for child in ast.iter_child_nodes(node):
                escape(child)

    def forget_target(node, *, inplace=False):
        if isinstance(node, ast.Name):
            known = names.get(node.id)
            if inplace and isinstance(known, tuple):
                forget_sequence(known)
            names.pop(node.id, None)
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            escape(node.value)
        else:
            for child in ast.iter_child_nodes(node):
                forget_target(child, inplace=inplace)

    def forget_imports(node):
        for alias in node.names:
            name = alias.asname or (
                alias.name if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
            )
            names.pop(name, None)

    def invalidate_call(node):
        if isinstance(node.func, ast.Name) and node.func.id in local_callables:
            # An opaque local callable may change captured/global containers.
            names.clear()
            unresolved.append("python_opaque_local_callable")
        if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            names.clear()
            unresolved.append("python_dynamic_namespace")
        if isinstance(node.func, ast.Attribute):
            receiver = value(node.func.value)
            if isinstance(receiver, tuple):
                forget_sequence(receiver)
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            escape(argument)

    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    controls = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.TryStar,
        ast.AsyncFor,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.BoolOp,
        ast.Match,
    )

    def invalidate_control(node):
        # Inspect possible rebinding/escape only; never count a branch's file
        # operations as having run. Alias creation inside a branch is opaque.
        if isinstance(node, definitions):
            if getattr(node, "name", None):
                local_callables.add(node.name)
            names.pop(getattr(node, "name", None), None)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forget_imports(node)
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            names.pop(node.name, None)
        if isinstance(node, ast.MatchMapping):
            names.pop(node.rest, None)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if node.value is not None:
                escape(node.value)
        if isinstance(node, ast.Call):
            invalidate_call(node)
        if isinstance(node, ast.AugAssign):
            forget_target(node.target, inplace=True)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.pop(node.id, None)
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            escape(node.value)
        for child in ast.iter_child_nodes(node):
            invalidate_control(child)

    def visit(node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forget_imports(node)
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            path_factories.update(
                alias.asname or alias.name for alias in node.names if alias.name in {"Path", "PurePath"}
            )
        if isinstance(node, definitions):
            # A definition or a copied test body is not evidence of its execution.
            if getattr(node, "name", None):
                local_callables.add(node.name)
            names.pop(getattr(node, "name", None), None)
            unresolved.append("python_definition_body_not_observed")
            return
        if isinstance(node, controls):
            invalidate_control(node)
            unresolved.append("python_dynamic_control_flow")
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                visit(node.value)
            resolved = value(node.value) if node.value is not None else None
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            callable_alias = isinstance(node.value, ast.Lambda) or (
                isinstance(node.value, ast.Name) and node.value.id in local_callables
            )
            if resolved is None and node.value is not None:
                escape(node.value)
            for target in targets:
                forget_target(target)
                if isinstance(target, ast.Name):
                    local_callables.discard(target.id)
                    if callable_alias:
                        local_callables.add(target.id)
                    if resolved is not None:
                        names[target.id] = resolved
            return
        if isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            forget_target(node.target, inplace=True)
            if node.value is not None:
                visit(node.value)
                escape(node.value)
            unresolved.append("python_assignment_unresolved")
            return
        if isinstance(node, ast.Delete):
            for target in node.targets:
                forget_target(target)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.pop(node.id, None)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                path = value(node.args[0]) if node.args else None
                mode = value(node.args[1]) if len(node.args) > 1 else "r"
                mode = next((value(k.value) for k in node.keywords if k.arg == "mode"), mode)
                if isinstance(path, str) and isinstance(mode, str):
                    accesses.append(("edit" if any(flag in mode for flag in "wax+") else "read", path))
                else:
                    unresolved.append("python_dynamic_open_path_or_mode")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                path = value(node.func.value)
                mode = value(node.args[0]) if node.args else "r"
                mode = next((value(k.value) for k in node.keywords if k.arg == "mode"), mode)
                if isinstance(path, str) and isinstance(mode, str):
                    accesses.append(("edit" if any(flag in mode for flag in "wax+") else "read", path))
                else:
                    unresolved.append("python_dynamic_open_path_or_mode")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "read_text",
                "read_bytes",
                "write_text",
                "write_bytes",
            }:
                path = value(node.func.value)
                if isinstance(path, str):
                    accesses.append(("edit" if node.func.attr.startswith("write") else "read", path))
                else:
                    unresolved.append("python_dynamic_path")
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"replace", "rename"}
                and len(node.args) == 1
                and not node.keywords
            ):
                src = value(node.func.value)
                dst = value(node.args[0])
                if isinstance(src, LiteralPath) and isinstance(dst, str):
                    accesses.append(("edit", src))
                    accesses.append(("edit", dst))
                elif src is None:
                    unresolved.append("python_dynamic_path")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and (
                    (node.func.value.id == "os" and node.func.attr in {"replace", "rename"})
                    or (
                        node.func.value.id == "shutil"
                        and node.func.attr in {"copy", "copy2", "copyfile", "move"}
                    )
                )
                and len(node.args) >= 2
            ):
                src = value(node.args[0])
                dst = value(node.args[1])
                if isinstance(src, str):
                    destructive = node.func.attr in {"replace", "rename", "move"}
                    accesses.append(("edit" if destructive else "read", src))
                else:
                    unresolved.append("python_dynamic_path")
                if isinstance(dst, str):
                    accesses.append(("edit", dst))
                else:
                    unresolved.append("python_dynamic_path")
            invalidate_call(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return accesses, sorted(set(unresolved))


def _shell_accesses(command: str, *, depth=0, remote=False, stdin=None):
    """Parse bounded shell structure without executing it or inspecting payload files."""
    if depth > 12 or len(command) > 262_144:
        return [], ["command_static_parser_limit"]
    accesses, unresolved = [], []
    # Quoted arguments belong to the receiving program; only recursive shell
    # and SSH wrappers may interpret their redirections. Cat/echo bodies stay data.
    heredoc_pattern = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\1[ \t]*\n")
    heredoc, quote, index = None, None, 0
    while index < len(command):
        char = command[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or command[index - 1] in " \t\r\n;&|()"):
            end = command.find("\n", index)
            if end < 0:
                break
            index = end
        elif char == "<":
            heredoc = heredoc_pattern.match(command, index)
            if heredoc:
                break
        index += 1
    if heredoc:
        end = re.search(rf"(?m)^\t*{re.escape(heredoc[2])}[ \t]*(?:\n|$)", command[heredoc.end() :])
        if end is None:
            return [], ["heredoc_boundary_unresolved"]
        body = command[heredoc.end() : heredoc.end() + end.start()]
        header = command[: heredoc.start()]
        tail = command[heredoc.end() + end.end() :]
        first, reasons = _shell_accesses(header, depth=depth + 1, remote=remote, stdin=body)
        rest, more = _shell_accesses(tail, depth=depth + 1, remote=remote) if tail.strip() else ([], [])
        return first + rest, reasons + more
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [], ["shell_syntax_unresolved"]
    segments, current = [], []
    for token in tokens:
        if token in {";", "&&", "||", "|", "&", "\n", "(", ")"}:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    for words in segments:
        while words and (
            re.match(r"^[A-Za-z_]\w*=", words[0]) or words[0] in {"env", "command", "exec"}
        ):
            words = words[1:]
        if not words:
            continue
        name = posixpath.basename(words[0])
        if name in {"sh", "bash", "zsh", "dash"}:
            option = next((i for i, word in enumerate(words) if word in {"-c", "-lc", "-ic"}), None)
            if option is not None and option + 1 < len(words):
                found, reasons = _shell_accesses(
                    words[option + 1], depth=depth + 1, remote=remote, stdin=stdin
                )
                accesses.extend(found)
                unresolved.extend(reasons)
            continue
        if name == "ssh":
            index = 1
            while index < len(words) and words[index].startswith("-"):
                index += 2 if words[index] in {"-p", "-o", "-i", "-l", "-F", "-J", "-S", "-W"} else 1
            remaining = words[index + 1 :]
            if remaining:
                script = remaining[0] if len(remaining) == 1 else shlex.join(remaining)
                found, reasons = _shell_accesses(script, depth=depth + 1, remote=True, stdin=stdin)
                accesses.extend(found)
                unresolved.extend(reasons)
            continue
        redirected = set()
        for i, word in enumerate(words[:-1]):
            if word in {">", ">>", "<"}:
                accesses.append(("read" if word == "<" else "edit", words[i + 1], remote))
                redirected.update({i, i + 1})
        words = [word for i, word in enumerate(words) if i not in redirected]
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name):
            code = (
                words[words.index("-c") + 1]
                if "-c" in words and words.index("-c") + 1 < len(words)
                else stdin
            )
            if code is not None:
                found, reasons = _python_accesses(code)
                accesses.extend((kind, path, remote) for kind, path in found)
                unresolved.extend(reasons)
            else:
                unresolved.append("python_script_or_dynamic_input_unobserved")
            continue
        if name in {"apply_patch", "patch"}:
            if stdin and name == "apply_patch":
                accesses.extend(
                    ("edit", match, remote)
                    for match in re.findall(r"(?m)^\*\*\* (?:Update|Add|Delete) File: (.+)$", stdin)
                )
            else:
                unresolved.append("patch_target_payload_unobserved")
            continue
        if name == "tee":
            for word in words[1:]:
                if not word.startswith("-"):
                    accesses.append(("edit", word, remote))
            continue
        if name in {"cp", "mv"}:
            if any(w == "-t" or w.startswith("--target-directory") for w in words[1:]):
                unresolved.append("copy_or_move_target_option_unresolved")
                continue
            operands = [w for w in words[1:] if not w.startswith("-")]
            if len(operands) >= 2:
                src_kind = "read" if name == "cp" else "edit"
                accesses.extend((src_kind, src, remote) for src in operands[:-1])
                accesses.append(("edit", operands[-1], remote))
            continue
        if name not in {"cat", "sed", "grep", "rg", "head", "tail", "perl"}:
            continue
        if name == "rg" and "--files" in words:
            continue
        operands, explicit_pattern, skip = [], False, False
        options_with_value = {
            "-e",
            "-f",
            "-g",
            "--glob",
            "-t",
            "--type",
            "-n",
            "-c",
            "-A",
            "-B",
            "-C",
            "-m",
            "--max-count",
            "--max-depth",
        }
        for i, word in enumerate(words[1:]):
            if skip:
                skip = False
                continue
            if word in {"-e", "-f"} or (name == "perl" and word == "-E"):
                explicit_pattern = True
                skip = True
            elif word in options_with_value and (word not in {"-n", "-c"} or name in {"head", "tail"}):
                skip = True
            elif word.startswith("-"):
                continue
            else:
                operands.append(word)
        if name in {"grep", "rg", "sed", "perl"} and not explicit_pattern and operands:
            operands = operands[1:]
        is_inplace_edit = (
            name == "sed" and any(word == "-i" or word.startswith("-i") for word in words[1:])
        ) or (
            name == "perl"
            and any(
                word == "-i"
                or word.startswith("-i")
                or (word.startswith("-") and not word.startswith("--") and "i" in word[1:])
                for word in words[1:]
            )
        )
        kind = "edit" if is_inplace_edit else "read"
        accesses.extend((kind, path, remote) for path in operands)
    return accesses, sorted(set(unresolved))


