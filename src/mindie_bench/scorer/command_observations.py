"""Bounded literal command-position observations, never hardware-run measurements."""

from __future__ import annotations

import posixpath
import re
import shlex

MAX_COMMAND_BYTES = 262_144
MAX_TOKENS = 4096
MAX_WRAPPER_DEPTH = 8
SCOPE = "one recognized command-attempt proxy per submitted command; not NPU/kernel completions"
_PYTHON = re.compile(r"python(?:\d+(?:\.\d+)*)?$")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*", re.DOTALL)
_HEREDOC = re.compile(r"<<(-?)[ \t]*(('[^'\n]*'|\"[^\"\n]*\"|[A-Za-z_]\w*))")
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
_SEPARATORS = {";", "&&", "||", "|", "&", "\n"}
_LITERAL = {char: chr(0xE000 + index) for index, char in enumerate(";&|<>(){}\n")}
_RESTORE = str.maketrans({value: key for key, value in _LITERAL.items()})


def _protect_literal_operators(text: str) -> str:
    # shlex drops quoting, so retain the distinction between `echo ';'` and `echo;`.
    if any(marker in text for marker in _LITERAL.values()):
        raise ValueError("reserved_literal_marker")
    output, quote, index = [], None, 0
    while index < len(text):
        char = text[index]
        if char == "\\" and quote != "'" and index + 1 < len(text):
            following = text[index + 1]
            output.append(_LITERAL.get(following, char + following))
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
            elif char in _LITERAL:
                char = _LITERAL[char]
        elif char in "'\"":
            quote = char
        output.append(char)
        index += 1
    return "".join(output)


def _without_heredocs(command: str) -> str:
    """Remove literal stdin bodies while preserving commands after their terminators."""
    output = list(command)
    pending, quote, index = [], None, 0
    while index < len(command):
        char = command[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or command[index - 1] in " \t\n;&|"):
            end = command.find("\n", index)
            end = len(command) if end < 0 else end
            output[index:end] = " " * (end - index)
            index = end
            continue
        elif (
            char == "<"
            and command[index : index + 2] == "<<"
            and (index == 0 or command[index - 1] != "<")
        ):
            match = _HEREDOC.match(command, index)
            if not match:
                if command[index : index + 3] != "<<<":
                    raise ValueError("unsupported_heredoc_delimiter")
            else:
                pending.append((bool(match[1]), shlex.split(match[2])[0]))
                index = match.end()
                continue
        if char == "\n" and quote is None and pending:
            position = index + 1
            for strip_tabs, delimiter in pending:
                while position < len(command):
                    end = command.find("\n", position)
                    end = len(command) if end < 0 else end
                    line = command[position:end]
                    output[position:end] = " " * (end - position)
                    position = end + (end < len(command))
                    if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                        break
                else:
                    raise ValueError("unterminated_heredoc")
                if (line.lstrip("\t") if strip_tabs else line) != delimiter:
                    raise ValueError("unterminated_heredoc")
            pending = []
            index = position
            continue
        index += 1
    if pending:
        raise ValueError("unterminated_heredoc")
    return "".join(output)


def observe_npu_command(command: str) -> dict:
    """Observe only known programs and literal wrappers; do not evaluate scripts or variables."""
    matched, unresolved = [], []

    def inspect(text, depth):
        if depth > MAX_WRAPPER_DEPTH:
            unresolved.append("wrapper_depth_limit")
            return
        if len(text.encode()) > MAX_COMMAND_BYTES:
            unresolved.append("command_size_limit")
            return
        try:
            text = _without_heredocs(text)
            if "$(" in text or "`" in text:
                unresolved.append("command_substitution_not_inspected")
                return
            lexer = shlex.shlex(
                _protect_literal_operators(text), posix=True, punctuation_chars=";&|<>()\n"
            )
            lexer.whitespace = " \t\r"
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = []
            for token in lexer:
                tokens.append(token)
                if len(tokens) > MAX_TOKENS:
                    unresolved.append("token_limit")
                    return
            if any(token and set(token) <= set("(){}") for token in tokens):
                unresolved.append("shell_group_or_function_not_inspected")
                return
        except ValueError as error:
            unresolved.append(str(error))
            return
        segment = []
        for token in tokens + [";"]:
            if token in _SEPARATORS or token and set(token) == {"\n"}:
                program(segment, depth)
                segment = []
            else:
                segment.append(token)

    def program(words, depth):
        if depth > MAX_WRAPPER_DEPTH:
            unresolved.append("wrapper_depth_limit")
            return
        # Redirection operands and literal environment assignments are not programs.
        args, index = [], 0
        while index < len(words):
            token = words[index]
            if token and set(token) <= {"<", ">", "&"} and token not in _SEPARATORS:
                index += 2
                continue
            args.append(token.translate(_RESTORE))
            index += 1
        while args and _ASSIGNMENT.fullmatch(args[0]):
            args.pop(0)
        if not args:
            return
        head = posixpath.basename(args[0])
        rest = args[1:]
        if any(char in args[0] for char in "$*?`") or head in {"eval", "source", "."}:
            unresolved.append("dynamic_program_not_inspected")
        elif head in {"env", "command", "exec", "nohup"}:
            if rest and rest[0] == "--":
                rest = rest[1:]
            if head == "env" and rest and rest[0] == "-i":
                rest = rest[1:]
            if rest and rest[0].startswith("-"):
                unresolved.append("unsupported_prefix_options")
            else:
                program(rest, depth + 1)
        elif head in _SHELLS:
            position = 0
            while position < len(rest):
                option = rest[position]
                if option in {"--help", "--version", "--noexec"} or (
                    option.startswith("-") and not option.startswith("--") and "n" in option
                ):
                    return
                if option in {"-o", "+o"}:
                    position += 2
                    continue
                if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", option):
                    if position + 1 < len(rest):
                        inspect(rest[position + 1], depth + 1)
                    else:
                        unresolved.append("missing_shell_command")
                    return
                if not option.startswith("-"):
                    break
                position += 1
            unresolved.append("shell_script_or_stdin_not_inspected")
        elif head == "ssh":
            position = 0
            while position < len(rest) and rest[position].startswith("-"):
                option = rest[position]
                if option == "--":
                    position += 1
                    break
                if option in {"-p", "-i", "-o", "-F", "-J", "-l", "-S"}:
                    position += 2
                elif re.fullmatch(r"-[46AaCfgKkMnqTtvXxYy]+", option) or (
                    len(option) > 2 and option[:2] in {"-p", "-i", "-o", "-F", "-J", "-l", "-S"}
                ):
                    position += 1
                else:
                    unresolved.append("unsupported_ssh_options")
                    return
            if position + 1 < len(rest):
                inspect(" ".join(rest[position + 1 :]), depth + 1)
            else:
                unresolved.append("ssh_without_literal_command")
        elif head in {"pytest", "torchrun", "msprof", "reproduce.py"}:
            matched.append(head)
        elif head == "vllm" and rest[:1] == ["serve"]:
            matched.append("vllm serve")
        elif _PYTHON.fullmatch(head):
            position = 0
            while position < len(rest):
                option = rest[position]
                if option in {"--help", "--version", "-h", "-V", "-VV"}:
                    return
                if option.startswith("-m"):
                    module = option[2:] or (rest[position + 1] if position + 1 < len(rest) else "")
                    if module in {"pytest", "unittest"}:
                        matched.append(head + " -m " + module)
                    return
                if option == "-" or option.startswith("-c"):
                    unresolved.append("python_inline_or_stdin_not_inspected")
                    return
                if option in {"-W", "-X"}:
                    position += 2
                elif re.fullmatch(r"-[bBdEiIOqRsSuvx]+", option) or (
                    option.startswith(("-W", "-X")) and len(option) > 2
                ):
                    position += 1
                elif option.startswith("-") and option != "--":
                    unresolved.append("unsupported_python_options")
                    return
                else:
                    if option == "--":
                        position += 1
                    if position < len(rest):
                        script = posixpath.basename(rest[position])
                        if script == "reproduce.py" or re.fullmatch(r"test_[\w.-]+\.py", script):
                            matched.append(head + " " + script)
                    return

    if not isinstance(command, str):
        unresolved.append("command_not_text")
    else:
        inspect(command, 0)
    return dict(
        recognized=bool(matched),
        matched_programs=list(dict.fromkeys(matched)),
        unresolved_reasons=list(dict.fromkeys(unresolved)),
        scope=SCOPE,
    )


def is_recognized_npu_command(command: str) -> bool:
    return observe_npu_command(command)["recognized"]
