"""One-shot SSH transport into the runner-owned, network-isolated NPU slot.

[Role] Forward SSH and knowledge RPC to the runner's local broker.
[Invariants] Broker requests stay local regardless of enterprise proxy settings.
[Timeouts & Locks] Each broker request retains its bounded 3600-second timeout.
[Side-Effect Policy] Submit once; the parent owns execution and receipt accounting.
"""

import base64
import json
import os
import select
import sys
import time
import urllib.error
import urllib.request


def _post(gateway, path, payload):
    request = urllib.request.Request(
        gateway + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 * 1024).decode(errors="replace")
        raise RuntimeError(f"benchmark SSH broker HTTP {exc.code}: {detail}") from exc


def _stream(gateway, command):
    opened = _post(gateway, "/bench/ssh-stream", dict(action="open", command=command))
    stream_id = opened["stream_id"]
    eof = False
    while True:
        data = b""
        close_input = False
        if not eof:
            readable, _, _ = select.select([sys.stdin.fileno()], [], [], 0.05)
            if readable:
                data = os.read(sys.stdin.fileno(), 65536)
                eof = not data
                close_input = eof
        else:
            time.sleep(0.05)
        result = _post(
            gateway,
            "/bench/ssh-stream",
            dict(
                action="exchange",
                stream_id=stream_id,
                stdin_b64=base64.b64encode(data).decode(),
                eof=close_input,
            ),
        )
        for name, target in [("stdout_b64", sys.stdout), ("stderr_b64", sys.stderr)]:
            target.buffer.write(base64.b64decode(result[name], validate=True))
            target.buffer.flush()
        exit_code = result["exit_code"]
        if exit_code is not None:
            if type(exit_code) is not int:
                raise ValueError("SSH stream returned an invalid exit code")
            return exit_code


def main():
    stdin_null = False
    args = sys.argv[1:]
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option == "--":
            break
        if option in {"-o", "-p", "-l", "-i", "-F", "-S"}:
            if not args:
                raise ValueError("SSH option requires a value")
            args.pop(0)
        elif option == "-n":
            stdin_null = True
        elif option not in {"-T", "-q", "-v"}:
            raise ValueError(f"unsupported benchmark SSH operation: {option}")
    if not args or args.pop(0) not in {"bench-slot", "root@bench-slot"} or not args:
        raise ValueError("only one-shot commands to bench-slot are available")
    command = " ".join(args)
    gateway = os.environ["MINDIE_BENCH_GATEWAY"]
    if (
        os.environ.get("MINDIE_BENCH_SSH_STREAM") == "1"
        or os.environ.get("REMOTE_DEV_HARNESS_PROFILE") == "dsh"
    ) and command != "bash -s":
        return _stream(gateway, command)
    if stdin_null or sys.stdin.isatty():
        stdin = b""
    else:
        readable, _, _ = select.select([sys.stdin.fileno()], [], [], 0.2)
        stdin = sys.stdin.buffer.read(8 * 1024 * 1024) if readable else b""
    payload = dict(command=command, stdin_b64=base64.b64encode(stdin).decode())
    result = _post(gateway, "/bench/ssh", payload)
    sys.stdout.buffer.write(base64.b64decode(result["stdout_b64"]))
    sys.stderr.buffer.write(base64.b64decode(result["stderr_b64"]))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
