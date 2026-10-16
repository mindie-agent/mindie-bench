"""One-shot SSH transport into the runner-owned, network-isolated NPU slot."""

import base64
import json
import os
import sys
import urllib.request


def main():
    args = sys.argv[1:]
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option in {"-o", "-p", "-l", "-i", "-F", "-S"}:
            if not args:
                raise ValueError("SSH option requires a value")
            args.pop(0)
        elif option not in {"-T", "-n", "-q"}:
            raise ValueError(f"unsupported benchmark SSH operation: {option}")
    if not args or args.pop(0) != "bench-slot" or not args:
        raise ValueError("only one-shot commands to bench-slot are available")
    stdin = b"" if sys.stdin.isatty() else sys.stdin.buffer.read(8 * 1024 * 1024)
    payload = dict(command=" ".join(args), stdin_b64=base64.b64encode(stdin).decode())
    request = urllib.request.Request(
        os.environ["MINDIE_BENCH_GATEWAY"] + "/bench/ssh",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    sys.stdout.buffer.write(base64.b64decode(result["stdout_b64"]))
    sys.stderr.buffer.write(base64.b64decode(result["stderr_b64"]))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
