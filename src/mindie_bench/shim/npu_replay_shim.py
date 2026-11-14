"""A thin client: oracle rules never enter the agent's process or filesystem.

[Role] Send replay arguments to the runner-owned local broker.
[Invariants] Local replay never uses an enterprise proxy.
[Timeouts & Locks] One bounded 60-second request, with no retry or client lock.
[Side-Effect Policy] The parent owns replay rules; failures remain infrastructure errors.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def main():
    cmd_name = os.path.basename(sys.argv[0])
    if os.name == "nt" and cmd_name.lower().endswith(".exe"):
        cmd_name = cmd_name[:-4]
    request = urllib.request.Request(
        os.environ["MINDIE_BENCH_GATEWAY"] + "/bench/replay",
        data=json.dumps({"argv": [cmd_name, *sys.argv[1:]]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=60) as response:
            result = json.load(response)
    except TimeoutError:
        raise
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        print(f"replay infrastructure error: {type(exc).__name__}", file=sys.stderr)
        return 125
    sys.stdout.buffer.write(result["stdout"].encode("utf-8"))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(result["stderr"].encode("utf-8"))
    sys.stderr.buffer.flush()
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
