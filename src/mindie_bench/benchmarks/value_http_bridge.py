#!/usr/bin/env python3
"""Explicit local-only gh transport fixture for the production reporter.

[Role] Forward gh-compatible argv and stdin to the scenario's actual HTTP receiver.
[Invariants] No credentials are read; the destination must be IPv4 loopback.
[Timeouts & Locks] One HTTP request has a five-second timeout.
[Side-Effect Policy] The only network destination is an owned loopback receiver.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main():
    endpoint = os.environ["MINDIE_VALUE_RECEIVER_URL"]
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("the benchmark transport only accepts owned IPv4 loopback HTTP")
    args = sys.argv[1:]
    method = args[args.index("--method") + 1]
    path = next(value for value in args if value.startswith("repos/"))
    data = sys.stdin.buffer.read() if "--input" in args else None
    request = urllib.request.Request(endpoint + "/" + path, data=data, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read()
        json.loads(body)
        header = f"HTTP/1.1 {response.status} {response.reason}\r\n"
        header += "".join(f"{key}: {value}\r\n" for key, value in response.headers.items())
        sys.stdout.buffer.write(header.encode() + b"\r\n" + body)
        return int(response.status >= 400)


if __name__ == "__main__":
    raise SystemExit(main())
