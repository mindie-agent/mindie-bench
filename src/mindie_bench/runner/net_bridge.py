# [Role] Relays loopback HTTP traffic to a Unix broker socket inside a network namespace.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Loopback-to-Unix broker relay inside a Linux network namespace."""

import select
import socket
import socketserver
import subprocess
import sys
import threading


def main():
    unix_path, port, *command = sys.argv[1:]

    class Relay(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                upstream.connect(unix_path)
                sockets = [self.request, upstream]
                while True:
                    ready, _, _ = select.select(sockets, [], [])
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (upstream if source is self.request else self.request).sendall(data)

    with socketserver.ThreadingTCPServer(("127.0.0.1", int(port)), Relay) as server:
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
