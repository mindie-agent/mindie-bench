# [Role] Owns a child lifetime pipe so runner-owned processes die with their owner.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""macOS owner-pipe pattern from MindIE's native process contract.

The pipe, unlike parent-PID polling, also handles a killed runner. Linux's
bubblewrap additionally terminates namespace descendants on owner exit.
"""

import os
import signal
import subprocess
import sys


def main():
    owner_pipe = int(sys.argv[1])
    command = sys.argv[2:]
    group = os.getpgrp()
    if group != os.getpid():
        return 125
    if os.fork() == 0:
        for descriptor in (0, 1, 2):
            os.close(descriptor)
        try:
            while os.read(owner_pipe, 1):
                pass
        finally:
            os.killpg(group, signal.SIGKILL)
        os._exit(125)
    os.close(owner_pipe)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
