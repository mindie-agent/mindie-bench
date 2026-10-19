"""Owned Kimi Code profile materialization for native trials.

[Role]
Creates the per-trial `KIMI_CODE_HOME` and environment wiring for the `kimi`
harness so the native CLI runs fully inside the benchmark-owned profile.
Called from `build_profile` when `config.harness.name == "kimi"`, mirroring
`build_dsh_profile`.

[Invariants]
- The owned home replaces `~/.kimi-code` via `KIMI_CODE_HOME`; the personal
  config, OAuth credentials, sessions and logs stay outside the sandbox and
  are never read or copied.
- Telemetry and the update preflight are disabled; `KIMI_CODE_TRUST_WORKSPACE`
  is never set, so project-level `.kimi-code/mcp.json` in a source snapshot
  stays disabled in headless runs.
- The harness executable's own directory leads PATH so the CLI resolves its
  bundled `rg`/`fd` siblings exactly as in the native installation.

[Timeouts & Locks]
- No waits, probes or locks; runtime-bound files (`config.toml`, `mcp.json`)
  are written later by `KimiDriver.command()` once the gateway URL exists.

[Side-Effect Policy]
- Writes only inside `profile root / "kimi"` and mutates only `env`.
"""

from __future__ import annotations

import os
from pathlib import Path


def build_kimi_profile(root: Path, env: dict[str, str], config) -> dict:
    home = root / "kimi"
    home.mkdir(mode=0o700, exist_ok=True)
    # Deterministic empty target for `--skills-dir` when the group declares no
    # skills, so auto-discovered user/project skills can never leak into G0/G1.
    (home / "no-skills").mkdir(exist_ok=True)
    env.update(
        KIMI_CODE_HOME=str(home),
        KIMI_DISABLE_TELEMETRY="1",
        KIMI_CODE_NO_AUTO_UPDATE="1",
    )
    executable_dir = str(Path(config.harness.executable).resolve(strict=True).parent)
    env["PATH"] = executable_dir + os.pathsep + env["PATH"]
    return {"home": str(home)}
