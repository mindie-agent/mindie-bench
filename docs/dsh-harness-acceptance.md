# DSH native-harness acceptance (2026-10-18)

Host: macOS arm64, local OpenAI-compatible model relay. Harness: DeepSeek
Harness `0.2.0-rc.2` at `/opt/homebrew/bin/dsh`, headless profile created fresh
inside each trial. Task: the materialized CPU L2 revision
`cpu-rope-rotary-dim-64` from `reports/acceptance-a2-a3-20261017/cpu-l2`.

The runner used the normal native gateway and OS sandbox. The trial profile set
`DSH_PERMISSION_MODE=danger-full-access` because the benchmark already confines
the whole process; a nested DSH Seatbelt invocation is unavailable on macOS.

| Group | Result | Observed |
| --- | --- | --- |
| G0 | `completed_pass` | 11 turns; bash/read/glob tools; 13 calls; verifier pass |
| G1 | `completed_pass` | 13 turns; remote-dev MCP configured and preflighted; the CPU task was solved locally, 0 remote calls |
| G2 | `completed_pass` | 18 turns; 1 `knowledge_query`, 1 `knowledge_explain`, a `skill` call, 1.27M cached input tokens; verifier pass |

The G2 run used the real knowledge MCP relay and the installed MindIE skill.
Tool names were normalized for the benchmark metrics (`read`→`Read`,
`edit`→`Edit`, `grep`→`Grep`, `glob`→`Glob`); MCP names retain their component.

Commands:

```sh
mindie-bench validate --tasks <task-revision> --config <dsh-config.toml>
mindie-bench run --tasks <task-revision> --config <dsh-config.toml> \
  --tier l2 --groups G0 --repetitions 1 --canary-first --output <run-dir>
```

Known boundaries:

- G3 native capture is unsupported for DSH; the capture adapter currently
  consumes Codex/Claude native Stop and transcript formats.
- G1's CPU canary does not exercise a real SSH/NPU remote; the remote MCP was
  launched and preflighted, but the model chose the local task path.
- The model credential is held by the benchmark gateway; DSH receives only the
  broker placeholder.
