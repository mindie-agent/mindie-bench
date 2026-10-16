# MindIE Bench

Independent, local-first evaluation of MindIE Agent's completion rate and total
engineering cost. This repository implements task/experiment contracts, native
Codex and Claude Code drivers, OS-isolated execution, external verification,
G0-G3 controls, replay, receipt scoring, frozen baselines, candidate mining, and
fail-closed publication staging.

Implementation and acceptance are distinct. Bundled fixtures are **synthetic**.
There is no claimed 12-task verified NPU dataset or measured MindIE speedup.

```sh
uv sync --locked
uv run mindie-bench --help
uv run pytest
```

See `docs/design.md` for scope and corrections to the research proposal,
`docs/usage.md` for runnable commands and adapters, and
`docs/acceptance.md` for actual evidence and remaining acceptance.
