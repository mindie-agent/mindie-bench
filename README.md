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

```sh
uv run mindie-bench demo --output reports/demo
uv run mindie-bench score --input reports/demo/run
```

The demo makes no model, SSH, or NPU calls. Native experiments require explicit
model credentials and component configurations; personal CLI logins are never
copied. Each run receives fresh profiles and source worktrees. Required failures
and uncertain effects remain visible in immutable receipts.

- [Research review and architecture](docs/design.md)
- [Task, experiment, capture and remote-worker contracts](docs/usage.md)
- [Actual acceptance evidence](docs/acceptance.md)
- [Nine-principle implementation review](docs/design-principles-review.md)
- [Twelve source-grounded candidates](tasks/candidates/README.md)

`export-hf` produces local redacted JSON, Parquet and a static HTML leaderboard.
It does not publish data. The public corpus metadata is attributed under
CC-BY-4.0; implementation code is MIT.
