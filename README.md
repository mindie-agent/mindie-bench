# MindIE Bench

Independent, local-first evaluation of MindIE Agent's completion rate and total
engineering cost. This repository implements task/experiment contracts, native
Codex and Claude Code drivers, OS-isolated execution, external verification,
G0-G3 controls, replay, receipt scoring, frozen baselines, candidate mining, and
fail-closed publication staging. A first-party integration connects native
Codex Stop evidence to MindIE capture, LangMem indexing, local Git publication,
and retrieval by an independent consumer.

Implementation and acceptance are distinct. Bundled fixtures are **synthetic**.
There is no claimed 12-task verified NPU dataset or measured MindIE speedup.
The current Ascend tasks cover one upstream RMSNorm defect family with two
trigger variants; their soundness checks do not establish G0-G3 effectiveness.
On A3 node 156, a real GPT-5.6-Sol seed has passed the NPU verifier and completed
native Stop capture, GPT-5.6-Luna indexing, local Git delivery and independent
consumer readback. A held-out null-trigger evaluation queried and read that
captured material, then passed the NPU verifier and completed its own local
capture. A later 24-cell cohort completed, but a runner PTY policy defect makes
it qualified historical evidence. The next 15-cell plan produced 14 receipts:
10 passed, 3 infrastructure failures and 1 unknown; one dependent evaluation was
not started. That run exposed opaque patch input incorrectly consuming the
command budget and incomplete provider settlement after a native disconnect.
Both defects are fixed in `d259695`, which passed 182 local tests. That source's
first admission check found occupied devices before any model request.
After a later clean readback, both runners passed fresh admission and started
the new 15-cell cohort at six-slot concurrency. Recurring foreign NPU occupancy
and an upstream model EOF left 12 receipts: 11 infrastructure failures and
1 unknown; three dependent evaluations were not started. Five business
verifiers passed 150 NPU cases and three seed captures completed, but required
postflight failed. The final revision `7691a48` additionally corrects the
seed-prerequisite diagnostic and export identity integrity, and passes 197 tests;
it does not relabel the
preceding cohort as evidence for its own digest. Full hardware acceptance
remains incomplete.
These partial, source-specific results do not establish an effectiveness gain.
Node 153's original
uncertain operation remains unresolved; node 156 results do not establish
recovery of node 153.

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
Local profiles use OS access restrictions. L3 runs trusted evaluation code in a
dedicated privileged Ascend container; its namespaces and cooperative NPU leases
are not a security boundary for hostile tenants.

- [Research review and architecture](docs/design.md)
- [Task, experiment, capture and remote-worker contracts](docs/usage.md)
- [Actual acceptance evidence](docs/acceptance.md)
- [Nine-principle implementation review](docs/design-principles-review.md)
- [Twelve source-grounded candidates](tasks/candidates/README.md)

`export-hf` produces local redacted JSON, Parquet and a static HTML leaderboard.
It does not publish data. The public corpus metadata is attributed under
CC-BY-4.0; implementation code is MIT.
