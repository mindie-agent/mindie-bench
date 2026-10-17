# MindIE Bench

Independent, local-first evaluation of MindIE Agent's completion rate and total
engineering cost. This repository implements task/experiment contracts, native
Codex and Claude Code drivers, OS-isolated execution, external verification,
G0-G3 controls, replay, receipt scoring, frozen baselines, candidate mining, and
fail-closed publication staging. A first-party integration connects native
Codex Stop evidence to MindIE capture, LangMem indexing, local Git publication,
and retrieval by an independent consumer.

Implementation and acceptance are distinct. Bundled fixtures are **synthetic**.
The real Ascend set contains three historical defect families with six
seed/evaluation variants: RMSNorm optional metadata, partial RoPE dimensions and
reduce-sampler candidates. Each hardware class has independent source-fail /
gold-pass NPU proof. Four separate CPU task recipes exercise real PyTorch source
contracts, with their own task identities and soundness requirements.

The current runtime fix is commit `620949d`, digest
`ddf64cba66341dbcf87d1ef8c87265fecb54c31a6fbf927f3a84778ebae4c278`.
The delivery suite passes 252 zero-model tests. The frozen H native
GPT-6.1-Sol / GPT-5.6-Luna matrix terminated with 88 receipts and two blocked
cells across its 90 planned A2/A3 cells. Its raw outcomes are 78 pass, five fail,
one infrastructure failure and four unknowns. Independent audit also found a
late remote-RPC unknown hidden by that infrastructure receipt; the original
receipt is preserved. H did not pass complete matrix acceptance.

The I runtime seals remote admission, waits a bounded time for admitted handlers
and checks uncertainty again before saving the outcome. Its new bounded
validation passed one RoPE seed/evaluation pair on each hardware class:
four native passes, real Stop/indexing/local Git/consumer loops, 72 external NPU
case executions and complete usage for all 78 provider requests. Separate
zero-model real RPC cases on both clusters verified normal settlement and
preserved intentional timeout uncertainty. This validates the finite repair
contract; it is not a rerun of the 90-cell matrix or proof of general benefit.
H remains bound to its original runtime
`f922f0dda22d052679523830782dca52736054978d5922d0233926fe62bb8bc9`.
The preceding G run exposed a task
verifier classification defect and insufficient capture diagnostics, corrected
and separately tested in H. Original unknown/infra receipts remain intact.
See [actual acceptance](docs/acceptance.md) for experiment identities,
terminal evidence and limits; planned cells are not completed cells.
Native Stop, real indexing, local Git delivery and independent consumer checks
remain distinct from observed foreground retrieval and effectiveness.
Four native negative routing tasks and a separate explicit retrieval canary
have passed. A real native CPU repair passed its independent 18-case judge.
These are finite additional observations, separate from the Ascend matrix.

There is no claimed initial twelve-task flagship dataset, universal MindIE speedup,
full-model serving, AscendC or multi-host HCCL acceptance. Original failed and
unknown attempts remain intact; new runs do not repair their history. Node 156
results do not establish recovery of node 153's unresolved operation.

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
- [Real Ascend task runbook](tasks/ascend/README.md)
- [Real CPU task recipes](tasks/cpu/README.md)
- [Routing and retrieval evidence contract](tasks/L1-L2-acceptance.md)
- [Twelve source-grounded candidates](tasks/candidates/README.md)

`export-hf` produces local redacted JSON, Parquet and a static HTML leaderboard.
It does not publish data. The public corpus metadata is attributed under
CC-BY-4.0; implementation code is MIT.
