# MindIE Bench

Independent evaluation and benchmark suite for MindIE Agent (`src/mindie_bench/{runner,scorer,evaluation,integrations,mining,hf_sync}`), not a runtime installed in business repositories. Authoritative product principles live in `../mindie-agent/docs/`.

## 1. Inner-Loop Development & Evaluation Discipline (Always Active)

- **Fast Code-Test Loop (Principle 1)**:
  - Edit source directly in this checkout. During development and debugging, run **only targeted module tests** (`uv run pytest tests/test_<module>.py -q`) and `uv run ruff check <changed_files>` for second-level feedback.
  - Run the full test suite (`uv run pytest` and `uv run ruff check .`) **only once** at final delivery or before PR submission.
- **No Mid-Iteration Ceremonies**:
  - Do **not** build wheels/sdists (`uv build` / `pip wheel`), run `mindie-bench freeze`, clone shadow/copy directories, or write `readback.json`/`restoration.json` audit files during inner-loop iteration.
  - When a newly written script, adapter, or task contract fails during development, fix it in place and re-run directly—do not treat development bugs as immutable historical trials.
- **Canary Before Full Matrix**:
  - Validate new tasks, harnesses, or remote NPU configs with a single cell (`--canary-first` or 1 task × 1 group × 1 repetition) to confirm MCP tool visibility and non-null metrics before running multi-arm `G0..G3` cohorts.
- **Evaluation Integrity & Error Visibility (Principle 9)**:
  - Keep private raw traces separate from publication artifacts. Never upload trial data, create public issues/PRs from trials, or automatically retry an `UNKNOWN` paid model/remote attempt.
  - Keep synthetic/replay (`L1`/`L2`) checks distinct from live NPU (`L3`) acceptance.

## 2. Progressive Disclosure Map (Read Only When Triggered)

Read reference documents only when your task touches that domain:

- **Benchmark Architecture, Isolation & Metrics** → [`docs/design.md`](docs/design.md)
- **CLI Commands & Study Execution Guide** → [`docs/usage.md`](docs/usage.md)
- **Milestone Acceptance Criteria (`v0.1.0` / `v0.2.0`)** → [`docs/acceptance.md`](docs/acceptance.md)
- **Gate — Submitting a Development PR**:
  - Only when preparing a PR, review the final diff against [`../mindie-agent/docs/design-principles.md#三开发-pr-提交前自查清单pre-pr-review`](../mindie-agent/docs/design-principles.md#三开发-pr-提交前自查清单pre-pr-review) and record the concise review summary in the PR description.
