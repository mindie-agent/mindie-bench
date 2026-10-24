# mindie-bench 0.4.3: Behavior-Executed Task Verifiers, Observed Continuation Accounting & Maintenance Patch Release Pipeline

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.3/docs/releases/mindie-bench-0.4.3.md)

`mindie-bench 0.4.3` upgrades six regression task verifiers to execute real behavioral checks, removes synthetic continuation fallbacks from strategy-selection evaluation while preserving observed costs on timed-out or faulted continuations, and adds the `release-kind=patch` maintenance publication path plus repository-authored community Release Notes support.

---

## 1. Key Fixes & Engineering Improvements

### 1. Six Task Verifiers Upgraded to Behavioral Execution (PR `#25`)

- Replaced permissive source checks across six benchmark tasks that previously accepted nonfunctional or constant edits:
  - `dspark-mrv2-drafter-crash-zh-vague`
  - `k3-idle-dp-gdn-state-zh-vague`
  - `l3-dspark-mrv2-drafter-crash-zh-vague`
  - `mtp-draft-slot-mapping-capacity-zh-vague`
  - `sfa-mtp-seq-lens-eval-zh-vague`
  - `sfa-mtp-seq-lens-seed-zh-vague`
- Each repaired verifier executes the tested workspace code under real behavioral checks, verifying:
  - Known faulty baseline fails (`before_reward=0`);
  - Golden repair patch passes (`after_reward=1`);
  - Zero model calls during verification (`model_calls=0`);
  - Nonfunctional or constant-return negative controls fail.

### 2. Strategy-Selection Continuation Isolation & Partial Evidence Preservation (`strategy_selection.py`, `knowledge_tiered_support.py`, `comparator.py`, `metrics.py`, PR `#24` / `#26`)

- **Removed Synthetic Continuation Fallbacks (PR `#24`)**: The default strategy selector no longer reads private judge answers to synthesize continuation outcomes or fabricated costs. When no real continuation executor is supplied, strategy selection and task continuation remain strictly separated (`not_executed` or `continuation_unverified`, keeping unmeasured continuation latency and device cost as `None`).
- **Observed Call Accounting & Partial Continuation Preservation (PR `#26`)**:
  - Deterministic strategy selection records `0` consumed model calls and tracks reserved budgets separately from observed calls;
  - When a single-cell continuation times out (`unknown`) or hits an infrastructure fault (`infra`), completed selector verdicts, consumed calls, and measured timeout durations are preserved rather than discarded.

### 3. Maintenance Patch Release Pipeline (`release-kind=patch`) & Repository-Authored Release Notes (`.github/workflows/publish-product.yml`, `collect_product_benchmarks.py`, `publish-release.yml`, PR `#27`)

- **Dedicated `release-kind=patch` Publication Path**:
  - `.github/workflows/publish-product.yml` and `.github/scripts/collect_product_benchmarks.py` add an explicit `release-kind=patch` mode restricted to genuine maintenance patches (`PATCH > 0` with no historical `value-suite-path`), while keeping the full `milestone` product value gate mandatory for `0.Y.0` feature milestones.
  - Under `release-kind=patch`, the pipeline never relabels historical `0.1.0` 109-trial reports as new `0.1.1` experiments, while still enforcing formal component GitHub Releases, immutable tag SHAs, package versions, per-repository `mindie-release-benchmark/2` reports, SHA256 digests, and exact component/version/SHA alignment between `channels/0.1.json` and `benchmark-manifest.json` (emitting `release-channel.json` and recording `channel_sha256` inside `product-release-provenance.json`).
- **Repository-Authored Community Release Notes**:
  - `.github/workflows/publish-release.yml` adds `release-notes-path` so formal component GitHub Releases publish reviewed repository release notes (bundled as `release-notes.md` and included in `SHA256SUMS`) rather than defaulting to raw benchmark tables or auto-generated commit lists.

---

## 2. Verification & Historical Data Boundaries

- **Verifier Soundness & Negative Controls**: All six repaired task verifiers passed `before_reward=0, after_reward=1, model_calls=0` soundness checks and nonfunctional negative-control tests (**`7 passed`**).
- **Release Pipeline & Collector Verification**: `mindie-bench #27` (merged at commit `484b80621aaf9293c559786d0ef2174c79a0dbb4`) passed cross-platform CI; local preflight and collector tests executing real Git tag checks passed **`30 passed` in `18.12s`**.
- **Historical `0.1.0` Evidence Boundary**: The 109 historical `0.1.0` trials have not yet been re-executed or rejudged under the repaired behavior verifiers. Consequently, historical `0.1.0` quantitative benefit claims remain suspended as current product acceptance evidence until rejudgment is completed.
