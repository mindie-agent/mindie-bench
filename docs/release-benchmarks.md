# Benchmarks on every release

Every component and native plugin owns a small `benchmarks/release.json` suite and
its workload script. `mindie-bench release-run` executes those workloads against
the candidate checkout and writes these public assets:

- `benchmark-results.json`: exact repository revision and version, dirty state,
  suite hash, environment, status of every attempt, raw metric samples, units and
  iteration counts.
- `benchmark-results.md`: readable results and explicit publication blockers.
- `benchmark-SHA256SUMS`: checksums of both public reports.

The `private/` directory holds raw subprocess logs and unfiltered measurement
files. It is never included in CI artifacts or release uploads. Do not attach
the whole output directory. Public scripts must emit only numeric measurements
and fixed reason identifiers, without user data, prompts, paths or credentials.
The suite's `publishable: true` is a repository-owned declaration of that scope,
not an automatic content scanner.

## Repository workload contract

See `benchmarks/release.json` and `benchmarks/release.py` for a working CPU
microbenchmark of production retrieval scoring. Each command runs in the
candidate checkout. `{python}` resolves to the explicitly selected interpreter;
install locked dependencies once before invoking the runner.

Each invocation writes `MINDIE_BENCH_RESULT_PATH`:

```json
{
  "status": "pass",
  "metrics": [
    {"name": "parse_latency", "unit": "us/op", "value": 12.3, "iterations": 1000}
  ]
}
```

The runner executes the declared repetitions, preserving every raw sample. A
script can warm up and batch operations internally; `iterations` records the
number of measured operations underlying the reported value. Use fixed
workloads and check their business results before reporting success.

Case kinds have distinct meanings:

- `functional`: contract/acceptance outcome, without a performance claim.
- `microbenchmark`: latency or throughput of a bounded component operation.
- `e2e`: real host/benchmark execution, retaining the existing MindIE Bench
  receipt and ablation semantics. A single canary does not prove productivity.

Statuses are `pass`, `fail`, `unknown`, and `unsupported`. Non-pass payloads
include `reason_code`, for example `hardware_unavailable`. Required non-pass
cases block publication. Optional unsupported cases remain visible.
Non-pass attempts are never automatically repeated. A local timeout fails the
case; `unknown` on timeout is reserved for a declared `external_write` whose
remote effect cannot be determined. Do not run paid or mutating suites without
the normal workload authorization.

```sh
mindie-bench release-run \
  --repo /path/to/component --repository mindie-agent/component \
  --version 0.0.2 --suite /path/to/component/benchmarks/release.json \
  --python /path/to/component/.venv/bin/python \
  --output /tmp/component-release-benchmark
```

The report blocks publication for dirty source, a source revision changed during
measurement, or a failed required case. Local development may still measure a
dirty checkout; those results must be rerun after committing the final source.
Do not treat a clean measurement as confirmation that a GitHub tag or released
wheel contains that revision; the paired-release procedure verifies delivery.

## Release workflow integration

The reusable `.github/workflows/release-benchmark.yml` checks out the caller's
candidate, installs its exact dependencies, measures it and uploads only the
three public assets. Call it by a reviewed immutable benchmark commit:

```yaml
jobs:
  benchmark:
    uses: mindie-agent/mindie-bench/.github/workflows/release-benchmark.yml@BENCH_COMMIT
    with:
      version: 0.0.2
      benchmark-ref: BENCH_COMMIT
      setup-command: uv sync --locked --all-extras
      python: .venv/bin/python
      artifact-name: release-benchmarks-linux
  publish:
    needs: benchmark
    # Existing publication job downloads that artifact and attaches all three
    # files to the same draft release as the distribution artifacts.
```

Use a unique artifact name for each platform. Windows callers select their
candidate interpreter (for example `.venv/Scripts/python.exe`) and dependency
setup. Hardware suites require the project's designated standing environment;
ordinary hosted runners do not establish A2/A3 acceptance.

Run the benchmark before making the release public. A separate upload step must
not use `if: always()` to bypass a failed benchmark gate. Failed evidence is
retained as a CI artifact for diagnosis. Public release assets should be uploaded
without overwriting results belonging to an existing version.

## Product collection

The documentation repository aggregates the ten runtime/plugin repositories;
it does not invent a document-rendering benchmark. Its manifest declares the
exact source and evidence required for each release:

```json
{
  "schema_version": "mindie-release-manifest/1",
  "product_version": "0.0.2",
  "repositories": [
    {
      "repository": "mindie-agent/remote-dev",
      "revision": "0000000000000000000000000000000000000000",
      "version": "0.9.6",
      "required_cases": ["rpc-control"],
      "platforms": ["linux", "darwin"]
    }
  ]
}
```

The zero revision is a documentation placeholder; use the actual locked commit.
Collect reports obtained from component release assets:

```sh
mindie-bench release-collect --manifest release-benchmarks.json \
  --input /tmp/remote-dev-linux --input /tmp/remote-dev-macos \
  --output /tmp/product-release-benchmark
```

Missing repositories/platforms/cases, duplicates, failed component gates and
revision/version mismatches produce a blocked JSON/Markdown report and exit 1.
This keeps benchmark delivery routine for future versions without automatically
publishing transcripts or substituting old hardware acceptance for new results.
Absolute latency samples describe the recorded environment; comparisons require
matching workloads and environments. Use the existing controlled `compare`
workflow for causal productivity or ablation claims.

## Product value suite and component attachments

Alongside per-component `benchmarks/release.json` workloads, each product release
evaluates the fixed product value suite in `benchmarks/product-value-v1.json`:

- **Real-task comparisons (`G0` / `G1` / `G2` / `G3`)**: compares native host
  baselines (`G0`), `remote-dev` remote execution (`G1`), `knowledge` retrieval
  with shipped Skills and retrieval guidance (`G2`), and cross-task experience
  capture/reuse (`G3`) per host and model without averaging across models.
- **Component scenarios**: repeated scenarios for `coordinator` (concurrent lease
  and port arbitration), `npu-top` (hardware capture replay and live container
  ownership), and `diagnostics` (local HTTP fault deduplication, redaction, and
  consent withdrawal).

Collect and verify the public value report from recorded trial receipts and
component scenario reports:

```sh
mindie-bench value-collect \
  --suite benchmarks/product-value-v1.json \
  --manifest /path/to/value-inputs.json \
  --output /tmp/product-value

mindie-bench value-verify \
  --suite benchmarks/product-value-v1.json \
  --evidence /tmp/product-value/value-evidence.json \
  --release-manifest /path/to/mindie-agent/docs/releases/0.0.2/benchmark-manifest.json \
  --output /tmp/product-value-verified
```

The output directory contains `value-suite.json`, `value-evidence.json`,
`value-results.json`, `value-results.md`, and the localized homepage blocks
(`value-homepage.md` and `value-homepage.en.md`). All completed passes, failures,
unknown outcomes, historical attempts, and separate setup/diagnostic model calls
remain accounted for in the public report.

Before publishing the product tag, a maintainer attaches the two versioned
component-view files (`product-<version>-value-results.json` and
`product-<version>-value-results.md`) to each component's formal GitHub Release:

```sh
uv run python .github/scripts/attach_product_value.py \
  --suite benchmarks/product-value-v1.json \
  --evidence /path/to/mindie-agent/docs/releases/0.0.2/value-evidence.json \
  --manifest /path/to/mindie-agent/docs/releases/0.0.2/benchmark-manifest.json \
  --output /tmp/product-value-components \
  --publish
```

## Automatic product publication

The documentation repository calls `.github/workflows/publish-product.yml` on
an existing `vMAJOR.MINOR.PATCH` tag and pins both the workflow and its
`benchmark-ref` input to the same reviewed benchmark commit. It contains no
runtime installation or collection code of its own:

```yaml
jobs:
  publish:
    permissions:
      contents: write
    uses: mindie-agent/mindie-bench/.github/workflows/publish-product.yml@BENCH_COMMIT
    with:
      benchmark-ref: BENCH_COMMIT
```

The product tag must contain `docs/releases/<version>/benchmark-manifest.json`,
`docs/releases/<version>/value-evidence.json`, and `docs/releases/<version>.md`.
Before creating the product Release, the workflow:

1. Recomputes the value report from `value-evidence.json` against
   `benchmarks/product-value-v1.json` and verifies that every component revision
   matches `benchmark-manifest.json`.
2. Verifies via `attach_product_value.py --require-existing` that every formal
   component Release already carries matching `product-<version>-value-results.json`
   and `product-<version>-value-results.md` attachments (using standard read
   access without requiring a new cross-repository write credential).
3. Downloads and verifies the public release benchmark assets from each target's
   formal `v<version>` GitHub Release and publishes the aggregated product Release
   with `SHA256SUMS`. Re-running an already published product tag does not
   overwrite that Release.
