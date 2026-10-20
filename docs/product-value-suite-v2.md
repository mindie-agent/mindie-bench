# Product value suite v2

`mindie-value-suite/2` is the product-matrix publication protocol. It is
dispatched by the existing `value-collect`, `value-verify`, `write_report` and
component-view interfaces; `mindie-value-suite/1` keeps its original
G0/G1/G2/G3 behavior unchanged.

The suite is declared **before outcomes**. It fixes per-cohort scenarios, task
IDs, product profiles, matched conditions and exact budgets, the formal P1
dependency identity, the final P2 release candidate, and which claims are
required or explicitly supplementary. Product arms are:

| Arm | Meaning |
| --- | --- |
| `P0_bare_host` | native host; no MindIE runtime modules, loaded-empty catalog proof |
| `P1_v002_baseline` | formal v0.0.2 product dependencies and native plugin; common benchmark runner |
| `P2_v003_full` | final v0.0.3 product candidate; ten repository targets |
| `P3_v003_flywheel` | real Seed → capture → C6 organizer → consumer → frozen corpus → independent review → different-family downstream lineage |

These labels are never translated into legacy `G0..G3` labels. A receipt's
internal `ablation_group` is only an execution capability mode; the product arm
comes from `experiment_context.cell`.

The checked-in runnable example is `tests/fixtures/product-value-v2/`, executed
by `tests/test_value_product_example.py`. The schema-specific tests are:

```sh
uv run pytest -q tests/test_value_product.py tests/test_value_product_binding.py \
  tests/test_value_product_views.py tests/test_value_product_example.py \
  tests/test_publish_product_workflow.py
```

## One plan per cohort

Different scenarios have different real task budgets. The suite therefore
declares **cohorts**, each with its own tasks, conditions, budget and P0/P1/P2
profiles. `build_product_plan` creates one real `ProductMatrixPlan` per cohort;
the adapter never squeezes two scenarios into one global budget.

Business completion and resource compliance are separate observations. A verified
artifact keeps its result when the agent stops at a known limit or total delivery
time exceeds its declaration. Reports retain measured delivery ratios, including
slowdowns, and publish successful overruns alongside resource measurements.
Matched allocations do not imply that actual spending complied: comparisons
report observed model-call, NPU-run, delivery-time, intervention and declared-cost
overruns, plus missing meters. Missing meters cannot establish compliance.
`claim_ready` describes complete comparison evidence, not success within budget.

```json
{
  "schema_version": "mindie-value-suite/2",
  "id": "product-value-0.0.3",
  "product_version": "0.0.3",
  "protocol": "matched-product/1",
  "repetitions": 1,
  "benchmark": {
    "version": "0.4.0",
    "revision": "<40-hex common mindie-bench commit>",
    "source_sha256": "<64-hex loaded runner source>"
  },
  "tasks": [
    {
      "task_id": "fused-rmsnorm-gate-dispatch-performance",
      "task_family": "performance",
      "hardware_tier": "A2",
      "task_digest": "<64-hex actual task tree>",
      "oracle_digest": "<64-hex actual oracle>",
      "execution_tier": "l3",
      "role": "eval",
      "source_budget": {
        "max_turns": 40,
        "max_npu_runs": 8,
        "timeout_sec": 1800.0,
        "verifier_timeout_sec": 600.0,
        "max_wall_clock_sec": 3361.65,
        "human_interventions": 0
      }
    }
  ],
  "cohorts": [
    {
      "id": "fused-shared-budget",
      "scenario": "fixed_case_optimization",
      "task_ids": ["fused-rmsnorm-gate-dispatch-performance"],
      "repetitions": 1,
      "required": true,
      "functional_claim": {"group": "P2_v003_full", "minimum_passes": 1},
      "profiles": ["<P0 profile>", "<P1 profile>", "<P2 profile>"]
    }
  ],
  "comparisons": [
    {"id": "fused-p0-p1", "cohort": "fused-shared-budget", "baseline": "P0_bare_host", "candidate": "P1_v002_baseline", "required": true},
    {"id": "fused-p1-p2", "cohort": "fused-shared-budget", "baseline": "P1_v002_baseline", "candidate": "P2_v003_full", "required": true}
  ]
}
```

The checked-in fixture also demonstrates the second cohort with the actual
2401.65 s idle-GDN budget. A required cohort must declare P0→P1 and P1→P2
comparisons and a predeclared functional claim. The gate checks completeness,
matched conditions, task functionality and accounting; it never requires a
speedup > 1. Slower/flat PASS results are publishable. All-FAIL required
outcomes fail the functional claim and remain a bounded negative result, not a
release-quality success.

## Separated source identities

Ten release repositories are **not** ten loaded Python modules. The actual
runtime exposes five Python modules from
`integrations/product_native.py:identities()`:

`knowledge`, `remote-dev`, `diagnostics`, `npu-top`, `coordinator`.

For P1/P2, `product_revision` and `product_artifact_sha256` are the runnable
DSH native-plugin revision and artifact, not this documentation repository's
commit or an archive containing value evidence. Each profile source separates:

- `runtime_modules`: the five actual installed distribution versions and
  direct-url Git commits;
- `product_dependencies`: the formal/final product subset (for 0.0.2 this is
  `knowledge`, `remote-dev`, `diagnostics`);
- `instrumentation`: the common subset (`coordinator`, `npu-top`), required to
  match across P1/P2 in a cohort;
- `native_plugin`: the actual DSH name, version and revision;
- `benchmark`: the common benchmark runner version/revision/source hash;
- `catalog`: the loaded catalog units and hash (P0 must be zero);

```json
{
  "source": {
    "product_version": "0.0.2",
    "product_revision": "<40-hex>",
    "product_artifact_sha256": "<64-hex>",
    "runtime_modules": {
      "knowledge": {"version": "0.12.2", "revision": "<40-hex>"},
      "remote-dev": {"version": "0.10.1", "revision": "<40-hex>"},
      "diagnostics": {"version": "0.4.3", "revision": "<40-hex>"},
      "coordinator": {"version": "0.5.3", "revision": "<40-hex>"},
      "npu-top": {"version": "0.1.9", "revision": "<40-hex>"}
    },
    "product_dependencies": ["knowledge", "remote-dev", "diagnostics"],
    "instrumentation": ["coordinator", "npu-top"],
    "native_plugin": {"name": "dsh", "version": "0.0.2", "revision": "d3a1e1282356b65b51fba49072c1842918746f8f", "contribution_enabled": false},
    "benchmark": {"version": "0.4.0", "revision": "<40-hex common commit>", "source_sha256": "<64-hex>"},
    "catalog": {"schema": "mindie-catalog/1", "units": 3, "sha256": "<64-hex>"},
    "catalog_sha256": "<64-hex>",
    "preparation_id": "current-p1"
  }
}
```

`preparation` retains preparation metadata by id. An entry with
`"status": "not_usable"` must not back a current profile. The historical
`p1-runtime-identities-observed` artifact is marked not usable because its
coordinator `0.5.2/daf7fa38` differs from formal 0.0.2
`0.5.3/34812c2` and its bench `0.2.0/fe87041` is not the common runner. Current
preparation is required. Common instrumentation revisions must match across P1
and P2 in a cohort; P1 product dependencies and native plugin must retain the
formal 0.0.2 identity.

## Formal P1 and candidate P2

`formal_baseline` requires a real published release provenance
(`kind=formal_release`, public release URL, `published_at`, source-manifest
SHA-256) and its exact ten-repository release manifest. `final_release` is an
**immutable product release candidate**, not an already-published main product.
Its `revision` and `artifact_sha256` are the **runnable DSH native plugin**
revision and artifact, exactly analogous to P1's formal `d3a1e128` plugin
artifact; they are **not** the documentation repository commit and not an
archive containing `value-evidence.json`. The exact ten-repository combination
is bound separately by `combination_manifest_sha256`.

```json
{
  "formal_baseline": {
    "product_version": "0.0.2",
    "revision": "d3a1e1282356b65b51fba49072c1842918746f8f",
    "artifact_sha256": "<64-hex formal plugin artifact>",
    "status": "formal_release",
    "provenance": {"kind": "formal_release", "tag": "v0.0.2"},
    "native_plugin": {"name": "dsh", "version": "0.0.2", "revision": "d3a1e1282356b65b51fba49072c1842918746f8f", "artifact_sha256": "<64-hex>"}
  },
  "final_release": {
    "product_version": "0.0.3",
    "status": "product_release_candidate",
    "tag": "v0.0.3",
    "revision": "<40-hex runnable DSH plugin revision>",
    "artifact_sha256": "<64-hex runnable DSH plugin artifact>",
    "combination_manifest_sha256": "<64-hex exact ten-repository + plugin digest>",
    "provenance": {"kind": "product_release_candidate", "tag": "v0.0.3", "source_manifest_sha256": "<same combination digest>"},
    "product_dependencies": ["knowledge", "remote-dev", "diagnostics"],
    "instrumentation": ["coordinator", "npu-top"],
    "native_plugin": {"name": "dsh", "version": "0.0.3", "revision": "<40-hex>", "artifact_sha256": "<64-hex>"},
    "repositories": ["<ten exact candidate targets with version/revision/artifact_sha256>"]
  }
}
```

The combination digest is computed over
`{product_version, native_plugin{name,version,revision,artifact_sha256},
repositories[...]{repository,component,version,revision,artifact_sha256}}`
(sorted by component). The DSH repository target's revision/artifact must equal
the native plugin identity, and `final_release.revision`/`artifact_sha256` must
equal that plugin identity. A candidate provenance must **not** claim a
`published_at`. Already-published native plugins/components keep their own
formal provenance where one exists. The main product tag is checked separately
during CI publication; the value suite does not demand an already-published
v0.0.3 or a tarball that contains the value evidence itself.
P1 and P2 must have distinct product versions, plugin revisions and plugin
artifact hashes. P2's `mindie-bench` repository must equal the common benchmark
revision.

## P3: private verification once, public verification later

Local `value-collect` loads `p3.lineage_manifest`, resolves relative locators
against the suite directory, and calls `P3LineageBinding.prepare` with
`verify_artifacts=True` on the original local Seed/capture/C6/review/corpus/
receipt bytes. Only after that succeeds does it write
`value-evidence.json:p3_public_lineage`, which contains:

- the locator-free `content_identity_sha256`, manifest/admission hashes and
  public admission (target, stages, cost ledger, canonical families, timeline,
  limitations, claim readiness);
- path-free public projections of the seed, capture, consumer and review
  receipts with their original SHA-256 values; consumer raw-capture/raw-trial
  references are republished as explicit SHA-256 fields only, never nested
  `{path, sha256}` objects;
- the downstream task hash and frozen-corpus tree hash;
- a `public_lineage_sha256` over the projection itself.

`value-verify` calls the public loader only. It recomputes the projection hash,
checks content identity, target, admission status/readiness, receipt/stage
hashes, cost-role receipt hashes, corpus hash, source identity and P2 matching
conditions/budget. It **never reopens private local originals** and never falls
back to the suite's private locator. A missing required public lineage blocks
the P3 claim; a supplementary one is a scoped limitation only. Seed/capture/C6/
review cost is reported separately from downstream cost and counted once.

## Historical attempts

Selected cells remain strict: current plan, profile, runtime manifest, exact
cohort budget and common benchmark runner. Historical/excluded attempts are
retained separately. `source_receipt_sha256` is SHA-256 of the **exact original
receipt bytes**; `canonical_receipt_sha256` separately identifies the parsed
payload. Historical records retain runtime manifest hash, old benchmark runner
identity, old component revisions and measured costs. They are not forced into
current pins and do not replace valid outcomes. A retry replacement may not
erase an UNKNOWN; missing costs remain null and incomplete.

## CLI and workflow input

```sh
uv run mindie-bench value-collect \
  --suite docs/releases/0.0.3/value-suite.json \
  --manifest /path/to/value-inputs.json \
  --output /tmp/product-value

uv run mindie-bench value-verify \
  --suite docs/releases/0.0.3/value-suite.json \
  --evidence /tmp/product-value/value-evidence.json \
  --release-manifest docs/releases/0.0.3/benchmark-manifest.json \
  --output /tmp/product-value-verified
```

`publish-product.yml` accepts optional `value-suite-path`. Empty keeps
`release-tool/benchmarks/product-value-v1.json`; non-empty is an exact relative
path inside the checked-out product tag, e.g.
`docs/releases/0.0.3/value-suite.json`. The selected file is copied to
`value-suite.json`, its SHA/schema/path is persisted in
`value-suite-identity.json`, and the same file is used for verification,
attachments and uploaded formal assets. There is no existence-based fallback.
Legacy `product-value-v1.json` remains an additional v1-compatible asset only.

### Stable public suite identity

`value-collect` distinguishes:

- `private_suite_sha256`: SHA-256 of the original private suite file bytes;
- `suite_identity`: SHA-256 of the **public canonical suite** written to
  `value-suite.json` after locator/path redaction, profile references abstracted
to path-free public profile identities, and safe evidence pointers.

Evidence/results and CI use the public canonical `suite_identity`. The private
suite SHA is retained only as provenance. This makes the flow
private suite → `value-collect` output suite/evidence → CI verify against the
**output public suite** work after original private files are unavailable.

For P3, the public suite removes `p3.lineage_manifest` entirely (path or inline
object) and carries `manifest_content_identity_sha256` plus
`public_lineage_evidence: "value-evidence.json:p3_public_lineage"`. It never
republishes private receipt/corpus locators. Public CI recomputes public
receipt numeric values and totals against the public cost ledger, verifies
Seed/Heldout canonical family distinction and receipt family hashes, checks
timeline stage order and absolute timestamps when available, and verifies the
consumer raw receipt SHA fields without reading private originals.

## Input manifest

`tests/fixtures/product-value-v2/value-inputs.json` is a complete example:

```json
{
  "schema_version": "mindie-value-inputs/2",
  "product_version": "0.0.3",
  "runner_release": {"version": "0.4.0", "revision": "<40-hex>", "source_sha256": "<64-hex>"},
  "component_revisions": {"<ten final candidate revisions>": "<40-hex>"},
  "runtime_manifests": ["runtime-<profile>.json"],
  "receipts": ["receipt-<n>.json"],
  "excluded_attempts": [{"reason": "...", "path": "historical.json", "replacement_trial_id": null}],
  "component_reports": [],
  "native_plugins": [],
  "setup_reports": [],
  "limitations": []
}
```

Each runtime manifest carries `profile_id`, `product_group`,
`config_sha256`, `derived_inputs` conditions/budget/task hashes,
`runtime_snapshot` with actual loaded module versions/commits, `native_plugin`
name/version/revision, and `catalog`. Each receipt carries the standard
`mindie-bench-receipt/1.0` fields plus
`environment_fingerprint.experiment_context.cell` and its actual runtime
identity. P3 adds `experiment_context.p3_admission` on the P3 trial receipt.

The public report recomputes from these projections: delivery time including
preparation/verifier/cleanup, model calls, token completeness, NPU runs,
manual interventions, budget/business verdict, irreversible unknown, exact
cohort budgets and actual profile runtime/source identities. Endpoint
addresses, commands, local paths, credentials and native logs are omitted.
