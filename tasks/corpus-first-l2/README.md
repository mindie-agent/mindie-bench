# Corpus-first real-source CPU protocol tasks

These six reusable recipes produce four evaluation tasks and two distinct seed
tasks from actual local Git history. They execute complete product modules,
SQLite state and, for overlap redaction, a real Gitleaks executable. They do not
extract target functions with AST or generate business source from a narrative.
The public reproduction and hidden judge are separate programs.

| Task | Role / knowledge condition | Actual historical contract |
| --- | --- | --- |
| `l2-placeholder-tree-seed` | seed / near_exact | Closed generated privacy tokens inside nested document keys and values are falsely flagged by the actual `scan_tree` path. |
| `l2-generated-placeholder` | eval / near_exact | Closed scalar tokens must survive `scan_text`; malformed tokens, outside text and technical slices remain independently checked. |
| `l2-overlapping-redaction` | eval / near_exact | Source redaction must cover the union of overlapping claims rather than retaining a longer sensitive tail after masking an inner identifier. |
| `l2-publication-cleanup-seed` | seed / analogical | Actual `PublicationWorker.submit` must retain its returned transport result when owned cleanup raises, and return it on the normal path. |
| `l2-publication-result` | eval / analogical | The outer `flush` path must preserve a confirmed result across mixed quarantine/receipt/cleanup failures and keep genuine uncertainty visible without replay. |
| `l2-summary-model-catalog` | eval / no_hit | The actual summary provider must project the selected metadata record, bind its policy hash and reject ambiguous/incomplete component records. |

The two G3 pairs cover different entry points and fault boundaries, rather than
renamed inputs. The first pair shares `cluster=l2-placeholder`; the second
shares `cluster=l2-publication`. Only these four paired tasks belong in a G3
run; the standalone overlap/catalog evaluations have no fabricated seed.

These are four repair contracts plus two same-family transfer triggers, not six
independent defects or additional Ascend operator families. The placeholder and
overlap defects are in the same upstream fix commit. Fabricated privacy strings
and transport receipts are test inputs for real historical bugs, not real user
secrets or actual remote publications.

## Frozen source and corpus evidence

The knowledge repository revisions are:

```text
base: e65b05c795e2c6e7073f331565df712c853d761f
privacy fix: 23b1516907edfcccf55243f141f414f1d4b8d535
publication base: 23b1516907edfcccf55243f141f414f1d4b8d535
publication fix: b1d8d9278a60394f0de28aa2986d02e0ecb8b115
```

The Codex adapter uses base
`4e4efcaf1d6511f735bc407c3b1927268ccb7806` and the local component fix
`1765180`. Each recipe records the full resolved fix SHA, original complete
module hashes and the externally supplied gold diff. An unmerged local fix is
not a published component Release.

Source material is bound to public corpus commit
`cec3249ccd75e7a73cf2f2f776117e0ec385795b`. The privacy tasks reference the
actual `a8366fec…` block describing overlapping-rule leaks and repeated
placeholder false positives. The publication tasks use the `dd069918…` body
describing completed-business/local-recording failures as **analogical**
guidance, rather than claiming it contains the exact later publication patch.
Recipes contain complete package/block IDs, body hashes and actual supporting
text. These useful MindIE histories are outside the original twelve mined NPU
dossiers; the selection was broadened within the same fixed public corpus.

These labels describe the evidenced relationship to the original source
corpus. They do not prove that a particular foreground profile mounted that
body. The existing cold-profile builder excludes `same_origin_task_hashes`;
its normal G2/seed configuration therefore removes the privacy origin
package. Report the actual filtered corpus and activation protocol rather
than calling that mounted condition an available near-exact answer. G3 may
then add only its own independently captured seed package, with that new
package's exact lineage checked separately.

The catalog task's `no_hit` condition is an explicitly **empty mounted
knowledge scope**, identified separately from the source corpus. It does not
assert that keyword search proved all public packages irrelevant. The operator
must freeze that empty corpus, include its actual commit/hash in the run
manifest, and mount it as `l2-no-hit`. Using the full corpus changes this
condition and cannot retain a no-hit claim.

There is no certified **stale** or **compositional** condition here. A stale
task needs an actually read historical recommendation, its old version
conditions, a current-version counterexample and a source-fail/gold-pass proof.
Ordinary old source, missing knowledge or a chosen label is insufficient.
This remains a concrete dataset gap.

## Materialize a fresh task revision

Use the benchmark repository's existing locked environment:

```sh
uv sync --locked
uv run python tasks/corpus-first-l2/materialize.py \
  --corpus /owned/fixed-public-corpus \
  --knowledge-repo /owned/knowledge-source-repository \
  --codex-repo /owned/codex-adapter-source-repository \
  --python /owned/locked-python/bin/python3 \
  --gitleaks /owned/gitleaks \
  --output /owned/new-l2-task-revision
```

The exact source commits must already exist in these local repositories.
The materializer does not fetch, install dependencies, read personal client
profiles or write to the source repositories. It binds the supplied Python and
Gitleaks binary hashes, observes Python/PyYAML versions, verifies every tracked
asset hash, and calls the existing `compile_recipe(corpus, recipe_manifest,
output)` API. Interpreter/library read grants must cover that exact Python
environment, its stdlib and Gitleaks; do not grant an entire source-parent
directory or expose the fixed component checkout.

Outputs are `<instance_id>/recipe.json`, `compile-receipt.json` and
`compiled/{task.toml,instruction.md,environment,verifier,oracle}`. Templates are
candidate definitions; materialization alone never makes them verified.
The gold patch, replay metadata and external judge stay in `oracle`/`verifier`;
only `environment` is copied into the agent's source snapshot. No gold source
or judge is copied into public assets, installed site-packages or a public
source archive.

All recipes intentionally set `replay_commands=[]` and declare replay disabled.
The actual CPU judge executes the historical source; no text replay substitutes
for it. A future NPU replay needs separately frozen physical evidence and its
own acceptance scope.

Validate and prove each fresh compiled task using the existing CLI:

```sh
uv run mindie-bench validate \
  --tasks /owned/new-l2-task-revision/INSTANCE/compiled
uv run mindie-bench soundness \
  --task /owned/new-l2-task-revision/INSTANCE/compiled \
  --config /owned/explicit-local-config.toml --tier l2 \
  --output /owned/new-l2-task-revision/INSTANCE/soundness --promote
```

Each proof must actually observe base reward 0 and patched reward 1 under the
external judge. A non-business runtime/scanner failure remains infrastructure;
do not turn it into reward 0. Changed interpreter, assets, paths or source
bindings change task identity and require new proof.

An optional `--proof-root /owned/previous-revision` copies only receipts at
`INSTANCE/compiled/soundness_receipt.json`. Compilation accepts them only if
their complete task hash, source commit, L2 tier, zero-model proof and 0→1
outcomes match exactly. It does not re-execute a task or repair an unknown
result. Mismatches fail visibly and preserve the attempted output.

Freeze task/config/runtime/corpus identities and a complete planned-cell
inventory before any model cohort. Use G0/G1/G2 for the four eval contracts;
use the two explicit pairs for G3, account for all required seed cost, and
retain failed/unknown/blocked attempts. These fixed normal budgets were not
tuned to obtain a desired G0 pass-rate band.

## Actual validation and limits

All six private source-bound revisions were independently executed under the
actual local verifier sandbox: base reward 0, gold reward 1, no model/SSH/NPU
calls and no recorded cleanup error. Initial generator/judge failures were
preserved before corrected fresh revisions. The formatted tracked templates
have their own fresh proof identity; they do not rewrite the earlier frozen
revision used by an independently launched cohort.

Earlier frozen cohort instructions incorrectly addressed the flattened public
reproduction as `environment/reproduce.py`; canonical instructions now use
source-root `reproduce.py`. The earlier catalog instruction also omitted the
selected-metadata policy-identity requirement exercised by its judge. The
current instruction states that requirement explicitly. Old results retain
those presentation/specification limitations; their receipts and judge are
not rewritten or retroactively rescored as results of the corrected task.

These proofs establish local CPU privacy/protocol behavior. Publication
transport is an explicit fixture around real worker/Store behavior, not a
GitHub write. Metadata inputs exercise component projection and policy
identity; this task does not certify the complete native CLI model schema or
make an inference request. Neither task acceptance, capture delivery nor a
corpus hit alone proves knowledge reuse, causal improvement, NPU correctness,
HCCL, graph compilation, whole-model precision or performance.

The original six Ascend function tasks and their hardware evidence remain a
separate smoke suite. L0's twelve positive retrieval questions are also a
different dataset. None is silently renamed the original twelve-task flagship.

## Nine-principle review

Reviewed against parent principles revision
`240248b18996904490cca1e1ecceb8d9ecd83e03`: use low-cost real CPU failure
reproductions (1); compile only explicitly supplied closed inputs (2); provide
one materialization entry with existing validation commands (3); keep mining,
compilation, execution and scoring separate (4); add no mandatory Skill steps
(5); preserve advisory history and missing stale evidence (6); permit explicit
knowledge activation without forcing every task to retrieve (7); reuse Git,
the locked environment and the actual existing verifier (8); and preserve
source/hash/runtime errors, partial business results and unknown effects
without automatic replay (9).
