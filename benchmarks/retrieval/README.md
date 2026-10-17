# Frozen positive-target retrieval questions

This is a small, manually authored **L0 component-retrieval** set over real
published knowledge material. It contains 12 engineering questions from 12
different `source_package` values in `tasks/candidates/`: six `paraphrase`
questions and six `symbol` questions.

It is **not** the original twelve-task L3 flagship, twelve independently
verified defects, a routing evaluation or an engineering-completion benchmark.
Historical material can contain proposals, failures and uncertain acceptance;
its presence in this set does not certify its engineering claims or endorse
historical VAWS entry points as current MindIE Agent instructions.

The source corpus is fixed at:

```text
cec3249ccd75e7a73cf2f2f776117e0ec385795b
```

The initial frozen `labels.jsonl` SHA-256 is:

```text
2343242780393a89e5b37f9b1b419da091764c535a281d770b018a9663c772c0
```

## Label construction and scope

The author read each package's navigation and the supporting passages in one
actual body block before writing its question. Each question has exactly one
known supporting target block; the navigation alone was not sufficient evidence.
For example, the updater question follows the actual body's
`prepared_source(root)`/dirty-checkout/identity-pending passage rather than
assuming that every topic in the navigation summary appears in that block.

All 24 cited source files were read and compared byte for byte with the fixed
Git blobs. The body SHA-256 values also matched their navigation declarations.
The questions are manually paraphrased engineering requests or contextualized
symbol questions, rather than package titles, block IDs or hashes as queries.
Their language is Chinese with original code/configuration symbols retained.
The six/six split describes this set, not a representative language distribution.

The selected passages support the following questions:

| Sample | Kind | Body support read before labelling |
| --- | --- | --- |
| `retrieval-positive-001` | paraphrase | Synchronization recovery had carry-over into the trailing control; reset/miss/reset and re-armed cases are needed for attribution. |
| `retrieval-positive-002` | symbol | `cu_seqlens[-1] == q.shape[1]` rejects real tokens inside a larger graph capacity, leaving `empty_like` output and active state unwritten. |
| `retrieval-positive-003` | paraphrase | The first nonfinite point is layer-10 KDA; raw beta versus backend-owned sigmoid differs across the migration. |
| `retrieval-positive-004` | symbol | The reduced DP2/TP2 GQA case changes heads and head dimension to match physical MLA/draft KV pages. |
| `retrieval-positive-005` | paraphrase | Optional stream, host buffer and rejected-token counts need local resource binding/grouped initialization assertions for type narrowing. |
| `retrieval-positive-006` | symbol | `prepared_source(root)` protects business/dirty/diverged checkouts, and `github_identity_required` remains a pending update boundary. |
| `retrieval-positive-007` | paraphrase | Target MLA, recurrent state and draft FullAttention share the prefix-hit boundary; missing draft groups lower the hit and tail replay rebuilds draft KV. |
| `retrieval-positive-008` | symbol | Session, reusable Environment and Run + Lease have separate lifetimes; cached monitoring is not allocation authority. |
| `retrieval-positive-009` | paraphrase | Verify the actual custom plugin entry/loading path and use an independent FP32 paged-KV reference instead of continuous-output self-comparison. |
| `retrieval-positive-010` | symbol | The broad `QuantType.NONE or SituActivationConfig` fallback affects unrelated nonquantized MoE; restrict the SiTU adaptation scope. |
| `retrieval-positive-011` | paraphrase | A missing upstream label permission leaves selected hardware tests untriggered, despite completed static code checks. |
| `retrieval-positive-012` | symbol | NPU `max().item()` adds a per-forward D2H synchronization; the existing CPU maximum can be reused, without proving all latency differences have that cause. |

Every row declares:

```json
{
  "sample_id": "retrieval-positive-001",
  "query": "An engineering question supported by the selected body",
  "relevant_task_ids": ["SOURCE_PACKAGE_ID"],
  "relevant_block_ids": ["SOURCE_PACKAGE_ID:BLOCK_ID"],
  "query_kind": "paraphrase",
  "source_commit": "cec3249ccd75e7a73cf2f2f776117e0ec385795b",
  "source_files_sha256": {
    "tasks/SOURCE_PACKAGE_ID/index.md": "SHA256_OF_NAVIGATION_BYTES",
    "tasks/SOURCE_PACKAGE_ID/blocks/BLOCK_ID.md": "SHA256_OF_BODY_BYTES"
  },
  "label_scope": "positive_only_target_recovery"
}
```

`query_kind` is `paraphrase` or `symbol`. Paths are relative to the corpus,
not to this benchmark repository. A block identity includes its source package
to avoid conflating equal block IDs across packages.

The relevance labels are deliberately **non-exhaustive**. Other corpus blocks
may also answer a question; unlabelled blocks are unjudged, not negatives.
Report known-positive target hit fractions and target reciprocal rank with the
actual sample count and kind split. Do not present these as global semantic
Recall, comprehensive relevance ranking, no-hit abstention, trigger recall,
task-solving improvement or NPU acceptance. A successful target retrieval also
does not prove that the agent understood or applied that historical advice.

## Freeze and execution contract

Labels were frozen before any component query or ranking run in this task.
No model, SSH or NPU operation was used to create the set. No labels were
selected or changed after inspecting retrieval scores.

The component runner should query the complete fixed corpus, rather than an
index containing only these 12 target packages, and record its actual package
count. Keep the first raw result for every sample, including misses and
errors. Preserve the labels hash, corpus commit, exact engine revision and
configuration, result refs/order, timings and source-hash verification with the
result. A source mismatch is an error, not an empty result or a reason to
silently substitute a different corpus.

Do not tune this frozen revision after seeing results. A later correction
requires a new labelled revision and an explanation while preserving the old
first result; a corrected dataset does not retroactively improve the first run.
This small author-judged positive set is diagnostic; a broader quality claim
needs independent relevance review, more queries, ambiguity adjudication,
negative/no-hit conditions and a separate held-out evaluation.

## Design-principles review

Reviewed against the current nine parent principles, revision
`240248b18996904490cca1e1ecceb8d9ecd83e03`: this finite zero-model set lowers
measurement cost (1); it measures the closed retrieval-target problem (2);
uses one JSONL contract with source identity (3); adds no runner or lifecycle
state (4); does not prescribe agent troubleshooting steps (5); treats historical
advice as fallible (6); neither forces foreground retrieval nor activates it in
routine work (7); reuses existing real corpus/package formats (8); and preserves
missing targets, failures, hash mismatches and first-run results visibly (9).
