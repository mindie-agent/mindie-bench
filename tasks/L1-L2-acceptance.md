# Real observation slices and CPU twins

These are additional evidence layers. An observed query, a CPU counterpart
and a live NPU engineering outcome answer different questions. None may be
silently substituted for another.

## L1 retrieval observations

Use only actual foreground `knowledge_query` / `knowledge_explain` events from
the selected frozen G2/G3 trials. Keep the original event path, raw bytes/hash,
trial/source/task/corpus identity, query text, ordered result refs and explain
responses in private evidence. A consumer script's readback is a separate
component check. Capture/indexer queries are not foreground queries.

First verify mechanical provenance:

1. Each returned package revision/block ref matches the actual Markdown bytes.
2. Each explain call uses the observed query ref and returned block hash.
3. In G3, the evaluation input corpus equals its own seed's captured corpus;
   no other repetition's seed or held-out capture enters it.

These checks establish delivery/reference identity, not retrieval relevance.
For a quality slice, judge relevance against the task's known failing source
and acceptance constraints. Record a relevance inventory before inspecting
result rank: relevant root cause, applicable runtime/version/configuration,
actual actionable evidence, and known stale/contradictory material. Read complete
redacted blocks rather than treating their navigation summary as truth. Preserve
disagreements and unjudged material instead of silently labelling it irrelevant.

A same-family seed block is an expected available reference, not automatically
a semantic gold label. A block that merely repeats the error without an
applicable diagnosis may be mechanically exact but not useful.

The existing scorer accepts JSONL such as:

```json
{"sample_id":"SOURCE/TRIAL/query-ordinal","evidence":"observed-native","ranked_ids":["BLOCK_REF_A","BLOCK_REF_B"],"relevant_ids":["BLOCK_REF_B"]}
```

```sh
uv run mindie-bench score-slices --kind retrieval --k 3 \
  --input /absolute/private-observed-and-judged-queries.jsonl
```

Use consistent IDs and no duplicate ranked entries. Keep task/hardware/group
coverage and raw-reference hashes in a separate judgement manifest; the small
scorer does not validate those facts. Report per-family and per-group results,
with counts of actual queries, observed explains, judged queries and omitted
unjudged cases. Multiple queries from one engineering attempt are correlated;
they are not independent tasks.

Recall requires a defensible relevant-set denominator. If the whole available
corpus has not been judged, do not call a limited seed-reference hit fraction
general corpus recall. An empty relevant set is a genuine no-hit scenario and
is outside this scorer's nonempty-label contract; preserve it in a separate
no-hit/abstention inventory instead of inventing a relevant block.

## L1 applicability and false-trigger observations

Keep separate event definitions for:

- Explicit native Skill invocation.
- Foreground knowledge query/explain.
- Remote execution or NPU test attempt.
- Approved background Stop capture.

Automatic Stop capture in an already authorized project is not a false Skill
trigger. Required MCP initialization is not a foreground knowledge query.
An optional knowledge query's absence is not a product failure merely because
the task is in the Ascend domain.

For a routing slice, state the chosen event and an explicit applicability label
derived from the actual user request and available tool capabilities. Positive
Ascend engineering prompts alone do not provide a false-positive denominator.
Zero observed false positives on zero clear negatives is unmeasured.

Include clear actual negative requests under the same native harness/tool
availability, such as a bounded local Python/text edit that requests no remote
server, NPU or Ascend knowledge. Such an optional additional paid L1 sample
requires a separate finite budget; existing positive traces cannot be relabelled
as negatives to avoid collecting the evidence. Do not start paid samples merely
to populate this document.

The current routing scorer consumes:

```json
{"sample_id":"SOURCE/ACTUAL_REQUEST/event-kind","evidence":"observed-native","expected_trigger":false,"actual_trigger":false}
```

```sh
uv run mindie-bench score-slices --kind routing \
  --input /absolute/private-observed-and-labelled-routing.jsonl
```

Its precision, recall and zero-false-trigger rate are secondary diagnostics.
Publish the number of clear
positives, negatives, ambiguous/unjudged requests and each event definition
alongside them. Do not claim zero false-trigger probability from a small finite
sample, or force knowledge use to satisfy an evaluator's expected label.

## L2 real CPU twins

The partial-RoPE cache and PyTorch reduce-sampler functions can execute on real
CPU PyTorch without torch_npu, custom kernels, model weights or fake NPU calls.
Their original historical source must still fail and the exact upstream gold
patch must pass. The CPU counterpart validates the same Python source contract,
not NPU numerical/backend behavior, AscendC kernels or HCCL.

For a reusable L2 task, create a separately frozen task definition with:

- The same upstream failing/gold source and family identity.
- An actual CPU public reproduction and self-contained CPU verifier, rather
  than pretending an NPU verifier succeeded through a fallback.
- `environment.l2` pointing to the exact source repository/base commit.
- An explicit installed CPU torch interpreter/runtime, locked package versions
  and hashes, isolated imports, and correct OS read grants. An ephemeral cache
  path or unpinned package install is insufficient for a repeatable operator run.
- A new tier-specific soundness proof and task digest; no copied L3 proof.

Run `mindie-bench soundness --tier l2` on those CPU task definitions before any
native model cohort. Real CPU execution should be judged from the resulting
source state. If replay feedback is also enabled, label that surrogate
separately; a text-pattern replay cannot replace the CPU final judge.

The four new L3 task definitions deliberately have no implicit CPU fallback.
Private CPU-counterpart checks during task development are useful feasibility
evidence, but are not promoted L2 tasks or a completed native L2 matrix.

The existing RMSNorm task imports an installed Ascend class and exercises NPU
operators. Replacing its forward with a homemade CPU formula or mocked NPU
operator would not establish the same real-source execution path. Keep that
family out of a claimed real CPU-twin set until an authentic CPU source path
and its own soundness proof exist.

## Minimum reusable delivery

Keep runnable task definitions/public reproducers/private verifiers/gold patches
under `tasks/`, and use the existing `validate`, `soundness`, `run`,
`score-slices`, `compare` and `export-hf` commands. The Ascend entry/runbook is
`tasks/ascend/README.md`. A particular run also needs a source-bound task/config/
image/component/model/corpus manifest, planned-cell inventory and raw-byte
receipt hashes. Private logs are supporting evidence, not the only entry point.

Missing observations, unavailable hardware and unknown paid effects remain
visible gaps. No candidate, synthetic fixture, query count or CPU diagnostic
is promoted to native L1/L2/L3 acceptance by its directory name.
