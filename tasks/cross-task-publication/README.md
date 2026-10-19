# Publication outcome transfer

This fixed CPU cohort tests one captured task against two later repair tasks.
It uses three existing historical business boundaries, not renamed examples of
one input:

| Task | Historical boundary |
| --- | --- |
| `publication-export-seed-v2` | `build_batch` must report rejected entries while retaining unrelated valid staging. Base `e65b05c`, fix `23b1516`, limited to `export.py`. |
| `publication-submit-eval` | `PublicationWorker.submit` must return its transport result and preserve it when later local cleanup fails. Base `23b1516`, fix `b1d8d92`. |
| `publication-flush-eval` | Outer `flush` must retain confirmed or genuinely unknown transport outcomes across quarantine, local receipt and cleanup errors without replay. Same historical revisions, different entry and composite contract. |

The two eval contracts reuse the existing corpus-first judges. Each task starts
from its own unmodified source snapshot. The seed gold patch contains no
`publication_worker.py` changes; eval bodies, oracles and solutions never enter
seed capture. The runner shares only the seed's naturally captured knowledge
with both evals, and never passes the first eval's capture to the second.

The fixed plan is three repetitions of G1, G2 and G3 with the same Codex model,
`home/codex/gpt-6.1-sol`, reasoning effort `xhigh`, 35 foreground calls and a
600-second foreground limit. Each group also performs the seed task, allowing
both eval-only and complete seed-plus-two-eval workflow accounting. G3 includes
all natural capture/indexing costs. Native summary uses the production small
model; its separately metered calls are not free.

G1 has the same remote tools and ordinary source/debugging access. G2 adds the
fixed historical corpus, including the existing analogical outcome-preservation
package. G3 starts from that same scope and adds only its own seed capture.
Retrieval and exact explain observations establish what was actually read.
Success rates, time, tokens and missing dollar pricing remain separate.

This is one historical publication failure family with fixture transport around
real Store/SQLite/worker behavior. It does not measure a live GitHub publication,
NPU performance, independent component benefits, or general cross-project
transfer. A ceiling result or a capture failure remains part of the result.

Seed v2 states the caller-facing exception attributes required by the judge.
The original seed omitted this public interface while checking it privately;
its recorded attempts are excluded from repair success rates, with their cost
retained separately. Eval task definitions and digests are unchanged.

Materialize with the existing entry:

```sh
uv run python tasks/corpus-first-l2/materialize.py \
  --templates tasks/cross-task-publication \
  --corpus /owned/fixed-corpus \
  --knowledge-repo /owned/knowledge \
  --codex-repo /owned/codex \
  --python /owned/cpu-judge/bin/python \
  --gitleaks /owned/gitleaks \
  --output /owned/knowledge-transfer/tasks
```

Run `mindie-bench soundness --promote` on each compiled task before model
admission. Then use the existing `mindie-bench run` command with `--groups
G1,G2,G3 --repetitions 3 --concurrency 2`. Final report directories retain the
exact bound config, source identities, planned cells, soundness, raw receipts,
and comparisons; task definitions are fixed before the first paid call.
