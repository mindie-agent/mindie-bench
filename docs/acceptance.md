# Acceptance evidence - 2026-10-16

Reviewed implementation SHA-256: `0d03c31600cce24f80f931de832aa9c69aa39317f7d4c935b0d0c78cb5731c29` (Python source tree).
This is development version `0.1.0`, not a published component or plugin Release.

| Check | Actual result | Boundary |
| --- | --- | --- |
| CPU contracts and integration | 79 passed in 25.89 seconds on this Mac | Includes real Seatbelt processes, mocked provider streams, and synthetic tasks; no live NPU |
| Ruff | Check and formatting passed | Local validation, not GitHub CI |
| Synthetic L2 execution | Three independent G0 trials passed; frozen ledger and identity comparison generated | Zero model calls and zero NPU runs; not a MindIE improvement claim |
| G3 lineage tests | Seed/capture/eval order and repetition isolation passed using a synthetic capture adapter | Production native Stop-hook/capture/indexer integration remains unaccepted |
| Installed Codex | `codex-cli 0.153.4` version check passed with isolated HOME and OS sandbox | No paid native solve was run |
| Installed Claude Code | `2.1.269 (Claude Code)` version check passed with isolated HOME and OS sandbox | No paid native solve was run |
| Public source discovery | 661 packages scanned, 497 matched domain heuristics, 12 candidates recorded | Not 12 verified tasks; each candidate lists missing evidence |
| Redacted export | Existing `mindie-knowledge 0.11.0` redactor plus `Gitleaks 8.30.1` passed | Local-only export of three synthetic receipts |
| Parquet | Three rows read back successfully | Synthetic evidence remains explicitly labelled |
| Static leaderboard | Generated from the same redacted summaries | No hosted Space or browser acceptance claimed |
| Packaging | wheel and sdist built; installed wheel CLI help passed in an isolated environment | No package upload, tag, Release or public PR |
| Distribution privacy | Built archives contain no reports, .git or .venv paths | Raw native traces remain private local artifacts |

The source digests recorded in the final synthetic receipts were read back and
matched the reviewed implementation digest. Previous development failures remain
in their separate ignored report directories; no paid or unknown native trial was
resubmitted.

## Inspectable local evidence

- `reports/release-candidate/run/`: raw private journals, normalized trajectories,
  four-state receipts, final source patches, and `eval_manifest.json`.
- `reports/release-candidate/baseline/`: frozen canonical receipt hashes.
- `reports/release-candidate/comparison.md`: identity comparison, explicitly
  labelled synthetic.
- `reports/native-version-smoke-final.json`: actual installed binary versions
  under the final sandbox policy, with zero model calls.
- `dist/hf-release-candidate/`: redacted JSON, Parquet, static HTML and manifest.
- `dist/mindie_bench-0.1.0-py3-none-any.whl` and the corresponding sdist.

## Unaccepted work and required inputs

1. No real Ascend task has a newly obtained FAIL-before/PASS-after proof. The 12
   source dossiers still need exact business source revisions, independent
   verifiers, gold fixes, transfer labels, device counts, and actual soundness.
2. Native full solves and G0-G3 comparisons need explicitly selected model
   deployments, scoped model API credentials, and installed component adapters
   configured for these fresh profiles. Personal CLI credentials were not read
   or copied. Native version checks are not native product-loop acceptance.
3. L3 requires a specified dedicated evaluation host/container/device pool, a
   validated memory/process/software JSON probe, and real topology acceptance.
   Its worker/probe contracts were tested on CPU; SSH, CANN, HCCL and NPU cleanup
   have not been exercised on hardware in this task.
4. macOS ownership covers the owned process group. Deliberately detached
   sessions are outside that guarantee; Linux PID namespaces provide a stronger
   containment boundary. Native Windows execution is explicitly unsupported.
5. A real corpus retrieval baseline, real routing observations, real summary
   usage, hardware performance scores, and comparative MindIE gains are unmeasured.
   GitHub CI, Harbor integration and Hugging Face hosting/upload are not accepted.

M1 engine, M2 replay, M3 contract/metric tooling, and M4 local export are
implemented. Their proposed verified dataset sizes and promotional NPU results
are not complete. Source historical status and summaries are navigation, not
certification.

Parent design principles: commit `240248b18996904490cca1e1ecceb8d9ecd83e03`, current working-file SHA-256
`642bb77daf460dd9c15931b70983fd972f5f8ee352863c2158432c1ca96b470a`. The parent's pre-existing document edits were preserved.
