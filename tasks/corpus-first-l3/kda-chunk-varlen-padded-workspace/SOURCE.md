# KDA native numerical-correctness candidate

This is one historical C++ operator defect, not eight independent tasks.
The complete source tree is the original upstream commit
`9a3a8fdeb248e9fe444fe80353ea3795d1030df0`; the original repair is
`8ddf72b3b697c29b8b0cd98401dd396fa319fd26`. The private oracle retains
only the original host-tiling patch. The complete independent historical test
module is from `f27f1658669bd22f20e8b2ed539b724ef3bbb5e3`, SHA256
`3dea5a1523accc1f7f3bc7102e35cb5e455c7b4eb246f37f1aafebbf6fb48370`.
No AST extraction or synthetic kernel implementation is used.

Real source builds and numerical operator proof were completed on one A3
Ascend910_9362 NPU. Before fails the finite-output assertion for the four
127/128-sequence cases; the four dense controls pass. Gold passes all eight
finite checks and independent FP32 CPU output/final-state comparisons.
Non-finite values were observed; NaN versus Inf was not separately counted.
The proof does not measure hardware kernel invocation count.

The initial task definition is deliberately `candidate`: the earlier
source-build/operator proof does not establish this later task definition's
isolated-worktree soundness, foreground leakage admission or model repair
results. Only an exact matching `soundness_receipt.json` establishes the
later task's own source-fail/gold-pass proof.
The first full-task cold-build attempt also passed source0/gold1, but local
test imports had added ignored bytecode to its task hash. That complete
historical revision and receipt remain private and do not establish clean
checkout acceptance. The canonical loader now avoids creating task bytecode.
The clean revision uses a separately locked generic Protobuf/Abseil dependency;
both KDA host and kernels are still rebuilt from the current source.
The private evidence is
`reports/benchmark-improvement-20261017/kda-workspace-task-build-1/`,
especially `operator-proof-final-readback.json`, the two numerical observed
receipts, the original CPack/install receipts and preserved failed attempts.
Do not copy their private paths, raw logs, gold package or original runtime
vendor into a foreground read grant.

The exact matching runtime and offline-input recipe are in `runtime/README.md`
and `runtime/dependencies.json`. A normal candidate run must rebuild its edited
complete `csrc` tree with `chunk_kda_fwd,kda_layout_swap12`, not reuse a gold
binary. The latter operation is a genuine native dependency, established by
normal ELF installer validation; no symbol stub or forced install is used.

The supplied complete-tree source repository must have only the base commit,
without parent history, alternates or the gold object. A new runtime image
must hide the runtime repository's C++ source/tests and old custom vendor from
the foreground. Its retained Python wrapper/extension is a fingerprinted ABI
dependency; only the freshly built custom OPP may supply this operator.
These constraints do not establish absence of answers in arbitrary compiled
vendor libraries.

Two exact corpus blocks were read and support `near_exact`:
`0bc94d24162470fe8ce3fc189b7eb239aa868d074ff53c78706cad0c5c608243:ff3250a1dc6c151404a3cd4d96ac5df37c6b77bcd4f91f957f799f9a68d2c515`
and
`bd76c09b015265e14e93da7611506d3566ca73c59a0e66c59d1555e41e400a6d:12b0143245219e443b4013315e4805370e14ab09325bc9ad5896f2a5b112ba25`.
Their source corpus is commit
`cec3249ccd75e7a73cf2f2f776117e0ec385795b`.
Other suggested analogies have not been independently labeled.

Scope is one actual native operator on one A3 NPU. No A2, ACLGraph, HCCL,
full-model accuracy, service, performance or flagship twelve-task acceptance
is implied.
