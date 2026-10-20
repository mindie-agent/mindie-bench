# Source and provenance

Owner: ORIGINAL native DSH Qwen author, native session
`session-1779cef2-9a4c-42ea-ab8d-63064c183f60`.

## Exact pins

- Model: `Qwen/Qwen3.5-0.8B`
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Full weight SHA256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- Config SHA256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`
- Index SHA256: `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e`
- vLLM: `4034c3d32e30d01639459edd3ab486f56993876d`
- Ascend source0: `a78a00e0b1b618f26a3bfce2519055b40cf4bf1e`
- Ascend after: `0f289fa2a8a381a644a5f329af61715a73ec2aec` (PR7109)

## Empty-target dependency correction

At vLLM `4034c3d`, `setup.py::_no_device()` selects
`requirements/common.txt`; it does not read `requirements/cuda.txt` and
`common.txt` has no nested `-r` includes. Therefore the CUDA torch pin is not an
empty-target dependency. The source0 Ascend pins (torch 2.9.0, torch-npu 2.9.0,
triton-ascend 3.2.0, transformers >=4.57.4) supply torch/torch-npu; vLLM
empty-target supplies `transformers >=4.56,<5` and common dependencies.

Admitted evidence:
- setup.py reference SHA256 `17d54d76504113841c899f27fc17519043e3fb8f50d810bb5f27b2c39366c99c`
- common.txt SHA256 `708faa85a40e3ce8d00df8cd0a5fcac5bdd894b5b97e81183649e73387a896fc`
- source bindings SHA256 `b0f8b54bd7a786a90d3a633892c7c33bb504e2f4e8082021829e5fdcb010b915`

The setup.py reference is masked and is **never executable**. Its three masks
are in precompiled-wheel helper path/filename text, not in the `_no_device()` or
`get_requirements()` branch logic; those branches are visible and are the admitted
basis for the empty-target correction.

## Reference

Accepted CPU FP32 reference report SHA256:
`61a1bb8bc869b583c2af6d8ad5098eca33dba85e5cc146465490c219cbd2f974`.
CPU raw arrays:
- teacher logits `06b3d086e2ff7e65e81ebe082deee083e1320f2beea5a6d2e23a2f5c0b619261`
- cache logits `02a7892ecf93f1210dc2234fe5e10c3cb4dc92ba6b9fa43e07022365c7b86290`
- full-prefix logits `43cc2f6f70a06e23114fcf7ddb3f61b7ace4d33dc4d74a4483f4e994d3da0064`
- generated IDs `fcde667461879acad17eef3cc112517708138c59a02f1c04ea7c489f209242bb`
- prompt IDs `7e8bda2f8f64856287296a18ed414261f768bff942afbc3d69153e1481227a0e`

The frozen support set is `support-set.json` with SHA256
`09d733f5c7156c4eff224113e26734711be61c141a07be3d096b0038c7dc872a`.

## Pending boundaries

- Runtime pending/unverified: no preprovisioned runtime ID, no verified reset
  snapshot, no target output, no left/right arm result.
- No `soundness_receipt.json`; evidence remains `candidate`.
- `remote_verifier.py` was absent at the allowlisted path; verifier interface is
  not confirmed. `verifier/test.sh` is a conservative shell contract that exits
  2 for missing target results and 1 for any mismatch.
- Cold provisioning is separate shared setup cost, not a TaskSpec/P2 benefit.
- No numeric NPU tolerance, no full-vocab/logit accuracy, no MTP/image result.

## Final repository location

Intended final task root:
`/Users/maoxx241/code/mindie-bench/tasks/corpus-first-l3/qwen35-0.8b-historical-language-prefill-decode/`

Relative pointers in this bundle use `environment/`, `verifier/`, `reference/`,
and `support-set.json` and remain valid after exact-byte handoff.
