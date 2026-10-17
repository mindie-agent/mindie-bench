In DP=2 deployments of Kimi K3 (hybrid Gated DeltaNet / GDN and MLA attention with MoE) on Ascend NPU, when one DP rank runs an idle step with no scheduled requests while the peer rank processes active requests, the idle rank's cached recurrent GDN state (`conv_state` / `ssm_state`) from earlier requests becomes corrupted, causing subsequent requests on that rank to diverge.

Reproduce the issue from the remote source root on the two assigned NPUs using `python3 bench_idle_dp_worker.py` (or `python3 test_idle_dp_npu.py`). Fix the idle DP execution path so that an idle rank leaves its cached GDN recurrent state tensors unchanged while preserving active-rank state updates, unrelated cache blocks, and cross-rank HCCL synchronization. Do not disable GDN/KDA layers or replace native kernels.

