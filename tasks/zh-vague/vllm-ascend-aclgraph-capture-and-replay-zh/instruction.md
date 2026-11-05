我们在 `vllm-ascend` 中开启 `FULL` 模式的 `ACLGraph`（`CUDAGraphMode.FULL`）跑 Decode 推理时遇到两个问题：

1. 首次进入图捕获阶段（`torch.npu.graph`）就报流同步错误（`ACL stream capture error: forbidden host-device synchronization`），导致图捕获直接中断；
2. 即便绕过捕获阶段的报错，在变长 Decode Batch（例如 `num_tokens=2` 或 `3` padding 到已捕获的 bucket `num_tokens_padded=4`）触发图回放（`replay`）时，输出结果和 KV Cache 内容都与 Eager 模式严重不一致，多步 Decode 后精度彻底劣化。

仓库里的 `tests/repro_aclgraph_runner.py` 可以复现从图捕获到多步变长 Batch 回放对比 Eager 的完整流程。请定位并修复 `vllm_ascend` 中的相关缺陷（请勿修改 `tests/repro_aclgraph_runner.py`），确保图捕获过程无非法同步，且变长 Batch 图回放的每步输出与 KV Cache 均与 Eager 模式严格一致。
