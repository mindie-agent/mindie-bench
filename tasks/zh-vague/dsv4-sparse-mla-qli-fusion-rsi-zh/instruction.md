我们在昇腾 910B (A2) 上调优 `DeepSeek-V4` (`Flash`/`Pro`, W8A8) 的 `SparseMLA` 解码路径（`vllm_ascend/models/deepseek_v4/sparse_mla_decode.py`）。

前期的第一阶段调优记录在 `logs/dsv4_stage1_l3_attempt.json`：我们已经按照常规 `L3` 工作流配置了 `DSV4_BLOCK_SIZES = 128` 与 `ENABLE_INDEXER_OVERLAP = True`，但连续两轮尝试均未收敛（`verdict: "unsupported"`）：
1. 每层 `qr_i8` 的反量化、`RMSNorm` 与吸收权重 `W_Q` 投影及 `RoPE` 切片拼接仍然产生 3 次多余的 GM `TensorMove` 拷贝；
2. 在调用 `npu_quant_lightning_indexer` 前，对 `q_indexer` 和 `k_indexer` 显式调用 `.contiguous()` 又产生了 2 次多余的 `TensorMove` 克隆；
3. 在 `npu_quant_lightning_indexer` 与 `npu_sparse_flash_attention` 交接处，主机端对 `seqused_k.max().item()` 做了同步读取（触发 D2H 同步），且当长序列尾部 `topk_indices` 含 `-1` 填充槽位时直接写入 `block_table`，导致 `AICore invalid GM address` 报错。

请你按以下步骤完成定位与深度探索重构：
1. 先使用 `mindie_knowledge_query`（默认 `mode="reuse"`）检索 `vllm-ascend` 领域中关于 `DeepSeek-V4 SparseMLA LightningIndexer` 的现有经验，并结合 `logs/dsv4_stage1_l3_attempt.json` 确认常规 `L3` 工作流为何在算子/张量视图层面受阻；
2. 确认常规 `L3` 工作流停滞后，调用 `mindie_knowledge_query(..., mode="explore", bottleneck_summary=..., ledger_entries=...)` 切换到 **RSI 独立探索模式**（剥离 `L3` 工作流定势，挂载 `L4` 代码优化变换 `mutator`），并用 `mindie_knowledge_explain` 精读命中的 `L4` Mutator 全文；
3. 严格按照该 `L4` Mutator 定义的三项可分离变换因子与留一法消融规范重构 `vllm_ascend/models/deepseek_v4/sparse_mla_decode.py`（保持 `_ascend_qli_runtime.py` 不变），实现三项因子的开关参数与 `run_ablation_suite()` 函数（返回四组消融实验结果）；
4. 运行消融验证并在仓库根目录生成实验台账 `experiment-ledger.jsonl`（记录 baseline、完整候选及三组留一法消融的 `run_id`、`arm`、`verdict` 与关键指标）和总结报告 `PROGRESS.md`。

（注：当前为本地验证环境，请使用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python` 执行本地 Python 脚本与消融验证，无需连接远端 NPU。）
