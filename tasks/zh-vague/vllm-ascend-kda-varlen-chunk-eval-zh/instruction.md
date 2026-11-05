在 vllm-ascend 的 KDA 算子（vllm_ascend/ops/triton/kda/kda.py）里跑变长 Prefill 与单步 Decode 混合批次续写（chunk_kda_varlen_mixed_continuation）时，前序变长序列尾块的跨边界渗漏同样污染了后续 Decode 请求的注意力输出与最终状态，帮我把混合批次的分块续写与输出投影路径也修干净。
