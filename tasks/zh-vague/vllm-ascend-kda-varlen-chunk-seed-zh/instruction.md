在 vllm-ascend 里跑 KDA 线性注意力算子（vllm_ascend/ops/triton/kda/kda.py）处理变长序列（cu_seqlens）预填充时，只要序列长度不是分块大小 BT 的整数倍，尾块就会越界读入下一条序列的 token 并算错循环状态 h，帮我把变长分块边界与尾块状态递推掩码修好。
