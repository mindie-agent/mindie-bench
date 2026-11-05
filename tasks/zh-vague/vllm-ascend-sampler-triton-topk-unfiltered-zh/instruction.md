在 vllm-ascend 的采样模块（vllm_ascend/sample/）里同时开启 top_k 和 top_p 采样时，遇到部分请求不启用 top_k（k<=0 或 k>=vocab_size）或者首个 token 概率直接超过 top_p 阈值时，会报越界或算出一整行 -inf 导致 softmax 变成 NaN，帮我把采样过滤与边界处理修好。
