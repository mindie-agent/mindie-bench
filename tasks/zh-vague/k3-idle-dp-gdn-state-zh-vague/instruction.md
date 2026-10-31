我在昇腾上用外部负载均衡多 DP（`data_parallel_external_lb`）跑 Kimi-K3 模型，只要某个 DP rank 遇到没有请求的空闲轮次跑了 dummy 步，后续请求的 GDN 状态就乱了导致输出异常，帮我解一下这个问题。
