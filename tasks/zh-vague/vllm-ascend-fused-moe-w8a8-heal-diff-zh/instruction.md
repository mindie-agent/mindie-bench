我们在 vllm-ascend 里跑多层 MoE 动态量化前向时，输出跟 FP32 基准比余弦相似度掉到了 0.79（复现入口 tests/repro_moe_quant_accuracy.py）；逐层打印看第 0 层路由完输出就有差异，但改路由根本没用，帮我定位真正算错的算子代码并修复。
