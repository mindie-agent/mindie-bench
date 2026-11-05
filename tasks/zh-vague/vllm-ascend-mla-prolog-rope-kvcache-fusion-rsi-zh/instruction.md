我在昇腾 A5（Ascend 950DT）上用下面的命令拉起 DeepSeek-V3.2 服务并压测 Decode 吞吐时，MLA 前置处理链路不仅单步耗时明显偏高，而且开启 FA 量化与 Prolog 后直接报下面的精度超差异常（现场日志见 `logs/vllm_mla_prolog_a5.log`），帮我把这个问题解一下：

```bash
# 服务端拉起命令
VLLM_USE_V1=1 vllm serve /models/DeepSeek-V3.2-MXFP8 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --additional-config '{"enable_mlapo": true, "fa_quant": true}' \
  --max-model-len 16384 --port 8000

# 触发问题的压测命令
vllm bench serve --model /models/DeepSeek-V3.2-MXFP8 \
  --dataset-name random --random-input-len 4096 --random-output-len 256 \
  --num-prompts 64 --request-rate 8 --port 8000
```

```text
AssertionError: [Ascend950DT] MLA prolog numerical check failed: max_abs_err=0.0625 > 2e-2
```
