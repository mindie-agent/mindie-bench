我在昇腾 A5（Ascend 950DT）上用下面的命令开启 Model Runner V2 和 DSpark 投机解码跑全图（`CUDAGraphMode.FULL`）推理时，当并发请求数小于捕获分桶大小（例如 3 条请求 pad 到 bucket=4）就会触发下面的 FIA TND 长度契约报错，而且单步投机回放存在同步退化（报错日志见 `logs/vllm_dspark_aclgraph_a5.log`），帮我把这个问题解一下：

```bash
# 服务端拉起命令
VLLM_USE_V1=1 VLLM_ASCEND_MODEL_RUNNER_V2=1 vllm serve /models/DeepSeek-V4 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' \
  -O '{"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [4, 8, 16, 32]}' \
  --max-model-len 16384 --port 8000

# 触发报错的请求命令（3 条并发请求 pad 到 bucket=4）
vllm bench serve --model /models/DeepSeek-V4 \
  --dataset-name random --random-input-len 1024 --random-output-len 64 \
  --num-prompts 3 --max-concurrency 3 --port 8000
```

```text
RuntimeError: [Ascend950DT] FIA TND contract violation: queryT (12) != last element of actualSequenceLengthQ (9)
```
