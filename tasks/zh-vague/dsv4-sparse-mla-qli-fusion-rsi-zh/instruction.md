我在昇腾 A5（Ascend 950DT）上用下面的命令拉起 DeepSeek-V4（W8A8 / Sparse C8）服务，跑长序列请求触发 SparseMLA 闪电索引（QLI）与稀疏注意力解码时，不仅解码延迟高，还会报下面的 AICore 非法地址错误导致 worker 挂掉（报错日志见 `logs/vllm_dsv4_sfa_a5.log`），帮我把这个问题解一下：

```bash
# 服务端拉起命令
VLLM_USE_V1=1 vllm serve /models/DeepSeek-V4-W8A8 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --additional-config '{"enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_mlapo": true}' \
  --max-model-len 32768 --port 8000

# 触发报错的压测命令
vllm bench serve --model /models/DeepSeek-V4-W8A8 \
  --dataset-name random --random-input-len 16384 --random-output-len 128 \
  --num-prompts 32 --request-rate 4 --port 8000
```

```text
RuntimeError: [Ascend950DT] AICore invalid GM address: negative block id (-1) reached npu_sparse_flash_attention
```
