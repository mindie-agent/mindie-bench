我们在多机 DP（外部启动器 `external_launcher`）环境下跑 Kimi-K3 / Qwen3-Next 这种带 GDN 线性注意力（Mamba 状态）的模型时，发现偶发输出乱码和精度崩坏。排查发现当某个 DP rank 没有分到真实请求（`total_num_scheduled_tokens == 0`）时，runner 为了参与 DP 集合通信会执行 1 个 token 的 `_dummy_run(1)`，结果把 GDN 的 recurrent state（`causal_conv1d` / KDA 状态）沿着旧请求遗留的 state indices 往前推进了一步，污染了正常请求的状态缓存。

报错与异常现场如下，帮我定位并修一下（要求保持空闲 DP rank 仍然执行 `_dummy_run(1)` 参与通信同步，但通过 `skip_gdn_state_update: bool = False` 将 GDN dummy metadata 的 `num_actual_tokens` 和 `max_query_len` 置 0、`nums_dict` 与 `spec_sequence_masks` 置 `None`、`non_spec_query_start_loc` 置全 0、`non_spec_state_indices_tensor` 置全 `-1`，并同步兼容 310P runner 的 `_dummy_run` 签名透传）：

```text
AssertionError: Idle DP rank _dummy_run(1) advanced GDN recurrent state on stale non_spec_state_indices_tensor (expected execute_model -> _dummy_run(1, skip_gdn_state_update=True) -> _build_attention_metadata(..., skip_gdn_state_update=skip_gdn_state_update) with num_actual_tokens=0, max_query_len=0, non_spec_query_start_loc=zeros_like, non_spec_state_indices_tensor=-1)
```

（注：当前为本地源码级修复验证环境，无需连接远端 NPU，直接在本地工作区完成代码修改并用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -m py_compile` 校验通过后即可直接收敛总结。）
