换了一组模型头维度配置后，在昇腾 NPU 上跑旋转位置编码缓存更新又崩了，报错如下，帮我直接定位并修复源码里的 RoPE 缓存维度计算逻辑：

```text
RuntimeError: The expanded size of the tensor (128) must match the existing size (32) at non-singleton dimension 3.  Target sizes: [1, 3, 1, 128].  Tensor sizes: [3, 1, 32]
```

（注：请在远端 NPU 工作区中直接修改 `vllm_ascend/` 源码并在分配的 NPU 上验证，不要修改本地工作区或 `tests/` 目录，修复验证通过后立即收敛总结。）
