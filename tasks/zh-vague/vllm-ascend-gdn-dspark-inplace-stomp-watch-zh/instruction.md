在 vllm-ascend 里跑混合线性注意力（GDN）加 DSpark 多步投机草稿生成时，全程不抛任何异常，但只要跑完第一步草稿，原本正常的请求序列状态和后续步的输出就全乱了（复现脚本在 tests/repro_gdn_dspark_step.py），帮我查出哪里把状态改坏了并修掉。
