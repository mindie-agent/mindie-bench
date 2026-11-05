vllm-ascend 的日志与崩溃诊断导出模块（vllm_ascend/logger.py）在处理运行期异常日志（样例见 logs/worker_crash_raw.log）时，不仅漏脱敏了内网节点 IP、密钥 Token 和用户主目录路径，而且在上报超时的时候把 TimeoutError 当普通 IO 错误重置回 pending 导致重复上报，帮我把脱敏规则和上报状态机修干净。
