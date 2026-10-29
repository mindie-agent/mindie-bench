# mindie-bench 0.4.5：原生知识评测接入与缺测统计修正

**中文** · [English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.5/docs/releases/mindie-bench-0.4.5.en.md)

原生 Codex 评测通过插件生产入口、真实 `UserPromptSubmit` 授权和私有知识 CLI 接入，保留独立 profile、指定模型与推理配置。DSH 原生配置携带宿主需要的 `mcpPluginUrl`，并在付费运行前检查真实挂载后的 catalog；不再用单独启动 Store 或 MCP 成功代替宿主加载成功。两端复用已安装组件命令和任务依赖路径，原生产物身份增加实际模块源码与 wheel 摘要。

Codex 私有知识调用不能被具名 MCP 事件完整覆盖时，绝对 query/explain 数量、知识时长及其 remainder 保持 `null`，单独保留观察到的具名调用下界。聚合与组件报告不把缺测或部分覆盖转成零，不把整个 shell 命令耗时当作知识耗时；实际模型 Token、墙钟与业务 oracle 独立报告。

原生终端将 zsh 的 `TMPPREFIX` 绑定至现有 profile 临时目录，修复 heredoc/here-string 仍尝试写入未授权 `/tmp/zsh` 的问题，不扩大 sandbox 权限。历史评测回执、失败日志、冻结包和付费请求用量保持原样。

已合入功能提交的定向本地验证为 73 passed，包含真实终端进程、原生事件解析及缺测值聚合边界；Ruff、LF 和跨平台 CI 通过。正式发布流水线仍会针对本 Tag 的精确源码执行锁文件安装、第一方依赖正式 Release 校验、本地合同测试和检索评分微基准，并附带结果与校验和。

本版本修复评测接入和测量口径，不宣称已证明知识策略整体收益。已有单点 CPU 配对出现降本信号，但复杂任务整体不退化门槛未通过；Codex 临时文件噪声、未完成尝试与缺失用量仍是结果的一部分。此次发布准备没有新增付费模型调用，也不重写历史结果。
