# mindie-bench 0.4.5：原生知识评测接入、Shell 临时目录隔离与缺测统计修正

**中文** · [English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.5/docs/releases/mindie-bench-0.4.5.en.md)

`mindie-bench 0.4.5` 统一了 Codex 与 DSH 原生 Harness 的生产级知识评测接入方式，修复了沙箱内 `zsh` heredoc 临时文件路径越界问题，并规范了私有知识调用在部分观测场景下的统计口径。

## 一、核心修复与工程改进

- **Codex 与 DSH 原生知识评测接入**：
  - Codex 原生评测走插件生产入口、真实 `UserPromptSubmit` 授权与私有知识 CLI，保持独立 profile、模型与推理强度配置。
  - DSH 原生配置注入宿主所需的 `mcpPluginUrl`，并在正式评测前校验宿主实际挂载的 catalog，不再把单独启动 Store 或 MCP 进程视为宿主挂载成功。
  - 两端统一复用环境已安装的组件命令与任务依赖路径，并在产物指纹中记录实际加载模块的源码与 wheel 摘要。
- **缺测与部分覆盖指标如实留空**：
  - 当具名 MCP 事件无法完整覆盖 Codex 的私有 CLI 知识调用时，绝对 `query`/`explain` 次数、知识检索耗时及剩余耗时保持 `null`，仅将已观测到的具名调用作为下界单独记录。
  - 汇总报告不再把缺测值折算为 `0`，也不再把整条 shell 命令耗时算作知识检索耗时；模型 Token、墙钟耗时与业务 Verifier 结果保持独立统计。
- **沙箱内 `zsh` `TMPPREFIX` 绑定**：
  - 原生终端启动时将 `zsh` 的 `TMPPREFIX` 指向当前 profile 的受管临时目录，解决 heredoc / here-string 默认写入未授权 `/tmp/zsh*` 触发沙箱拦截的问题，且不扩大沙箱写权限范围。

## 二、测试与验证

- 定向回归测试 **`73 passed`**，覆盖真实终端进程、原生事件解析与缺测值聚合边界；`ruff`、Unix `LF` 行尾检查与跨平台 CI 全部通过。
