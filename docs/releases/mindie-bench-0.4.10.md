# mindie-bench 0.4.10：本地与远端 NPU 存储/工作树生命周期加固

[English](mindie-bench-0.4.10.en.md)

本版本完成 Wave 1 存储与工作树生命周期泄漏修复：所有由评测器创建、但此前会被异常路径或过期截止时间跳过的临时工作区、远端 Git worktree、容器内大体积产物与测试语料快照，现在都有明确的归属判断、关闭顺序和强制回收路径。

- **分层知识评测临时工作区**：`evaluate_applicability_and_distractors(..., workspace=None)` 会在自有 `mindie-tiered-*` 目录中安装分级语料并打开 3 个 SQLite `Store`。现在先关闭全部 Store（Windows 上必须在删除 `.sqlite3` 前完成），再在 `finally` 中强制删除自有目录；成功与异常路径均不残留，调用方传入的 `workspace` 保持不变。
- **远端 NPU worktree 与 `.bench-home` 缓存**：`POSTFLIGHT` 将补丁/未跟踪文件检查放入 `try/finally`。即使 agent 创建了意外的 `.git` 目录、补丁超过 `32 MiB` 或未跟踪清单越界，也会清理 `<slot>` 工作树、`.bench-home` 缓存，并 `git worktree prune` 掉 `<repo>/.git/worktrees/<slot>`；清理错误与主检查错误一并回传，不再静默遗留多 GB 远端目录。
- **远端 `<slot>.final.json` 产物**：`RemoteSlot.close()` 在本地 `remote_cleanup.json` 持久化后，于 15 秒清理宽限内删除远端 `<slot>.final.json`（含 base64 补丁时约 85 MiB），并顺带清理当前用户拥有的、超过 48 小时且无活动 slot 的 `<slots_root>/*.final.json`。删除失败只记录 `cleanup_warning`，不改写主 trial 判决。
- **过期截止时间下的本地 worktree 清理**：`WorktreeSlot.cleanup()` 预留 `CLEANUP_GRACE_S = 15.0` 最小清理宽限。即使 trial 截止时间已耗尽，也会先移除 agent 伪造的 `.git` 目录、重挂元数据并执行 `git worktree remove --force`；失败时删除自有工作树并运行 `git worktree prune`，保证 `<repo>/.git/worktrees/<name>` 不泄漏。续跑执行器把该宽限计入外层预算。
- **有界缓存与测试夹具**：KDA 构建缓存在 `copy2`/哈希失败时删除 PID 临时目录，并把 `kda_build_out` 限制为最多 16 个 LRU 条目；测试用 operators 干净快照改到隔离临时目录并在进程退出时删除，同时清理当前用户遗留的过期快照。

本版本独立发布，不更新历史产品通道，也不改变组件依赖版本或冻结基准任务资产。
