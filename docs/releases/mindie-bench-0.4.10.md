# mindie-bench 0.4.10：本地与远端 NPU 存储/工作树生命周期加固

[English](mindie-bench-0.4.10.en.md)

本版本完成 Wave 1 存储与工作树生命周期泄漏修复：所有由评测器创建、但此前会被异常路径或过期截止时间跳过的临时工作区、远端 Git worktree、容器内大体积产物与测试语料快照，现在都有明确的归属判断、关闭顺序和强制回收路径。

- **分层知识评测临时工作区**：`evaluate_applicability_and_distractors(..., workspace=None)` 会在自有 `mindie-tiered-*` 目录中安装分级语料并打开 3 个 SQLite `Store`。现在先关闭全部 Store（Windows 上必须在删除 `.sqlite3` 前完成），再在 `finally` 中强制删除自有目录；成功与异常路径均不残留，调用方传入的 `workspace` 保持不变。
- **远端 NPU worktree 与 `.bench-home` 缓存**：`POSTFLIGHT` 将补丁/未跟踪文件检查放入 `try/finally`，并通过 `<admin>/gitdir` 精确匹配本槽位路径来定位唯一的 `<repo>/.git/worktrees/<slot>` 管理目录；即使 agent 创建了意外的 `.git` 目录、补丁超过 `32 MiB` 或未跟踪清单越界，也只删除本槽位的工作树、`.bench-home` 缓存和该精确管理目录，绝不对共享仓库执行仓库级清理，因而不会误删并发活跃槽位的元数据。清理错误与主检查错误一并回传，不再静默遗留多 GB 远端目录。
- **远端 `POSTFLIGHT` 超时算术**：清理脚本获得 `30s` 单调预算，拆分为补丁/状态提取、`10s` 有界 teardown 和 `5s` 最终产物预留；`<slot>.final.json` 在 teardown 前原子写出，teardown 超时也不会丢失，若仍有余量则在 teardown 后原子更新清理状态。外层 SSH 等待 `50s >= 30s + 15s`，调用方不再先于 `.final.json` 超时。
- **远端 `<slot>.final.json` 产物**：`RemoteSlot.close()` 先以同目录临时文件 + `flush` + `fsync` + `os.replace` 持久化本地 `remote_cleanup.json`，成功后才在 15 秒清理宽限内删除远端 `<slot>.final.json`（含 base64 补丁时约 85 MiB）；`fsync`/替换失败时保留远端产物。并顺带清理当前用户拥有的、超过 48 小时且无活动 slot 的 `<slots_root>/*.final.json`。
- **过期截止时间下的本地 worktree 清理**：`WorktreeSlot.cleanup()` 预留 `CLEANUP_GRACE_S = 15.0`，其中 `10s` 用于重挂元数据与 `git worktree remove --force`，剩余 `5s` 专用于纯 Python fallback：删除自有工作树并从 `.git` 指针记录的精确 `self._admin_dir` 删除本槽位管理目录。即使 remove 子预算已耗尽也不影响 fallback，且绝不触碰 `<repo>/.git/worktrees/` 下的兄弟条目。续跑执行器把 `patch_capture + CLEANUP_GRACE_S + verifier drain` 预先计入 `outer_timeout`（默认 `750s = 600s + 120s + 10s + 15s + 5s`）。
- **有界缓存与测试夹具**：KDA 构建缓存在 `copy2`/哈希失败时删除 PID 临时目录，并把 `kda_build_out` 限制为最多 16 个已完成条目，命中时刷新 `mtime` 以实现真实 LRU；进行中的 `.tmp-<pid>` 目录从不计入淘汰候选，只有 PID 已退出且超过 1 小时的陈旧临时目录才通过原子 `.trash-*` 重命名回收。测试用 operators 干净快照改到隔离临时目录并在进程退出时删除，同时清理当前用户遗留的过期快照。

本版本独立发布，不更新历史产品通道，也不改变组件依赖版本或冻结基准任务资产。
