# mindie-bench 0.4.9：原生 Windows 与 WSL2 跨平台契约、进程树回收及三端 CI 门禁

[English](mindie-bench-0.4.9.en.md)

`mindie-bench` 现将 `windows-latest` 正式接入主分支与 PR 的 `release-ready` CI 门禁（与
`ubuntu-latest`、`macos-latest` 并行），并补齐原生 Windows（含中文 OEM `CP936` 环境、
无 WSL 环境）与 WSL2 的跨平台宿主执行、进程清理及路径归一化契约。

运行器与本地裁判引入 `host_platform` 原生平台层：在 WSL2 下拒绝 `/mnt/[a-z]` 挂载盘中
混入的 Windows `.exe`/`.cmd` 互操作包装器，在 Windows 下过滤 `WindowsApps` 应用执行别名
零字节桩与 `System32\bash.exe` WSL 空壳，按需解析用户态 Git for Windows 与 WinGet `uv`。
所有后台与本地子进程统一启用 `CREATE_NO_WINDOW`，并通过 Win32 Job Object
（`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`）与有界管道读写线程确保超时后孙进程树彻底回收且
保留已排空的标准输出证据。

评分器、知识代理与全仓读写统一显式使用 UTF-8 与 Unix `LF`（`\n`）换行，剥离 Windows
`\\?\` 扩展路径前缀并支持 `file:///C:/...` 构建产物 URI 哈希核验。

本版本独立发布，不更新历史产品通道，也不改变组件依赖版本或冻结基准任务资产。
