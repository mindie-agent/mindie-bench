# mindie-bench 0.4.22：`host-accept --all-hosts` 有界双并发防 CPU 饥饿与产品整包诊断透出

[English](mindie-bench-0.4.22.en.md)

`mindie-bench v0.4.22` 修复三平台产品整包 `qualify-product` 在 3–4 核云端运行器上执行 `host-accept --all-hosts` 时的 CPU 并发争用与报错截断问题：

- **统一有界双并发与单宿主顺序场景调度（`src/mindie_bench/host_acceptance/cli.py`）**：将跨宿主并发上限在全平台统一收敛为 `host_workers = min(2, len(clean_hosts))`，且每个宿主内 4 大验收场景顺序执行（复用运行期共享 `uv` 缓存），消除此前在 POSIX 上 `7 宿主 × 2 场景 = 14` 路并发 `uv`/MCP 争抢 3–4 个虚拟核导致 `Stop` 钩子 `6s` 预算超时饥饿的问题。
- **整包发版失败摘要前置直出（`.github/workflows/publish-product.yml` · `.github/workflows/publish-release.yml`）**：当 `host-accept --all-hosts` 返回非零退出码时，优先解析报告 JSON 中未通过宿主与场景的 `error` 字段并置于异常消息首部，避免被尾部 `stdout` 截断遮蔽。
