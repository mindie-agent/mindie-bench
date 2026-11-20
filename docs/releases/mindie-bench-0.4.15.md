# mindie-bench 0.4.15：7 端原生协议验收器（`host-accept`）与三平台全栈发版可用性门禁

[English](mindie-bench-0.4.15.en.md)

`mindie-bench v0.4.15` 为 MindIE Agent 全部 17 个代码仓库上线零 Mock 端到端发版可用性防线（Gate Layer 2 & Gate Layer 3）：

- **7 端宿主原生协议验收器（`src/mindie_bench/host_acceptance/` · `mindie-bench host-accept`）**：无需闭源 IDE 客户端二进制，直接按全部 7 款原生宿主插件（`codex`、`cc`、`kimi`、`dsh` Cordis ESM、`agy`、`wb`、`oc` ESM）的真实发布清单与打包产物，在 Linux、macOS 与 Windows 上执行 4 大必检场景：① 冷启动首字节 `stdout` 纯净度、并发表面握手、Schema 1:1 对齐与真实 `tools/call`；② 公开 `install.py`（挂载 `git` 拦截陷阱）开箱首轮可用性；③ Hook 快速路径与 `local-only` 合成密钥脱敏采集（未授权场景零 SQLite 创建）；④ `n-1` 正式版向候选版平滑升级切流（保留 `mindie-consent.json` 与自定义命令重绑、修复退役引擎与缺失路径）。
- **候选 Wheelhouse 替换组装器与可复用工作流（`assemble_candidate_wheelhouse.py` · `fullstack-smoke.yml` · `verify-host-plugin.yml`）**：支持在单组件或 `harness` 独立发版时将待发候选 `.whl` 确定性叠加至正式兄弟组件与 `reme-ai` 预构建包（同时强校 `HAR-ABI` `dependencies == []`），并升级 `upgrade_acceptance.py`（`wheelhouse.update_runtime` + `selftest --mode full`）与 `native_upgrade_acceptance.py`（真实执行 `mcpServer/tool/call` 校验 `knowledge_query` 与 `mindie_doctor`）。
- **单仓发版与 21 格产品整包发版前置阻断（`publish-release.yml` · `publish-product.yml`）**：单仓发版在创建草稿与不可变 Tag 前强制通过三平台 `usability-gate`；产品整包发版拆分为 `stage-product` -> `qualify-product`（解包待发布的真实 `mindie-wheels-bundle.tar.gz`，跨 `3 OS × 7 宿主 = 21` 格全量验收）-> `publish`，任一格未通过即阻断发布。
