# mindie-bench 0.4.13：零 Git 预构建 wheelhouse、发布流水线耗时收敛与 `remote-dev` 1:1 原生工具面评分对齐

[English](mindie-bench-0.4.13.en.md)

`mindie-bench v0.4.13` 收敛产品发布、组件/插件发布流水线耗时与 `remote-dev` 原生工具面的契约边界：

- **零 Git 预构建 wheelhouse 包**：新增 `.github/scripts/build_wheels_bundle.py`，从发行通道 JSON 读取 9 个共享组件，逐个校验正式 tag revision、`SHA256SUMS`、wheel 流式 SHA-256 与规范化 `METADATA` `Name`/`Version`，并在隔离临时目录以剥离写令牌的环境构建纯 Python `reme-ai[as]` wheel，最终生成确定性的 `mindie-wheels-bundle.tar.gz`（扁平成员、排序、`mtime=0`、`mode=0644`、`< 3 MiB`）。
- **统一 `release-channel.json` 与可选 `install.py` 发布**：`publish-product.yml` 在 `milestone` 与 `patch` 两条路径上统一校验 `channels/<track>.json`，无条件传入 `collect_product_benchmarks.py` 并写入 `release-channel.json`/`channel_sha256`，在计算 `SHA256SUMS` 之前构建 wheelhouse 包并按需暂存 `product/install.py`，最终把通道文件、wheelhouse 包与安装器一并挂到正式 Release。
- **发布流水线缓存复用与冗余重建裁剪**：`publish-release.yml`、`release-benchmark.yml` 与 `publish-product.yml` 全面开启 `astral-sh/setup-uv@v6` 依赖缓存，`upgrade_acceptance.py` 与 `native_upgrade_acceptance.py` 跨阶段共享同一 `UV_CACHE_DIR`；`verify-public` 作业由 25–45 分钟二次全量重建收敛为 `< 30s` 的 `.github/scripts/verify_public_release.py` 公开 CDN 下载与 `SHA256SUMS` 全资产完整性校验；`check_release_dependencies.py` 同时支持官方 GitHub Release wheel URL 收据与仅依赖 `mindie-harness` 的薄宿主插件。
- **wheel provenance 库存探测与 1:1 原生 `remote-dev` 评分对齐**：`native_upgrade_acceptance.py` 支持 `allow_wheel_provenance=True`（兼容旧版消费端回退）、`mindie_doctor` 发现与退役 `remote_BashOutput`/`remote_KillShell` 拦截；`mcp_probe.py`、`target_access.py` 与 `tool_observations.py` 对齐全宿主 1:1 原生远端工具名与参数（含数值型 `session_id`、`image_path`、`notebook_path` 及 `remote_manage_task` `Action="status"` 唯一确认语义）。
