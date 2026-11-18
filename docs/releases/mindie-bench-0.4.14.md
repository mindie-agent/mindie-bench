# mindie-bench 0.4.14：校准预编译 `mindie-wheels-bundle.tar.gz` 体积上限（`16 MiB`）

[English](mindie-bench-0.4.14.en.md)

`mindie-bench v0.4.14` 根据 9 个共享组件正式纯 Python wheel 与 `reme-ai[as]` wheel 的实测归档体积（`5.64 MiB`，其中 `mindie_sim` 轮子 `2.48 MiB`）校准 `.github/scripts/build_wheels_bundle.py` 的 `MAX_BUNDLE_BYTES` 上限：

- **校准 `MAX_BUNDLE_BYTES = 16 * 1024 * 1024`（`16 MiB`）**：由于标准 `.whl` 已采用 DEFLATE 压缩，10 个纯 Python 轮子与清单打入确定性 `PAX` `.tar.gz` 后实测体积为 `5.64 MiB`，原 `3 MiB` 紧上界会导致正式产品通道打包误拒；现上调至 `16 MiB` 并与顶层 `install.py` 保持一致，同时保留流式 `SHA256SUMS`、`METADATA` 规范化与零写令牌隔离构建全部安全不变量。
