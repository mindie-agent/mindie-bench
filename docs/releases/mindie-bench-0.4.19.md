# mindie-bench 0.4.19：`host-accept` 单轮隔离 `uv` 缓存共享、Windows 并发收敛与超时算术校准

[English](mindie-bench-0.4.19.en.md)

`mindie-bench v0.4.19` 消除三平台（尤其是 Windows Server NTFS + Defender）在 4 场景全栈虚拟环境构建中的重复三方依赖下载与超时抖动：

- **单次运行级隔离 `uv` 缓存共享（`host_acceptance/cli.py` · `__init__.py`）**：`run_host_acceptance` 在临时目录创建单轮独占 `mindie-ha-uvcache-*` 并在 `finally` 统一回收，使 `cold_native_launch`、`public_install_py`、`hooks_and_capture` 与 `upgrade_cutover`（及 `--all-hosts` 7 宿主）共享同一同卷预解压缓存，彻底避免重复下载与解压 180+ 个公共 PyPI 轮子。
- **Windows 磁盘并发控制与超时算术对齐（`cli.py` · `verify-host-plugin.yml` · `publish-release.yml` · `publish-product.yml`）**：Windows 下按宿主持有单场景顺序热身、跨宿主并发上限收敛为 `2`，并将单宿主预算提升至 `600s`（外层步长 `30m–45m`、作业上限 `30m–55m`），严格满足 `outer_timeout >= sum(inner_budgets) + cleanup_grace`。
