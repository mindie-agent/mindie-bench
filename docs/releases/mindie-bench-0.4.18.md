# mindie-bench 0.4.18：`host-accept` 规范化 `local-only` 授权配置、Windows 超长路径读取与冷态 `remote_auth` 校验对齐

[English](mindie-bench-0.4.18.en.md)

`mindie-bench v0.4.18` 对齐 `host-accept` 在全部 7 款真实宿主插件上的端到端协议与采集校验契约：

- **规范化 `mindie-consent/1` 与 `mindie-community-config/1` 授权字段（`hooks_capture.py`）**：为正向 `local-only` `Stop` 采集场景写入合法 32 位十六进制 `choice_revision`、`generation` 及 `"local_only": true`，精确对齐 `mindie_knowledge.loop.settings` 生产权威校验。
- **Windows `MAX_PATH` 扩展路径遍历与文件锁避让（`hooks_capture.py`）**：使用 `\\?\` 扩展路径扫描 `<MINDIE_DATA_HOME>` 下含双 64 位 SHA-256 子目录（`> 260` 字符）的已提交脱敏材料块，并跳过活动 `.lock` 锁句柄。
- **冷沙箱未配置默认机器时的 `remote_auth(action="status")` 结构化回执判定（`mcp_wire.py`）**：允许冷态无默认主机时返回结构化 `error_code == "endpoint_incomplete"` 回执。
