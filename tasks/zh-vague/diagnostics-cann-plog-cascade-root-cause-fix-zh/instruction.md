我们在排查多卡推理故障归因与诊断上报模块 `vllm_ascend_diag/plog_fault_reconciler.py`（配套复现数据位于 `vllm_ascend_diag/fixtures/diagnostics_case.json`）。

当前 `reconcile_plog_and_outbox()` 的实现存在三处严重缺陷：
1. **将下游 HCCL 等待超时误判为首发根因 (`triage`)**：日志第一行是 `rank[1]` 的 `HCCL_E_WAIT_TIMEOUT`，第四行是 `rank[1]` 的 `ErrCode=EI0006` (`hccl_collective`)，但这是因为 `rank[0]` 在 `14:54:51.635` / `14:54:52.000` 发生了 `HBM ECC uncorrectable error` (`failed operator: MatMulV2`, `ErrCode=E89999`, `subsystem = "driver_memory"`, `fault_domain = "device_oom_or_driver"`)。当前代码直接取日志首行作为根因，漏判了 `root_cause_rank = "0"`, `device_id = "0"`, `failed_operator = "MatMulV2"`, `error_codes = ["E89999", "EI0006"]`，且未将 `rank[1]` 的两条 HCCL 报错标记为级联超时 (`cascading_timeout_detected = True`, `cascading_rank = "1"`, `cascading_timeout_lines`)，也缺少 `"npu-smi"` 与 `"HCCL_ENTRY_LOG_ENABLE=1"` 等 `recommended_tools`；
2. **R2 隐私脱敏不完整 (`sanitized_log`)**：当前仅替换了 `Bearer` 令牌，遗漏了 `sk-...` API Key、内网地址 `10.24.7.19`、编号主机名 `node-17` 以及用户目录 `/home/exampleuser/project/model`（脱敏后应保留 `torch_npu` 与 `MatMulV2` 等技术上下文，且 `remaining_secrets == []`、`scan_finding_count == 0`）；
3. **Outbox 外部写入超时后盲目重置为 `pending` (`outbox_audit`)**：当事件入队并执行 `begin_post` 发起外部 GitHub Issue POST 时，一旦发生超时 (`simulate_post_timeout=True`)，状态必须保持为 `"uncertain"`（严禁重置回 `"pending"` 导致重复发帖），且后续同一 `fingerprint` 的新事件 (`operation_id_2`) 再次 `enqueue` 时，应将 `occurrences` 累加至 `2` 并把两个 `operation_id` 合并进 `incident_ids`，同时维持 `state == "uncertain"` (`blind_retry_prevented = True`)。

请修复 `vllm_ascend_diag/plog_fault_reconciler.py` 中的 `reconcile_plog_and_outbox()`。
（注：当前为本地离线复现环境，请优先检索 MindIE 领域知识库或直接复用环境中已安装的 `mindie_diagnostics` 组件完成修复与验证；若当前环境未安装对应组件，请按上述规范直接修复模块代码并用 `python3` 自测，无需连接远端 NPU。）
