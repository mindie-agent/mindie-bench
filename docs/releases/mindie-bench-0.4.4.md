# mindie-bench 0.4.4：HTTP/1.1 分块流结算并发关闭竞态修复

[English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.4/docs/releases/mindie-bench-0.4.4.en.md)

`mindie-bench 0.4.4` 修复了模型代理网关在结算取消阶段中断活跃 `HTTP/1.1 chunked` 响应流时的跨线程关闭竞态，确保未完成流在中断后仍由请求处理线程完整落盘结构化 `model_error` 与 `model_usage` 日志。

## 一、问题背景与修复内容

- **触发场景与历史回执保留**：在使用 `mindie-bench 0.4.3` 执行新短模糊题面的首个 Codex G0 Canary 时，任务在 `480s` 前台预算到期时尚未收完模型响应，原始回执记录为 `terminal_status=unknown`、`cost_complete=False`（该真实回执原样保留、未做重放）。在结算阶段中断该流式连接时，暴露出 `HTTP/1.1 chunked` 响应的跨线程关闭竞态：结算线程直接调用 `connection.close()` 会在请求处理线程执行 `response.read1()` 期间清空 `HTTPResponse.fp`，在 Python 3.13 下可能触发 `AttributeError`（`NoneType.close`），导致结构化 `model_error` 记录丢失。
- **单所有者关闭与 Socket 中断（`src/mindie_bench/runner/gateway.py`）**：`Gateway._close_model_connections` 在结算到期时仅对已登记的底层传输 socket 执行 `shutdown(socket.SHUT_RDWR)` 以中断阻塞读取，不再跨线程调用 `connection.close()`；`HTTPConnection` / `HTTPResponse.close()` 由请求处理线程自身的 `finally` 块独占执行。
- **不变量与预算边界保持不变**：本次修复不改变 `480s` 前台预算及其他生产预算，不自动重试未知外部副作用的付费请求；遇到未完整响应时仍保持 `unknown` 判定与已观测 usage 记录，绝不将超时未完成请求伪造为成功或补造缺失的 Token 账单。

## 二、验证与适用范围

- **回归验证**：模型流传输与连接生命周期两个零付费本地回环测试模块共 **`18 passed`（`13.24s`）**；其中 `tests/test_model_delivery_settlement.py` 新增的 `HTTP/1.1 chunked` 读取暂停交错回归测试（`test_chunked_settlement_keeps_live_response_close_with_its_reader`）在 **`0.60s`** 内确定性通过，并稳定拒绝旧版跨线程 `close` 负对照。
- **范围边界**：本版本仅修复流结算时的连接关闭竞态与日志完整性；既有 `0.4.3` Canary 的 `unknown` 与费用未完整缺口保持原状，真实业务评测任务与 G5 知识演进实验尚未完成。
