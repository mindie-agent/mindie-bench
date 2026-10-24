# mindie-bench 0.4.4: HTTP/1.1 Chunked Response Settlement Close-Race Fix

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.4/docs/releases/mindie-bench-0.4.4.md)

`mindie-bench 0.4.4` fixes a cross-thread close race in the model proxy gateway when settlement interrupts an active `HTTP/1.1 chunked` response stream, ensuring interrupted streams still record structured `model_error` and `model_usage` journal entries from the owning request handler.

## 1. Root Cause & Fix

- **Observed Trigger & Preserved Receipt**: During the first Codex G0 canary on the new short vague prompt under `mindie-bench 0.4.3`, the `480s` foreground budget expired while a model response was still in flight. Its original receipt (`terminal_status=unknown`, `cost_complete=False`) is preserved as-is without replay. Interrupting the stream during settlement exposed an `HTTP/1.1 chunked` concurrency bug: calling `connection.close()` from the settler thread cleared `HTTPResponse.fp` while the handler thread was inside `response.read1()`, which on Python 3.13 could raise `AttributeError` (`NoneType.close`) and drop the structured `model_error` record.
- **Socket Shutdown Without Cross-Thread Close (`src/mindie_bench/runner/gateway.py`)**: `Gateway._close_model_connections` now only calls `shutdown(socket.SHUT_RDWR)` on the registered raw socket to interrupt blocked I/O, leaving `HTTPConnection` / `HTTPResponse.close()` exclusively to the request handler thread's `finally` block.
- **Unchanged Honesty & Budget Invariants**: Foreground (`480s`) and production budgets remain unchanged, and zero automatic paid retries are performed. Incomplete responses remain classified as `unknown` with only observed usage preserved; the gateway neither fabricates task success nor backfills missing token billing.

## 2. Verification & Scope

- **Regression Verification**: The two zero-cost loopback test modules covering model delivery and socket lifecycle passed **`18 passed` (`13.24s`)**. The new deterministic `HTTP/1.1 chunked` reader-pause interleaving regression in `tests/test_model_delivery_settlement.py` (`test_chunked_settlement_keeps_live_response_close_with_its_reader`) passed in **`0.60s`** and reliably rejects the previous cross-thread `close` behavior.
- **Scope Boundary**: This release only fixes settlement stream teardown and journal recording; the prior `0.4.3` canary `unknown` status and incomplete-cost gap remain preserved, and full benchmark task runs and G5 experiments are not yet completed.
