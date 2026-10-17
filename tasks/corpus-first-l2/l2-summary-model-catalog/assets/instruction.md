# Real-source CPU protocol repair

An isolated summary worker inherits its selected provider route but loses the selected model's native metadata. Repair the real summary_provider module so a unique small-model alias and its complete native record survive the private worker-home projection. Keep other models, hooks and authentication files outside that projection. Reject incomplete/ambiguous records visibly. This is a local CPU protocol test; do not call a model or a remote service.

The selected native metadata is part of the route/policy identity: changing that record must change `route_sha256` even when the provider endpoint and model name remain the same.

Run `@L2_PYTHON@ reproduce.py` from the source root.
