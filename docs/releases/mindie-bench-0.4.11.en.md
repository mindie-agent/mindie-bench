# mindie-bench 0.4.11: Sub-800-Line Runner Modularization, `mindie-harness` Birth Release Support, and `bench` Governance

[中文](mindie-bench-0.4.11.md)

`mindie-bench v0.4.11` completes clean-architecture modularization across the evaluation runner and extends the organization-wide release controller:

- **Sub-800-Line Modular Decomposition & Dead-Island Removal**: Splits oversized runner modules (`gate.py`, `harness.py`, `release.py`, `value.py`) into focused `< 800`-line submodules with 4-part invariant headers and prunes unreachable legacy helpers while preserving 100% of the 20 behavioral-oracle tasks and 109 3-state/helper controls.
- **Four-State Verdict Honesty & Explicit `TimeoutError` Propagation**: Preserves strict separation across `completed_pass`, `completed_fail`, `infra_failed`, and `unknown`, preceding `except OSError:` boundaries with `except TimeoutError: raise`.
- **`mindie-harness` Birth Release Policy & `mindie-agent-oc` Consumer Contract**: Extends `release-policy.json`, `check_release_dependencies.py`, `upgrade_acceptance.py`, and `consumer_contract.py` to validate `mindie-agent/harness` birth releases and `mindie-agent/mindie-agent-oc` native plugin release metadata, aligned with `--profile bench` PR governance.
