# Contributing to `mindie-bench`

`mindie-bench` uses the **`bench`** repository governance profile (`BENCH-ORACLE`, `BENCH-VERDICT`, `BENCH-PROVENANCE`, `BENCH-ISOLATION`).

## 1. Flat Sibling + `.worktrees/` Development Layout

Keep the base `../mindie-bench` checkout clean and develop inside a linked Git worktree:

```bash
git -C ../mindie-bench fetch origin
git -C ../mindie-bench worktree add -b feat/<topic> ../.worktrees/mindie-bench-<topic> origin/main
```

When launching a sandboxed agent from `mindie-agent/`, grant write access to **both** `../mindie-bench` (Git `commondir`) and `../.worktrees/mindie-bench-<topic>`.

## 2. Local Verification & Installed-Wheel Smoke Test

1. **Lint, Tier 0/1 isolated contracts, and synthetic demo**:
   ```bash
   uv sync --locked --all-extras
   uv run ruff check .
   uv run pytest -m "not live_npu"
   uv run mindie-bench demo --output /tmp/mindie-bench-demo
   ```
   Tests automatically isolate `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `XDG_*`, `MINDIE_*`, `REMOTE_DEV_*`, `UV_CACHE_DIR`, and `TMPDIR` via `tests/conftest.py`.

2. **Installed-wheel smoke test outside the checkout**:
   ```bash
   uv build --out-dir /tmp/mindie-bench-dist
   uv venv /tmp/mindie-bench-smoke
   uv pip install --python /tmp/mindie-bench-smoke/bin/python /tmp/mindie-bench-dist/*.whl
   (cd /tmp && /tmp/mindie-bench-smoke/bin/python -c \
     "import mindie_bench; from mindie_bench.runner import evaluate_rsi_candidate_gates; assert 'site-packages' in mindie_bench.__file__" && \
     /tmp/mindie-bench-smoke/bin/mindie-bench --help >/dev/null)
   ```

3. **Tier 2 standing A2/A3 NPU verification** (when lab network is reachable):
   ```bash
   uv run pytest -m live_npu
   ```

4. **Unix `LF` & Pre-PR Checklist (`bench`)**:
   ```bash
   python3 .github/scripts/check_lf_eol.py
   python3 .github/scripts/check_pr_checklist.py --profile bench --body-file /path/to/pr_body.md
   ```
   Documentation-only PRs may record `not-applicable: docs-only` on unaffected runtime items.
