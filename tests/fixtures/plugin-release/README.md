# Native compatibility captures

These are complete source declarations captured on November 12, 2026 for the
shared publisher's generated-artifact tests. They are compatibility inputs,
not hardware responses or fabricated upgrade results.

| Fixture | Source repository | Source |
| --- | --- | --- |
| codex.json | mindie-agent/mindie-agent-codex | 91aec540aa42d733cb2382aad9cc34ad416242b1 |
| claude-code.json | mindie-agent/mindie-agent-cc | 9fcaadb74938cea629e2e8eac3e29c9fbe1260a7 |
| kimi-code.json | mindie-agent/mindie-agent-kimi | 136c4ca042cc8996a45f4280c25ca3e92bff2f70 |
| antigravity.json | mindie-agent/mindie-agent-agy | 46ebe8e35e6a6525230b03bb84d9aa7615e56970 |
| workbuddy.json | mindie-agent/mindie-agent-wb | af77cda6b0d5c526c8308087e7bae5ad245b1168 |
| opencode.json | mindie-agent/mindie-agent-oc | c500a45166221d1d536cf6f0bee9d2e2bf924e89 |
| dsh.json | mindie-agent/mindie-agent-dsh | Canonical declaration migration for unpublished 0.1.17 |

Codex stores its declaration under `plugins/mindie-agent/`; the other six
repositories store it at their root. Runtime validation and source paths come
from `load_consumer`, rather than from this explanatory table.
