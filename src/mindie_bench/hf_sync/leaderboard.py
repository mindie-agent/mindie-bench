# [Role] Builds and publishes the redacted public leaderboard view.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Static viewer for already-redacted summaries; no external dependencies."""

from html import escape


def render(summaries: dict) -> str:
    def display(value):
        return "unmeasured" if value is None else f"{value:.4g}"

    rows = []
    for name, summary in sorted(summaries.items()):
        values = [
            name,
            str(summary["attempts"]),
            str(summary["valid_attempts"]),
            display(summary["pass_at_1"]),
            display(summary["pass_pow_k"]),
            display(summary.get("mean_delivery_wall_clock_sec")),
            display(summary["cost_per_resolved_task"]),
            (
                f"{display(summary.get('budget', {}).get('delivery_budget_exceeded_attempts'))} / "
                f"{display(summary.get('budget', {}).get('delivery_budget_observed_attempts'))}"
            ),
            display(summary.get("budget", {}).get("pass_at_1_within_delivery_budget")),
            str(summary["statuses"]),
        ]
        rows.append("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in values) + "</tr>")
    return (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MindIE Bench evidence</title><style>
:root{color-scheme:light;--ink:#202c27;--paper:#f5f2e8;--line:#c6cbbd;--accent:#245c43}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:18px Georgia,serif}
main{max-width:1350px;margin:auto;padding:clamp(24px,5vw,72px)}
small{font:13px ui-monospace,monospace;letter-spacing:.1em;color:var(--accent)}
h1{font-weight:normal;font-size:clamp(36px,5vw,68px);margin:20px 0}p{max-width:850px;line-height:1.6}
.table{overflow-x:auto;border-top:3px solid var(--ink);margin:40px 0}
table{border-collapse:collapse;width:100%;font:14px ui-monospace,monospace}
th,td{text-align:left;padding:18px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--accent);white-space:nowrap}td:last-child{min-width:290px;line-height:1.7}
footer{font-size:15px;color:#48584e;max-width:850px;line-height:1.7}
</style><main><small>MINDIE / EVALUATION EVIDENCE</small><h1>Completion. Cost. Coverage.</h1>
<p>Every row identifies its tier and evidence class. Synthetic and CPU replay results
do not establish live NPU performance. Infrastructure failures and unknown attempts
remain visible alongside valid-attempt scores.</p><div class="table"><table><thead><tr>
<th>Tier / evidence / group</th><th>Attempts</th><th>Valid</th><th>Pass@1</th><th>Pass^k estimate</th>
<th>Delivery seconds</th><th>$/resolved</th><th>Delivery overruns / observed</th>
<th>Pass@1 within delivery budget</th><th>All terminal counts</th>
</tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div><footer>
Pass^k is a per-task estimate requiring k valid repetitions, not a guarantee.
Pass@1 retains verified business outcomes, including over-budget completions.
Delivery overruns and within-budget success are reported separately.
Unmeasured values are not zero. G3 seed rows are separate; the JSON ledger additionally
records seed setup cost and cost per resolved evaluation including setup.
Inspect receipts.json, leaderboard.json, trials.parquet and eval_manifest.json
for source fingerprints and metric coverage. No raw private trajectories are included.
</footer></main></html>
"""
    )
