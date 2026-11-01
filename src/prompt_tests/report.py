"""Self-contained HTML report: one file, no external assets, works offline and in CI artifacts."""

from __future__ import annotations

import html
import json
from datetime import datetime

from .runner import CaseResult, summarize

CSS = """
:root{--bg:#fbfbfa;--fg:#1b1b1a;--muted:#6b6b66;--line:#e4e4df;--card:#fff;--pass:#1f7a4d;--pass-bg:#e5f3ec;--fail:#b3261e;--fail-bg:#fbe9e7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecece8;--muted:#9a9a93;--line:#2c2c2a;--card:#1c1c1b;--pass:#6fcf97;--pass-bg:#173326;--fail:#f28b82;--fail-bg:#3a1c1a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1200px;margin:0 auto;padding:32px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}.card b{font-size:28px;display:block;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num{font-variant-numeric:tabular-nums;text-align:right}.pass{color:var(--pass);background:var(--pass-bg)}.fail{color:var(--fail);background:var(--fail-bg)}
.cell{cursor:pointer;font-weight:600}.detail{display:none;font-size:13px}.detail.open{display:table-row}
pre{white-space:pre-wrap;margin:6px 0;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);padding:8px;border-radius:6px;max-height:280px;overflow:auto}
.wrap{overflow-x:auto}
"""

JS = """
document.querySelectorAll('.cell').forEach(c=>c.addEventListener('click',()=>{
  document.getElementById(c.dataset.target).classList.toggle('open')}));
"""


def _e(s) -> str:
    return html.escape(str(s))


def render(suite_name: str, results: list[CaseResult]) -> str:
    summary = summarize(results)
    models = list(summary)
    cards = ""
    for m, s in summary.items():
        errors = f" · {s['errors']} errors" if s["errors"] else ""
        cards += (
            f'<div class="card"><div class="muted">{_e(m)}</div><b>{s["pass_rate"]:.0%}</b>'
            f'<div class="muted">pass rate · mean score {s["mean_score"]:.2f} · '
            f"p50 {s['p50_latency']:.1f}s · p95 {s['p95_latency']:.1f}s{errors}</div></div>"
        )
    tags = sorted({t for s in summary.values() for t in s["by_tag"]})
    tag_rows = "".join(
        f"<tr><td>{_e(t)}</td>"
        + "".join(f'<td class="num">{summary[m]["by_tag"].get(t, 0):.0%}</td>' for m in models)
        + "</tr>"
        for t in tags
    )
    case_ids = list(dict.fromkeys(r.case.id for r in results))
    by_key: dict[tuple[str, str], list[CaseResult]] = {}
    for r in results:
        by_key.setdefault((r.case.id, r.model), []).append(r)

    rows = []
    for i, cid in enumerate(case_ids):
        first = next(r for r in results if r.case.id == cid)
        cells, details = [], []
        for m in models:
            rs = by_key.get((cid, m), [])
            passed = sum(r.passed for r in rs)
            cls = "pass" if passed == len(rs) else "fail"
            label = (
                "PASS" if passed == len(rs) else ("FAIL" if passed == 0 else f"{passed}/{len(rs)}")
            )
            target = f"d{i}"
            cells.append(f'<td class="cell {cls}" data-target="{target}">{label}</td>')
            for r in rs[:1]:
                score_lines = (
                    "".join(
                        f"<div>{'✓' if s.passed else '✗'} <b>{_e(s.name)}</b> {s.value:.2f} {_e(s.reason)}</div>"
                        for s in r.scores
                    )
                    or f"<div>error: {_e(r.error)}</div>"
                )
                details.append(
                    f"<div><b>{_e(m)}</b> <span class='muted'>{r.latency_s:.1f}s"
                    f"{' (cached)' if r.cached else ''}</span><pre>{_e(r.output)}</pre>{score_lines}</div>"
                )
        rows.append(
            f'<tr><td><b>{_e(cid)}</b><div class="muted">{_e(", ".join(first.case.tags))}</div></td>{"".join(cells)}</tr>'
            f'<tr class="detail" id="d{i}"><td colspan="{len(models) + 1}"><div class="muted">Input</div>'
            f"<pre>{_e(first.case.input)}</pre>{''.join(details)}</td></tr>"
        )

    head = "".join(f"<th>{_e(m)}</th>" for m in models)
    data = json.dumps(summary).replace("</", "<\\/")  # safe inside <script>
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>prompt-tests · {_e(suite_name)}</title>
<style>{CSS}</style></head><body><main>
<h1>{_e(suite_name)}</h1><div class="muted">{len(case_ids)} cases · {len(models)} model(s) · {datetime.now():%Y-%m-%d %H:%M}</div>
<div class="cards">{cards}</div>
<h2>By tag</h2><div class="wrap"><table><tr><th>Tag</th>{head}</tr>{tag_rows}</table></div>
<h2>Cases <span class="muted">(click a result for output and scores)</span></h2>
<div class="wrap"><table><tr><th>Case</th>{head}</tr>{"".join(rows)}</table></div>
<p class="muted" style="margin-top:32px">Generated by prompt-tests · built by <a href="https://abbishekkdgs.tech">Abbishekk DGS</a></p>
<script type="application/json" id="summary">{data}</script><script>{JS}</script>
</main></body></html>"""
