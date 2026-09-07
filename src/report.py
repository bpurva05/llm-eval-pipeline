"""
Phase 4.1: the diff report.

Pure-Python HTML generation (Jinja2 + inline SVG for the trend chart) so the
report is a single self-contained file -- no JS CDN dependency, works as a
GitHub Actions artifact, works opened locally, works linked from Slack.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from .diff import RunDiff, Severity
from .drift import DriftReport
from .eval_runner import EvalRun

TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Report - {{ run.prompt_version }} - {{ run.run_id[:8] }}</title>
<style>
  :root {
    --ok: #1a7f37; --warn: #9a6700; --crit: #cf222e;
    --bg: #f6f8fa; --border: #d0d7de; --text: #1f2328; --muted: #656d76;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text); margin: 0; padding: 32px; }
  .container { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
           font-size: 12px; font-weight: 600; color: white; }
  .badge.ok { background: var(--ok); }
  .badge.warning { background: var(--warn); }
  .badge.critical { background: var(--crit); }
  .card { background: white; border: 1px solid var(--border); border-radius: 8px;
          padding: 20px; margin-bottom: 20px; }
  .scorecard { display: flex; gap: 16px; flex-wrap: wrap; }
  .stat { flex: 1; min-width: 140px; }
  .stat .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em;}
  .stat .value { font-size: 26px; font-weight: 700; margin-top: 2px; }
  .delta-up { color: var(--ok); } .delta-down { color: var(--crit); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  tr.regressed { background: #fff1f0; }
  tr.improved { background: #f0fff4; }
  code.pill { background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
              padding: 1px 6px; font-size: 12px; }
  h2 { font-size: 16px; margin-top: 0;}
  .empty { color: var(--muted); font-style: italic; font-size: 13px; }
  .drift-box { border-left: 4px solid var(--warn); padding: 8px 12px; background: #fff8f0; font-size: 13px; }
  .drift-box.ok { border-left-color: var(--ok); background: #f0fff4; }
</style>
</head>
<body>
<div class="container">
  <h1>Eval Report <span class="badge {{ diff.severity.value }}">{{ diff.severity.value | upper }}</span></h1>
  <div class="meta">
    Prompt <code class="pill">{{ run.prompt_version }}</code> ·
    Model <code class="pill">{{ run.model }}</code> ·
    Dataset <code class="pill">{{ run.dataset_version }}</code> ·
    Run <code class="pill">{{ run.run_id[:8] }}</code> ·
    {{ run.timestamp }}
  </div>

  <div class="card scorecard">
    <div class="stat">
      <div class="label">Pass Rate</div>
      <div class="value">{{ "%.1f"|format(run.overall_pass_rate*100) }}%
        {% if not diff.is_first_run %}
          <span class="{{ 'delta-up' if diff.pass_rate_delta >= 0 else 'delta-down' }}" style="font-size:14px;">
            ({{ "%+.1f"|format(diff.pass_rate_delta*100) }} pts)
          </span>
        {% endif %}
      </div>
    </div>
    <div class="stat">
      <div class="label">Avg Summary Score</div>
      <div class="value">{{ "%.2f"|format(run.avg_summary_score) }}/5</div>
    </div>
    <div class="stat">
      <div class="label">Avg Latency</div>
      <div class="value">{{ "%.0f"|format(run.avg_latency_ms) }}ms</div>
    </div>
    <div class="stat">
      <div class="label">Total Tokens</div>
      <div class="value">{{ run.total_tokens }}</div>
    </div>
  </div>

  <div class="card">
    <h2>Category Accuracy</h2>
    <table>
      <tr><th>Category</th><th>Accuracy</th><th>vs. Baseline</th></tr>
      {% for cat, acc in run.category_accuracy.items() %}
      <tr>
        <td>{{ cat }}</td>
        <td>{{ "%.1f"|format(acc*100) }}%</td>
        <td>
          {% if diff.category_accuracy_delta.get(cat) is not none %}
            <span class="{{ 'delta-up' if diff.category_accuracy_delta[cat] >= 0 else 'delta-down' }}">
              {{ "%+.1f"|format(diff.category_accuracy_delta[cat]*100) }} pts
            </span>
          {% else %}—{% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <div class="card">
    <h2>Regressions ({{ diff.regressions | length }})</h2>
    {% if diff.regressions %}
    <table>
      <tr><th>Case</th><th>Difficulty</th><th>Baseline Output</th><th>Current Output</th></tr>
      {% for r in diff.regressions %}
      <tr class="regressed">
        <td>{{ r.case_id }}</td>
        <td>{{ r.difficulty }}</td>
        <td><code class="pill">{{ r.baseline_category }}</code><br>{{ r.baseline_summary }} <span class="empty">(score {{ r.baseline_summary_score }}/5)</span></td>
        <td><code class="pill">{{ r.current_category }}</code><br>{{ r.current_summary }} <span class="empty">(score {{ r.current_summary_score }}/5)</span></td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty">No regressions vs. baseline.</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>Improvements ({{ diff.improvements | length }})</h2>
    {% if diff.improvements %}
    <table>
      <tr><th>Case</th><th>Difficulty</th><th>Baseline Output</th><th>Current Output</th></tr>
      {% for i in diff.improvements %}
      <tr class="improved">
        <td>{{ i.case_id }}</td>
        <td>{{ i.difficulty }}</td>
        <td><code class="pill">{{ i.baseline_category }}</code><br>{{ i.baseline_summary }} <span class="empty">(score {{ i.baseline_summary_score }}/5)</span></td>
        <td><code class="pill">{{ i.current_category }}</code><br>{{ i.current_summary }} <span class="empty">(score {{ i.current_summary_score }}/5)</span></td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty">No improvements vs. baseline.</div>
    {% endif %}
  </div>

  {% if drift %}
  <div class="card">
    <h2>Drift Check</h2>
    <div class="drift-box {{ 'ok' if not drift.is_drifting else '' }}">{{ drift.message }}</div>
  </div>
  {% endif %}

  {% if trend_svg %}
  <div class="card">
    <h2>Pass Rate Trend (last {{ trend_run_count }} runs)</h2>
    {{ trend_svg | safe }}
  </div>
  {% endif %}
</div>
</body>
</html>
"""
)


def _render_trend_svg(runs: list[EvalRun], width: int = 880, height: int = 180) -> str:
    if len(runs) < 2:
        return ""
    pad = 30
    xs = [pad + i * (width - 2 * pad) / (len(runs) - 1) for i in range(len(runs))]
    ys = [
        height - pad - r.overall_pass_rate * (height - 2 * pad)
        for r in runs
    ]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#0969da"><title>{r.run_id[:8]} - {r.overall_pass_rate:.1%}</title></circle>'
        for x, y, r in zip(xs, ys, runs)
    )
    baseline_y = height - pad - (height - 2 * pad)  # 100% line
    zero_y = height - pad  # 0% line
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
      <line x1="{pad}" y1="{zero_y}" x2="{width-pad}" y2="{zero_y}" stroke="#d0d7de" stroke-width="1"/>
      <line x1="{pad}" y1="{baseline_y}" x2="{width-pad}" y2="{baseline_y}" stroke="#d0d7de" stroke-width="1" stroke-dasharray="4,4"/>
      <polyline points="{points}" fill="none" stroke="#0969da" stroke-width="2"/>
      {circles}
    </svg>
    """


def render_report(
    run: EvalRun,
    diff: RunDiff,
    drift: DriftReport | None = None,
    trend_runs: list[EvalRun] | None = None,
) -> str:
    trend_svg = _render_trend_svg(trend_runs) if trend_runs else ""
    return TEMPLATE.render(
        run=run,
        diff=diff,
        drift=drift,
        trend_svg=trend_svg,
        trend_run_count=len(trend_runs) if trend_runs else 0,
    )


def write_report(html: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
