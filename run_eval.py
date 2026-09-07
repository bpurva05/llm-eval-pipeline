#!/usr/bin/env python3
"""
Main CLI entry point. This is what CI (and you, locally) invoke.

Usage:
    python run_eval.py --prompt-version v2
    python run_eval.py --prompt-version v2 --no-slack
    python run_eval.py --prompt-version v2 --warning-threshold 0.05 --critical-threshold 0.10

Exit codes:
    0 = ok or warning (does not block merge)
    1 = critical regression detected (blocks merge in CI)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from src.config import GoldenDataset, PromptConfig, REPORTS_DIR
from src.diff import Severity, diff_runs
from src.drift import detect_drift
from src.eval_runner import run_eval as run_eval_async
from src.report import render_report, write_report
from src.slack_alert import send_slack_alert
from src.storage import get_all_runs, get_latest_run, get_recent_runs, save_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM eval pipeline.")
    parser.add_argument(
        "--prompt-version",
        default=os.environ.get("PROMPT_VERSION"),
        help="e.g. v2. Defaults to the $PROMPT_VERSION env var, then the latest prompt file.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-slack", action="store_true", help="Skip Slack notification even if webhook is configured.")
    parser.add_argument("--warning-threshold", type=float, default=0.03)
    parser.add_argument("--critical-threshold", type=float, default=0.08)
    parser.add_argument("--report-url-base", default=None, help="Public base URL where the HTML report will be hosted (for Slack link).")
    return parser.parse_args()


async def main() -> int:
    load_dotenv()
    args = parse_args()

    config = PromptConfig.load_version(args.prompt_version) if args.prompt_version else PromptConfig.latest()
    dataset = GoldenDataset.load()

    print(f"Running eval: prompt={config.version} model={config.model} cases={len(dataset.cases)}")
    run = await run_eval_async(config, dataset, concurrency=args.concurrency)

    baseline = get_latest_run()
    diff = diff_runs(
        run, baseline,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold,
    )

    save_run(run)

    all_runs = get_all_runs()  # includes the one we just saved, chronological
    drift = detect_drift(all_runs)

    report_path = REPORTS_DIR / f"report_{run.run_id[:8]}.html"
    trend_runs = get_recent_runs(n=15)
    html = render_report(run, diff, drift=drift, trend_runs=trend_runs)
    write_report(html, report_path)

    report_url = f"{args.report_url_base.rstrip('/')}/{report_path.name}" if args.report_url_base else str(report_path)

    print(f"\n{'='*60}")
    print(f"Run ID:        {run.run_id}")
    print(f"Prompt:        {run.prompt_version}  Model: {run.model}")
    print(f"Pass rate:     {run.overall_pass_rate:.1%}"
          + ("" if diff.is_first_run else f"  ({diff.pass_rate_delta:+.1%} vs. baseline)"))
    print(f"Avg summary:   {run.avg_summary_score:.2f}/5")
    print(f"Severity:      {diff.severity.value.upper()}")
    print(f"Regressions:   {len(diff.regressions)}")
    print(f"Improvements:  {len(diff.improvements)}")
    print(f"Drift check:   {drift.message}")
    print(f"Report:        {report_path}")
    print(f"{'='*60}\n")

    if not args.no_slack:
        send_slack_alert(run, diff, report_url=report_url)

    if diff.severity == Severity.CRITICAL:
        print("CRITICAL regression detected. Failing build.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
