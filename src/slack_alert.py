"""
Phase 4.2: Slack alerting via incoming webhook.

Deliberately simple: a POST of a Block Kit payload to a webhook URL. No
Slack app, no OAuth, no bot token -- what most small/mid teams actually
wire up for CI notifications.
"""
from __future__ import annotations

import os

import requests

from .diff import RunDiff, Severity
from .eval_runner import EvalRun

SEVERITY_EMOJI = {
    Severity.OK: "✅",
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "🚨",
}


def build_slack_payload(run: EvalRun, diff: RunDiff, report_url: str | None = None) -> dict:
    emoji = SEVERITY_EMOJI[diff.severity]

    if diff.is_first_run:
        headline = f"First recorded eval run for prompt `{run.prompt_version}`: {run.overall_pass_rate:.1%} pass rate."
    else:
        headline = (
            f"{len(diff.regressions)} regression(s) detected. "
            f"Pass rate {'dropped' if diff.pass_rate_delta < 0 else 'moved'} from "
            f"{diff.baseline_pass_rate:.1%} to {diff.current_pass_rate:.1%} "
            f"({diff.pass_rate_delta:+.1%})."
        )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji} Eval run: {diff.severity.value.upper()}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Prompt:* `{run.prompt_version}`  |  *Model:* `{run.model}`\n"
                    f"{headline}"
                ),
            },
        },
    ]

    if not diff.is_first_run and (diff.regressions or diff.improvements):
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Regressions:*\n{len(diff.regressions)}"},
                {"type": "mrkdwn", "text": f"*Improvements:*\n{len(diff.improvements)}"},
            ],
        })

    if report_url:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"<{report_url}|View full diff report>"},
        })

    return {"blocks": blocks}


def send_slack_alert(run: EvalRun, diff: RunDiff, report_url: str | None = None) -> bool:
    """
    Returns True if the alert was sent (or skipped intentionally because no
    webhook is configured), False if the send itself failed.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[slack_alert] SLACK_WEBHOOK_URL not set; skipping Slack notification.")
        return True

    payload = build_slack_payload(run, diff, report_url)
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[slack_alert] Failed to send Slack alert: {e}")
        return False
