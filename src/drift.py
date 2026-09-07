"""
Phase 4.3: slow drift detection.

Per-run diffing catches sharp regressions (a bad prompt edit). It does NOT
catch gradual degradation -- e.g. a model provider silently updates the
underlying weights behind "gpt-4o-mini" and your pass rate creeps down
0.5% at a time. No single run trips WARNING_THRESHOLD, but the trend is real.

We track a rolling average over the last N runs and flag it if it drops
below the average of the N runs before that by more than a small threshold.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .eval_runner import EvalRun

DEFAULT_WINDOW = 7
DEFAULT_DRIFT_THRESHOLD = 0.05  # 5% drop in rolling average vs. prior window


@dataclass
class DriftReport:
    has_enough_data: bool
    window_size: int
    recent_avg_pass_rate: float | None
    prior_avg_pass_rate: float | None
    drift_delta: float | None
    is_drifting: bool
    message: str


def detect_drift(
    historical_runs: list[EvalRun],
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DriftReport:
    """
    historical_runs must be in chronological order (oldest first), and should
    include the just-completed run as the last element.
    """
    if len(historical_runs) < window * 2:
        return DriftReport(
            has_enough_data=False,
            window_size=window,
            recent_avg_pass_rate=None,
            prior_avg_pass_rate=None,
            drift_delta=None,
            is_drifting=False,
            message=(
                f"Need at least {window * 2} runs to detect drift "
                f"(have {len(historical_runs)}). Skipping drift check."
            ),
        )

    recent_window = historical_runs[-window:]
    prior_window = historical_runs[-window * 2 : -window]

    recent_avg = statistics.mean(r.overall_pass_rate for r in recent_window)
    prior_avg = statistics.mean(r.overall_pass_rate for r in prior_window)
    delta = recent_avg - prior_avg

    is_drifting = delta <= -threshold

    if is_drifting:
        message = (
            f"Slow drift detected: {window}-run rolling pass rate dropped from "
            f"{prior_avg:.1%} to {recent_avg:.1%} ({delta:+.1%}), even though no "
            f"single run crossed the per-run alert threshold."
        )
    else:
        message = (
            f"No slow drift: {window}-run rolling pass rate is {recent_avg:.1%} "
            f"vs. {prior_avg:.1%} in the prior window ({delta:+.1%})."
        )

    return DriftReport(
        has_enough_data=True,
        window_size=window,
        recent_avg_pass_rate=recent_avg,
        prior_avg_pass_rate=prior_avg,
        drift_delta=delta,
        is_drifting=is_drifting,
        message=message,
    )
