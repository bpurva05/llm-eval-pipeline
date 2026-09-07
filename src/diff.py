"""
Phase 3.3-3.4: the diffing engine. This is the actual product.

Anyone can run an eval once. The value is in comparing this run to the last
one: what regressed, what improved, and is the delta big enough to be signal
rather than noise from LLM sampling variance (even at temperature=0, model
providers aren't perfectly deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .eval_runner import EvalRun
from .scoring import CaseScore

# Configurable thresholds (also overridable via env vars, see run_eval.py CLI)
WARNING_THRESHOLD = 0.03  # 3% pass-rate delta
CRITICAL_THRESHOLD = 0.08  # 8% pass-rate delta


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CaseFlip:
    case_id: str
    difficulty: str
    baseline_passed: bool
    current_passed: bool
    baseline_category: str | None
    current_category: str | None
    baseline_summary: str | None
    current_summary: str | None
    baseline_summary_score: int
    current_summary_score: int


@dataclass
class RunDiff:
    baseline_run_id: str | None
    current_run_id: str
    pass_rate_delta: float
    baseline_pass_rate: float
    current_pass_rate: float
    category_accuracy_delta: dict[str, float]
    regressions: list[CaseFlip] = field(default_factory=list)
    improvements: list[CaseFlip] = field(default_factory=list)
    severity: Severity = Severity.OK

    @property
    def is_first_run(self) -> bool:
        return self.baseline_run_id is None


def _index_by_id(scores: list[CaseScore]) -> dict[str, CaseScore]:
    return {s.case_id: s for s in scores}


def _classify_severity(pass_rate_delta: float, has_critical_flip: bool = False) -> Severity:
    magnitude = abs(pass_rate_delta)
    if magnitude >= CRITICAL_THRESHOLD or has_critical_flip:
        return Severity.CRITICAL
    if magnitude >= WARNING_THRESHOLD:
        return Severity.WARNING
    return Severity.OK


def diff_runs(
    current: EvalRun,
    baseline: EvalRun | None,
    warning_threshold: float = WARNING_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
) -> RunDiff:
    if baseline is None:
        # First run ever: nothing to regress against. Still useful as a baseline snapshot.
        return RunDiff(
            baseline_run_id=None,
            current_run_id=current.run_id,
            pass_rate_delta=0.0,
            baseline_pass_rate=0.0,
            current_pass_rate=current.overall_pass_rate,
            category_accuracy_delta={},
            severity=Severity.OK,
        )

    baseline_by_id = _index_by_id(baseline.case_scores)
    current_by_id = _index_by_id(current.case_scores)

    regressions: list[CaseFlip] = []
    improvements: list[CaseFlip] = []

    shared_ids = set(baseline_by_id) & set(current_by_id)
    for case_id in shared_ids:
        b, c = baseline_by_id[case_id], current_by_id[case_id]
        if b.passed == c.passed:
            continue
        flip = CaseFlip(
            case_id=case_id,
            difficulty=c.difficulty,
            baseline_passed=b.passed,
            current_passed=c.passed,
            baseline_category=b.predicted_category,
            current_category=c.predicted_category,
            baseline_summary=b.predicted_summary,
            current_summary=c.predicted_summary,
            baseline_summary_score=b.summary_score,
            current_summary_score=c.summary_score,
        )
        if b.passed and not c.passed:
            regressions.append(flip)
        elif not b.passed and c.passed:
            improvements.append(flip)

    pass_rate_delta = current.overall_pass_rate - baseline.overall_pass_rate

    cat_delta: dict[str, float] = {}
    baseline_acc = baseline.category_accuracy
    current_acc = current.category_accuracy
    for cat in set(baseline_acc) | set(current_acc):
        cat_delta[cat] = current_acc.get(cat, 0.0) - baseline_acc.get(cat, 0.0)

    # A single flip on a case tagged "edge" or "hard" flipping in bulk (>=3 regressions
    # on previously-passing cases) is treated as critical even if the aggregate
    # percentage looks small on a tiny dataset.
    has_critical_flip = len(regressions) >= 3

    severity = _classify_severity(pass_rate_delta, has_critical_flip)
    # Only escalate on regressions (drops), never on pure improvements
    if pass_rate_delta > 0 and not has_critical_flip:
        severity = Severity.OK

    return RunDiff(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        pass_rate_delta=pass_rate_delta,
        baseline_pass_rate=baseline.overall_pass_rate,
        current_pass_rate=current.overall_pass_rate,
        category_accuracy_delta=cat_delta,
        regressions=regressions,
        improvements=improvements,
        severity=severity,
    )
