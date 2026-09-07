"""
Unit tests for the parts of the pipeline that don't require an API key:
diffing, drift detection, and dataset/prompt loading. Run with:

    pytest tests/ -v

(Scoring and classifier tests need OPENAI_API_KEY and are intentionally
left as integration tests you run against a real key, not mocked here --
mocking the LLM call would defeat the point of an eval pipeline.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diff import Severity, diff_runs
from src.drift import detect_drift
from src.eval_runner import EvalRun
from src.scoring import CaseScore, score_composite


def make_score(cid, passed, category_correct=True, summary_score=5, difficulty="easy"):
    return CaseScore(
        case_id=cid, difficulty=difficulty,
        predicted_category="billing" if category_correct else "technical",
        expected_category="billing", category_correct=category_correct,
        predicted_summary="x", expected_summary="y", summary_score=summary_score,
        judge_reasoning="", latency_ms=100.0, prompt_tokens=10, completion_tokens=5,
        error=None, passed=passed,
    )


def test_first_run_has_no_diff():
    run = EvalRun(run_id="r1", timestamp="t", prompt_version="v1", model="m", dataset_version="v1",
                  case_scores=[make_score("c1", True)])
    diff = diff_runs(run, None)
    assert diff.is_first_run
    assert diff.severity == Severity.OK


def test_regression_detected():
    baseline = EvalRun(run_id="base", timestamp="t0", prompt_version="v1", model="m", dataset_version="v1",
                        case_scores=[make_score("c1", True), make_score("c2", True)])
    current = EvalRun(run_id="cur", timestamp="t1", prompt_version="v2", model="m", dataset_version="v1",
                       case_scores=[make_score("c1", True), make_score("c2", False)])
    diff = diff_runs(current, baseline)
    assert len(diff.regressions) == 1
    assert diff.regressions[0].case_id == "c2"
    assert diff.pass_rate_delta < 0


def test_bulk_flip_forces_critical_even_with_small_percentage():
    cases = [f"c{i}" for i in range(100)]
    baseline = EvalRun(run_id="base", timestamp="t0", prompt_version="v1", model="m", dataset_version="v1",
                        case_scores=[make_score(c, True) for c in cases])
    # 3 flips out of 100 = 3% delta, below CRITICAL_THRESHOLD (8%) on magnitude alone
    current_scores = [make_score(c, True) for c in cases]
    for i in range(3):
        current_scores[i] = make_score(cases[i], False)
    current = EvalRun(run_id="cur", timestamp="t1", prompt_version="v2", model="m", dataset_version="v1",
                       case_scores=current_scores)
    diff = diff_runs(current, baseline)
    assert diff.severity == Severity.CRITICAL


def test_score_composite_requires_category_and_summary_bar():
    assert score_composite(category_correct=True, summary_score=3, error=None) is True
    assert score_composite(category_correct=True, summary_score=2, error=None) is False
    assert score_composite(category_correct=False, summary_score=5, error=None) is False
    assert score_composite(category_correct=True, summary_score=5, error="boom") is False


def test_drift_requires_minimum_history():
    runs = [EvalRun(run_id=f"r{i}", timestamp=f"t{i}", prompt_version="v1", model="m", dataset_version="v1",
                     case_scores=[make_score("c1", True)]) for i in range(5)]
    report = detect_drift(runs, window=7)
    assert report.has_enough_data is False


def test_drift_flags_gradual_decline():
    runs = []
    for i in range(16):
        passed = i < 8  # first half all pass, second half all fail -> steep decline
        runs.append(EvalRun(run_id=f"r{i}", timestamp=f"t{i}", prompt_version="v1", model="m",
                             dataset_version="v1", case_scores=[make_score("c1", passed)]))
    report = detect_drift(runs, window=7)
    assert report.has_enough_data
    assert report.is_drifting is True
