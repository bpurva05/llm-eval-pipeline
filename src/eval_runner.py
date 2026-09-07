"""
Phase 3.1: the test runner.

Takes a PromptConfig + GoldenDataset, runs every case through the classifier
feature concurrently (bounded by a semaphore so we don't blow through rate
limits or spend a fortune), scores each one, and returns a full EvalRun.
"""
from __future__ import annotations

import asyncio
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .classifier import classify_email
from .config import GoldenDataset, PromptConfig
from .scoring import CaseScore, score_case

DEFAULT_CONCURRENCY = 8


@dataclass
class EvalRun:
    run_id: str
    timestamp: str
    prompt_version: str
    model: str
    dataset_version: str
    case_scores: list[CaseScore] = field(default_factory=list)

    @property
    def overall_pass_rate(self) -> float:
        if not self.case_scores:
            return 0.0
        return sum(1 for c in self.case_scores if c.passed) / len(self.case_scores)

    @property
    def category_accuracy(self) -> dict[str, float]:
        by_cat: dict[str, list[bool]] = {}
        for c in self.case_scores:
            by_cat.setdefault(c.expected_category, []).append(c.category_correct)
        return {cat: sum(v) / len(v) for cat, v in by_cat.items()}

    @property
    def avg_summary_score(self) -> float:
        scores = [c.summary_score for c in self.case_scores if c.summary_score > 0]
        return statistics.mean(scores) if scores else 0.0

    @property
    def avg_latency_ms(self) -> float:
        latencies = [c.latency_ms for c in self.case_scores]
        return statistics.mean(latencies) if latencies else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.case_scores)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "dataset_version": self.dataset_version,
            "overall_pass_rate": self.overall_pass_rate,
            "category_accuracy": self.category_accuracy,
            "avg_summary_score": self.avg_summary_score,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "case_scores": [c.__dict__ for c in self.case_scores],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalRun":
        run = cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            prompt_version=d["prompt_version"],
            model=d["model"],
            dataset_version=d["dataset_version"],
        )
        run.case_scores = [CaseScore(**c) for c in d["case_scores"]]
        return run


async def _run_one_case(sem: asyncio.Semaphore, config: PromptConfig, case) -> CaseScore:
    async with sem:
        result = await classify_email(config, case.input_email)
        return await score_case(case, result)


async def run_eval(
    config: PromptConfig,
    dataset: GoldenDataset,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> EvalRun:
    sem = asyncio.Semaphore(concurrency)
    tasks = [_run_one_case(sem, config, case) for case in dataset.cases]
    case_scores = await asyncio.gather(*tasks)

    return EvalRun(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_version=config.version,
        model=config.model,
        dataset_version=dataset.dataset_version,
        case_scores=list(case_scores),
    )
