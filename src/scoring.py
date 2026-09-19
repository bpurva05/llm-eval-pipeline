"""
Phase 3.2: multi-dimensional scoring.

Category correctness is a cheap binary check. Summary quality is fuzzier --
two summaries can both be "correct" while differing in phrasing -- so we use
an LLM-as-judge rubric instead of exact string matching.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .classifier import ClassificationResult
from .config import GoldenCase

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")

JUDGE_SYSTEM_PROMPT = """You are grading a customer-support email summary for accuracy.
You will be given the original email, a reference (ideal) summary, and a candidate
summary produced by another AI system. Score the candidate from 1-5:

5 = Captures the same core issue/request as the reference, no hallucinated details.
4 = Captures the core issue, minor omission or slightly different emphasis.
3 = Partially correct: misses an important detail or is vague.
2 = Mostly wrong: misidentifies the core issue but is tangentially related.
1 = Completely wrong or hallucinates details not present in the email.

Respond ONLY with JSON: {"score": <1-5 integer>, "reasoning": "<one short sentence>"}
"""


@dataclass
class CaseScore:
    case_id: str
    difficulty: str
    predicted_category: str | None
    expected_category: str
    category_correct: bool
    predicted_summary: str | None
    expected_summary: str
    summary_score: int  # 1-5, 0 if it couldn't be judged (e.g. no output at all)
    judge_reasoning: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None
    passed: bool  # composite pass/fail used for regression diffing


async def judge_summary(email_text: str, reference_summary: str, candidate_summary: str) -> tuple[int, str]:
    """LLM-as-judge: rate the candidate summary 1-5 against the reference, retrying on rate limits."""
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"EMAIL:\n{email_text}\n\n"
                            f"REFERENCE SUMMARY:\n{reference_summary}\n\n"
                            f"CANDIDATE SUMMARY:\n{candidate_summary}"
                        ),
                    },
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return int(parsed.get("score", 0)), str(parsed.get("reasoning", ""))
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                match = re.search(r"try again in ([\d.]+)s", error_str)
                wait_seconds = float(match.group(1)) + 0.5 if match else 5.0
                await asyncio.sleep(wait_seconds)
                continue
            return 0, f"judge_error: {error_str}"
    return 0, "judge_error: max retries exceeded"


def score_composite(category_correct: bool, summary_score: int, error: str | None) -> bool:
    """
    A case 'passes' if there was no hard error, the category is correct, and
    the summary is judged at least a 3/5. Tune this bar as your product bar changes.
    """
    if error:
        return False
    return category_correct and summary_score >= 3


async def score_case(case: GoldenCase, result: ClassificationResult) -> CaseScore:
    if result.output is None:
        return CaseScore(
            case_id=case.id,
            difficulty=case.difficulty,
            predicted_category=None,
            expected_category=case.expected_category,
            category_correct=False,
            predicted_summary=None,
            expected_summary=case.expected_summary,
            summary_score=0,
            judge_reasoning="no valid output",
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            error=result.error or "empty output",
            passed=False,
        )

    category_correct = result.output.category == case.expected_category
    summary_score, reasoning = await judge_summary(
        case.input_email, case.expected_summary, result.output.summary
    )
    passed = score_composite(category_correct, summary_score, result.error)

    return CaseScore(
        case_id=case.id,
        difficulty=case.difficulty,
        predicted_category=result.output.category,
        expected_category=case.expected_category,
        category_correct=category_correct,
        predicted_summary=result.output.summary,
        expected_summary=case.expected_summary,
        summary_score=summary_score,
        judge_reasoning=reasoning,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        error=result.error,
        passed=passed,
    )