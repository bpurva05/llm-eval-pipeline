"""
Phase 1: the actual LLM-powered feature being evaluated.

This is intentionally a single, small, swappable function. The eval pipeline
in Phases 3-5 doesn't know or care which provider this calls -- it only knows
it gets a ClassificationOutput back for a given PromptConfig + email string.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import ClassificationOutput, PromptConfig

_client: AsyncOpenAI | None = None

MAX_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 8.0
JSON_RETRY_BACKOFF_SECONDS = 1.0


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or put it in a .env file."
            )
        base_url = os.environ.get("OPENAI_BASE_URL")  # e.g. Groq's OpenAI-compatible endpoint
        _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _client


def _parse_retry_after(error_message: str) -> float:
    """Providers like Groq include 'Please try again in 2.445s' in 429 bodies."""
    match = re.search(r"try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1)) + 0.5  # small buffer
    return DEFAULT_BACKOFF_SECONDS


def retry_delay(error_message: str) -> float | None:
    """
    Seconds to wait before retrying, or None if the error is not retryable.

    - 429 rate limits: provider-side throttling, not a prompt-quality signal.
    - json_validate_failed: the provider's JSON mode rejected one generation;
      a fresh attempt usually succeeds, so retry quickly.
    """
    lowered = error_message.lower()
    if "429" in error_message or "rate_limit" in lowered:
        return _parse_retry_after(error_message)
    if "json_validate_failed" in lowered:
        return JSON_RETRY_BACKOFF_SECONDS
    return None


@dataclass
class ClassificationResult:
    output: ClassificationOutput | None
    raw_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


def _build_messages(config: PromptConfig, email_text: str) -> list[dict]:
    messages = [{"role": "system", "content": config.system_prompt}]
    for ex in config.few_shot_examples:
        messages.append({"role": "user", "content": ex.input})
        messages.append({"role": "assistant", "content": ex.output})
    messages.append({"role": "user", "content": email_text})
    return messages


async def classify_email(config: PromptConfig, email_text: str) -> ClassificationResult:
    """
    Run one email through the classifier feature under a given prompt version.

    Returns a ClassificationResult even on failure (bad JSON, API error) so
    the eval runner can score "the model returned garbage" as a failure
    rather than crashing the whole run. Transient provider errors (rate
    limits, rejected JSON generations) are retried first.
    """
    client = _get_client()
    messages = _build_messages(config, email_text)

    start = time.perf_counter()
    last_error: str | None = None
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            last_error = str(e)
            delay = retry_delay(last_error)
            if delay is not None and attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                continue
            break

    latency_ms = (time.perf_counter() - start) * 1000

    if response is None:
        return ClassificationResult(
            output=None, raw_text="", latency_ms=latency_ms,
            prompt_tokens=0, completion_tokens=0, error=last_error,
        )

    raw_text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    try:
        parsed = json.loads(raw_text)
        output = ClassificationOutput(**parsed)
        error = None
    except Exception as e:
        output = None
        error = f"Failed to parse model output as ClassificationOutput: {e}"

    return ClassificationResult(
        output=output,
        raw_text=raw_text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
    )
