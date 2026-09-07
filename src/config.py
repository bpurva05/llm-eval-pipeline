"""
Typed interface contract for the eval pipeline.

Everything the eval engine consumes or produces is a Pydantic model so that
malformed LLM output fails loudly (and gets scored as a failure) instead of
silently corrupting a run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "technical", "account", "general"]

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
DATASET_DIR = REPO_ROOT / "golden_dataset"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


class FewShotExample(BaseModel):
    input: str
    output: str


class PromptConfig(BaseModel):
    """A single versioned prompt. This is the 'code' the CI pipeline diffs against."""

    version: str
    created_at: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    author: Optional[str] = None
    change_notes: Optional[str] = None
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "PromptConfig":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def load_version(cls, version: str) -> "PromptConfig":
        """Load prompts/classifier_<version>.yaml, e.g. load_version('v2')."""
        path = PROMPTS_DIR / f"classifier_{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No prompt file for version '{version}' at {path}")
        return cls.load(path)

    @classmethod
    def latest(cls) -> "PromptConfig":
        """Load the highest-numbered classifier_vN.yaml in /prompts."""
        candidates = sorted(PROMPTS_DIR.glob("classifier_v*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No prompt files found in {PROMPTS_DIR}")
        return cls.load(candidates[-1])


class ClassificationOutput(BaseModel):
    """The structured output contract the classifier function must satisfy."""

    category: Category
    summary: str

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("summary must not be empty")
        return v.strip()


class GoldenCase(BaseModel):
    id: str
    input_email: str
    expected_category: Category
    expected_summary: str
    difficulty: Literal["easy", "medium", "hard", "edge"]
    notes: str = ""


class GoldenDataset(BaseModel):
    dataset_version: str
    created_at: str
    description: str
    cases: list[GoldenCase]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GoldenDataset":
        if path is None:
            candidates = sorted(DATASET_DIR.glob("dataset_v*.json"))
            if not candidates:
                raise FileNotFoundError(f"No dataset files found in {DATASET_DIR}")
            path = candidates[-1]
        import json

        with open(path, "r") as f:
            raw = json.load(f)
        return cls(**raw)
