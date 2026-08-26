"""Pydantic data models for test cases, suites, configurations, and run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, field_validator


class Expectations(BaseModel):
    """Qualitative and quantitative assertions for a test case."""

    must_mention: list[str] = Field(
        default_factory=list,
        description="Substrings or keywords that the model response must include.",
    )
    must_not_mention: list[str] = Field(
        default_factory=list,
        description="Substrings or keywords that the model response must not include.",
    )
    must_not_refuse: bool = Field(
        default=True,
        description="Whether the response is expected not to refuse the request.",
    )
    max_length_chars: int | None = Field(
        default=None,
        ge=1,
        description="Maximum allowed character length for the output.",
    )


class TestCase(BaseModel):
    """An individual test case definition."""

    id: str = Field(..., description="Unique identifier for the test case.")
    description: str | None = Field(
        default=None, description="Human-readable description of the scenario."
    )
    input: str = Field(..., description="Input prompt/message sent to the model.")
    expectations: Expectations = Field(
        default_factory=Expectations,
        description="Assertions evaluated against the model output.",
    )


class TargetConfig(BaseModel):
    """Configuration of the model/prompt target being evaluated."""

    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model identifier to invoke.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt defining the persona or behavior of the model.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature between 0.0 and 1.0.",
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens to generate.",
    )


class TestSuite(BaseModel):
    """Complete test suite specification loaded from YAML."""

    version: str = Field(
        default="1",
        description="Schema version.",
    )
    name: str = Field(..., description="Name of the test suite.")
    description: str | None = Field(
        default=None, description="Overview of the test suite."
    )
    target: TargetConfig = Field(
        default_factory=TargetConfig,
        description="Target configuration for the suite.",
    )
    test_cases: list[TestCase] = Field(
        default_factory=list,
        description="List of test cases to execute.",
    )

    @field_validator("test_cases")
    @classmethod
    def validate_unique_ids(cls, test_cases: list[TestCase]) -> list[TestCase]:
        seen_ids = set()
        for tc in test_cases:
            if tc.id in seen_ids:
                raise ValueError(f"Duplicate test case id found: '{tc.id}'")
            seen_ids.add(tc.id)
        return test_cases

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestSuite:
        """Load and validate a TestSuite from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Test suite file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {file_path}: Expected a mapping/dict at root.")

        return cls.model_validate(data)


class TestCaseResult(BaseModel):
    """Result of running a single test case against the model."""

    test_case_id: str
    input: str
    output: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class RunOutput(BaseModel):
    """Full execution run capture serialized to JSON."""

    run_id: str
    timestamp: str
    suite_name: str
    target: TargetConfig
    results: list[TestCaseResult] = Field(default_factory=list)
