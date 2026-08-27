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

    __test__ = False

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

    provider: str = Field(
        default="gemini",
        description="LLM provider backend ('gemini' or 'anthropic').",
    )
    model: str = Field(
        default="gemini-3.6-flash",
        description="Model identifier to invoke.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt defining the persona or behavior of the model.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional sampling temperature for providers/models that support it.",
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens to generate.",
    )


class TestSuite(BaseModel):
    """Complete test suite specification loaded from YAML."""

    __test__ = False

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

    __test__ = False

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


class ExpectationsCheckResult(BaseModel):
    """Result of programmatic expectation assertions."""

    passed: bool = True
    failures: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Structured verdict from LLM-as-judge evaluation."""

    reasoning: str
    verdict: str = Field(
        ...,
        description="Outcome of comparison: 'better', 'worse', or 'equivalent'.",
    )
    category: str = Field(
        default="none",
        description="Classification of difference: meaning_shift, tone_shift, new_refusal, length_violation, etc.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TestCaseDiffResult(BaseModel):
    """Comparison result for a single test case between baseline and new run."""

    __test__ = False

    test_case_id: str
    input: str
    status: str = Field(
        ...,
        description="Overall status: 'pass', 'regressed', 'improved', or 'error'.",
    )
    similarity_score: float | None = None
    baseline_output: str | None = None
    new_output: str | None = None
    expectations_result: ExpectationsCheckResult = Field(
        default_factory=ExpectationsCheckResult
    )
    flagged_for_judge: bool = False
    flag_reasons: list[str] = Field(default_factory=list)
    judge_verdict: JudgeVerdict | None = None
    latency_delta_ms: float = 0.0
    token_delta: int = 0
    error: str | None = None


class DiffReport(BaseModel):
    """Aggregated comparison report between a baseline run and a new run."""

    suite_name: str
    baseline_run_id: str
    baseline_timestamp: str
    new_run_id: str
    new_timestamp: str
    target: TargetConfig
    total_cases: int = 0
    passed_cases: int = 0
    regressed_cases: int = 0
    improved_cases: int = 0
    error_cases: int = 0
    judge_calls_count: int = 0
    case_diffs: list[TestCaseDiffResult] = Field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True if any test case regressed or produced an error."""
        return self.regressed_cases > 0 or self.error_cases > 0

