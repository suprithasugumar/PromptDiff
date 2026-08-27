"""Unit tests for DiffEngine orchestration."""

from promptdiff.diff import DiffEngine
from promptdiff.embeddings import MockEmbedder
from promptdiff.judge import LLMJudge
from promptdiff.models import (
    Expectations,
    RunOutput,
    TargetConfig,
    TestCase,
    TestCaseResult,
    TestSuite,
)
from promptdiff.providers.base import GenerationResult


class MockJudgeProvider:
    def __init__(self, verdict: str = "equivalent", category: str = "none"):
        self.verdict = verdict
        self.category = category

    def generate(self, user_input: str, **kwargs) -> GenerationResult:
        return GenerationResult(
            text=f'{{"reasoning": "Mock evaluation.", "verdict": "{self.verdict}", "category": "{self.category}", "confidence": 0.9}}'
        )


def test_diff_identical_runs():
    suite = TestSuite(
        name="test_suite",
        test_cases=[
            TestCase(
                id="case1",
                input="Hello",
                expectations=Expectations(must_mention=["hello"]),
            )
        ],
    )

    baseline = RunOutput(
        run_id="run_base",
        timestamp="2026-08-27T10:00:00Z",
        suite_name="test_suite",
        target=TargetConfig(),
        results=[
            TestCaseResult(
                test_case_id="case1",
                input="Hello",
                output="Hello there, how can I help you?",
            )
        ],
    )

    current = RunOutput(
        run_id="run_current",
        timestamp="2026-08-27T10:05:00Z",
        suite_name="test_suite",
        target=TargetConfig(),
        results=[
            TestCaseResult(
                test_case_id="case1",
                input="Hello",
                output="Hello there, how can I help you?",
            )
        ],
    )

    engine = DiffEngine(
        embedder=MockEmbedder(),
        judge=LLMJudge(provider=MockJudgeProvider()),
        similarity_threshold=0.88,
    )

    report = engine.diff(baseline, current, suite, dry_run=False)

    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.regressed_cases == 0
    assert report.improved_cases == 0
    assert report.judge_calls_count == 0  # skipped because identical similarity = 1.0 >= 0.88
    assert report.has_regressions is False


def test_diff_flagged_regression():
    suite = TestSuite(
        name="test_suite",
        test_cases=[
            TestCase(
                id="case1",
                input="I need a refund",
                expectations=Expectations(must_mention=["refund"]),
            )
        ],
    )

    baseline = RunOutput(
        run_id="run_base",
        timestamp="2026-08-27T10:00:00Z",
        suite_name="test_suite",
        target=TargetConfig(),
        results=[
            TestCaseResult(
                test_case_id="case1",
                input="I need a refund",
                output="We will process your refund promptly.",
            )
        ],
    )

    # Current output violates expectation (missing keyword 'refund')
    current = RunOutput(
        run_id="run_current",
        timestamp="2026-08-27T10:05:00Z",
        suite_name="test_suite",
        target=TargetConfig(),
        results=[
            TestCaseResult(
                test_case_id="case1",
                input="I need a refund",
                output="Sorry, we cannot help you.",
            )
        ],
    )

    engine = DiffEngine(
        embedder=MockEmbedder(),
        judge=LLMJudge(provider=MockJudgeProvider(verdict="worse", category="new_refusal")),
        similarity_threshold=0.88,
    )

    report = engine.diff(baseline, current, suite, dry_run=False)

    assert report.total_cases == 1
    assert report.regressed_cases == 1
    assert report.has_regressions is True
    assert report.judge_calls_count == 1
    assert report.case_diffs[0].status == "regressed"
    assert report.case_diffs[0].judge_verdict.verdict == "worse"
