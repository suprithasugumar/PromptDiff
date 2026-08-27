"""Unit tests for report rendering and markdown report generation."""

from promptdiff.models import (
    DiffReport,
    ExpectationsCheckResult,
    JudgeVerdict,
    TargetConfig,
    TestCaseDiffResult,
)
from promptdiff.report import generate_markdown_report


def test_generate_markdown_report_passed():
    report = DiffReport(
        suite_name="test-suite",
        baseline_run_id="run-base-001",
        baseline_timestamp="2026-08-27T10:00:00Z",
        new_run_id="run-new-002",
        new_timestamp="2026-08-27T10:05:00Z",
        target=TargetConfig(provider="gemini", model="gemini-2.5-flash"),
        total_cases=2,
        passed_cases=2,
        regressed_cases=0,
        improved_cases=0,
        error_cases=0,
        judge_calls_count=0,
        case_diffs=[
            TestCaseDiffResult(
                test_case_id="tc_1",
                input="Hello",
                status="pass",
                similarity_score=0.98,
                baseline_output="Hi there!",
                new_output="Hello there!",
            ),
            TestCaseDiffResult(
                test_case_id="tc_2",
                input="Help",
                status="pass",
                similarity_score=0.95,
                baseline_output="How can I help?",
                new_output="How may I assist you?",
            ),
        ],
    )

    md = generate_markdown_report(report)
    assert "## 🟢 PromptDiff: All Checks Passed" in md
    assert "**PASSED**" in md
    assert "`test-suite`" in md
    assert "`gemini (gemini-2.5-flash)`" in md
    assert "| **2** | 2 | 0 | 0 | 0 | 0 | **PASSED** |" in md


def test_generate_markdown_report_with_regressions_and_improvements():
    report = DiffReport(
        suite_name="support-eval",
        baseline_run_id="run-base-100",
        baseline_timestamp="2026-08-27T10:00:00Z",
        new_run_id="run-new-200",
        new_timestamp="2026-08-27T10:10:00Z",
        target=TargetConfig(provider="gemini", model="gemini-2.5-flash"),
        total_cases=3,
        passed_cases=1,
        regressed_cases=1,
        improved_cases=1,
        error_cases=0,
        judge_calls_count=2,
        case_diffs=[
            TestCaseDiffResult(
                test_case_id="tc_pass",
                input="Greeting",
                status="pass",
                similarity_score=0.96,
                baseline_output="Hello!",
                new_output="Hello! How are you?",
            ),
            TestCaseDiffResult(
                test_case_id="tc_regressed",
                input="Refund please",
                status="regressed",
                similarity_score=0.72,
                baseline_output="I am sorry for the trouble. Your refund has been processed in 3-5 days.",
                new_output="Refund processed.",
                expectations_result=ExpectationsCheckResult(
                    passed=False, failures=["Must mention: 'sorry'"]
                ),
                flagged_for_judge=True,
                judge_verdict=JudgeVerdict(
                    verdict="worse",
                    reasoning="Omitted standard apology and refund timeline.",
                    category="tone_shift",
                    confidence=0.92,
                ),
            ),
            TestCaseDiffResult(
                test_case_id="tc_improved",
                input="How to reset password?",
                status="improved",
                similarity_score=0.84,
                baseline_output="Click forgot password.",
                new_output="Go to settings, click security, and select reset password.",
                flagged_for_judge=True,
                judge_verdict=JudgeVerdict(
                    verdict="better",
                    reasoning="Provided clearer step-by-step instructions.",
                    category="more_detailed",
                    confidence=0.88,
                ),
            ),
        ],
    )

    md = generate_markdown_report(report)
    assert "## 🔴 PromptDiff: Regression Detected" in md
    assert "**FAILED**" in md
    assert "### ⚠️ Regressed Cases (1)" in md
    assert "#### ❌ `tc_regressed`" in md
    assert "Cosine Similarity:** `0.720`" in md
    assert "Judge Verdict:** `WORSE`" in md
    assert "Must mention: 'sorry'" in md
    assert "<details>" in md
    assert "Baseline Output:" in md
    assert "### 🟢 Improved Cases (1)" in md
    assert "`tc_improved`" in md
