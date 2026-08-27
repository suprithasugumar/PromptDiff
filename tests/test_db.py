"""Unit tests for SQLite database persistence and foreign key enforcement."""

import sqlite3
from pathlib import Path
from promptdiff.db import (
    get_connection,
    get_run_detail,
    get_runs_for_suite,
    get_suites,
    init_db,
    record_diff_report,
    record_run,
)
from promptdiff.models import (
    DiffReport,
    ExpectationsCheckResult,
    JudgeVerdict,
    RunOutput,
    TargetConfig,
    TestCaseDiffResult,
    TestCaseResult,
)


def test_sqlite_foreign_keys_pragma_enabled(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.execute("PRAGMA foreign_keys;")
    result = cursor.fetchone()
    assert result[0] == 1  # Must explicitly be enabled (1)


def test_record_and_query_runs(tmp_path: Path):
    db_path = tmp_path / "test.db"
    run_output = RunOutput(
        run_id="run_101",
        timestamp="2026-08-27T10:00:00Z",
        suite_name="test-suite-a",
        target=TargetConfig(provider="gemini", model="gemini-3.6-flash"),
        results=[
            TestCaseResult(
                test_case_id="tc_1",
                input="Test prompt",
                output="Test output",
                latency_ms=15.0,
                prompt_tokens=10,
                completion_tokens=8,
            )
        ],
    )

    # 1. Record baseline run
    record_run(run_output, is_baseline=True, db_path=db_path)

    suites = get_suites(db_path=db_path)
    assert len(suites) == 1
    assert suites[0]["suite_name"] == "test-suite-a"
    assert suites[0]["total_runs"] == 1
    assert suites[0]["baseline_runs"] == 1

    runs = get_runs_for_suite("test-suite-a", db_path=db_path)
    assert len(runs) == 1
    assert runs[0]["id"] == "run_101"
    assert runs[0]["status"] == "BASELINE"

    detail = get_run_detail("run_101", db_path=db_path)
    assert detail is not None
    assert detail["id"] == "run_101"
    assert len(detail["cases"]) == 1
    assert detail["cases"][0]["test_case_id"] == "tc_1"


def test_record_diff_report_and_cascade(tmp_path: Path):
    db_path = tmp_path / "test.db"
    report = DiffReport(
        suite_name="test-suite-b",
        baseline_run_id="base_run_1",
        baseline_timestamp="2026-08-27T09:00:00Z",
        new_run_id="diff_run_2",
        new_timestamp="2026-08-27T09:05:00Z",
        target=TargetConfig(provider="gemini", model="gemini-3.6-flash"),
        total_cases=1,
        passed_cases=0,
        regressed_cases=1,
        improved_cases=0,
        error_cases=0,
        judge_calls_count=1,
        case_diffs=[
            TestCaseDiffResult(
                test_case_id="tc_regressed",
                input="Can you refund me?",
                status="regressed",
                similarity_score=0.65,
                baseline_output="Sure, refunding.",
                new_output="No refunds.",
                expectations_result=ExpectationsCheckResult(passed=False, failures=["Missing refund"]),
                flagged_for_judge=True,
                judge_verdict=JudgeVerdict(
                    reasoning="Output refuses valid refund.",
                    verdict="worse",
                    category="new_refusal",
                    confidence=0.95,
                ),
            )
        ],
    )

    record_diff_report(report, db_path=db_path)

    detail = get_run_detail("diff_run_2", db_path=db_path)
    assert detail is not None
    assert detail["status"] == "REGRESSED"
    assert detail["regressed_cases"] == 1
    assert len(detail["cases"]) == 1
    assert detail["cases"][0]["judge_verdict"] == "worse"
    assert detail["cases"][0]["judge_category"] == "new_refusal"

    # Test ON DELETE CASCADE
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM runs WHERE id = 'diff_run_2'")

    with get_connection(db_path) as conn:
        remaining_cases = conn.execute("SELECT COUNT(*) FROM test_case_results WHERE run_id = 'diff_run_2'").fetchone()[0]
        assert remaining_cases == 0
