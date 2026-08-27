"""SQLite persistence and query module for PromptDiff run history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from promptdiff.models import DiffReport, RunOutput, TestCaseDiffResult, TestCaseResult

DEFAULT_DB_PATH = "promptdiff.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    suite_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    is_baseline BOOLEAN NOT NULL DEFAULT 0,
    baseline_run_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT,
    total_cases INTEGER NOT NULL DEFAULT 0,
    passed_cases INTEGER NOT NULL DEFAULT 0,
    regressed_cases INTEGER NOT NULL DEFAULT 0,
    improved_cases INTEGER NOT NULL DEFAULT 0,
    error_cases INTEGER NOT NULL DEFAULT 0,
    judge_calls_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_suite ON runs(suite_name);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp DESC);

CREATE TABLE IF NOT EXISTS test_case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    test_case_id TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT,
    baseline_output TEXT,
    status TEXT NOT NULL,
    similarity_score REAL,
    flagged_for_judge BOOLEAN NOT NULL DEFAULT 0,
    judge_verdict TEXT,
    judge_category TEXT,
    judge_reasoning TEXT,
    judge_confidence REAL,
    expectation_passed BOOLEAN NOT NULL DEFAULT 1,
    expectation_failures TEXT,
    latency_ms REAL DEFAULT 0.0,
    latency_delta_ms REAL DEFAULT 0.0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_tc_run_id ON test_case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_tc_case_id ON test_case_results(test_case_id);
"""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a connection with foreign key support enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Crucial requirement: explicitly enable foreign keys on every SQLite connection
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite tables and indexes."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def record_run(
    run_output: RunOutput,
    is_baseline: bool = False,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Record a raw or baseline execution run in SQLite."""
    init_db(db_path)

    total_cases = len(run_output.results)
    error_cases = sum(1 for r in run_output.results if r.error)
    passed_cases = total_cases - error_cases

    status = "BASELINE" if is_baseline else ("ERROR" if error_cases > 0 else "PASS")

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                id, suite_name, timestamp, is_baseline, baseline_run_id,
                provider, model, system_prompt, total_cases, passed_cases,
                regressed_cases, improved_cases, error_cases, judge_calls_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_output.run_id,
                run_output.suite_name,
                run_output.timestamp,
                1 if is_baseline else 0,
                None,
                run_output.target.provider,
                run_output.target.model,
                run_output.target.system_prompt,
                total_cases,
                passed_cases,
                0,
                0,
                error_cases,
                0,
                status,
            ),
        )

        for res in run_output.results:
            tc_status = "error" if res.error else ("baseline" if is_baseline else "pass")
            conn.execute(
                """
                INSERT INTO test_case_results (
                    run_id, test_case_id, input, output, baseline_output,
                    status, similarity_score, flagged_for_judge, judge_verdict,
                    judge_category, judge_reasoning, judge_confidence,
                    expectation_passed, expectation_failures,
                    latency_ms, latency_delta_ms, prompt_tokens, completion_tokens, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_output.run_id,
                    res.test_case_id,
                    res.input,
                    res.output,
                    res.output if is_baseline else None,
                    tc_status,
                    1.0 if is_baseline else None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    1,
                    json.dumps([]),
                    res.latency_ms,
                    0.0,
                    res.prompt_tokens,
                    res.completion_tokens,
                    res.error,
                ),
            )


def record_diff_report(
    report: DiffReport,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Record a full diff evaluation report with per-case details in SQLite."""
    init_db(db_path)

    status = (
        "ERROR"
        if report.error_cases > 0
        else ("REGRESSED" if report.regressed_cases > 0 else "PASS")
    )

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                id, suite_name, timestamp, is_baseline, baseline_run_id,
                provider, model, system_prompt, total_cases, passed_cases,
                regressed_cases, improved_cases, error_cases, judge_calls_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.new_run_id,
                report.suite_name,
                report.new_timestamp,
                0,
                report.baseline_run_id,
                report.target.provider,
                report.target.model,
                report.target.system_prompt,
                report.total_cases,
                report.passed_cases,
                report.regressed_cases,
                report.improved_cases,
                report.error_cases,
                report.judge_calls_count,
                status,
            ),
        )

        for case_diff in report.case_diffs:
            judge_verdict = case_diff.judge_verdict.verdict if case_diff.judge_verdict else None
            judge_category = case_diff.judge_verdict.category if case_diff.judge_verdict else None
            judge_reasoning = case_diff.judge_verdict.reasoning if case_diff.judge_verdict else None
            judge_confidence = case_diff.judge_verdict.confidence if case_diff.judge_verdict else None

            conn.execute(
                """
                INSERT INTO test_case_results (
                    run_id, test_case_id, input, output, baseline_output,
                    status, similarity_score, flagged_for_judge, judge_verdict,
                    judge_category, judge_reasoning, judge_confidence,
                    expectation_passed, expectation_failures,
                    latency_ms, latency_delta_ms, prompt_tokens, completion_tokens, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.new_run_id,
                    case_diff.test_case_id,
                    case_diff.input,
                    case_diff.new_output,
                    case_diff.baseline_output,
                    case_diff.status,
                    case_diff.similarity_score,
                    1 if case_diff.flagged_for_judge else 0,
                    judge_verdict,
                    judge_category,
                    judge_reasoning,
                    judge_confidence,
                    1 if case_diff.expectations_result.passed else 0,
                    json.dumps(case_diff.expectations_result.failures),
                    0.0,
                    case_diff.latency_delta_ms,
                    0,
                    0,
                    case_diff.error,
                ),
            )


def get_suites(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all test suites with summary stats from SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT 
                suite_name,
                COUNT(*) as total_runs,
                MAX(timestamp) as last_run_at,
                SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) as passed_runs,
                SUM(CASE WHEN status = 'REGRESSED' THEN 1 ELSE 0 END) as regressed_runs,
                SUM(CASE WHEN is_baseline = 1 THEN 1 ELSE 0 END) as baseline_runs
            FROM runs
            GROUP BY suite_name
            ORDER BY last_run_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_runs_for_suite(
    suite_name: str | None = None,
    limit: int = 50,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve history of runs for a specific suite (or all suites if None)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if suite_name:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE suite_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (suite_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_run_detail(
    run_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Retrieve complete run metadata and its per-test-case results."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run_row:
            return None

        cases_rows = conn.execute(
            "SELECT * FROM test_case_results WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()

        run_dict = dict(run_row)
        cases_list = []
        for c in cases_rows:
            cd = dict(c)
            if cd.get("expectation_failures"):
                try:
                    cd["expectation_failures"] = json.loads(cd["expectation_failures"])
                except Exception:
                    pass
            cases_list.append(cd)

        run_dict["cases"] = cases_list
        return run_dict
