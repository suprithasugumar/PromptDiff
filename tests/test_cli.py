"""Unit tests for CLI execution, baseline management, and diff reporting."""

from pathlib import Path
from typer.testing import CliRunner
from promptdiff.cli import app
from promptdiff.models import TestSuite
from promptdiff.runner import SuiteRunner

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PromptDiff" in result.stdout
    assert "run" in result.stdout


def test_cli_run_no_baseline_notice(tmp_path: Path):
    yaml_path = "examples/support_bot/test_cases.yaml"
    result = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "support-reply-generator" in result.stdout
    assert "No baseline found for suite" in result.stdout

    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1


def test_cli_run_baseline_and_diff_with_clean_suite(tmp_path: Path):
    # Create a minimal test suite YAML
    clean_yaml = tmp_path / "clean_suite.yaml"
    clean_yaml.write_text(
        """version: "1"
name: "clean-suite"
target:
  provider: "gemini"
  model: "gemini-3.6-flash"
test_cases:
  - id: "tc_simple"
    input: "Hello world"
""",
        encoding="utf-8",
    )

    # 1. Run with --baseline
    base_result = runner.invoke(
        app,
        ["run", str(clean_yaml), "--output-dir", str(tmp_path), "--baseline", "--dry-run"],
    )
    assert base_result.exit_code == 0
    assert "Baseline established" in base_result.stdout

    baseline_files = list(tmp_path.glob("baseline_*.json"))
    assert len(baseline_files) == 1

    # 2. Run again without --baseline (triggers diff)
    diff_result = runner.invoke(
        app,
        ["run", str(clean_yaml), "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert diff_result.exit_code == 0
    assert "Diff Report" in diff_result.stdout
    assert "Total Cases" in diff_result.stdout
    assert "SUCCESS" in diff_result.stdout


def test_cli_run_diff_catches_regression(tmp_path: Path):
    yaml_path = "examples/support_bot/test_cases.yaml"

    # Establish baseline
    base_result = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--baseline", "--dry-run"],
    )
    assert base_result.exit_code == 0

    # Diff against baseline (support bot dry run triggers expectation failures -> regression)
    diff_result = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert diff_result.exit_code == 1
    assert "Diff Report" in diff_result.stdout
    assert "FAILURE" in diff_result.stdout
    assert "Regressed" in diff_result.stdout



def test_runner_dry_run():
    suite = TestSuite.from_yaml("examples/support_bot/test_cases.yaml")
    suite_runner = SuiteRunner()
    run_output = suite_runner.run_suite(suite, dry_run=True)

    assert len(run_output.results) == 4
    assert all(r.error is None for r in run_output.results)
    assert all("[DRY RUN Mock Output" in (r.output or "") for r in run_output.results)


def test_cli_history_empty(tmp_path: Path):
    db_path = tmp_path / "empty.db"
    result = runner.invoke(app, ["history", "--db", str(db_path)])
    assert result.exit_code == 0
    assert "No runs found" in result.stdout


def test_cli_history_and_run_detail(tmp_path: Path):
    yaml_path = "examples/support_bot/test_cases.yaml"
    db_path = tmp_path / "hist.db"

    # 1. Establish baseline
    base_res = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--db", str(db_path), "--baseline", "--dry-run"],
    )
    assert base_res.exit_code == 0

    # 2. Run diff
    diff_res = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--db", str(db_path), "--dry-run"],
    )
    # Regression in dry run causes exit code 1
    assert diff_res.exit_code == 1

    # 3. Query history table
    hist_res = runner.invoke(app, ["history", "--db", str(db_path)])
    assert hist_res.exit_code == 0
    assert "Run History" in hist_res.stdout
    assert "support" in hist_res.stdout
    assert "BASELINE" in hist_res.stdout
    assert "REGRESSED" in hist_res.stdout

    # 4. Query history with suite name filter
    suite_hist_res = runner.invoke(app, ["history", "support-reply-generator", "--db", str(db_path)])
    assert suite_hist_res.exit_code == 0
    assert "support" in suite_hist_res.stdout

    # 5. Query granular run detail via --run-id
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    run_id = conn.execute("SELECT id FROM runs LIMIT 1").fetchone()[0]
    conn.close()

    detail_res = runner.invoke(app, ["history", "--run-id", run_id, "--db", str(db_path)])
    assert detail_res.exit_code == 0
    assert "Run Detail" in detail_res.stdout
    assert "Test Case Breakdown" in detail_res.stdout
    assert "billing" in detail_res.stdout


def test_cli_run_markdown_report_and_fail_on(tmp_path: Path):
    yaml_path = "examples/support_bot/test_cases.yaml"
    md_file = tmp_path / "pr_comment.md"

    # Establish baseline
    base_res = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--baseline", "--dry-run"],
    )
    assert base_res.exit_code == 0

    # 1. Run with --fail-on none and --markdown-report (should exit 0 despite regressions)
    diff_res_none = runner.invoke(
        app,
        [
            "run",
            yaml_path,
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--markdown-report",
            str(md_file),
            "--fail-on",
            "none",
        ],
    )
    assert diff_res_none.exit_code == 0
    assert md_file.exists()
    md_content = md_file.read_text(encoding="utf-8")
    assert "PromptDiff" in md_content
    assert "Summary" in md_content

    # 2. Run with --max-regressions 10 (should exit 0)
    diff_res_max = runner.invoke(
        app,
        [
            "run",
            yaml_path,
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--max-regressions",
            "10",
        ],
    )
    assert diff_res_max.exit_code == 0
