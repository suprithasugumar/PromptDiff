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

