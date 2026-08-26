"""Unit tests for CLI execution and storage."""

from pathlib import Path
from typer.testing import CliRunner
from promptdiff.cli import app
from promptdiff.models import TestSuite
from promptdiff.runner import SuiteRunner
from promptdiff.storage import save_run

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PromptDiff" in result.stdout
    assert "run" in result.stdout


def test_cli_run_dry_run(tmp_path: Path):
    yaml_path = "examples/support_bot/test_cases.yaml"
    result = runner.invoke(
        app,
        ["run", yaml_path, "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "support-reply-generator" in result.stdout
    assert "Run results saved to" in result.stdout

    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1
    assert "support-reply-generator" in saved_files[0].name


def test_runner_dry_run():
    suite = TestSuite.from_yaml("examples/support_bot/test_cases.yaml")
    suite_runner = SuiteRunner()
    run_output = suite_runner.run_suite(suite, dry_run=True)

    assert len(run_output.results) == 4
    assert all(r.error is None for r in run_output.results)
    assert all("[DRY RUN Mock Output" in (r.output or "") for r in run_output.results)
