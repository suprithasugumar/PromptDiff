"""Command-line interface for PromptDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptdiff import __version__
from promptdiff.diff import DiffEngine
from promptdiff.models import TestCase, TestCaseResult, TestSuite
from promptdiff.report import render_diff_report
from promptdiff.runner import SuiteRunner
from promptdiff.storage import get_latest_baseline, save_baseline, save_run

app = typer.Typer(
    name="promptdiff",
    help="PromptDiff: CLI tool for LLM regression testing and observability.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]PromptDiff[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """PromptDiff CLI."""
    pass


@app.command(name="run")
def run(
    config: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the test suite YAML configuration file.",
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        "-b",
        help="Save this execution run as the primary baseline for the test suite.",
    ),
    threshold: float = typer.Option(
        0.88,
        "--threshold",
        "-t",
        help="Cosine similarity threshold (0.0 - 1.0) below which cases are flagged for LLM judge.",
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        "--output-dir",
        "-o",
        help="Directory where run JSON results are stored.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and test cases without calling external model APIs.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable detailed logging of model requests and responses.",
    ),
):
    """Run a test suite against the target model and diff against the baseline."""
    try:
        suite = TestSuite.from_yaml(config)
    except Exception as exc:
        console.print(f"[bold red]Error loading test suite:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Suite:[/bold cyan] {suite.name}\n"
            f"[bold cyan]Provider:[/bold cyan] {suite.target.provider}\n"
            f"[bold cyan]Target Model:[/bold cyan] {suite.target.model}\n"
            f"[bold cyan]Test Cases:[/bold cyan] {len(suite.test_cases)}\n"
            f"[bold cyan]Baseline Run:[/bold cyan] {'Yes (Saving as Baseline)' if baseline else 'No (Diffing against existing)'}\n"
            f"[bold cyan]Dry Run:[/bold cyan] {'Yes' if dry_run else 'No'}",
            title="[bold green]PromptDiff Run[/bold green]",
            expand=False,
        )
    )

    runner = SuiteRunner()

    table = Table(title="Execution Progress", show_header=True, header_style="bold magenta")
    table.add_column("Test Case ID", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens (in/out)", justify="right")

    def on_case_complete(case: TestCase, result: TestCaseResult):
        if result.error:
            status_text = "[bold red]ERROR[/bold red]"
            tokens_text = "-"
        else:
            status_text = "[bold green]OK[/bold green]"
            tokens_text = f"{result.prompt_tokens}/{result.completion_tokens}"

        table.add_row(
            case.id,
            status_text,
            f"{result.latency_ms:.1f}",
            tokens_text,
        )

        if verbose:
            console.print(f"\n[bold yellow]--- {case.id} ---[/bold yellow]")
            console.print(f"[dim]Input:[/dim] {case.input}")
            if result.output:
                console.print(f"[dim]Output:[/dim]\n{result.output}")
            if result.error:
                console.print(f"[red]Error:[/red] {result.error}")

    with console.status("[bold blue]Executing test cases...[/bold blue]"):
        try:
            run_output = runner.run_suite(
                suite,
                dry_run=dry_run,
                progress_callback=on_case_complete,
            )
        except Exception as exc:
            console.print(f"[bold red]Execution error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print(table)

    # Always save the individual run output
    saved_path = save_run(run_output, output_dir=output_dir)
    console.print(f"\n[dim]Run results saved to: {saved_path}[/dim]")

    if baseline:
        baseline_path = save_baseline(run_output, output_dir=output_dir)
        console.print(
            f"[bold green]Baseline established:[/bold green] Saved as suite baseline at [bold underline]{baseline_path}[/bold underline]\n"
        )
        raise typer.Exit(code=0)

    # If not a baseline run, automatically diff against the latest baseline
    existing_baseline = get_latest_baseline(suite.name, output_dir=output_dir)
    if existing_baseline is None:
        console.print(
            f"\n[bold yellow]Notice:[/bold yellow] No baseline found for suite '{suite.name}'. "
            f"Run [bold cyan]promptdiff run <config> --baseline[/bold cyan] to establish a baseline.\n"
        )
        raise typer.Exit(code=0)

    # Perform embedding drift diffing + LLM Judge scoring
    console.print("\n[bold blue]Diffing current run against baseline...[/bold blue]")
    diff_engine = DiffEngine(similarity_threshold=threshold)

    with console.status("[bold blue]Computing embedding drift and running judge...[/bold blue]"):
        try:
            report = diff_engine.diff(
                baseline_run=existing_baseline,
                current_run=run_output,
                suite=suite,
                dry_run=dry_run,
            )
        except Exception as exc:
            console.print(f"[bold red]Diffing error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print()
    render_diff_report(report, console=console)

    if report.has_regressions:
        console.print(
            f"\n[bold red]FAILURE:[/bold red] Detected regressions or errors in test suite '{suite.name}'.\n"
        )
        raise typer.Exit(code=1)
    else:
        console.print(
            f"\n[bold green]SUCCESS:[/bold green] No regressions detected across all test cases.\n"
        )
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()

