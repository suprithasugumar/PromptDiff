"""Command-line interface for PromptDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import webbrowser
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptdiff import __version__
from promptdiff.db import (
    DEFAULT_DB_PATH,
    get_run_detail,
    get_runs_for_suite,
    get_suites,
    record_diff_report,
    record_run,
)
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
    db: Path = typer.Option(
        Path(DEFAULT_DB_PATH),
        "--db",
        help="Path to the SQLite database file for run history.",
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

    # Always save the individual run output to disk
    saved_path = save_run(run_output, output_dir=output_dir)
    console.print(f"\n[dim]Run results saved to JSON: {saved_path}[/dim]")

    if baseline:
        baseline_path = save_baseline(run_output, output_dir=output_dir)
        record_run(run_output, is_baseline=True, db_path=db)
        console.print(
            f"[bold green]Baseline established:[/bold green] Saved as suite baseline at [bold underline]{baseline_path}[/bold underline] and recorded in SQLite ([bold]{db}[/bold])\n"
        )
        raise typer.Exit(code=0)

    # If not a baseline run, automatically diff against the latest baseline
    existing_baseline = get_latest_baseline(suite.name, output_dir=output_dir)
    if existing_baseline is None:
        # Record standalone non-baseline run into SQLite
        record_run(run_output, is_baseline=False, db_path=db)
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

    # Persist diff report to SQLite
    record_diff_report(report, db_path=db)

    console.print()
    render_diff_report(report, console=console)

    if report.has_regressions:
        console.print(
            f"\n[bold red]FAILURE:[/bold red] Detected regressions or errors in test suite '{suite.name}'. (Recorded to {db})\n"
        )
        raise typer.Exit(code=1)
    else:
        console.print(
            f"\n[bold green]SUCCESS:[/bold green] No regressions detected across all test cases. (Recorded to {db})\n"
        )
        raise typer.Exit(code=0)


@app.command(name="history")
def history(
    suite_name: Optional[str] = typer.Argument(
        None,
        help="Optional test suite name to filter history.",
    ),
    limit: int = typer.Option(
        15,
        "--limit",
        "-n",
        help="Maximum number of historical runs to display.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Inspect granular test case results for a specific run ID.",
    ),
    db: Path = typer.Option(
        Path(DEFAULT_DB_PATH),
        "--db",
        help="Path to SQLite database file.",
    ),
):
    """Query past test runs and regression trends from SQLite history."""
    if run_id:
        detail = get_run_detail(run_id, db_path=db)
        if not detail:
            console.print(f"[bold red]Error:[/bold red] Run ID '{run_id}' not found in database {db}.")
            raise typer.Exit(code=1)

        console.print(
            Panel(
                f"[bold cyan]Run ID:[/bold cyan] {detail['id']}\n"
                f"[bold cyan]Suite:[/bold cyan] {detail['suite_name']}\n"
                f"[bold cyan]Timestamp:[/bold cyan] {detail['timestamp']}\n"
                f"[bold cyan]Model:[/bold cyan] {detail['provider']} ({detail['model']})\n"
                f"[bold cyan]Status:[/bold cyan] {detail['status']}",
                title="[bold green]Run Detail[/bold green]",
                expand=False,
            )
        )

        table = Table(title="Test Case Breakdown", show_header=True, header_style="bold magenta")
        table.add_column("Test Case ID", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Similarity", justify="right")
        table.add_column("Judge Verdict", justify="center")
        table.add_column("Reason / Failures", style="dim", max_width=50)

        for c in detail.get("cases", []):
            if c["status"] == "pass":
                sc = "[bold green]PASS[/bold green]"
            elif c["status"] == "regressed":
                sc = "[bold red]REGRESSED[/bold red]"
            elif c["status"] == "improved":
                sc = "[bold cyan]IMPROVED[/bold cyan]"
            elif c["status"] == "baseline":
                sc = "[bold purple]BASELINE[/bold purple]"
            else:
                sc = "[bold red]ERROR[/bold red]"

            sim = f"{c['similarity_score']:.3f}" if c["similarity_score"] is not None else "-"
            jv = c["judge_verdict"] or "-"
            if jv == "worse":
                jv = "[bold red]WORSE[/bold red]"
            elif jv == "better":
                jv = "[bold cyan]BETTER[/bold cyan]"
            elif jv == "equivalent":
                jv = "[yellow]EQUIV[/yellow]"

            reason = c.get("judge_reasoning") or ""
            if c.get("expectation_failures"):
                reason = "; ".join(c["expectation_failures"])

            table.add_row(c["test_case_id"], sc, sim, jv, reason)

        console.print(table)
        return

    runs = get_runs_for_suite(suite_name, limit=limit, db_path=db)
    if not runs:
        console.print(
            f"[bold yellow]No runs found in database {db}.[/bold yellow] Execute [bold cyan]promptdiff run <config>[/bold cyan] to generate history."
        )
        return

    table = Table(
        title=f"Run History ({f'Suite: {suite_name}' if suite_name else 'All Suites'})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Run ID", style="cyan", min_width=24)
    table.add_column("Timestamp (UTC)", style="dim")
    table.add_column("Suite", style="white")
    table.add_column("Model", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("P / I / R", justify="center")
    table.add_column("Judge", justify="right")

    for r in runs:
        if r["status"] == "PASS":
            status_chip = "[bold green]PASS[/bold green]"
        elif r["status"] == "REGRESSED":
            status_chip = "[bold red]REGRESSED[/bold red]"
        elif r["status"] == "BASELINE":
            status_chip = "[bold purple]BASELINE[/bold purple]"
        else:
            status_chip = "[bold red]ERROR[/bold red]"

        pir = f"[green]{r['passed_cases']}[/green]/[cyan]{r['improved_cases']}[/cyan]/[red]{r['regressed_cases']}[/red]"
        timestamp_str = r["timestamp"].replace("T", " ")[:19]

        table.add_row(
            r["id"],
            timestamp_str,
            r["suite_name"],
            f"{r['provider']}/{r['model']}",
            status_chip,
            pir,
            str(r["judge_calls_count"]),
        )

    console.print(table)


@app.command(name="dashboard")
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the dashboard on."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not automatically open the browser."),
    db: Path = typer.Option(Path(DEFAULT_DB_PATH), "--db", help="Path to SQLite database."),
):
    """Launch the local web dashboard for trend visualization and diff inspection."""
    import uvicorn

    url = f"http://{host}:{port}"
    console.print(
        Panel(
            f"[bold cyan]PromptDiff Dashboard[/bold cyan]\n\n"
            f"• URL: [bold underline green]{url}[/bold underline green]\n"
            f"• Database: [bold]{db.resolve()}[/bold]\n\n"
            f"[dim]Press Ctrl+C in terminal to stop the server.[/dim]",
            title="[bold green]Web Dashboard[/bold green]",
            expand=False,
        )
    )

    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(
        "promptdiff.dashboard.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    app()


