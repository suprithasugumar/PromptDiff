"""Command-line interface for PromptDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptdiff import __version__
from promptdiff.models import TestCase, TestCaseResult, TestSuite
from promptdiff.runner import SuiteRunner
from promptdiff.storage import save_run

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
    output_dir: Path = typer.Option(
        Path("runs"),
        "--output-dir",
        "-o",
        help="Directory where raw run JSON results will be stored.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and test cases without calling the model API.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable detailed logging of model requests and responses.",
    ),
):
    """Run a test suite against the target model and save raw outputs."""
    try:
        suite = TestSuite.from_yaml(config)
    except Exception as exc:
        console.print(f"[bold red]Error loading test suite:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Suite:[/bold cyan] {suite.name}\n"
            f"[bold cyan]Target Model:[/bold cyan] {suite.target.model}\n"
            f"[bold cyan]Test Cases:[/bold cyan] {len(suite.test_cases)}\n"
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
            console.print(f"\n[bold yellow]─── {case.id} ───[/bold yellow]")
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

    saved_path = save_run(run_output, output_dir=output_dir)
    console.print(f"\n[bold green]✓[/bold green] Run results saved to [bold underline]{saved_path}[/bold underline]")


if __name__ == "__main__":
    app()
