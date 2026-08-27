"""Rich terminal reporting for PromptDiff diff results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from promptdiff.models import DiffReport, TestCaseDiffResult


def render_diff_report(report: DiffReport, console: Console | None = None) -> None:
    """Render a styled Rich terminal report from a DiffReport."""
    if console is None:
        console = Console()

    # 1. Header panel
    header_content = (
        f"[bold cyan]Suite:[/bold cyan] {report.suite_name}\n"
        f"[bold cyan]Baseline Run:[/bold cyan] {report.baseline_run_id} ({report.baseline_timestamp[:19].replace('T', ' ')} UTC)\n"
        f"[bold cyan]Current Run:[/bold cyan]  {report.new_run_id}\n"
        f"[bold cyan]Target:[/bold cyan]       {report.target.provider} ({report.target.model})"
    )
    console.print(
        Panel(
            header_content,
            title="[bold green]PromptDiff Diff Report[/bold green]",
            expand=False,
        )
    )

    # 2. Summary stats line
    summary_parts = [
        f"[bold]{report.total_cases}[/bold] Total Cases",
        f"[bold green]{report.passed_cases} Passed[/bold green]",
    ]
    if report.regressed_cases > 0:
        summary_parts.append(f"[bold red]{report.regressed_cases} Regressed[/bold red]")
    if report.improved_cases > 0:
        summary_parts.append(f"[bold cyan]{report.improved_cases} Improved[/bold cyan]")
    if report.error_cases > 0:
        summary_parts.append(f"[bold magenta]{report.error_cases} Errors[/bold magenta]")

    summary_parts.append(f"[dim]({report.judge_calls_count} judge calls triggered)[/dim]")
    summary_text = " | ".join(summary_parts)
    console.print(f"\n[bold]Summary:[/bold] {summary_text}\n")

    # 3. Diff Table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test Case ID", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Similarity", justify="right", min_width=10)
    table.add_column("Judge Call", justify="center", min_width=12)
    table.add_column("Details / Reason", style="dim", max_width=55)

    for case_diff in report.case_diffs:
        # Status styling
        if case_diff.status == "pass":
            status_cell = "[bold green]PASS[/bold green]"
        elif case_diff.status == "regressed":
            status_cell = "[bold red]REGRESSED[/bold red]"
        elif case_diff.status == "improved":
            status_cell = "[bold cyan]IMPROVED[/bold cyan]"
        else:
            status_cell = "[bold red]ERROR[/bold red]"

        # Similarity formatting
        if case_diff.similarity_score is not None:
            sim_cell = f"{case_diff.similarity_score:.3f}"
        else:
            sim_cell = "-"

        # Judge cell & Details formatting
        if case_diff.judge_verdict:
            verdict = case_diff.judge_verdict.verdict
            if verdict == "worse":
                judge_cell = "[bold red]WORSE[/bold red]"
            elif verdict == "better":
                judge_cell = "[bold cyan]BETTER[/bold cyan]"
            else:
                judge_cell = "[yellow]EQUIV[/yellow]"
            details = case_diff.judge_verdict.reasoning
        elif case_diff.error:
            judge_cell = "-"
            details = f"[red]{case_diff.error}[/red]"
        elif not case_diff.expectations_result.passed:
            judge_cell = "[bold red]FAIL[/bold red]"
            details = "; ".join(case_diff.expectations_result.failures)
        else:
            judge_cell = "[dim]Skipped[/dim]"
            details = "High semantic similarity (no drift detected)"

        table.add_row(
            case_diff.test_case_id,
            status_cell,
            sim_cell,
            judge_cell,
            details,
        )

    console.print(table)

    # 4. Detailed Breakdown for Flagged / Regressed / Improved cases
    flagged_cases = [
        c
        for c in report.case_diffs
        if c.flagged_for_judge or c.status in {"regressed", "improved", "error"}
    ]
    if flagged_cases:
        console.print()
        console.rule("[bold yellow]Flagged Case Details[/bold yellow]")
        console.print()
        for c in flagged_cases:
            _render_case_detail(c, console)


def _render_case_detail(c: TestCaseDiffResult, console: Console) -> None:
    """Render detailed diagnostic card for a flagged test case."""
    if c.status == "regressed":
        tag = "[bold red]REGRESSED[/bold red]"
    elif c.status == "improved":
        tag = "[bold cyan]IMPROVED[/bold cyan]"
    elif c.status == "error":
        tag = "[bold red]ERROR[/bold red]"
    else:
        tag = "[bold yellow]FLAGGED[/bold yellow]"

    sim_str = f"{c.similarity_score:.3f}" if c.similarity_score is not None else "N/A"
    console.print(f"{tag} [bold]{c.test_case_id}[/bold] (Cosine Similarity: {sim_str})")

    if c.judge_verdict:
        console.print(f"  [dim]Category:[/dim]   {c.judge_verdict.category}")
        console.print(f"  [dim]Confidence:[/dim] {c.judge_verdict.confidence:.2f}")
        console.print(f"  [dim]Judge Note:[/dim] {c.judge_verdict.reasoning}")

    if not c.expectations_result.passed:
        console.print(
            f"  [red]Failed Assertions:[/red] {'; '.join(c.expectations_result.failures)}"
        )

    # Show baseline vs new comparison snippet
    if c.baseline_output:
        base_snippet = (
            c.baseline_output[:250] + "..."
            if len(c.baseline_output) > 250
            else c.baseline_output
        )
        console.print(
            Panel(
                base_snippet,
                title="[dim]Baseline Output (Output A)[/dim]",
                expand=False,
                border_style="dim",
            )
        )
    if c.new_output:
        new_snippet = (
            c.new_output[:250] + "..."
            if len(c.new_output) > 250
            else c.new_output
        )
        border = "red" if c.status == "regressed" else "cyan" if c.status == "improved" else "yellow"
        console.print(
            Panel(
                new_snippet,
                title="[dim]New Output (Output B)[/dim]",
                expand=False,
                border_style=border,
            )
        )
    console.print()
