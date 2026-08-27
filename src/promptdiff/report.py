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


def generate_markdown_report(report: DiffReport) -> str:
    """Generate a GitHub-flavored Markdown report suitable for PR comments and CI summaries."""
    status_icon = "🔴" if report.has_regressions else "🟢"
    status_text = "Regression Detected" if report.has_regressions else "All Checks Passed"

    lines: list[str] = [
        f"## {status_icon} PromptDiff: {status_text}",
        "",
        f"**Suite:** `{report.suite_name}` &nbsp;|&nbsp; "
        f"**Target:** `{report.target.provider} ({report.target.model})` &nbsp;|&nbsp; "
        f"**Baseline:** `{report.baseline_run_id}`",
        "",
        "### 📊 Summary",
        "| Total Cases | Passed | Regressed | Improved | Errors | Judge Calls | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **{report.total_cases}** | {report.passed_cases} | "
        f"{'🔴 **' + str(report.regressed_cases) + '**' if report.regressed_cases > 0 else str(report.regressed_cases)} | "
        f"{'🟢 ' + str(report.improved_cases) if report.improved_cases > 0 else str(report.improved_cases)} | "
        f"{'⚠️ ' + str(report.error_cases) if report.error_cases > 0 else str(report.error_cases)} | "
        f"{report.judge_calls_count} | "
        f"{'**FAILED**' if report.has_regressions else '**PASSED**'} |",
        "",
    ]

    regressed = [c for c in report.case_diffs if c.status == "regressed"]
    errors = [c for c in report.case_diffs if c.status == "error"]
    improved = [c for c in report.case_diffs if c.status == "improved"]

    if regressed:
        lines.append(f"### ⚠️ Regressed Cases ({len(regressed)})")
        lines.append("")
        for c in regressed:
            sim_str = f"`{c.similarity_score:.3f}`" if c.similarity_score is not None else "`N/A`"
            lines.append(f"#### ❌ `{c.test_case_id}`")
            lines.append(f"- **Cosine Similarity:** {sim_str}")
            if c.judge_verdict:
                lines.append(
                    f"- **Judge Verdict:** `{c.judge_verdict.verdict.upper()}` &nbsp;"
                    f"(Confidence: `{c.judge_verdict.confidence:.2f}` &nbsp;|&nbsp; Category: `{c.judge_verdict.category}`)"
                )
                lines.append(f"- **Judge Reasoning:** *{c.judge_verdict.reasoning}*")
            if not c.expectations_result.passed:
                lines.append(f"- **Failed Expectations:** ⚠️ `{'; '.join(c.expectations_result.failures)}`")

            # Diff snippet
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>🔍 <b>View Output Comparison</b></summary>")
            lines.append("")
            if c.baseline_output:
                lines.append("**Baseline Output:**")
                lines.append("```text")
                lines.append(c.baseline_output)
                lines.append("```")
                lines.append("")
            if c.new_output:
                lines.append("**New Output:**")
                lines.append("```text")
                lines.append(c.new_output)
                lines.append("```")
            lines.append("</details>")
            lines.append("")
        lines.append("---")
        lines.append("")

    if errors:
        lines.append(f"### 🛑 Error Cases ({len(errors)})")
        lines.append("")
        for c in errors:
            lines.append(f"- **`{c.test_case_id}`**: {c.error or 'Unknown error'}")
        lines.append("")
        lines.append("---")
        lines.append("")

    if improved:
        lines.append(f"### 🟢 Improved Cases ({len(improved)})")
        lines.append("")
        for c in improved:
            cat = c.judge_verdict.category if c.judge_verdict else "quality_gain"
            reason = c.judge_verdict.reasoning if c.judge_verdict else ""
            reason_text = f" — *{reason}*" if reason else ""
            lines.append(f"- **`{c.test_case_id}`**: Verdict `BETTER` (Category: `{cat}`){reason_text}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(
        "<sub>Generated by <a href=\"https://github.com/suprithasugumar/PromptDiff\">PromptDiff</a> • LLM Regression & Observability</sub>"
    )

    return "\n".join(lines)
