"""Programmatic evaluation of test case expectations."""

from __future__ import annotations

import re
from promptdiff.models import Expectations, ExpectationsCheckResult

# Common patterns indicating an LLM refusal
REFUSAL_PATTERNS = [
    r"\bi cannot (assist|help|fulfill|provide|do that|comply)\b",
    r"\bi am unable to (assist|help|fulfill|provide|do that|comply)\b",
    r"\bi'm sorry, but i cannot\b",
    r"\bi apologize, but i cannot\b",
    r"\bas an ai,\s*i cannot\b",
    r"\bas an ai language model,\s*i cannot\b",
    r"\bi must decline\b",
]

_REFUSAL_REGEX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def detect_refusal(text: str) -> bool:
    """Check whether text contains common refusal phrases."""
    return bool(_REFUSAL_REGEX.search(text))


def evaluate_expectations(
    output: str | None,
    expectations: Expectations,
) -> ExpectationsCheckResult:
    """Evaluate assertions against a generated model output."""
    if output is None:
        return ExpectationsCheckResult(
            passed=False,
            failures=["Model output is empty or resulted in an execution error."],
        )

    failures: list[str] = []
    output_lower = output.lower()

    # 1. Check must_mention keywords
    for keyword in expectations.must_mention:
        if keyword.lower() not in output_lower:
            failures.append(f"Missing required keyword: '{keyword}'")

    # 2. Check must_not_mention keywords
    for keyword in expectations.must_not_mention:
        if keyword.lower() in output_lower:
            failures.append(f"Contains forbidden keyword: '{keyword}'")

    # 3. Check max_length_chars
    if expectations.max_length_chars is not None:
        if len(output) > expectations.max_length_chars:
            failures.append(
                f"Output length ({len(output)} chars) exceeded limit ({expectations.max_length_chars} chars)"
            )

    # 4. Check refusal
    if expectations.must_not_refuse and detect_refusal(output):
        failures.append("Unexpected model refusal detected.")

    return ExpectationsCheckResult(
        passed=len(failures) == 0,
        failures=failures,
    )
