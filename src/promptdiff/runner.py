"""Execution runner for PromptDiff test suites using provider abstractions."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from promptdiff.models import RunOutput, TestCase, TestCaseResult, TestSuite
from promptdiff.providers import LLMProvider, get_provider


def _slugify(text: str) -> str:
    """Helper to convert string to a filesystem-friendly identifier."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_").lower()


class SuiteRunner:
    """Executes test suites against pluggable LLM provider backends."""

    def __init__(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.api_key = api_key
        self._provider_cache: dict[str, LLMProvider] = {}

    def get_provider(self, name: str) -> LLMProvider:
        """Get or initialize the specified provider."""
        if name not in self._provider_cache:
            self._provider_cache[name] = get_provider(name, api_key=self.api_key)
        return self._provider_cache[name]

    def run_case(
        self,
        case: TestCase,
        suite: TestSuite,
        dry_run: bool = False,
    ) -> TestCaseResult:
        """Run an individual test case against the target provider and model."""
        if dry_run:
            return TestCaseResult(
                test_case_id=case.id,
                input=case.input,
                output=f"[DRY RUN Mock Output for: '{case.input[:40]}...']",
                latency_ms=12.5,
                prompt_tokens=15,
                completion_tokens=10,
                error=None,
            )

        provider_name = self.provider_name or suite.target.provider
        start_time = time.perf_counter()
        try:
            provider = self.get_provider(provider_name)
            result = provider.generate(
                user_input=case.input,
                system_prompt=suite.target.system_prompt,
                model=suite.target.model,
                temperature=suite.target.temperature,
                max_tokens=suite.target.max_tokens,
            )

            return TestCaseResult(
                test_case_id=case.id,
                input=case.input,
                output=result.text,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                error=None,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return TestCaseResult(
                test_case_id=case.id,
                input=case.input,
                output=None,
                latency_ms=round(elapsed_ms, 2),
                error=str(exc),
            )

    def run_suite(
        self,
        suite: TestSuite,
        dry_run: bool = False,
        progress_callback=None,
    ) -> RunOutput:
        """Run all test cases in a suite and assemble a RunOutput."""
        now = datetime.now(timezone.utc)
        timestamp_str = now.isoformat()
        run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{_slugify(suite.name)}"

        results: list[TestCaseResult] = []
        for case in suite.test_cases:
            result = self.run_case(case, suite, dry_run=dry_run)
            results.append(result)
            if progress_callback:
                progress_callback(case, result)

        return RunOutput(
            run_id=run_id,
            timestamp=timestamp_str,
            suite_name=suite.name,
            target=suite.target,
            results=results,
        )
