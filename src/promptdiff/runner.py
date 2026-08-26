"""Execution runner for PromptDiff test suites."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
import dotenv

import anthropic
from promptdiff.models import RunOutput, TestCase, TestCaseResult, TestSuite


def _slugify(text: str) -> str:
    """Helper to convert string to a filesystem-friendly identifier."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_").lower()


class SuiteRunner:
    """Executes test suites against LLM endpoints."""

    def __init__(self, api_key: str | None = None) -> None:
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Please set it in your environment or in a .env file."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def run_case(
        self,
        case: TestCase,
        suite: TestSuite,
        dry_run: bool = False,
    ) -> TestCaseResult:
        """Run an individual test case against the target model."""
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

        start_time = time.perf_counter()
        try:
            kwargs: dict = {
                "model": suite.target.model,
                "max_tokens": suite.target.max_tokens,
                "temperature": suite.target.temperature,
                "messages": [{"role": "user", "content": case.input}],
            }
            if suite.target.system_prompt:
                kwargs["system"] = suite.target.system_prompt

            response = self.client.messages.create(**kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract response text
            output_parts = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            output_text = "\n".join(output_parts)

            prompt_tokens = getattr(response.usage, "input_tokens", 0)
            completion_tokens = getattr(response.usage, "output_tokens", 0)

            return TestCaseResult(
                test_case_id=case.id,
                input=case.input,
                output=output_text,
                latency_ms=round(elapsed_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
