"""Base protocol and data models for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    """Standardized result returned by any LLM provider."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for pluggable LLM provider backends."""

    def generate(
        self,
        user_input: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate a completion for the given prompt."""
        ...
