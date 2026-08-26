"""Anthropic provider implementation using anthropic SDK."""

from __future__ import annotations

import os
import time
import dotenv
import anthropic

from promptdiff.providers.base import GenerationResult


class AnthropicProvider:
    """LLM provider implementation for Anthropic Claude models."""

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

    def generate(
        self,
        user_input: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate response text and token counts using Claude."""
        target_model = model or "claude-3-5-sonnet-20241022"
        max_toks = max_tokens or 1000

        kwargs: dict = {
            "model": target_model,
            "max_tokens": max_toks,
            "messages": [{"role": "user", "content": user_input}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["extra_body"] = {"temperature": temperature}

        start_time = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        output_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        output_text = "\n".join(output_parts)

        prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "output_tokens", 0) or 0

        return GenerationResult(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
        )
