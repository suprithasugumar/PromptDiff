"""Google Gemini provider implementation using google-genai SDK."""

from __future__ import annotations

import os
import time
import dotenv
from google import genai
from google.genai import types

from promptdiff.providers.base import GenerationResult


class GeminiProvider:
    """LLM provider implementation for Google Gemini models."""

    def __init__(self, api_key: str | None = None) -> None:
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set it in your environment or in a .env file."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        user_input: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate response text and token counts using Gemini."""
        target_model = model or "gemini-3.6-flash"

        config_kwargs: dict = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        # For Gemini 3.x models, sampling parameters (temperature, top_p, top_k) are deprecated.
        # Only pass temperature if explicitly requested on non-Gemini-3 models.
        if temperature is not None and "gemini-3" not in target_model.lower():
            config_kwargs["temperature"] = temperature

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        start_time = time.perf_counter()
        response = self.client.models.generate_content(
            model=target_model,
            contents=user_input,
            config=config,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        output_text = response.text or ""
        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        return GenerationResult(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
        )
