"""LLM providers factory and registry for PromptDiff."""

from __future__ import annotations

from promptdiff.providers.base import GenerationResult, LLMProvider
from promptdiff.providers.gemini import GeminiProvider
from promptdiff.providers.anthropic import AnthropicProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(provider_name: str = "gemini", api_key: str | None = None) -> LLMProvider:
    """Retrieve an initialized LLMProvider instance."""
    normalized_name = provider_name.strip().lower()
    provider_cls = PROVIDER_REGISTRY.get(normalized_name)
    if not provider_cls:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported provider '{provider_name}'. Available providers: {available}"
        )
    return provider_cls(api_key=api_key)


__all__ = [
    "GenerationResult",
    "LLMProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "get_provider",
    "PROVIDER_REGISTRY",
]
