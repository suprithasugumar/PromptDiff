"""Unit tests for provider abstraction layer."""

import pytest
from promptdiff.providers import (
    GenerationResult,
    LLMProvider,
    GeminiProvider,
    AnthropicProvider,
    get_provider,
)


def test_provider_protocol_conformance():
    gemini = GeminiProvider(api_key="mock_key")
    anthropic = AnthropicProvider(api_key="mock_key")

    assert isinstance(gemini, LLMProvider)
    assert isinstance(anthropic, LLMProvider)


def test_get_provider_registry():
    gemini = get_provider("gemini", api_key="mock_key")
    assert isinstance(gemini, GeminiProvider)

    claude = get_provider("anthropic", api_key="mock_key")
    assert isinstance(claude, AnthropicProvider)

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("openai", api_key="mock_key")
