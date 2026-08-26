"""Example test subject: Acme Support Reply Generator."""

from __future__ import annotations

import os
import dotenv
from promptdiff.providers import get_provider

SUPPORT_SYSTEM_PROMPT = """You are a helpful, empathetic customer support agent for Acme Cloud.
Guidelines:
1. Always be concise, polite, and professional.
2. If a customer is asking for a refund under $100, guide them through account verification and reassure them.
3. If a customer asks for a refund over $100, explain that manager escalation is required and offer to open an escalation ticket.
4. If a query is completely unrelated to Acme Cloud services, politely state that you can only assist with Acme Cloud products.
"""


def generate_support_reply(
    user_message: str,
    provider_name: str = "gemini",
    model: str = "gemini-3.6-flash",
    temperature: float | None = None,
    max_tokens: int = 500,
) -> str:
    """Generate a support reply using the configured LLM provider."""
    dotenv.load_dotenv()
    provider = get_provider(provider_name)
    result = provider.generate(
        user_input=user_message,
        system_prompt=SUPPORT_SYSTEM_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result.text


if __name__ == "__main__":
    sample_query = "I was charged twice for my subscription this month. Can I get a refund?"
    print(f"Query: {sample_query}\n---")
    print(generate_support_reply(sample_query))
