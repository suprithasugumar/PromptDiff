"""Example test subject: Acme Support Reply Generator."""

from __future__ import annotations

import os
import anthropic
import dotenv

SUPPORT_SYSTEM_PROMPT = """You are a helpful, empathetic customer support agent for Acme Cloud.
Guidelines:
1. Always be concise, polite, and professional.
2. If a customer is asking for a refund under $100, guide them through account verification and reassure them.
3. If a customer asks for a refund over $100, explain that manager escalation is required and offer to open an escalation ticket.
4. If a query is completely unrelated to Acme Cloud services, politely state that you can only assist with Acme Cloud products.
"""


def generate_support_reply(
    user_message: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.2,
) -> str:
    """Standalone helper function to generate a support reply using Claude."""
    dotenv.load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment or .env")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=temperature,
        system=SUPPORT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "\n".join(
        block.text for block in response.content if hasattr(block, "text")
    )


if __name__ == "__main__":
    sample_query = "I was charged twice for my subscription this month. Can I get a refund?"
    print(f"Query: {sample_query}\n---")
    print(generate_support_reply(sample_query))
