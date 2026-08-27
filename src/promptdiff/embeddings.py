"""Embedding provider abstraction and cosine similarity utilities."""

from __future__ import annotations

import math
import os
from typing import Protocol, runtime_checkable
import dotenv


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding generation backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of strings."""
        ...


class GeminiEmbedder:
    """Embedding backend using Google Gemini gemini-embedding-001."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
    ) -> None:
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set it in your environment or in a .env file."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given list of texts."""
        if not texts:
            return []

        # Replace empty strings with a single space as embeddings require non-empty text
        sanitized_texts = [t if t.strip() else " " for t in texts]

        response = self.client.models.embed_content(
            model=self.model,
            contents=sanitized_texts,
        )

        embeddings_list: list[list[float]] = []
        if hasattr(response, "embeddings") and response.embeddings:
            for item in response.embeddings:
                embeddings_list.append(list(item.values))
        elif hasattr(response, "embedding") and response.embedding:
            embeddings_list.append(list(response.embedding.values))
        else:
            raise RuntimeError(f"Unexpected response format from embedding API: {response}")

        return embeddings_list


class MockEmbedder:
    """Deterministic mock embedder for tests and dry-runs."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-embeddings based on text length and character hashes."""
        results: list[list[float]] = []
        for text in texts:
            vec: list[float] = []
            seed = sum(ord(c) for c in text)
            for i in range(self.dimension):
                val = math.sin(seed + i * 0.1)
                vec.append(val)

            # Normalize to unit length
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            results.append([v / norm for v in vec])
        return results


def get_embedder(
    provider_name: str = "gemini",
    api_key: str | None = None,
    mock: bool = False,
) -> Embedder:
    """Factory helper to obtain the appropriate embedding engine."""
    if mock or provider_name == "mock":
        return MockEmbedder()

    if provider_name == "gemini":
        return GeminiEmbedder(api_key=api_key)

    raise ValueError(f"Unsupported embedding provider: '{provider_name}'")
