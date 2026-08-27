"""Unit tests for embedding utilities and mock embedder."""

import math
from promptdiff.embeddings import MockEmbedder, cosine_similarity, get_embedder


def test_cosine_similarity_identical():
    vec = [1.0, 2.0, 3.0]
    sim = cosine_similarity(vec, vec)
    assert math.isclose(sim, 1.0, rel_tol=1e-5)


def test_cosine_similarity_orthogonal():
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    sim = cosine_similarity(vec_a, vec_b)
    assert math.isclose(sim, 0.0, rel_tol=1e-5)


def test_cosine_similarity_opposite():
    vec_a = [1.0, 2.0]
    vec_b = [-1.0, -2.0]
    sim = cosine_similarity(vec_a, vec_b)
    assert math.isclose(sim, -1.0, rel_tol=1e-5)


def test_cosine_similarity_empty():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_mock_embedder_deterministic():
    embedder = MockEmbedder(dimension=32)
    texts = ["hello world", "hello world", "completely different sentence"]
    vecs = embedder.embed(texts)

    assert len(vecs) == 3
    assert len(vecs[0]) == 32

    # Same text produces identical embeddings
    sim_identical = cosine_similarity(vecs[0], vecs[1])
    assert math.isclose(sim_identical, 1.0, rel_tol=1e-5)


def test_get_embedder_mock():
    embedder = get_embedder(mock=True)
    assert isinstance(embedder, MockEmbedder)


def test_gemini_embedder_default_model():
    from promptdiff.embeddings import GeminiEmbedder

    embedder = GeminiEmbedder(api_key="dummy-key")
    assert embedder.model == "gemini-embedding-001"

