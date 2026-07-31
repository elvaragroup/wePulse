from __future__ import annotations

import asyncio

import pytest

from src.embeddings import EmbeddingError, FakeEmbeddingClient, cosine_similarity


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_parallel_same_direction_is_one():
    assert cosine_similarity([2.0, 2.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_length():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0], [1.0, 0.0])


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_fake_embedding_client_returns_responder_output():
    client = FakeEmbeddingClient(lambda text: [float(len(text)), 0.0])
    result = asyncio.run(client.embed(["hi", "hello"], model="fake-model"))
    assert result == [[2.0, 0.0], [5.0, 0.0]]


def test_fake_embedding_client_caches_per_text(tmp_path):
    calls = {"n": 0}

    def responder(text: str) -> list[float]:
        calls["n"] += 1
        return [1.0, 0.0]

    client = FakeEmbeddingClient(responder, cache_dir=tmp_path / "cache")
    asyncio.run(client.embed(["same text"], model="fake-model"))
    asyncio.run(client.embed(["same text"], model="fake-model"))
    assert calls["n"] == 1
    assert client.cache_hits == 1


def test_fake_embedding_client_cache_is_keyed_by_model_and_input_type(tmp_path):
    client = FakeEmbeddingClient(lambda text: [float(len(text))], cache_dir=tmp_path / "cache")
    asyncio.run(client.embed(["x"], model="model-a", input_type="document"))
    asyncio.run(client.embed(["x"], model="model-b", input_type="document"))
    asyncio.run(client.embed(["x"], model="model-a", input_type="query"))
    assert client.cache_hits == 0
    assert client.calls_made == 3


def test_fake_embedding_client_preserves_order_with_partial_cache_hits(tmp_path):
    client = FakeEmbeddingClient(lambda text: [float(ord(text[0]))], cache_dir=tmp_path / "cache")
    asyncio.run(client.embed(["a"], model="m"))  # primes the cache for "a"
    result = asyncio.run(client.embed(["b", "a", "c"], model="m"))
    assert result == [[98.0], [97.0], [99.0]]


def test_embed_empty_list_returns_empty():
    client = FakeEmbeddingClient(lambda text: [1.0])
    assert asyncio.run(client.embed([], model="m")) == []


def test_voyage_client_raises_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from src.embeddings import VoyageEmbeddingClient

    with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingClient()
