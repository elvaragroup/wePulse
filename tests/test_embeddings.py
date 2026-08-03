from __future__ import annotations

import asyncio

import httpx
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


def test_embedding_client_applies_concurrency_limit():
    """Verify that concurrency parameter actually throttles concurrent _raw() calls."""
    concurrency_state = {"in_flight": 0, "max_observed": 0}

    class ConcurrencyTrackingClient(FakeEmbeddingClient):
        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            concurrency_state["in_flight"] += 1
            concurrency_state["max_observed"] = max(
                concurrency_state["max_observed"], concurrency_state["in_flight"]
            )

            # Yield control to allow concurrent tasks to interleave
            await asyncio.sleep(0)

            concurrency_state["in_flight"] -= 1
            return await super()._raw(texts, model=model, input_type=input_type)

    client = ConcurrencyTrackingClient(lambda text: [1.0], concurrency=1)

    async def run_test():
        # Each embed() call below is a cache miss for a distinct text, so each
        # triggers its own _raw() call -- with concurrency=1 only one may run
        # at a time despite being launched concurrently via gather.
        await asyncio.gather(
            client.embed(["text1"], model="m"),
            client.embed(["text2"], model="m"),
            client.embed(["text3"], model="m"),
        )

    asyncio.run(run_test())

    assert concurrency_state["max_observed"] == 1


# --- batching ---


def test_embed_chunks_misses_into_batch_size_groups():
    """~144 texts in one request has no size limit protection today -- misses
    must be chunked into at most batch_size per _raw() call, with each
    chunk's results cached immediately (so an unrelated later chunk's
    failure never loses earlier chunks' already-computed embeddings)."""
    call_sizes: list[int] = []

    class TrackingEmbeddingClient(FakeEmbeddingClient):
        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            call_sizes.append(len(texts))
            return await super()._raw(texts, model=model, input_type=input_type)

    client = TrackingEmbeddingClient(lambda t: [float(len(t))], batch_size=2)
    result = asyncio.run(client.embed([f"text{i}" for i in range(5)], model="m"))

    assert call_sizes == [2, 2, 1]
    assert result == [[float(len(f"text{i}"))] for i in range(5)]


def test_embed_caches_each_batch_as_it_completes(tmp_path):
    """Even mid-flight, a chunk that succeeds must be cached right away --
    not held in memory until every chunk finishes."""
    seen_chunks: list[list[str]] = []

    class RecordingEmbeddingClient(FakeEmbeddingClient):
        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            seen_chunks.append(list(texts))
            return await super()._raw(texts, model=model, input_type=input_type)

    client = RecordingEmbeddingClient(
        lambda t: [float(len(t))], cache_dir=tmp_path / "cache", batch_size=1
    )
    asyncio.run(client.embed(["aa", "bbb"], model="m"))
    assert seen_chunks == [["aa"], ["bbb"]]

    # Both texts must now be cached individually, independent of one another.
    second_call_client = RecordingEmbeddingClient(
        lambda t: (_ for _ in ()).throw(AssertionError("should be served from cache")),
        cache_dir=tmp_path / "cache",
        batch_size=1,
    )
    result = asyncio.run(second_call_client.embed(["aa", "bbb"], model="m"))
    assert result == [[2.0], [3.0]]
    assert second_call_client.cache_hits == 2


# --- retry with backoff ---


def test_embed_retries_retryable_transport_error_then_succeeds():
    state = {"failures": 2}

    class FlakyEmbeddingClient(FakeEmbeddingClient):
        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            if state["failures"]:
                state["failures"] -= 1
                raise httpx.ConnectError("boom")
            return await super()._raw(texts, model=model, input_type=input_type)

    client = FlakyEmbeddingClient(lambda t: [1.0], max_retries=3, backoff_base=0.001)
    result = asyncio.run(client.embed(["x"], model="m"))
    assert result == [[1.0]]


def test_embed_retryable_http_status_error_retries():
    request = httpx.Request("POST", "https://api.voyageai.com/v1/embeddings")

    def make_error() -> httpx.HTTPStatusError:
        response = httpx.Response(429, request=request)
        return httpx.HTTPStatusError("rate limited", request=request, response=response)

    state = {"failures": 1}

    class FlakyEmbeddingClient(FakeEmbeddingClient):
        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            if state["failures"]:
                state["failures"] -= 1
                raise make_error()
            return await super()._raw(texts, model=model, input_type=input_type)

    client = FlakyEmbeddingClient(lambda t: [1.0], max_retries=3, backoff_base=0.001)
    result = asyncio.run(client.embed(["x"], model="m"))
    assert result == [[1.0]]


def test_embed_non_retryable_error_propagates_immediately():
    class BadRequestEmbeddingClient(FakeEmbeddingClient):
        calls = 0

        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            type(self).calls += 1
            request = httpx.Request("POST", "https://api.voyageai.com/v1/embeddings")
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

    client = BadRequestEmbeddingClient(lambda t: [1.0], max_retries=3, backoff_base=0.001)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.embed(["x"], model="m"))
    assert BadRequestEmbeddingClient.calls == 1


def test_embed_exhausts_retries_and_raises():
    class AlwaysFlakyEmbeddingClient(FakeEmbeddingClient):
        calls = 0

        async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
            type(self).calls += 1
            raise httpx.ConnectError("still broken")

    client = AlwaysFlakyEmbeddingClient(lambda t: [1.0], max_retries=3, backoff_base=0.001)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(client.embed(["x"], model="m"))
    assert AlwaysFlakyEmbeddingClient.calls == 3
