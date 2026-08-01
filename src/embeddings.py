"""Embedding client layer, parallel to src/llm.py: cache -> call.

Cached per text (sha256(model + input_type + text)), not per whole request
like llm.py's LLMRequest.cache_key() -- diagnostics re-embeds an evolving,
overlapping set of quotes across runs, so whole-request caching would bust
the entire cache on any quote-set change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import httpx


class EmbeddingError(RuntimeError):
    pass


def _is_retryable(exc: Exception) -> bool:
    """Retryable HTTP-transport failures: rate limiting / server errors, or a
    transport-level failure (connect/read timeouts, connection resets --
    httpx.TransportError is their common base class). Modeled on
    src/llm.py's _is_retryable."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compute cosine similarity of a zero vector")
    return dot / (norm_a * norm_b)


def _cache_key(text: str, *, model: str, input_type: str) -> str:
    canonical = f"{model}\0{input_type}\0{text}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingClient(ABC):
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        concurrency: int = 8,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        batch_size: int = 64,
    ) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0
        self.cache_hits = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.batch_size = batch_size

    @abstractmethod
    async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
        """Return one embedding per text, same order. `texts` never repeats a
        cached text -- callers pre-filter."""

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _cache_read(self, text: str, *, model: str, input_type: str) -> list[float] | None:
        path = self._cache_path(_cache_key(text, model=model, input_type=input_type))
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["embedding"]

    def _cache_write(self, text: str, embedding: list[float], *, model: str, input_type: str) -> None:
        path = self._cache_path(_cache_key(text, model=model, input_type=input_type))
        if path is None:
            return
        path.write_text(
            json.dumps({"model": model, "input_type": input_type, "embedding": embedding}),
            encoding="utf-8",
        )

    async def embed(
        self, texts: list[str], *, model: str, input_type: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            cached = self._cache_read(text, model=model, input_type=input_type)
            if cached is not None:
                self.cache_hits += 1
                results[i] = cached
            else:
                misses.append((i, text))

        for start in range(0, len(misses), self.batch_size):
            chunk = misses[start : start + self.batch_size]
            self.calls_made += 1
            async with self._semaphore:
                fresh = await self._call_with_backoff(
                    [t for _, t in chunk], model=model, input_type=input_type
                )
            if len(fresh) != len(chunk):
                raise EmbeddingError(f"expected {len(chunk)} embedding(s), got {len(fresh)}")
            # Write this chunk's results to cache immediately -- so a later
            # chunk's failure never loses the embeddings this chunk already
            # paid for.
            for (i, text), embedding in zip(chunk, fresh):
                results[i] = embedding
                self._cache_write(text, embedding, model=model, input_type=input_type)

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    async def _call_with_backoff(
        self, texts: list[str], *, model: str, input_type: str
    ) -> list[list[float]]:
        """Retry loop around _raw() for one chunk, same shape as
        src/llm.py's LLMClient._call_with_backoff. Propagates the exception
        once retries are exhausted -- callers must not swallow it, since a
        chunk that never succeeded must not be silently treated as
        embedded."""
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._raw(texts, model=model, input_type=input_type)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise
                last = exc
                delay = self.backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, self.backoff_base))
        raise last  # pragma: no cover - loop always returns or raises


class FakeEmbeddingClient(EmbeddingClient):
    """responder(text) -> list[float]. Deterministic, offline."""

    def __init__(self, responder: Callable[[str], list[float]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.responder = responder
        self.requests: list[str] = []

    async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
        self.requests.extend(texts)
        return [self.responder(t) for t in texts]


class VoyageEmbeddingClient(EmbeddingClient):
    """Real client. Constructed lazily so tests never need a key."""

    ENDPOINT = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise EmbeddingError(
                "VOYAGE_API_KEY is not set. Copy .env.example to .env and fill it in, then "
                "run with `uv run --env-file .env ...` -- uv does not read .env automatically -- "
                "or export the variable directly."
            )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=60.0,
        )

    async def _raw(self, texts: list[str], *, model: str, input_type: str) -> list[list[float]]:
        response = await self._client.post(
            self.ENDPOINT, json={"input": texts, "model": model, "input_type": input_type}
        )
        response.raise_for_status()
        payload = response.json()
        by_index = {item["index"]: item["embedding"] for item in payload["data"]}
        return [by_index[i] for i in range(len(texts))]

    async def aclose(self) -> None:
        await self._client.aclose()
