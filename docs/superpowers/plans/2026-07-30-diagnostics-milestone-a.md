# Diagnostics Milestone A (persona-collapse baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a re-runnable `diagnostics.py` module that turns the persona-collapse symptoms measured in a real 23-event run (shared idioms, 96%+ per-event category consensus, near-constant `intensity`, inverted archetype/engagement correlation) into seven formal, comparable metrics per the v2 persona-engine brief's Phase 5 table, and record them as the required baseline before any engine change.

**Architecture:** Two new pure-function/data modules (`src/embeddings.py` — a Voyage AI client parallel to `src/llm.py`; `src/diagnostics_metrics.py` — one function per metric) plus one orchestration+CLI module (`src/diagnostics.py`, mirroring `src/score.py`'s shape: load → compute → write → compare-to-baseline).

**Tech Stack:** Python 3.12, existing `anthropic`/`pydantic`/`numpy`/`scipy`, plus new `httpx` (already a transitive dependency via `anthropic`, promoted to direct since this plan imports it explicitly) and new `scikit-learn>=1.3` (its built-in `HDBSCAN`).

## Global Constraints

- Zero real API/network calls in any test — every test uses `FakeEmbeddingClient`/`FakeLLMClient`, matching the whole codebase's existing convention (`tests/test_llm.py`, `tests/test_probe_leakage.py`, etc.).
- Every new module follows this codebase's established shape: module-level `REPO = Path(__file__).resolve().parent.parent`, a domain `<Name>Error(RuntimeError)` class, pure computation separated from I/O, `main(argv: list[str] | None = None) -> int` CLI entry points (see `src/score.py`, `src/probe_leakage.py`).
- Embeddings are cached **per text** (`sha256(model + "\0" + input_type + "\0" + text)`), not per whole request like `src/llm.py`'s `LLMRequest.cache_key()` — diagnostics re-embeds an evolving, overlapping quote set across runs, so whole-request caching would bust the entire cache on any change.
- `structured_schema()` (`src/llm.py:95-121`) strips numeric bounds (`ge`/`le`) from the JSON schema sent to Anthropic's structured-outputs API and forces every property into `required` — this plan has no new pydantic models sent through that path (Milestone A makes no new structured LLM calls beyond reusing the existing `PersonaReaction` schema for the Stability check), but if extended later, remember bounds are client-side-only.
- `VOYAGE_API_KEY` follows the exact pattern `ANTHROPIC_API_KEY` already uses in `.env.example` and `src/llm.py`'s `AnthropicLLMClient.__init__` (lazy import, clear error naming the missing var and the `uv run --env-file .env` requirement — see `src/llm.py:264-274`).
- The Voyage embedding model name (`DEFAULT_EMBEDDING_MODEL` in Task 1) must be verified against Voyage's current model list at implementation time — model names on hosted embedding APIs change; do not trust the name in this plan blindly, confirm it resolves before Task 7's real-data run.

---

### Task 1: Embedding client layer — `src/embeddings.py`

**Files:**
- Create: `src/embeddings.py`
- Test: `tests/test_embeddings.py`
- Modify: `pyproject.toml` (add `httpx>=0.27` as a direct dependency — already installed transitively via `anthropic`, but this task imports it directly for the first time)
- Modify: `.env.example` (add `VOYAGE_API_KEY=` line)

**Interfaces:**
- Produces:
  - `class EmbeddingError(RuntimeError)`
  - `cosine_similarity(a: list[float], b: list[float]) -> float`
  - `class EmbeddingClient(ABC)` with `async def embed(self, texts: list[str], *, model: str, input_type: str = "document") -> list[list[float]]`
  - `class FakeEmbeddingClient(EmbeddingClient)` — `FakeEmbeddingClient(responder: Callable[[str], list[float]], **kwargs)`, exposes `.requests: list[str]`
  - `class VoyageEmbeddingClient(EmbeddingClient)` — `VoyageEmbeddingClient(*, api_key: str | None = None, **kwargs)`, has `async def aclose(self) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.embeddings'`.

- [ ] **Step 3: Add the `httpx` dependency and `VOYAGE_API_KEY` to `.env.example`**

In `pyproject.toml`, add `"httpx>=0.27",` to the `dependencies` list (alongside `anthropic`/`pydantic`/`pyyaml`/`scipy`/`numpy`).

In `.env.example`, append after the existing `ANTHROPIC_API_KEY=` block:

```
# Required for src/diagnostics.py's Homogeneity/Redundancy metrics.
VOYAGE_API_KEY=
```

Run `uv sync` after editing `pyproject.toml` so the lockfile picks up the explicit dependency.

- [ ] **Step 4: Implement `src/embeddings.py`**

```python
"""Embedding client layer, parallel to src/llm.py: cache -> call.

Cached per text (sha256(model + input_type + text)), not per whole request
like llm.py's LLMRequest.cache_key() -- diagnostics re-embeds an evolving,
overlapping set of quotes across runs, so whole-request caching would bust
the entire cache on any quote-set change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable


class EmbeddingError(RuntimeError):
    pass


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
    def __init__(self, *, cache_dir: Path | None = None, concurrency: int = 8) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0
        self.cache_hits = 0

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

        if misses:
            self.calls_made += 1
            fresh = await self._raw([t for _, t in misses], model=model, input_type=input_type)
            if len(fresh) != len(misses):
                raise EmbeddingError(f"expected {len(misses)} embedding(s), got {len(fresh)}")
            for (i, text), embedding in zip(misses, fresh):
                results[i] = embedding
                self._cache_write(text, embedding, model=model, input_type=input_type)

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]


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
        import httpx

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/embeddings.py tests/test_embeddings.py pyproject.toml uv.lock .env.example
git commit -m "Add embeddings.py: Voyage AI client with per-text caching"
```

---

### Task 2: Register variance, Specificity, Distribution-match metrics — `src/diagnostics_metrics.py` (part 1)

**Files:**
- Create: `src/diagnostics_metrics.py`
- Test: `tests/test_diagnostics_metrics.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class RegisterVarianceResult: n, word_count_mean, word_count_stdev, word_count_min, word_count_max, sentence_count_stdev, exclamation_rate, em_dash_rate`
  - `register_variance(quotes: list[str]) -> RegisterVarianceResult`
  - `@dataclass(frozen=True) class SpecificityResult: false_positive_rate, n_null_events, n_false_positive`
  - `specificity(backlash_predicted_by_event: dict[str, bool], null_event_ids: set[str]) -> SpecificityResult`
  - `@dataclass(frozen=True) class DistributionMatchResult: status, total_variation_distance, categories_compared`
  - `distribution_match_partial(simulated_mix: dict[str, float], ground_truth_mix: dict[str, float]) -> DistributionMatchResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnostics_metrics.py`:

```python
from __future__ import annotations

import pytest

from src.diagnostics_metrics import (
    distribution_match_partial,
    register_variance,
    specificity,
)


# --- register_variance ---


def test_register_variance_word_count_stats():
    quotes = ["one two three four", "five six"]
    result = register_variance(quotes)
    assert result.n == 2
    assert result.word_count_mean == pytest.approx(3.0)
    assert result.word_count_stdev == pytest.approx(1.4142135623730951)
    assert result.word_count_min == 2
    assert result.word_count_max == 4


def test_register_variance_sentence_count_stdev():
    quotes = ["One. Two. Three.", "No punctuation here"]
    result = register_variance(quotes)
    # sentence counts: [3, 1] -- three ". "/". "-at-end matches vs the
    # no-punctuation fallback of 1
    assert result.sentence_count_stdev == pytest.approx(1.4142135623730951)


def test_register_variance_exclamation_and_dash_rate():
    quotes = ["Wait!", "No punctuation here", "Great—truly great", "Another one!"]
    result = register_variance(quotes)
    assert result.exclamation_rate == pytest.approx(0.5)
    assert result.em_dash_rate == pytest.approx(0.25)


def test_register_variance_rejects_empty():
    with pytest.raises(ValueError, match="zero quotes"):
        register_variance([])


# --- specificity ---


def test_specificity_no_false_positives():
    predicted = {"evt_001": True, "evt_002": False, "evt_003": False}
    result = specificity(predicted, {"evt_002", "evt_003"})
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.n_null_events == 2
    assert result.n_false_positive == 0


def test_specificity_one_false_positive():
    predicted = {"evt_001": True, "evt_002": True, "evt_003": False}
    result = specificity(predicted, {"evt_002", "evt_003"})
    assert result.false_positive_rate == pytest.approx(0.5)
    assert result.n_false_positive == 1


def test_specificity_missing_event_defaults_to_not_flagged():
    result = specificity({}, {"evt_005"})
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.n_false_positive == 0


def test_specificity_rejects_empty_null_set():
    with pytest.raises(ValueError, match="no null events"):
        specificity({"evt_001": True}, set())


# --- distribution_match_partial ---


def test_distribution_match_identical_mixes_is_zero():
    result = distribution_match_partial({"privacy": 0.6, "none": 0.4}, {"privacy": 0.6, "none": 0.4})
    assert result.status == "measured_partial"
    assert result.total_variation_distance == pytest.approx(0.0)


def test_distribution_match_disjoint_mixes_is_one():
    result = distribution_match_partial({"privacy": 1.0}, {"none": 1.0})
    assert result.total_variation_distance == pytest.approx(1.0)
    assert result.categories_compared == 2


def test_distribution_match_both_empty():
    result = distribution_match_partial({}, {})
    assert result.total_variation_distance is None
    assert result.categories_compared == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.diagnostics_metrics'`.

- [ ] **Step 3: Implement `src/diagnostics_metrics.py` (part 1)**

```python
"""Pure statistical functions for the anti-collapse diagnostics suite (see
docs/superpowers/plans/2026-07-30-diagnostics-milestone-a.md). No I/O, no
LLM/embedding calls -- src/diagnostics.py owns loading data and calling
embedding/LLM clients; these functions only compute over already-collected
values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_EM_DASH = "—"


def _stdev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5


@dataclass(frozen=True)
class RegisterVarianceResult:
    n: int
    word_count_mean: float
    word_count_stdev: float
    word_count_min: int
    word_count_max: int
    sentence_count_stdev: float
    exclamation_rate: float
    em_dash_rate: float


def register_variance(quotes: list[str]) -> RegisterVarianceResult:
    """Word-count spread, sentence-count spread, and two structural-uniformity
    markers (exclamation marks, em-dashes) across a set of persona quotes.
    Low word-count stdev / a hard floor / near-zero exclamation-mark rate are
    all collapse symptoms -- measured against a real 23-event v1 run: 23-word
    floor, 0% exclamation rate, 68% em-dash rate."""
    if not quotes:
        raise ValueError("cannot compute register variance over zero quotes")

    word_counts = [len(q.split()) for q in quotes]
    sentence_counts = [len(_SENTENCE_SPLIT.findall(q)) or 1 for q in quotes]
    n = len(quotes)

    return RegisterVarianceResult(
        n=n,
        word_count_mean=sum(word_counts) / n,
        word_count_stdev=_stdev(word_counts),
        word_count_min=min(word_counts),
        word_count_max=max(word_counts),
        sentence_count_stdev=_stdev(sentence_counts),
        exclamation_rate=sum("!" in q for q in quotes) / n,
        em_dash_rate=sum(_EM_DASH in q for q in quotes) / n,
    )


@dataclass(frozen=True)
class SpecificityResult:
    false_positive_rate: float
    n_null_events: int
    n_false_positive: int


def specificity(
    backlash_predicted_by_event: dict[str, bool], null_event_ids: set[str]
) -> SpecificityResult:
    """False-positive rate over events the user labeled expected_null=True:
    the fraction where backlash was predicted anyway. Mirrors src/score.py's
    false_positive_rate, but at the run level (any-arm) rather than per-arm."""
    if not null_event_ids:
        raise ValueError("no null events to measure specificity against")

    flagged = [backlash_predicted_by_event.get(eid, False) for eid in null_event_ids]
    n_fp = sum(flagged)
    return SpecificityResult(
        false_positive_rate=n_fp / len(null_event_ids),
        n_null_events=len(null_event_ids),
        n_false_positive=n_fp,
    )


@dataclass(frozen=True)
class DistributionMatchResult:
    status: str  # "measured_partial" -- see docstring below
    total_variation_distance: float | None
    categories_compared: int


def distribution_match_partial(
    simulated_mix: dict[str, float], ground_truth_mix: dict[str, float]
) -> DistributionMatchResult:
    """Total variation distance between two category-share distributions.
    Marked 'measured_partial': the real Distribution Match metric (v2 brief
    Phase 5) compares against a corpus-derived real-world grievance mix,
    which doesn't exist without Phase 1's scraped corpus (out of scope for
    this plan). This proxy compares against ground_truth/labeled/*.json's
    present_categories mix instead, which already exists for any scored
    run -- different semantics, useful only as an early, provisional
    signal."""
    categories = set(simulated_mix) | set(ground_truth_mix)
    if not categories:
        return DistributionMatchResult(
            status="measured_partial", total_variation_distance=None, categories_compared=0
        )

    tvd = 0.5 * sum(abs(simulated_mix.get(c, 0.0) - ground_truth_mix.get(c, 0.0)) for c in categories)
    return DistributionMatchResult(
        status="measured_partial", total_variation_distance=tvd, categories_compared=len(categories)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics_metrics.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/diagnostics_metrics.py tests/test_diagnostics_metrics.py
git commit -m "Add diagnostics_metrics.py: register variance, specificity, distribution match"
```

---

### Task 3: Homogeneity, Redundancy, Span dispersion, Stability metrics — `src/diagnostics_metrics.py` (part 2)

**Files:**
- Modify: `src/diagnostics_metrics.py` (append)
- Modify: `tests/test_diagnostics_metrics.py` (append)
- Modify: `pyproject.toml` (add `scikit-learn>=1.3`)

**Interfaces:**
- Consumes: `_stdev` (Task 2, private helper reused here)
- Produces:
  - `@dataclass(frozen=True) class HomogeneityResult: mean_pairwise_cosine, n_quotes, n_pairs`
  - `homogeneity(embeddings: list[list[float]]) -> HomogeneityResult`
  - `@dataclass(frozen=True) class RedundancyResult: n_clusters, n_reacting_personas, ratio, n_noise`
  - `redundancy(embeddings: list[list[float]], *, n_reacting_personas: int, min_cluster_size: int = 2) -> RedundancyResult`
  - `@dataclass(frozen=True) class SpanDispersionResult: normalized_position_stdev, distinct_span_fraction`
  - `span_dispersion(citations: list[tuple[int, int]], announcement_len: int) -> SpanDispersionResult`
  - `@dataclass(frozen=True) class StabilityResult: n_pairs_sampled, n_reruns, category_agreement_rate`
  - `stability(samples: dict[tuple[str, str], list[tuple[str, frozenset]]]) -> StabilityResult`

- [ ] **Step 1: Add the `scikit-learn` dependency**

In `pyproject.toml`, add `"scikit-learn>=1.3",` to `dependencies`. Run `uv sync`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_diagnostics_metrics.py` (add this import alongside the
existing `from src.diagnostics_metrics import ...` line at the top of the
file, then add the test functions below at the end of the file):

```python
from src.diagnostics_metrics import homogeneity, redundancy, span_dispersion, stability


# --- homogeneity ---


def test_homogeneity_mean_pairwise_cosine():
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    result = homogeneity(embeddings)
    # pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=0.0 -> mean 1/3
    assert result.mean_pairwise_cosine == pytest.approx(1 / 3)
    assert result.n_quotes == 3
    assert result.n_pairs == 3


def test_homogeneity_rejects_fewer_than_two():
    with pytest.raises(ValueError, match="at least 2"):
        homogeneity([[1.0, 0.0]])


# --- redundancy ---


def test_redundancy_fewer_than_min_cluster_size_returns_one_cluster_per_point():
    result = redundancy([[1.0, 0.0]], n_reacting_personas=1, min_cluster_size=2)
    assert result.n_clusters == 1
    assert result.ratio == pytest.approx(1.0)
    assert result.n_noise == 0


def test_redundancy_finds_well_separated_clusters():
    embeddings = [
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.01],
        [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.01],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.01],
    ]
    result = redundancy(embeddings, n_reacting_personas=6, min_cluster_size=2)
    assert result.n_clusters == 3
    assert result.n_noise == 0
    assert result.ratio == pytest.approx(0.5)


# --- span_dispersion ---


def test_span_dispersion_all_same_span_has_zero_stdev():
    result = span_dispersion([(0, 10), (0, 10), (0, 10)], announcement_len=100)
    assert result.normalized_position_stdev == pytest.approx(0.0)
    assert result.distinct_span_fraction == pytest.approx(1 / 3)


def test_span_dispersion_spread_across_document():
    result = span_dispersion([(0, 10), (45, 55), (90, 100)], announcement_len=100)
    # midpoints normalized: 0.05, 0.50, 0.95 -> rounded*1000: 50, 500, 950
    # stdev([50, 500, 950]) = 450.0 -> normalized_position_stdev = 0.45
    assert result.normalized_position_stdev == pytest.approx(0.45)
    assert result.distinct_span_fraction == pytest.approx(1.0)


def test_span_dispersion_rejects_empty_citations():
    with pytest.raises(ValueError, match="no citations"):
        span_dispersion([], announcement_len=100)


def test_span_dispersion_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="announcement_len"):
        span_dispersion([(0, 10)], announcement_len=0)


# --- stability ---


def test_stability_perfect_agreement():
    samples = {("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5}
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(1.0)
    assert result.n_pairs_sampled == 1
    assert result.n_reruns == 5


def test_stability_disagreement_lowers_rate():
    samples = {
        ("001", "evt_001"): [
            ("criticize", frozenset({"privacy"})),
            ("ignore", frozenset()),
            ("criticize", frozenset({"privacy"})),
            ("criticize", frozenset({"privacy"})),
            ("criticize", frozenset({"privacy"})),
        ]
    }
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(0.0)


def test_stability_mixed_pairs():
    samples = {
        ("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5,
        ("002", "evt_002"): [("ignore", frozenset())] * 4 + [("criticize", frozenset({"labor"}))],
    }
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(0.5)


def test_stability_rejects_empty_samples():
    with pytest.raises(ValueError, match="no samples"):
        stability({})


def test_stability_rejects_mismatched_rerun_counts():
    samples = {
        ("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5,
        ("002", "evt_002"): [("ignore", frozenset())] * 3,
    }
    with pytest.raises(ValueError, match="same rerun count"):
        stability(samples)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'homogeneity'`.

- [ ] **Step 4: Implement part 2, appended to `src/diagnostics_metrics.py`**

Also add `from src.embeddings import cosine_similarity` to the top-of-file imports.

```python
@dataclass(frozen=True)
class HomogeneityResult:
    mean_pairwise_cosine: float
    n_quotes: int
    n_pairs: int


def homogeneity(embeddings: list[list[float]]) -> HomogeneityResult:
    """Mean pairwise cosine similarity across every non-ignore quote's
    embedding. High similarity means many personas are one voice wearing
    different subject-matter vocabulary."""
    n = len(embeddings)
    if n < 2:
        raise ValueError("need at least 2 embeddings to compute pairwise similarity")

    total = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_similarity(embeddings[i], embeddings[j])
            n_pairs += 1

    return HomogeneityResult(mean_pairwise_cosine=total / n_pairs, n_quotes=n, n_pairs=n_pairs)


def _cosine_distance_matrix(embeddings: list[list[float]]):
    import numpy as np

    matrix = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    distance = 1.0 - similarity
    return np.clip(distance, 0.0, 2.0)


@dataclass(frozen=True)
class RedundancyResult:
    n_clusters: int
    n_reacting_personas: int
    ratio: float  # n_clusters / n_reacting_personas; low ratio = collapse
    n_noise: int


def redundancy(
    embeddings: list[list[float]], *, n_reacting_personas: int, min_cluster_size: int = 2
) -> RedundancyResult:
    """Cluster persona-reaction embeddings; compare cluster count to the
    number of reacting personas. clusters << personas means many personas
    land in the same handful of semantic clusters -- collapse."""
    from sklearn.cluster import HDBSCAN

    if len(embeddings) < min_cluster_size:
        return RedundancyResult(
            n_clusters=len(embeddings), n_reacting_personas=n_reacting_personas, ratio=1.0, n_noise=0
        )

    distance_matrix = _cosine_distance_matrix(embeddings)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed").fit_predict(distance_matrix)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if n_noise else 0)

    return RedundancyResult(
        n_clusters=n_clusters,
        n_reacting_personas=n_reacting_personas,
        ratio=n_clusters / n_reacting_personas if n_reacting_personas else 0.0,
        n_noise=n_noise,
    )


@dataclass(frozen=True)
class SpanDispersionResult:
    normalized_position_stdev: float
    distinct_span_fraction: float


def span_dispersion(citations: list[tuple[int, int]], announcement_len: int) -> SpanDispersionResult:
    """How spread out cited (char_start, char_end) spans are across the
    document. All personas citing one span -> normalized_position_stdev near
    0. Only meaningful once span-grounded citations exist (a future
    Milestone B); v1 runs have no spans, so src/diagnostics.py passes None
    instead of calling this for them."""
    if not citations:
        raise ValueError("no citations to compute span dispersion over")
    if announcement_len <= 0:
        raise ValueError("announcement_len must be positive")

    midpoints = [((start + end) / 2) / announcement_len for start, end in citations]
    distinct = len({(start, end) for start, end in citations})

    return SpanDispersionResult(
        normalized_position_stdev=_stdev([round(m * 1000) for m in midpoints]) / 1000,
        distinct_span_fraction=distinct / len(citations),
    )


@dataclass(frozen=True)
class StabilityResult:
    n_pairs_sampled: int
    n_reruns: int
    category_agreement_rate: float


def stability(samples: dict[tuple[str, str], list[tuple[str, frozenset]]]) -> StabilityResult:
    """samples maps (persona_id, event_id) -> list of (reaction, category_set)
    tuples, one per rerun. A pair 'agrees' iff every rerun produced the exact
    same tuple. Only computes agreement over already-collected reruns --
    src/diagnostics.py owns making the repeated LLM calls."""
    if not samples:
        raise ValueError("no samples to compute stability over")

    n_reruns_seen = {len(v) for v in samples.values()}
    if len(n_reruns_seen) != 1:
        raise ValueError(f"all sampled pairs must have the same rerun count, got {sorted(n_reruns_seen)}")
    n_reruns = n_reruns_seen.pop()

    agreements = sum(1 for reruns in samples.values() if len(set(reruns)) == 1)
    return StabilityResult(
        n_pairs_sampled=len(samples), n_reruns=n_reruns, category_agreement_rate=agreements / len(samples)
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics_metrics.py -v`
Expected: all tests PASS (24 total in this file). If `test_redundancy_finds_well_separated_clusters` is flaky due to `HDBSCAN` version differences, increase the separation between the three synthetic clusters (e.g. widen the `0.01` offsets) rather than loosening the assertion.

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all previously-passing tests still PASS, plus the new ones.

- [ ] **Step 7: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/diagnostics_metrics.py tests/test_diagnostics_metrics.py pyproject.toml uv.lock
git commit -m "Add homogeneity, redundancy, span dispersion, and stability metrics"
```

---

### Task 4: Row loading — `src/diagnostics.py` (part 1)

**Files:**
- Create: `src/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `src.personas.Persona`, `src.personas.load_personas`
- Produces:
  - `REPO: Path`
  - `class DiagnosticsError(RuntimeError)`
  - `@dataclass(frozen=True) class DiagnosticsRow: event_id, persona_id, reaction, text, category, char_start, char_end`
  - `load_v1_rows(run_dir: Path, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]`
  - `load_rows(run_dir: Path, *, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnostics.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.diagnostics import DiagnosticsError, load_rows, load_v1_rows


def write_reaction(path: Path, **kwargs) -> None:
    defaults = dict(reaction="ignore", categories=[], intensity=0.0, quote=None, reasoning="r")
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults), encoding="utf-8")


def test_load_v1_rows_reads_every_persona_reaction(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(
        tmp_path / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize", categories=["privacy"], intensity=0.8, quote="q",
    )
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "002.json")

    rows = load_v1_rows(tmp_path, personas_by_id)
    assert len(rows) == 2
    by_id = {r.persona_id: r for r in rows}
    assert by_id["001"].event_id == "evt_001"
    assert by_id["001"].reaction == "criticize"
    assert by_id["001"].text == "q"
    assert by_id["001"].category == "privacy"
    assert by_id["001"].char_start is None
    assert by_id["002"].reaction == "ignore"
    assert by_id["002"].text is None
    assert by_id["002"].category is None


def test_load_v1_rows_across_multiple_events(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    write_reaction(tmp_path / "raw" / "B30" / "evt_002" / "001.json")
    rows = load_v1_rows(tmp_path, personas_by_id)
    assert {r.event_id for r in rows} == {"evt_001", "evt_002"}


def test_load_v1_rows_raises_when_b30_missing(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    with pytest.raises(DiagnosticsError, match="no raw/B30"):
        load_v1_rows(tmp_path, personas_by_id)


def test_load_v1_rows_raises_when_no_reactions_present(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    (tmp_path / "raw" / "B30").mkdir(parents=True)
    with pytest.raises(DiagnosticsError, match="no reactions found"):
        load_v1_rows(tmp_path, personas_by_id)


def test_load_v1_rows_raises_for_unknown_persona_id(tmp_path):
    """personas_by_id is a real validation, not a decorative parameter: a
    persona_id in raw/B30/ that isn't in the current roster (persona set
    changed since this run happened) must fail loudly rather than silently
    skew the diagnostics numbers with data from a persona nobody can audit
    anymore."""
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "999.json")
    with pytest.raises(DiagnosticsError, match="999.*not found in the current persona set"):
        load_v1_rows(tmp_path, personas_by_id={})


def test_load_rows_dispatches_to_v1(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    rows = load_rows(tmp_path, personas_by_id=personas_by_id)
    assert len(rows) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.diagnostics'`.

- [ ] **Step 3: Implement `src/diagnostics.py` (part 1)**

```python
"""Anti-collapse diagnostics: turns a run's raw output into the seven
metrics from the v2 persona-engine brief's Phase 5 table, comparable across
engine changes. See
docs/superpowers/plans/2026-07-30-diagnostics-milestone-a.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.personas import Persona

REPO = Path(__file__).resolve().parent.parent


class DiagnosticsError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiagnosticsRow:
    """Run-shape-agnostic view every metric function consumes. char_start/
    char_end are always None for v1 rows (no spans exist); populated once a
    future Milestone B extends load_rows() to detect pipeline-shaped runs."""

    event_id: str
    persona_id: str
    reaction: str
    text: str | None       # the quote, None iff reaction == "ignore"
    category: str | None   # first category if any, else None
    char_start: int | None
    char_end: int | None


def load_v1_rows(run_dir: Path, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]:
    """Reads every runs/<id>/raw/B30/<event_id>/<persona_id>.json -- B30 holds
    every persona's reaction regardless of which arms ran (same read pattern
    as src/dashboard.py's load_event_reactions). Every persona_id found must
    exist in personas_by_id: silently including a reaction from a persona no
    longer in the current roster (persona set changed since this run) would
    skew every downstream metric without anyone noticing -- fail loudly
    instead, matching this codebase's "malformed data raises, never skips
    silently" convention (see src/events.py's parser)."""
    b30_dir = run_dir / "raw" / "B30"
    if not b30_dir.exists():
        raise DiagnosticsError(f"no raw/B30 reactions at {b30_dir}")

    rows: list[DiagnosticsRow] = []
    for event_dir in sorted(b30_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        for path in sorted(event_dir.glob("*.json")):
            persona_id = path.stem
            if persona_id not in personas_by_id:
                raise DiagnosticsError(
                    f"persona {persona_id!r} not found in the current persona set "
                    f"(reaction at {path}) -- diagnostics refuses to silently analyze "
                    "reactions from personas outside the current roster"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            categories = data.get("categories") or []
            rows.append(
                DiagnosticsRow(
                    event_id=event_dir.name,
                    persona_id=path.stem,
                    reaction=data["reaction"],
                    text=data.get("quote"),
                    category=categories[0] if categories else None,
                    char_start=None,
                    char_end=None,
                )
            )
    if not rows:
        raise DiagnosticsError(f"no reactions found under {b30_dir}")
    return rows


def load_rows(run_dir: Path, *, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]:
    """Today this always loads the v1 shape (raw/B30/...) -- it's the only
    shape any run produces yet. A future Milestone B plan extends this to
    shape-detect and dispatch to a pipeline-shaped loader once
    runs/<id>/pipeline/ exists."""
    return load_v1_rows(run_dir, personas_by_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/diagnostics.py tests/test_diagnostics.py
git commit -m "Add diagnostics.py row loading: DiagnosticsRow, load_v1_rows, load_rows"
```

---

### Task 5: Orchestration — `run_diagnostics`, `DiagnosticsReport`, `write_report` (part 2)

**Files:**
- Modify: `src/diagnostics.py` (append)
- Modify: `tests/test_diagnostics.py` (append)

**Interfaces:**
- Consumes: everything from Task 4; `src.diagnostics_metrics.*` (Tasks 2-3); `src.embeddings.EmbeddingClient` (Task 1); `src.events.load_events`, `src.events.Event`; `src.config.load_config`; `src.taxonomy.load_taxonomy`; `src.personas.load_personas`, `src.personas.render_system_prompt`; `src.llm.LLMClient`, `src.llm.LLMRequest`, `src.llm.structured_schema`; `src.models.PersonaReaction`
- Produces:
  - `DEFAULT_STABILITY_SAMPLE = 25`, `DEFAULT_EMBEDDING_MODEL = "voyage-3-lite"` (verify this model id against Voyage's current docs before Task 7 — see Global Constraints), `STABILITY_RERUNS = 5`
  - `@dataclass(frozen=True) class DiagnosticsReport: run_id, n_rows, register_variance, homogeneity, redundancy, span_dispersion, stability, specificity, distribution_match` with `.to_json_dict() -> dict`
  - `async def run_diagnostics(*, repo: Path = REPO, run_id: str, embedding_client, llm_client=None, stability_sample: int = DEFAULT_STABILITY_SAMPLE, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> DiagnosticsReport`
  - `write_report(report: DiagnosticsReport, run_dir: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diagnostics.py`:

```python
import asyncio
import shutil

from src.diagnostics import DiagnosticsReport, run_diagnostics, write_report
from src.embeddings import FakeEmbeddingClient
from src.llm import FakeLLMClient, LLMRequest


def write_prediction(path: Path, **kwargs) -> None:
    defaults = dict(
        arm="B30", event_id="evt_001", ranked_categories=["privacy", "legal", "pricing"],
        scores={}, backlash_predicted=True,
    )
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults), encoding="utf-8")


EVENTS_TEXT = (
    "=== EVENT ===\n"
    "id: evt_001\n"
    "company: Acme Corp\n"
    "sector: consumer_tech\n"
    "date: 2026-06-14\n"
    "headline: Acme launches an AI layer\n"
    "expected_null: false\n"
    "---\n"
    "Acme Corp today announced an AI layer, enabled by default.\n"
    "=== END EVENT ===\n"
    "\n"
    "=== EVENT ===\n"
    "id: evt_002\n"
    "company: Northwind\n"
    "sector: industrial\n"
    "date: 2026-06-20\n"
    "headline: Northwind opens a warehouse\n"
    "expected_null: true\n"
    "---\n"
    "Northwind opened a distribution centre, adding 400 jobs.\n"
    "=== END EVENT ===\n"
)


@pytest.fixture
def diagnostics_sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copytree(repo / "personas", root / "personas")
    shutil.copy(repo / "config.yaml", root / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", root / "taxonomy.txt")
    (root / "inputs").mkdir()
    (root / "inputs" / "events.txt").write_text(EVENTS_TEXT, encoding="utf-8")

    run_dir = root / "runs" / "run_001"
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize", categories=["privacy"], intensity=0.8, quote="This is a real complaint about defaults.",
    )
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "002.json",
        reaction="mild_concern", categories=["overclaim"], intensity=0.5, quote="Slightly overstated marketing copy here.",
    )
    write_reaction(run_dir / "raw" / "B30" / "evt_002" / "001.json")
    write_reaction(run_dir / "raw" / "B30" / "evt_002" / "002.json")
    write_prediction(run_dir / "predictions" / "evt_001__B30.json", event_id="evt_001", backlash_predicted=True)
    write_prediction(run_dir / "predictions" / "evt_002__B30.json", event_id="evt_002", backlash_predicted=False, ranked_categories=["none", "labor", "environment"])

    return root


def fake_embedding_responder(text: str) -> list[float]:
    # Deterministic, distinguishable-by-length embedding so homogeneity/
    # redundancy produce non-degenerate but reproducible numbers.
    return [float(len(text) % 7), float(len(text) % 5), 1.0]


def test_run_diagnostics_computes_report_without_stability(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    assert report.run_id == "run_001"
    assert report.n_rows == 4
    assert report.register_variance.n == 2  # only the 2 non-ignore rows have quotes
    assert report.homogeneity.n_quotes == 2
    assert report.specificity.n_null_events == 1  # evt_002 is the only expected_null event
    assert report.specificity.false_positive_rate == pytest.approx(0.0)  # B30 correctly said no backlash
    assert report.stability is None
    assert report.span_dispersion is None


def test_run_diagnostics_raises_without_manifest_dir(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    with pytest.raises(DiagnosticsError, match="no run at"):
        asyncio.run(
            run_diagnostics(repo=diagnostics_sandbox, run_id="does_not_exist", embedding_client=embedding_client)
        )


def test_run_diagnostics_runs_stability_when_llm_client_given(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)

    def responder(request: LLMRequest, attempt: int) -> str:
        return json.dumps(
            {"reaction": "criticize", "categories": ["privacy"], "intensity": 0.8, "quote": "q", "reasoning": "r"}
        )

    llm_client = FakeLLMClient(responder)
    report = asyncio.run(
        run_diagnostics(
            repo=diagnostics_sandbox, run_id="run_001",
            embedding_client=embedding_client, llm_client=llm_client, stability_sample=1,
        )
    )
    assert report.stability is not None
    assert report.stability.n_reruns == 5
    assert report.stability.category_agreement_rate == pytest.approx(1.0)


def test_write_report_creates_json_and_txt(diagnostics_sandbox, tmp_path):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    run_dir = diagnostics_sandbox / "runs" / "run_001"
    write_report(report, run_dir)
    assert (run_dir / "diagnostics_report.json").exists()
    assert (run_dir / "diagnostics_report.txt").exists()
    text = (run_dir / "diagnostics_report.txt").read_text(encoding="utf-8")
    assert "Homogeneity" in text
    assert "Stability: not measured" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics.py -v -k "run_diagnostics or write_report"`
Expected: FAIL with `ImportError: cannot import name 'run_diagnostics'`.

- [ ] **Step 3: Implement part 2, appended to `src/diagnostics.py`**

Add these imports to the top of `src/diagnostics.py`, alongside the existing ones:

```python
from dataclasses import asdict
```

Append:

```python
from src.diagnostics_metrics import (
    DistributionMatchResult,
    HomogeneityResult,
    RedundancyResult,
    RegisterVarianceResult,
    SpanDispersionResult,
    SpecificityResult,
    StabilityResult,
    distribution_match_partial,
    homogeneity,
    redundancy,
    register_variance,
    specificity,
    stability,
)
from src.personas import load_personas

DEFAULT_STABILITY_SAMPLE = 25
DEFAULT_EMBEDDING_MODEL = "voyage-3-lite"  # verify against Voyage's current model list before real use
STABILITY_RERUNS = 5
STABILITY_SAMPLE_SEED = 20260730


@dataclass(frozen=True)
class DiagnosticsReport:
    run_id: str
    n_rows: int
    register_variance: RegisterVarianceResult
    homogeneity: HomogeneityResult
    redundancy: RedundancyResult
    span_dispersion: SpanDispersionResult | None
    stability: StabilityResult | None
    specificity: SpecificityResult
    distribution_match: DistributionMatchResult | None

    def to_json_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_rows": self.n_rows,
            "register_variance": asdict(self.register_variance),
            "homogeneity": asdict(self.homogeneity),
            "redundancy": asdict(self.redundancy),
            "span_dispersion": asdict(self.span_dispersion) if self.span_dispersion else None,
            "stability": asdict(self.stability) if self.stability else None,
            "specificity": asdict(self.specificity),
            "distribution_match": asdict(self.distribution_match) if self.distribution_match else None,
        }


def _backlash_predicted_by_event(repo: Path, run_id: str, event_ids: set[str]) -> dict[str, bool]:
    predictions_dir = repo / "runs" / run_id / "predictions"
    result: dict[str, bool] = {}
    for event_id in event_ids:
        path = predictions_dir / f"{event_id}__B30.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            result[event_id] = bool(data["backlash_predicted"])
    return result


def _ground_truth_mix(repo: Path, event_ids: set[str]) -> dict[str, float]:
    labeled_dir = repo / "ground_truth" / "labeled"
    counts: dict[str, int] = {}
    total = 0
    for event_id in event_ids:
        path = labeled_dir / f"{event_id}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for category in data["present_categories"]:
            counts[category] = counts.get(category, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {category: count / total for category, count in counts.items()}


def _simulated_mix(rows: list[DiagnosticsRow]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        if row.category is None:
            continue
        counts[row.category] = counts.get(row.category, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {category: count / total for category, count in counts.items()}


async def _run_stability_check(
    *, client, config, taxonomy, personas_by_id, events_by_id, rows: list[DiagnosticsRow], sample_size: int
) -> StabilityResult | None:
    import random

    from src.llm import LLMRequest, structured_schema
    from src.models import PersonaReaction
    from src.personas import render_system_prompt

    candidates = sorted({(r.persona_id, r.event_id) for r in rows if r.reaction != "ignore"})
    if not candidates:
        return None

    rng = random.Random(STABILITY_SAMPLE_SEED)
    sample = rng.sample(candidates, k=min(sample_size, len(candidates)))

    samples: dict[tuple[str, str], list[tuple[str, frozenset]]] = {}
    for persona_id, event_id in sample:
        persona = personas_by_id[persona_id]
        event = events_by_id[event_id]
        reruns: list[tuple[str, frozenset]] = []
        for attempt in range(1, STABILITY_RERUNS + 1):
            system = render_system_prompt(persona, taxonomy, prior_statements=event.prior_statements)
            system = f"{system}\n<!-- diagnostics stability attempt {attempt}/{STABILITY_RERUNS} -->"
            request = LLMRequest(
                role="persona",
                model=config.models["persona"],
                system=system,
                user=event.to_prompt(),
                max_tokens=config.max_tokens["persona"],
                temperature=config.temperature_for("persona"),
                thinking=config.thinking_for("persona"),
                schema_name="PersonaReaction",
                schema=structured_schema(PersonaReaction),
            )
            reaction = await client.complete(request, PersonaReaction)
            reruns.append((reaction.reaction, frozenset(reaction.categories)))
        samples[(persona_id, event_id)] = reruns

    return stability(samples)


async def run_diagnostics(
    *,
    repo: Path = REPO,
    run_id: str,
    embedding_client,
    llm_client=None,
    stability_sample: int = DEFAULT_STABILITY_SAMPLE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> DiagnosticsReport:
    from src.config import load_config
    from src.events import load_events
    from src.taxonomy import load_taxonomy

    run_dir = repo / "runs" / run_id
    if not run_dir.exists():
        raise DiagnosticsError(f"no run at {run_dir}")

    personas_by_id = {p.id: p for p in load_personas(repo / "personas")}
    events = load_events(repo / "inputs" / "events.txt")
    events_by_id = {e.id: e for e in events}
    null_event_ids = {e.id for e in events if e.expected_null}

    rows = load_rows(run_dir, personas_by_id=personas_by_id)
    non_ignore = [r for r in rows if r.reaction != "ignore" and r.text]
    if not non_ignore:
        raise DiagnosticsError(f"{run_dir}: every reaction was 'ignore' -- nothing to measure")

    quotes = [r.text for r in non_ignore]  # type: ignore[misc]
    register = register_variance(quotes)

    embeddings = await embedding_client.embed(quotes, model=embedding_model)
    homog = homogeneity(embeddings)
    redund = redundancy(embeddings, n_reacting_personas=len({r.persona_id for r in non_ignore}))

    event_ids = {r.event_id for r in rows}
    backlash_by_event = _backlash_predicted_by_event(repo, run_id, event_ids)
    relevant_null_ids = null_event_ids & event_ids
    if not relevant_null_ids:
        raise DiagnosticsError(
            f"{run_dir}: no expected_null events overlap this run's events -- specificity cannot "
            "be measured (spec 0.6 requires >=40% null events in the study design)"
        )
    spec_result = specificity(backlash_by_event, relevant_null_ids)

    gt_mix = _ground_truth_mix(repo, event_ids)
    dist_match = distribution_match_partial(_simulated_mix(rows), gt_mix) if gt_mix else None

    stability_result = None
    if llm_client is not None:
        config = load_config(repo / "config.yaml")
        taxonomy = load_taxonomy(repo / "taxonomy.txt")
        stability_result = await _run_stability_check(
            client=llm_client, config=config, taxonomy=taxonomy,
            personas_by_id=personas_by_id, events_by_id=events_by_id,
            rows=rows, sample_size=stability_sample,
        )

    return DiagnosticsReport(
        run_id=run_id, n_rows=len(rows), register_variance=register,
        homogeneity=homog, redundancy=redund, span_dispersion=None,
        stability=stability_result, specificity=spec_result, distribution_match=dist_match,
    )


def write_report(report: DiagnosticsReport, run_dir: Path) -> None:
    (run_dir / "diagnostics_report.json").write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"Diagnostics -- run {report.run_id} ({report.n_rows} rows)",
        "",
        f"Register variance: word_count mean={report.register_variance.word_count_mean:.1f} "
        f"stdev={report.register_variance.word_count_stdev:.2f} "
        f"min={report.register_variance.word_count_min} max={report.register_variance.word_count_max}; "
        f"exclamation_rate={report.register_variance.exclamation_rate:.2%} "
        f"em_dash_rate={report.register_variance.em_dash_rate:.2%}",
        f"Homogeneity: mean_pairwise_cosine={report.homogeneity.mean_pairwise_cosine:.4f} "
        f"(n={report.homogeneity.n_quotes})",
        f"Redundancy: {report.redundancy.n_clusters} clusters / "
        f"{report.redundancy.n_reacting_personas} reacting personas "
        f"(ratio={report.redundancy.ratio:.3f}, noise={report.redundancy.n_noise})",
        f"Specificity: false_positive_rate={report.specificity.false_positive_rate:.2%} "
        f"({report.specificity.n_false_positive}/{report.specificity.n_null_events} null events)",
    ]
    if report.stability is not None:
        lines.append(
            f"Stability: category_agreement_rate={report.stability.category_agreement_rate:.2%} "
            f"over {report.stability.n_pairs_sampled} pair(s) x {report.stability.n_reruns} reruns"
        )
    else:
        lines.append("Stability: not measured (no llm_client passed)")
    if report.distribution_match is not None:
        lines.append(
            f"Distribution match (partial, vs ground_truth/labeled): "
            f"tvd={report.distribution_match.total_variation_distance:.4f} "
            f"over {report.distribution_match.categories_compared} categor(y/ies)"
        )
    else:
        lines.append("Distribution match: not measured (no ground truth labeled for this run's events)")
    lines.append("Span dispersion: not measured (v1 runs have no span citations)")

    (run_dir / "diagnostics_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: all tests PASS (11 total in this file).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/diagnostics.py tests/test_diagnostics.py
git commit -m "Add diagnostics.py orchestration: run_diagnostics, DiagnosticsReport, write_report"
```

---

### Task 6: Baseline persistence and CLI — `src/diagnostics.py` (part 3)

**Files:**
- Modify: `src/diagnostics.py` (append)
- Modify: `tests/test_diagnostics.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 4-5
- Produces:
  - `load_baseline(repo: Path) -> dict | None`
  - `write_baseline(report: DiagnosticsReport, repo: Path) -> None`
  - `compare_to_baseline(report: DiagnosticsReport, baseline: dict) -> str`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diagnostics.py`:

```python
from src.diagnostics import compare_to_baseline, load_baseline, main, write_baseline


def test_load_baseline_returns_none_when_absent(tmp_path):
    assert load_baseline(tmp_path) is None


def test_write_and_load_baseline_round_trips(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    write_baseline(report, diagnostics_sandbox)
    baseline = load_baseline(diagnostics_sandbox)
    assert baseline is not None
    assert baseline["run_id"] == "run_001"
    assert (diagnostics_sandbox / "results" / "diagnostics_baseline.json").exists()


def test_compare_to_baseline_mentions_both_numbers(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    write_baseline(report, diagnostics_sandbox)
    baseline = load_baseline(diagnostics_sandbox)
    text = compare_to_baseline(report, baseline)
    assert "run_001" in text
    assert "Homogeneity" in text
    assert "Redundancy" in text
```

Note: `main()` itself (the `argparse` CLI wiring real `VoyageEmbeddingClient`/`AnthropicLLMClient`) is intentionally **not** unit-tested here — it requires real API keys and makes real network calls by construction, same as `src/run_sim.py`'s and `src/probe_leakage.py`'s `main()` functions, neither of which has a dedicated test. It's exercised for real in Task 7.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics.py -v -k baseline`
Expected: FAIL with `ImportError: cannot import name 'load_baseline'`.

- [ ] **Step 3: Implement part 3, appended to `src/diagnostics.py`**

Add `import argparse`, `import sys` to the top-of-file imports.

```python
def load_baseline(repo: Path) -> dict | None:
    path = repo / "results" / "diagnostics_baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(report: DiagnosticsReport, repo: Path) -> None:
    path = repo / "results" / "diagnostics_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def compare_to_baseline(report: DiagnosticsReport, baseline: dict) -> str:
    lines = [f"Comparison against results/diagnostics_baseline.json (run {baseline['run_id']}):", ""]
    lines.append(
        f"Homogeneity: {report.homogeneity.mean_pairwise_cosine:.4f} vs "
        f"baseline {baseline['homogeneity']['mean_pairwise_cosine']:.4f}"
    )
    lines.append(
        f"Redundancy ratio: {report.redundancy.ratio:.3f} vs baseline {baseline['redundancy']['ratio']:.3f}"
    )
    lines.append(
        f"Specificity false_positive_rate: {report.specificity.false_positive_rate:.2%} vs "
        f"baseline {baseline['specificity']['false_positive_rate']:.2%}"
    )
    lines.append(
        f"Register variance word_count_stdev: {report.register_variance.word_count_stdev:.2f} vs "
        f"baseline {baseline['register_variance']['word_count_stdev']:.2f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, dest="run_id")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--stability-sample", type=int, default=DEFAULT_STABILITY_SAMPLE)
    parser.add_argument("--skip-stability", action="store_true", help="skip the extra LLM calls Stability needs")
    args = parser.parse_args(argv)

    async def _run() -> DiagnosticsReport:
        from src.embeddings import VoyageEmbeddingClient
        from src.llm import AnthropicLLMClient

        run_dir = REPO / "runs" / args.run_id
        embedding_client = VoyageEmbeddingClient(cache_dir=run_dir / "cache" / "embeddings")
        llm_client = None if args.skip_stability else AnthropicLLMClient(cache_dir=run_dir / "cache")
        try:
            return await run_diagnostics(
                run_id=args.run_id, embedding_client=embedding_client, llm_client=llm_client,
                stability_sample=args.stability_sample,
            )
        finally:
            if llm_client is not None:
                await llm_client.aclose()

    try:
        report = asyncio.run(_run())
    except DiagnosticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run_dir = REPO / "runs" / args.run_id
    write_report(report, run_dir)
    print((run_dir / "diagnostics_report.txt").read_text(encoding="utf-8"))

    if args.write_baseline:
        write_baseline(report, REPO)
        print("\nwrote baseline to results/diagnostics_baseline.json")

    if args.compare_baseline:
        baseline = load_baseline(REPO)
        if baseline is None:
            print(
                "\nerror: no baseline at results/diagnostics_baseline.json -- run --write-baseline first",
                file=sys.stderr,
            )
            return 1
        print()
        print(compare_to_baseline(report, baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: all tests PASS (14 total in this file).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests PASS, no regressions (255 existing + all new tests from this plan).

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/diagnostics.py tests/test_diagnostics.py
git commit -m "Add diagnostics.py baseline persistence and CLI entry point"
```

---

### Task 7: Real-data verification against the two existing runs

**Not a subagent task — the controller runs this directly, since it needs real `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` values that a subagent has no access to.**

- [ ] **Step 1: Confirm `VOYAGE_API_KEY` is set**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
test -f .env && grep -q "^VOYAGE_API_KEY=.\+" .env && echo "key set" || echo "MISSING -- get one from voyageai.com and add it to .env"
```

If missing, stop here and ask the human partner to add it before continuing (the same pattern used for `ANTHROPIC_API_KEY` earlier in this project).

- [ ] **Step 2: Verify the embedding model name resolves**

Before running against real data, confirm `DEFAULT_EMBEDDING_MODEL` (`"voyage-3-lite"` as written in this plan) is still a valid, current Voyage model id — check https://docs.voyageai.com or run one manual embed call and inspect the response for an error. Update the constant in `src/diagnostics.py` if it's changed.

- [ ] **Step 3: Run against the 23-event real run, writing the baseline**

```bash
uv run --env-file .env python -m src.diagnostics --run 20260730T002314.423Z_2ec30ae6 --write-baseline
```

Confirm the printed report's numbers land in the range this session's manual analysis found: homogeneity mean pairwise cosine noticeably above what a genuinely diverse voice set would show, redundancy ratio well below 1.0 (few clusters relative to ~15-17 effective reacting personas), specificity false_positive_rate near 0.0 (10/10 null events correctly predicted no backlash), register variance word_count_min at or near 23. `--skip-stability` is a reasonable first pass to avoid the extra 125 API calls while confirming everything else works; drop it for the real baseline run once satisfied.

- [ ] **Step 4: Run against the 2-event smoke-test run for comparison**

```bash
uv run --env-file .env python -m src.diagnostics --run 20260729T165142.726Z_2ec30ae6 --compare-baseline
```

Confirms `--compare-baseline` works against a second, smaller real run and that the comparison output reads sensibly (expect noisier numbers given only 2 events).

- [ ] **Step 5: Verify `results/diagnostics_baseline.json` and `runs/<id>/diagnostics_report.{json,txt}` are sensible**

Read both files directly. Confirm the JSON round-trips (`python -c "import json; json.load(open('results/diagnostics_baseline.json'))"` succeeds) and the `.txt` summary reads clearly.

- [ ] **Step 6: Report the actual baseline numbers back to the human partner**

This is the deliverable of the entire milestone — state the recorded Homogeneity, Redundancy, Specificity, and Register-variance numbers plainly, as the "before" side of the eventual v1-vs-pipeline comparison Milestone B will produce.

---

## Self-Review Notes

- **Spec coverage:** All seven brief Phase-5 metrics have a task: Homogeneity/Redundancy (Task 3), Span dispersion (Task 3, correctly deferred/`None` for v1 — no spans exist until a future Milestone B), Stability (Task 3 + Task 5's orchestration), Specificity (Task 2), Distribution match (Task 2, explicitly `measured_partial`), Register variance (Task 2). The brief's required gate ("diagnostics/ module against current v1 panel → gate: baseline numbers recorded") is Task 7. CLI shape (`--run`, `--write-baseline`, `--compare-baseline`) matches the plan's Global Constraints and the master plan's design.
- **Placeholders:** none — every step has runnable code; the one intentionally-unimplemented piece (pipeline-shape row loading) is explicitly out of scope for this milestone, documented in `load_rows`'s own docstring rather than left as a silent gap.
- **Type consistency:** `DiagnosticsRow` (Task 4) fields are used identically by `run_diagnostics` (Task 5); `DiagnosticsReport` (Task 5) fields match `write_report`/`write_baseline`/`compare_to_baseline` (Tasks 5-6) field-for-field; `StabilityResult`/`stability()` (Task 3) signature matches `_run_stability_check`'s usage (Task 5) exactly (`dict[tuple[str, str], list[tuple[str, frozenset]]]`).
