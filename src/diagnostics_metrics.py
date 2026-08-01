"""Pure statistical functions for the anti-collapse diagnostics suite (see
docs/superpowers/plans/2026-07-30-diagnostics-milestone-a.md). No I/O, no
LLM/embedding calls -- src/diagnostics.py owns loading data and calling
embedding/LLM clients; these functions only compute over already-collected
values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.embeddings import cosine_similarity

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


@dataclass(frozen=True)
class HomogeneityResult:
    mean_pairwise_cosine: float
    n_quotes: int
    n_pairs: int


def homogeneity(embeddings: list[list[float]]) -> HomogeneityResult:
    """Mean pairwise cosine similarity across every non-ignore quote's
    embedding. High similarity means many personas are one voice wearing
    different subject-matter vocabulary.

    Caveat: when called on a whole-run's pooled quotes spanning multiple
    events (as src/diagnostics.py's run_diagnostics does), this number
    blends cross-event topical (dis)similarity with within-topic
    voice-similarity -- a low value could reflect genuine topic diversity
    across events rather than genuine voice diversity within a topic."""
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
    land in the same handful of semantic clusters -- collapse.

    Caveat: when called on a whole-run's pooled quotes spanning multiple
    events (as src/diagnostics.py's run_diagnostics does), a high
    ratio could reflect genuine topic diversity across events rather than
    genuine voice diversity within a topic, and vice versa."""
    from sklearn.cluster import HDBSCAN

    if len(embeddings) < min_cluster_size:
        ratio = len(embeddings) / n_reacting_personas if n_reacting_personas else 0.0
        return RedundancyResult(
            n_clusters=len(embeddings),
            n_reacting_personas=n_reacting_personas,
            ratio=ratio,
            n_noise=0,
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
