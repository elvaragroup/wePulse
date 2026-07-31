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
