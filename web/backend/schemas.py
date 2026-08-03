"""Pydantic response models mirroring `service.py`/`transform.py` dataclasses.

Field-for-field mirrors of the dataclasses produced by Task 1/2's pure-Python
layer, plus adapter functions that convert those dataclasses into the
pydantic models FastAPI serializes. Keeping the adapter logic here (rather
than inline in `main.py`) keeps route handlers thin.
"""

from __future__ import annotations

from pydantic import BaseModel

from web.backend.service import (
    EnsembleResult,
    EventContext,
    EventResult,
    EventSummary,
    NaiveResult,
)
from web.backend.transform import CategoryDiff, CategoryScore, GroundTruthDisplay, QuoteItem

__all__ = [
    "EventSummaryOut",
    "EventContextOut",
    "CategoryScoreOut",
    "NaiveResultOut",
    "QuoteItemOut",
    "EnsembleResultOut",
    "ComparisonOut",
    "GroundTruthOut",
    "EventResultResponse",
    "EventsListResponse",
    "to_event_summary_out",
    "to_event_context_out",
    "to_category_score_out",
    "to_naive_result_out",
    "to_quote_item_out",
    "to_ensemble_result_out",
    "to_comparison_out",
    "to_ground_truth_out",
    "to_event_result_response",
    "to_events_list_response",
]


class EventSummaryOut(BaseModel):
    id: str
    company: str
    headline: str
    date: str
    sector: str


class EventContextOut(BaseModel):
    id: str
    company: str
    headline: str
    date: str
    sector: str
    source_url: str | None
    announcement: str


class CategoryScoreOut(BaseModel):
    id: str
    label: str
    confidence: float | None


class NaiveResultOut(BaseModel):
    backlash_predicted: bool
    top_categories: list[CategoryScoreOut]


class QuoteItemOut(BaseModel):
    archetype_label: str
    platform: str
    reaction: str
    intensity: float
    quote: str
    categories: tuple[str, ...]


class EnsembleResultOut(BaseModel):
    backlash_predicted: bool
    top_categories: list[CategoryScoreOut]
    reaction_mix_summary: str
    reaction_counts: dict[str, int]
    sample_quotes: list[QuoteItemOut]


class ComparisonOut(BaseModel):
    agreed: list[CategoryScoreOut]
    ensemble_only: list[CategoryScoreOut]
    naive_only: list[CategoryScoreOut]
    backlash_agreement: bool


class GroundTruthOut(BaseModel):
    dominant_category: CategoryScoreOut
    present_categories: list[CategoryScoreOut]
    backlash_occurred: bool
    judge_confidence: float
    summary: str


class EventResultResponse(BaseModel):
    event: EventContextOut
    naive: NaiveResultOut
    ensemble: EnsembleResultOut
    comparison: ComparisonOut
    ground_truth: GroundTruthOut | None


class EventsListResponse(BaseModel):
    events: list[EventSummaryOut]


def to_event_summary_out(summary: EventSummary) -> EventSummaryOut:
    return EventSummaryOut(
        id=summary.id,
        company=summary.company,
        headline=summary.headline,
        date=summary.date,
        sector=summary.sector,
    )


def to_event_context_out(event: EventContext) -> EventContextOut:
    return EventContextOut(
        id=event.id,
        company=event.company,
        headline=event.headline,
        date=event.date,
        sector=event.sector,
        source_url=event.source_url,
        announcement=event.announcement,
    )


def to_category_score_out(score: CategoryScore) -> CategoryScoreOut:
    return CategoryScoreOut(id=score.id, label=score.label, confidence=score.confidence)


def to_naive_result_out(naive: NaiveResult) -> NaiveResultOut:
    return NaiveResultOut(
        backlash_predicted=naive.backlash_predicted,
        top_categories=[to_category_score_out(c) for c in naive.top_categories],
    )


def to_quote_item_out(quote: QuoteItem) -> QuoteItemOut:
    return QuoteItemOut(
        archetype_label=quote.archetype_label,
        platform=quote.platform,
        reaction=quote.reaction,
        intensity=quote.intensity,
        quote=quote.quote,
        categories=quote.categories,
    )


def to_ensemble_result_out(ensemble: EnsembleResult) -> EnsembleResultOut:
    return EnsembleResultOut(
        backlash_predicted=ensemble.backlash_predicted,
        top_categories=[to_category_score_out(c) for c in ensemble.top_categories],
        reaction_mix_summary=ensemble.reaction_mix_summary,
        reaction_counts=ensemble.reaction_counts,
        sample_quotes=[to_quote_item_out(q) for q in ensemble.sample_quotes],
    )


def to_comparison_out(comparison: CategoryDiff) -> ComparisonOut:
    return ComparisonOut(
        agreed=[to_category_score_out(c) for c in comparison.agreed],
        ensemble_only=[to_category_score_out(c) for c in comparison.ensemble_only],
        naive_only=[to_category_score_out(c) for c in comparison.naive_only],
        backlash_agreement=comparison.backlash_agreement,
    )


def to_ground_truth_out(ground_truth: GroundTruthDisplay | None) -> GroundTruthOut | None:
    if ground_truth is None:
        return None
    return GroundTruthOut(
        dominant_category=to_category_score_out(ground_truth.dominant_category),
        present_categories=[to_category_score_out(c) for c in ground_truth.present_categories],
        backlash_occurred=ground_truth.backlash_occurred,
        judge_confidence=ground_truth.judge_confidence,
        summary=ground_truth.summary,
    )


def to_event_result_response(result: EventResult) -> EventResultResponse:
    return EventResultResponse(
        event=to_event_context_out(result.event),
        naive=to_naive_result_out(result.naive),
        ensemble=to_ensemble_result_out(result.ensemble),
        comparison=to_comparison_out(result.comparison),
        ground_truth=to_ground_truth_out(result.ground_truth),
    )


def to_events_list_response(summaries: list[EventSummary]) -> EventsListResponse:
    return EventsListResponse(events=[to_event_summary_out(s) for s in summaries])
