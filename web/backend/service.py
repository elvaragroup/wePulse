"""Data-orchestration layer for the client-facing demo web app.

Wires the pre-existing, unmodified `src.dashboard` / `src.events` /
`src.personas` / `src.taxonomy` loaders together with Task 1's `runs.py`
(run-directory discovery) and `transform.py` (pure display transforms) to
produce the two shapes the web frontend needs: a list of event summaries and
a fully assembled per-event result (naive baseline vs. ensemble, plus their
diff).

Malformed or missing data raises rather than silently degrading a panel to
None -- the same "fail loudly" convention `src/events.py`, `src/personas.py`,
`src/taxonomy.py`, and `web/backend/runs.py` already follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.dashboard import (
    load_event_predictions,
    load_event_reactions,
    reaction_mix_counts,
    split_reacted_and_ignored,
)
from src.events import Event, load_events
from src.models import GroundTruthLabel
from src.personas import Persona, load_personas
from src.taxonomy import Taxonomy, load_taxonomy
from web.backend.runs import REPO, WebDataError, get_run_dir
from web.backend.transform import (
    CategoryDiff,
    CategoryScore,
    GroundTruthDisplay,
    QuoteItem,
    compare_predictions,
    reaction_mix_summary,
    select_curated_quotes,
    to_ground_truth_display,
    top_categories,
)

__all__ = [
    "REPO",
    "WebDataError",
    "EventSummary",
    "EventContext",
    "NaiveResult",
    "EnsembleResult",
    "EventResult",
    "list_event_summaries",
    "build_event_result",
    "resolve_ground_truth",
]


@dataclass(frozen=True)
class EventSummary:
    id: str
    company: str
    headline: str
    date: str
    sector: str


@dataclass(frozen=True)
class EventContext:
    id: str
    company: str
    headline: str
    date: str
    sector: str
    source_url: str | None
    announcement: str


@dataclass(frozen=True)
class NaiveResult:
    backlash_predicted: bool
    top_categories: list[CategoryScore]


@dataclass(frozen=True)
class EnsembleResult:
    backlash_predicted: bool
    top_categories: list[CategoryScore]
    reaction_mix_summary: str
    reaction_counts: dict[str, int]
    sample_quotes: list[QuoteItem]


@dataclass(frozen=True)
class EventResult:
    event: EventContext
    naive: NaiveResult
    ensemble: EnsembleResult
    comparison: CategoryDiff
    ground_truth: GroundTruthDisplay | None


@lru_cache(maxsize=None)
def _load_events_cached(repo: Path) -> tuple[Event, ...]:
    """inputs/events.txt is static per-process; avoid re-parsing on every call."""
    return tuple(load_events(repo / "inputs" / "events.txt"))


@lru_cache(maxsize=None)
def _load_personas_by_id_cached(repo: Path) -> dict[str, Persona]:
    return {p.id: p for p in load_personas(repo / "personas")}


@lru_cache(maxsize=None)
def _load_taxonomy_cached(repo: Path) -> Taxonomy:
    return load_taxonomy(repo / "taxonomy.txt")


def _find_event(events: tuple[Event, ...], event_id: str) -> Event:
    for event in events:
        if event.id == event_id:
            return event
    raise WebDataError(f"unknown event_id {event_id!r}: not found in inputs/events.txt")


def resolve_ground_truth(event_id: str, repo: Path = REPO) -> GroundTruthLabel | None:
    """Load a real, judge-produced ground-truth label if one exists.

    Returns None when `ground_truth/labeled/<event_id>.json` doesn't exist --
    the honest, universal case today (see ground_truth/README.md). The moment
    `label_truth.py` is run for real on an event, this starts returning data
    with no further changes needed anywhere in this module.
    """
    path = repo / "ground_truth" / "labeled" / f"{event_id}.json"
    if not path.exists():
        return None
    return GroundTruthLabel.model_validate_json(path.read_text(encoding="utf-8"))


def list_event_summaries(repo: Path = REPO) -> list[EventSummary]:
    """Every event from inputs/events.txt, sorted by id (evt_001..evt_023).

    inputs/events.txt's on-disk order does not actually match numeric id
    order (see commit efae2b4, which spliced evt_021-023 in early, before
    evt_004-020), so we sort explicitly here rather than relying on file
    order. This also matches the plan's API contract for GET /api/events:
    "all 23, sorted by id" -- the right UX for a client-facing dropdown
    regardless of how the source file happens to be ordered.
    """
    events = _load_events_cached(repo)
    summaries = [
        EventSummary(id=e.id, company=e.company, headline=e.headline, date=e.date, sector=e.sector)
        for e in events
    ]
    return sorted(summaries, key=lambda s: s.id)


def build_event_result(event_id: str, repo: Path = REPO) -> EventResult:
    """Assemble the naive-vs-ensemble comparison for one event.

    Raises WebDataError if event_id isn't in inputs/events.txt, or if either
    the 'A' (naive) or 'B30' (ensemble) prediction is missing from the run's
    predictions/ directory.
    """
    events = _load_events_cached(repo)
    event = _find_event(events, event_id)

    taxonomy = _load_taxonomy_cached(repo)
    personas_by_id = _load_personas_by_id_cached(repo)
    run_dir = get_run_dir(repo)

    predictions_by_arm = {p.arm: p for p in load_event_predictions(run_dir, event_id)}

    naive_pred = predictions_by_arm.get("A")
    if naive_pred is None:
        raise WebDataError(f"missing arm 'A' prediction for {event_id!r} in {run_dir}")

    ensemble_pred = predictions_by_arm.get("B30")
    if ensemble_pred is None:
        raise WebDataError(f"missing arm 'B30' prediction for {event_id!r} in {run_dir}")

    reactions = load_event_reactions(run_dir, event_id, personas_by_id)
    reacted, _ignored = split_reacted_and_ignored(reactions)
    counts = reaction_mix_counts(reactions)

    event_context = EventContext(
        id=event.id,
        company=event.company,
        headline=event.headline,
        date=event.date,
        sector=event.sector,
        source_url=event.source_url,
        announcement=event.announcement,
    )

    naive_result = NaiveResult(
        backlash_predicted=naive_pred.backlash_predicted,
        top_categories=top_categories(naive_pred, taxonomy),
    )

    ensemble_result = EnsembleResult(
        backlash_predicted=ensemble_pred.backlash_predicted,
        top_categories=top_categories(ensemble_pred, taxonomy),
        reaction_mix_summary=reaction_mix_summary(counts),
        reaction_counts=counts,
        sample_quotes=select_curated_quotes(reacted, personas_by_id),
    )

    comparison = compare_predictions(naive_pred, ensemble_pred, taxonomy)
    ground_truth_label = resolve_ground_truth(event_id, repo)
    ground_truth = to_ground_truth_display(ground_truth_label, taxonomy)

    return EventResult(
        event=event_context,
        naive=naive_result,
        ensemble=ensemble_result,
        comparison=comparison,
        ground_truth=ground_truth,
    )
