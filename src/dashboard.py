"""Static HTML dashboard over one run's events, persona reactions, and
aggregated predictions (docs/superpowers/specs/2026-07-29-dashboard-design.md).

Read-only: every function here loads files that run_sim.py / score.py already
produce. No API calls, no new dependencies, no server -- `main` writes one
self-contained HTML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models import ArmPrediction, PersonaReaction
from src.personas import Persona

REPO = Path(__file__).resolve().parent.parent
ARM_ORDER = ("A", "B3", "B8", "B15", "B30", "C")


class DashboardError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReactionRow:
    persona_id: str
    persona_name: str
    archetype: str
    reaction: str
    categories: tuple[str, ...]
    intensity: float
    quote: str | None


def load_event_reactions(
    run_dir: Path, event_id: str, personas_by_id: dict[str, Persona]
) -> list[ReactionRow]:
    """Every persona's reaction to one event, from raw/B30/<event_id>/*.json --
    B30 always holds every persona's reaction regardless of which arms ran,
    since run_sim.py writes reactions once and reuses them across arms."""
    event_dir = run_dir / "raw" / "B30" / event_id
    if not event_dir.exists():
        return []

    rows: list[ReactionRow] = []
    for path in sorted(event_dir.glob("*.json")):
        persona_id = path.stem
        reaction = PersonaReaction.model_validate_json(path.read_text(encoding="utf-8"))
        persona = personas_by_id.get(persona_id)
        rows.append(
            ReactionRow(
                persona_id=persona_id,
                persona_name=persona.name if persona else f"(persona file not found: {persona_id})",
                archetype=persona.archetype if persona else "unknown",
                reaction=reaction.reaction,
                categories=tuple(reaction.categories),
                intensity=reaction.intensity,
                quote=reaction.quote,
            )
        )
    return rows


def load_event_predictions(run_dir: Path, event_id: str) -> list[ArmPrediction]:
    """Every arm's aggregated prediction for one event, ordered by ARM_ORDER."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return []

    predictions = [
        ArmPrediction.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(predictions_dir.glob(f"{event_id}__*.json"))
    ]

    def sort_key(prediction: ArmPrediction) -> tuple[int, str]:
        try:
            return (ARM_ORDER.index(prediction.arm), "")
        except ValueError:
            return (len(ARM_ORDER), prediction.arm)

    return sorted(predictions, key=sort_key)


REACTION_KEYS = ("ignore", "mild_concern", "criticize", "outrage")


def reaction_mix_counts(rows: list[ReactionRow]) -> dict[str, int]:
    counts = dict.fromkeys(REACTION_KEYS, 0)
    for row in rows:
        counts[row.reaction] += 1
    return counts


def split_reacted_and_ignored(
    rows: list[ReactionRow],
) -> tuple[list[ReactionRow], list[ReactionRow]]:
    """Reacted personas first (what you actually want to read), sorted by how
    strongly they felt; ignored personas collapsed to their own list so 20+
    'ignore' rows don't bury the signal."""
    reacted = sorted(
        (r for r in rows if r.reaction != "ignore"),
        key=lambda r: (-r.intensity, r.persona_id),
    )
    ignored = sorted((r for r in rows if r.reaction == "ignore"), key=lambda r: r.persona_id)
    return reacted, ignored
