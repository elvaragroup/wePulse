"""Static HTML dashboard over one run's events, persona reactions, and
aggregated predictions (docs/superpowers/specs/2026-07-29-dashboard-design.md).

Read-only: every function here loads files that run_sim.py / score.py already
produce. No API calls, no new dependencies, no server -- `main` writes one
self-contained HTML file.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from src.events import Event, EventParseError, load_events
from src.models import Arm, ArmPrediction, PersonaReaction, Reaction
from src.personas import Persona, PersonaError, load_personas

REPO = Path(__file__).resolve().parent.parent
ARM_ORDER = get_args(Arm)


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


REACTION_KEYS = get_args(Reaction)


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


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _persona_label(row: ReactionRow) -> str:
    return f"{_escape(row.persona_id)} {_escape(row.persona_name)}"


def render_event_card(
    event: Event,
    reacted: list[ReactionRow],
    ignored: list[ReactionRow],
    predictions: list[ArmPrediction],
    counts: dict[str, int],
) -> str:
    prediction_rows = "\n".join(
        f'<div class="arm-row"><span class="arm-label">{_escape(p.arm)}</span>'
        f'<span class="arm-cats">{_escape(", ".join(p.ranked_categories))}</span>'
        f'<span class="arm-flag">{"backlash" if p.backlash_predicted else "no backlash"}</span></div>'
        for p in predictions
    )
    reacted_rows = "\n".join(
        f"<tr><td>{_persona_label(r)}</td><td>{_escape(r.archetype)}</td>"
        f"<td>{_escape(r.reaction)}</td><td>{_escape(', '.join(r.categories))}</td>"
        f"<td>{r.intensity:.2f}</td><td>{_escape(r.quote or '')}</td></tr>"
        for r in reacted
    )
    ignored_list = ", ".join(_persona_label(r) for r in ignored)
    counts_line = " | ".join(f"{key}: {count}" for key, count in counts.items())

    return f"""
<details class="event-card">
  <summary>
    <div class="event-header">
      <span class="event-id">{_escape(event.id)}</span>
      <span class="event-company">{_escape(event.company)}</span>
      <span class="event-headline">{_escape(event.headline)}</span>
      <span class="badge">expected_null: {event.expected_null}</span>
    </div>
    <div class="predictions">
      {prediction_rows}
    </div>
  </summary>
  <pre class="announcement">{_escape(event.announcement)}</pre>
  <div class="reaction-mix">{counts_line}</div>
  <table class="reactions">
    <thead><tr><th>Persona</th><th>Archetype</th><th>Reaction</th><th>Categories</th><th>Intensity</th><th>Quote</th></tr></thead>
    <tbody>
      {reacted_rows}
    </tbody>
  </table>
  <details class="ignored-list">
    <summary>{len(ignored)} persona(s) ignored</summary>
    <p>{ignored_list}</p>
  </details>
</details>
""".strip()


def render_missing_event_card(event: Event) -> str:
    return f"""
<details class="event-card missing">
  <summary>
    <div class="event-header">
      <span class="event-id">{_escape(event.id)}</span>
      <span class="event-company">{_escape(event.company)}</span>
      <span class="event-headline">{_escape(event.headline)}</span>
      <span class="badge missing">not yet run</span>
    </div>
  </summary>
  <p>No predictions or persona reactions found for this event in this run.</p>
</details>
""".strip()


PAGE_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
.event-card { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1rem; padding: 0.75rem 1rem; }
.event-card summary { cursor: pointer; }
.event-header { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
.event-id { font-weight: 700; }
.badge { margin-left: auto; font-size: 0.8rem; padding: 0.1rem 0.5rem; border-radius: 999px; background: #eee; }
.badge.missing { background: #fee; }
.predictions { margin-top: 0.5rem; }
.arm-row { display: flex; gap: 0.75rem; font-family: monospace; font-size: 0.9rem; flex-wrap: wrap; }
.arm-label { font-weight: 700; width: 3.5rem; }
table.reactions { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
table.reactions th, table.reactions td { border-bottom: 1px solid #eee; padding: 0.25rem 0.5rem; text-align: left; font-size: 0.9rem; }
.announcement { white-space: pre-wrap; background: #fafafa; padding: 0.5rem; border-radius: 4px; font-size: 0.85rem; }
"""


def render_page(run_id: str, cards_html: list[str]) -> str:
    cards = "\n".join(cards_html)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>crisis-sim dashboard: {_escape(run_id)}</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<h1>Run {_escape(run_id)}</h1>
{cards}
</body>
</html>
"""


def build_dashboard(*, repo: Path = REPO, run_id: str) -> str:
    run_dir = repo / "runs" / run_id
    if not (run_dir / "manifest.json").exists():
        raise DashboardError(f"no run at {run_dir}")

    events = load_events(repo / "inputs" / "events.txt")
    personas_by_id = {p.id: p for p in load_personas(repo / "personas")}

    cards: list[str] = []
    for event in events:
        predictions = load_event_predictions(run_dir, event.id)
        reactions = load_event_reactions(run_dir, event.id, personas_by_id)
        if not predictions and not reactions:
            cards.append(render_missing_event_card(event))
            continue
        counts = reaction_mix_counts(reactions)
        reacted, ignored = split_reacted_and_ignored(reactions)
        cards.append(render_event_card(event, reacted, ignored, predictions, counts))

    return render_page(run_id, cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        page = build_dashboard(repo=args.repo, run_id=args.run_id)
    except (DashboardError, EventParseError, PersonaError, OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = args.out or (args.repo / "runs" / args.run_id / "dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
