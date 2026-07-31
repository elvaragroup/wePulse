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
