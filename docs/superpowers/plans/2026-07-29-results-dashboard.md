# Results Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static-HTML dashboard (`python -m src.dashboard <run_id>`) that renders one run's events, all 30 personas' reactions, and each arm's aggregated top-3 prediction into a single self-contained HTML file, so results can be scanned in a browser instead of grepped from raw JSON.

**Architecture:** A single new module, `src/dashboard.py`, following the existing CLI-script convention (`run_sim.py`, `score.py`): module-level `REPO`, an `<Name>Error` exception class, pure data-loading/aggregation functions, pure string-based HTML-rendering functions, and a `main(argv)` entry point. No new dependencies — reads existing pydantic models (`Event`, `Persona`, `PersonaReaction`, `ArmPrediction`) via their existing loaders, and writes one HTML file with `html.escape` for all interpolated text.

**Tech Stack:** Python 3.12 stdlib only (`argparse`, `html`, `dataclasses`, `pathlib`). No new PyPI dependency. `pytest` for tests, same as the rest of the repo.

## Global Constraints

- Zero new dependencies — spec's non-goal list explicitly rules out adding infrastructure (`CRISIS_SIM_VALIDATION_SPEC.md` §0).
- No server process, no build step — one script writes one static HTML file (design spec, "Goals").
- One `run_id` per invocation — no cross-run comparison in this version (design spec, "Non-goals").
- No ground truth, `results/scores.csv` metrics, or leakage verdicts shown — reactions + aggregated predictions only (design spec, "Non-goals").
- Output defaults to `runs/<run_id>/dashboard.html`; overridable via `--out` (design spec, "CLI").
- One event's missing data must render a "not yet run" notice, never crash the whole generator (design spec, "Error handling").
- All interpolated text (announcement, quotes, headlines) must be HTML-escaped — this is a static file rendering third-party/model-generated text.
- Follow the existing module convention: `REPO = Path(__file__).resolve().parent.parent`, a module-specific `Error(RuntimeError)` class, `argparse`-based `main(argv: list[str] | None = None) -> int`.

---

### Task 1: Loaders — `ReactionRow`, `load_event_reactions`, `load_event_predictions`

**Files:**
- Create: `src/dashboard.py`
- Modify: `tests/conftest.py` (add a session-scoped `personas` fixture, following the existing `taxonomy`/`config` fixture pattern)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `src.events.Event`, `src.events.load_events`; `src.personas.Persona`, `src.personas.load_personas`; `src.models.PersonaReaction`, `src.models.ArmPrediction`.
- Produces:
  - `REPO: Path`
  - `ARM_ORDER: tuple[str, ...] = ("A", "B3", "B8", "B15", "B30", "C")`
  - `class DashboardError(RuntimeError): pass`
  - `@dataclass(frozen=True) class ReactionRow: persona_id: str; persona_name: str; archetype: str; reaction: str; categories: tuple[str, ...]; intensity: float; quote: str | None`
  - `load_event_reactions(run_dir: Path, event_id: str, personas_by_id: dict[str, Persona]) -> list[ReactionRow]`
  - `load_event_predictions(run_dir: Path, event_id: str) -> list[ArmPrediction]`

- [ ] **Step 1: Add a `personas` fixture to `tests/conftest.py`**

Open `tests/conftest.py` and add, next to the existing `taxonomy`/`config` fixtures:

```python
from src.personas import load_personas


@pytest.fixture(scope="session")
def personas():
    return load_personas(REPO / "personas")
```

(Add the import at the top alongside the existing `from src.config import load_config` / `from src.taxonomy import load_taxonomy` lines.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dashboard.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from src.dashboard import ARM_ORDER, ReactionRow, load_event_predictions, load_event_reactions
from src.models import ArmPrediction, PersonaReaction


def write_reaction(path: Path, **kwargs) -> None:
    defaults = dict(reaction="ignore", categories=[], intensity=0.0, quote=None, reasoning="r")
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PersonaReaction(**defaults).model_dump_json(), encoding="utf-8")


def write_prediction(path: Path, **kwargs) -> None:
    defaults = dict(
        arm="A",
        event_id="evt_001",
        ranked_categories=["privacy", "legal", "pricing"],
        scores={},
        backlash_predicted=True,
    )
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ArmPrediction(**defaults).model_dump_json(), encoding="utf-8")


def test_load_event_reactions_maps_persona_metadata(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(
        tmp_path / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize",
        categories=["privacy"],
        intensity=0.8,
        quote="q",
    )
    rows = load_event_reactions(tmp_path, "evt_001", personas_by_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.persona_id == "001"
    assert row.persona_name == "privacy_hawk"
    assert row.archetype == "critic"
    assert row.reaction == "criticize"
    assert row.categories == ("privacy",)
    assert row.intensity == 0.8
    assert row.quote == "q"


def test_load_event_reactions_falls_back_for_unknown_persona_id(tmp_path):
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "999.json")
    rows = load_event_reactions(tmp_path, "evt_001", {})
    assert len(rows) == 1
    assert rows[0].persona_id == "999"
    assert "not found" in rows[0].persona_name


def test_load_event_reactions_empty_when_directory_missing(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    rows = load_event_reactions(tmp_path, "evt_absent", personas_by_id)
    assert rows == []


def test_load_event_reactions_sorted_by_persona_id(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "002.json")
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    rows = load_event_reactions(tmp_path, "evt_001", personas_by_id)
    assert [r.persona_id for r in rows] == ["001", "002"]


def test_load_event_predictions_sorted_by_arm_order(tmp_path):
    write_prediction(
        tmp_path / "predictions" / "evt_001__C.json",
        arm="C",
        ranked_categories=["none", "labor", "environment"],
        backlash_predicted=False,
    )
    write_prediction(tmp_path / "predictions" / "evt_001__A.json", arm="A")
    write_prediction(tmp_path / "predictions" / "evt_001__B30.json", arm="B30")
    preds = load_event_predictions(tmp_path, "evt_001")
    assert [p.arm for p in preds] == ["A", "B30", "C"]
    assert ARM_ORDER == ("A", "B3", "B8", "B15", "B30", "C")


def test_load_event_predictions_empty_when_no_matches(tmp_path):
    assert load_event_predictions(tmp_path, "evt_absent") == []


def test_load_event_predictions_ignores_other_events(tmp_path):
    write_prediction(tmp_path / "predictions" / "evt_002__A.json", event_id="evt_002")
    write_prediction(tmp_path / "predictions" / "evt_001__A.json", event_id="evt_001")
    preds = load_event_predictions(tmp_path, "evt_001")
    assert len(preds) == 1
    assert preds[0].event_id == "evt_001"
```

- [ ] **Step 3: Run tests, verify they fail with an import error**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: `ModuleNotFoundError: No module named 'src.dashboard'` (or collection error) for every test.

- [ ] **Step 4: Implement `src/dashboard.py`**

```python
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
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/dashboard.py tests/test_dashboard.py tests/conftest.py
git commit -m "Add dashboard loaders: ReactionRow, load_event_reactions, load_event_predictions"
```

---

### Task 2: Aggregation — `reaction_mix_counts`, `split_reacted_and_ignored`

**Files:**
- Modify: `src/dashboard.py` (append)
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `ReactionRow` (Task 1)
- Produces:
  - `REACTION_KEYS: tuple[str, ...] = ("ignore", "mild_concern", "criticize", "outrage")`
  - `reaction_mix_counts(rows: list[ReactionRow]) -> dict[str, int]`
  - `split_reacted_and_ignored(rows: list[ReactionRow]) -> tuple[list[ReactionRow], list[ReactionRow]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
from src.dashboard import REACTION_KEYS, ReactionRow, reaction_mix_counts, split_reacted_and_ignored


def make_row(persona_id: str, reaction: str, intensity: float = 0.0) -> ReactionRow:
    return ReactionRow(
        persona_id=persona_id,
        persona_name=f"name{persona_id}",
        archetype="critic",
        reaction=reaction,
        categories=() if reaction == "ignore" else ("privacy",),
        intensity=intensity,
        quote=None if reaction == "ignore" else "q",
    )


def test_reaction_mix_counts_all_four_keys_always_present():
    counts = reaction_mix_counts([make_row("001", "criticize", 0.5)])
    assert counts == {"ignore": 0, "mild_concern": 0, "criticize": 1, "outrage": 0}


def test_reaction_mix_counts_tallies_across_rows():
    rows = [
        make_row("001", "ignore"),
        make_row("002", "ignore"),
        make_row("003", "outrage", 1.0),
        make_row("004", "mild_concern", 0.3),
    ]
    assert reaction_mix_counts(rows) == {"ignore": 2, "mild_concern": 1, "criticize": 0, "outrage": 1}


def test_reaction_mix_counts_key_order_matches_reaction_keys():
    assert list(reaction_mix_counts([]).keys()) == list(REACTION_KEYS)


def test_split_separates_ignored_from_reacted():
    rows = [make_row("001", "ignore"), make_row("002", "criticize", 0.5)]
    reacted, ignored = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["002"]
    assert [r.persona_id for r in ignored] == ["001"]


def test_split_reacted_sorted_by_intensity_descending():
    rows = [make_row("001", "criticize", 0.3), make_row("002", "outrage", 0.9), make_row("003", "mild_concern", 0.5)]
    reacted, _ = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["002", "003", "001"]


def test_split_ties_broken_by_persona_id():
    rows = [make_row("003", "criticize", 0.5), make_row("001", "criticize", 0.5)]
    reacted, _ = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["001", "003"]


def test_split_ignored_sorted_by_persona_id():
    rows = [make_row("003", "ignore"), make_row("001", "ignore")]
    _, ignored = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in ignored] == ["001", "003"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_dashboard.py -k "reaction_mix or split_" -v`
Expected: FAIL with `ImportError: cannot import name 'REACTION_KEYS'`.

- [ ] **Step 3: Implement the aggregation functions**

Append to `src/dashboard.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: all tests PASS (14 total so far).

- [ ] **Step 5: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/dashboard.py tests/test_dashboard.py
git commit -m "Add dashboard aggregation: reaction_mix_counts, split_reacted_and_ignored"
```

---

### Task 3: HTML rendering — `render_event_card`, `render_missing_event_card`, `render_page`

**Files:**
- Modify: `src/dashboard.py` (append)
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `Event` (`src.events`), `ArmPrediction` (`src.models`), `ReactionRow`/`ARM_ORDER` (Task 1), `reaction_mix_counts`/`split_reacted_and_ignored` (Task 2)
- Produces:
  - `render_event_card(event: Event, reacted: list[ReactionRow], ignored: list[ReactionRow], predictions: list[ArmPrediction], counts: dict[str, int]) -> str`
  - `render_missing_event_card(event: Event) -> str`
  - `render_page(run_id: str, cards_html: list[str]) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
from src.dashboard import render_event_card, render_missing_event_card, render_page
from src.events import Event


def make_event(**overrides) -> Event:
    defaults = dict(
        id="evt_001",
        company="Acme Corp",
        sector="consumer_tech",
        date="2026-06-14",
        headline="Acme introduces Acme Assist",
        source_url=None,
        expected_null=False,
        announcement="Acme Corp today announced Acme Assist.",
        prior_statements=None,
    )
    defaults.update(overrides)
    return Event(**defaults)


def test_render_event_card_includes_headline_and_arm_predictions():
    event = make_event()
    prediction = ArmPrediction(
        arm="B30",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "financial"],
        scores={},
        backlash_predicted=True,
    )
    reacted = [make_row("001", "criticize", 0.8)]
    html_out = render_event_card(event, reacted, [], [prediction], reaction_mix_counts(reacted))
    assert "evt_001" in html_out
    assert "Acme introduces Acme Assist" in html_out
    assert "B30" in html_out
    assert "privacy" in html_out and "overclaim" in html_out


def test_render_event_card_escapes_html_in_announcement_and_quotes():
    event = make_event(announcement="Uses <script>alert(1)</script> data.")
    reacted = [
        ReactionRow(
            persona_id="001",
            persona_name="privacy_hawk",
            archetype="critic",
            reaction="criticize",
            categories=("privacy",),
            intensity=0.8,
            quote="<b>quote</b>",
        )
    ]
    html_out = render_event_card(event, reacted, [], [], reaction_mix_counts(reacted))
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<b>quote</b>" not in html_out


def test_render_event_card_lists_ignored_personas_compactly():
    event = make_event()
    ignored = [
        ReactionRow(
            persona_id="002",
            persona_name="labor_advocate",
            archetype="critic",
            reaction="ignore",
            categories=(),
            intensity=0.0,
            quote=None,
        )
    ]
    html_out = render_event_card(event, [], ignored, [], reaction_mix_counts([]))
    assert "labor_advocate" in html_out


def test_render_missing_event_card_notes_not_yet_run():
    event = make_event(id="evt_003")
    html_out = render_missing_event_card(event)
    assert "evt_003" in html_out
    assert "not yet run" in html_out.lower()


def test_render_page_wraps_cards_with_run_id():
    page = render_page("run_001", ["<details>card A</details>", "<details>card B</details>"])
    assert "run_001" in page
    assert "card A" in page and "card B" in page
    assert "<html" in page
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_dashboard.py -k render -v`
Expected: FAIL with `ImportError: cannot import name 'render_event_card'`.

- [ ] **Step 3: Implement the rendering functions**

Append to `src/dashboard.py` (add `import html` to the top-of-file imports alongside the existing `dataclasses`/`pathlib` imports):

```python
def _escape(text: str) -> str:
    return html.escape(text, quote=True)


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
        f"<tr><td>{_escape(r.persona_id)} {_escape(r.persona_name)}</td><td>{_escape(r.archetype)}</td>"
        f"<td>{_escape(r.reaction)}</td><td>{_escape(', '.join(r.categories))}</td>"
        f"<td>{r.intensity:.2f}</td><td>{_escape(r.quote or '')}</td></tr>"
        for r in reacted
    )
    ignored_list = ", ".join(f"{_escape(r.persona_id)} {_escape(r.persona_name)}" for r in ignored)
    counts_line = " | ".join(f"{key}: {count}" for key, count in counts.items())

    return f"""
<details class="event-card">
  <summary>
    <span class="event-id">{_escape(event.id)}</span>
    <span class="event-company">{_escape(event.company)}</span>
    <span class="event-headline">{_escape(event.headline)}</span>
    <span class="badge">expected_null: {event.expected_null}</span>
  </summary>
  <div class="predictions">
    {prediction_rows}
  </div>
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
    <span class="event-id">{_escape(event.id)}</span>
    <span class="event-company">{_escape(event.company)}</span>
    <span class="event-headline">{_escape(event.headline)}</span>
    <span class="badge missing">not yet run</span>
  </summary>
  <p>No predictions or persona reactions found for this event in this run.</p>
</details>
""".strip()


PAGE_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
.event-card { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1rem; padding: 0.75rem 1rem; }
.event-card summary { cursor: pointer; display: flex; gap: 0.75rem; align-items: center; }
.event-id { font-weight: 700; }
.badge { margin-left: auto; font-size: 0.8rem; padding: 0.1rem 0.5rem; border-radius: 999px; background: #eee; }
.badge.missing { background: #fee; }
.arm-row { display: flex; gap: 0.75rem; font-family: monospace; font-size: 0.9rem; }
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
```

Also add `from src.events import Event` to the imports at the top of `src/dashboard.py`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: all tests PASS (19 total so far).

- [ ] **Step 5: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/dashboard.py tests/test_dashboard.py
git commit -m "Add dashboard HTML rendering: event cards and page wrapper"
```

---

### Task 4: Orchestration and CLI — `build_dashboard`, `main`

**Files:**
- Modify: `src/dashboard.py` (append)
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-3; `src.events.load_events`; `src.personas.load_personas`
- Produces:
  - `build_dashboard(*, repo: Path = REPO, run_id: str) -> str`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
import shutil

from src.dashboard import DashboardError, build_dashboard, main


@pytest.fixture
def sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copytree(repo / "personas", root / "personas")
    (root / "inputs").mkdir()
    (root / "inputs" / "events.txt").write_text(
        "=== EVENT ===\n"
        "id: evt_001\n"
        "company: Acme Corp\n"
        "sector: consumer_tech\n"
        "date: 2026-06-14\n"
        "headline: Acme launches an AI layer\n"
        "expected_null: false\n"
        "---\n"
        "Acme Corp today announced an AI layer.\n"
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
        "Northwind opened a distribution centre.\n"
        "=== END EVENT ===\n",
        encoding="utf-8",
    )
    return root


def test_build_dashboard_raises_without_manifest(sandbox):
    with pytest.raises(DashboardError, match="no run at"):
        build_dashboard(repo=sandbox, run_id="run_001")


def test_build_dashboard_renders_missing_card_for_unrun_event(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    page = build_dashboard(repo=sandbox, run_id="run_001")
    assert "evt_001" in page
    assert "evt_002" in page
    assert "not yet run" in page.lower()


def test_build_dashboard_renders_full_card_when_data_present(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize",
        categories=["privacy"],
        intensity=0.8,
        quote="q",
    )
    write_prediction(run_dir / "predictions" / "evt_001__A.json", event_id="evt_001", arm="A")

    page = build_dashboard(repo=sandbox, run_id="run_001")
    assert "privacy_hawk" in page
    assert "not yet run" in page.lower()  # evt_002 still has no data


def test_main_writes_dashboard_html_to_default_path(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    exit_code = main(["run_001", "--repo", str(sandbox)])
    assert exit_code == 0
    out_path = run_dir / "dashboard.html"
    assert out_path.exists()
    assert "evt_001" in out_path.read_text(encoding="utf-8")


def test_main_returns_nonzero_for_missing_run(sandbox, capsys):
    exit_code = main(["does_not_exist", "--repo", str(sandbox)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_dashboard.py -k "build_dashboard or main" -v`
Expected: FAIL with `ImportError: cannot import name 'build_dashboard'`.

- [ ] **Step 3: Implement `build_dashboard` and `main`**

Append to `src/dashboard.py` (add `import argparse`, `import sys`, and `from src.events import Event, load_events` — merge with the `Event` import already added in Task 3 — and `from src.personas import Persona, load_personas` — merge with the existing `Persona` import from Task 1):

```python
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
    except DashboardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = args.out or (args.repo / "runs" / args.run_id / "dashboard.html")
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: all tests PASS (24 total).

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests PASS (224 existing + 24 new = 248).

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim"
git add src/dashboard.py tests/test_dashboard.py
git commit -m "Add dashboard build_dashboard orchestration and CLI entry point"
```

- [ ] **Step 7: Manually verify against the real smoke-test run**

Run against the actual run_id produced during the earlier live smoke test, to confirm the real data (not fixtures) renders correctly:

```bash
uv run python -m src.dashboard 20260729T165142.726Z_2ec30ae6
open runs/20260729T165142.726Z_2ec30ae6/dashboard.html
```

Confirm in the browser: both events show, evt_001's B30 prediction reads `privacy, overclaim, financial` (or similar), expanding evt_001 shows ~19 reacted personas sorted by intensity with `privacy_hawk` criticizing, and the ignored-personas list is collapsed and shows ~11 names for evt_001 / ~28 for evt_002.

---

## Self-Review Notes

- **Spec coverage:** CLI (`<run_id> [--repo] [--out]`) → Task 4. Data sources table → Tasks 1 & 4. Page structure (collapsed summary, expanded reactions/predictions, reaction-mix counts, reacted-sorted-by-intensity, collapsed ignored list) → Tasks 2-3. Error handling (missing manifest, missing event data, missing persona file) → Tasks 1 & 4. Testing approach (pure-function fixtures, HTML smoke tests, no browser automation) → all four tasks.
- **Placeholders:** none — every step has runnable code.
- **Type consistency:** `ReactionRow` (Task 1) fields are used identically in Tasks 2-4; `ARM_ORDER` (Task 1) is consumed only inside `load_event_predictions` itself; `build_dashboard`'s signature (`repo`, `run_id` keyword-only) matches every test call site across Task 4.
