# Client-Facing Demo Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, client-facing demo web app: pick a past event from a dropdown, pick a "kind of simulation" to view (naive AI baseline vs. our 30-persona ensemble vs. a side-by-side comparison), and instantly see business-friendly results, replaying already-computed historical data with zero live API cost or wait.

**Architecture:** New top-level `web/` directory, sibling to `src/`. `web/backend/` is a FastAPI app in four small modules (`runs.py` → `transform.py` → `service.py` → `schemas.py`/`main.py`, each layer built on the one before). `web/frontend/` is plain HTML/CSS/JS (ES modules, no build step) served as static files by the same FastAPI app.

**Tech Stack:** Python 3.12, existing `pydantic`, plus new `fastapi` + `uvicorn[standard]` (new `web` extra) and `httpx` (added to the existing `dev` extra, required by `fastapi.testclient.TestClient`).

## Global Constraints

- Zero live LLM/API calls anywhere in this feature, including tests — everything reads already-computed files under `runs/<run_id>/` and `inputs/events.txt`. This is a read-only reporting layer over existing artifacts, matching `src/dashboard.py`'s and `src/score.py`'s own "read-only, no API calls" convention.
- Reuse `src/dashboard.py`'s loaders (`load_event_reactions`, `load_event_predictions`, `reaction_mix_counts`, `split_reacted_and_ignored`) rather than re-parsing `runs/<id>/raw/**/*.json` or `runs/<id>/predictions/*.json` directly. Reuse `src/events.load_events`, `src/personas.load_personas`, `src/taxonomy.load_taxonomy`.
- Every new Python module follows this codebase's established shape: a domain `<Name>Error(RuntimeError)` class per module family (`WebDataError` in `web/backend/service.py` and `web/backend/runs.py`), pure computation (`transform.py`) separated from I/O (`service.py`), frozen dataclasses for internal value objects (see `src/dashboard.py`'s `ReactionRow`, `src/score.py`'s `EventScore`).
- `Event.expected_null` (the study's internal a-priori prior) must never appear in any client-facing API response — exclude it explicitly from every schema/transform that touches `Event`.
- No feature here depends on or reads `ground_truth/` (currently empty) — do not build any "predicted vs. actually happened" comparison.
- `Reaction` (from `src/models.py`) is `Literal["ignore", "mild_concern", "criticize", "outrage"]` — there is no `"praise"` reaction. Any plain-language summary must handle exactly these four values.
- `ArmPrediction.scores` (`src/models.py`) is `dict[str, float]`, always empty (`{}`) for arm `A` (naive baseline has no per-category confidence) and populated for arm `B30`. Every function that reads `scores` must treat a missing key as `None`, not `0.0` — `0.0` would misleadingly imply "the ensemble actively scored this near zero" when the real meaning is "this arm doesn't produce a score at all" (for arm A) or "this category wasn't in the top-3" (for arm B30 categories outside its own ranked list — not applicable here since we only ever look up scores for a prediction's own `ranked_categories`).
- FastAPI route declaration order matters: `/api/*` routes must be declared on the `FastAPI()` app **before** `app.mount("/", StaticFiles(...))` — Starlette matches mounts/routes in declaration order, and a catch-all mount declared first would swallow every `/api/*` request.
- The one real run directory is `runs/20260730T002314.423Z_2ec30ae6` (23 events, all with arms A/B3/B8/B15/B30/C already scored). This is the only run with `manifest.json` status `"complete"` today.

---

### Task 1: `web/backend/runs.py` + `web/backend/transform.py`

**Files:**
- Create: `web/__init__.py` (empty), `web/backend/__init__.py` (empty)
- Create: `web/backend/runs.py`
- Create: `web/backend/transform.py`
- Test: `tests/test_web_transform.py`

**Interfaces:**
- Produces (`runs.py`):
  - `REPO = Path(__file__).resolve().parents[2]` (from `web/backend/runs.py` up to repo root)
  - `class WebDataError(RuntimeError)`
  - `get_run_dir(repo: Path = REPO) -> Path` — scans `repo/runs/*/manifest.json`, returns the directory of the one whose `manifest.json` has `"status": "complete"`. Raise `WebDataError` if zero such runs exist, or if more than one does (today there is exactly one; ambiguity is a real error, not a "pick the first" default — this is the seam a future "which run to serve" feature would extend, so it must fail loudly rather than silently guess).
- Produces (`transform.py`, pure — no filesystem access, no imports from `runs.py` or `service.py`):
  - `ARCHETYPE_LABELS: dict[str, str] = {"critic": "Critic", "neutral": "Neutral observer", "sympathetic": "Sympathetic voice", "insider": "Industry insider"}`
  - `@dataclass(frozen=True) class CategoryScore: id: str; label: str; confidence: float | None`
  - `category_label(taxonomy: Taxonomy, category_id: str) -> str` — looks up the label from `taxonomy.entries` (raise `KeyError` or a clear `ValueError` if the id isn't found; this should never happen in practice since categories come from validated `Category` literals, but don't silently return the id itself as a fallback — that would hide a real taxonomy/category drift bug).
  - `top_categories(prediction: ArmPrediction, taxonomy: Taxonomy) -> list[CategoryScore]` — one `CategoryScore` per entry in `prediction.ranked_categories`, in the same order, with `confidence = prediction.scores.get(category_id)` (so arm A's `scores={}` gives `confidence=None` for every entry).
  - `reaction_mix_summary(counts: dict[str, int]) -> str` — a single plain-language sentence built from a `reaction_counts` dict keyed by the four `Reaction` values (as produced by `src.dashboard.reaction_mix_counts`). Order mentioned: outrage, then criticize, then mild_concern, then ignore. Omit any reaction with count 0. Use correct singular/plural ("1 persona" vs "N personas"). Group `outrage` and `criticize` together as "objected" per the example below; `mild_concern` is its own clause; `ignore` is "stayed silent". Exact behavior, given as test cases below.
  - `@dataclass(frozen=True) class QuoteItem: archetype_label: str; platform: str; reaction: str; intensity: float; quote: str; categories: tuple[str, ...]`
  - `select_curated_quotes(reacted: list[ReactionRow], personas_by_id: dict[str, Persona], limit: int = 8) -> list[QuoteItem]` — `reacted` is already non-ignoring and sorted by intensity descending (the shape `src.dashboard.split_reacted_and_ignored(...)[0]` returns). Algorithm: (1) walk `reacted` in order, taking the first row seen for each distinct `archetype` not yet taken, until either every archetype present has one row or `limit` is reached; (2) if slots remain under `limit`, fill them with the next highest-intensity rows not already taken, regardless of archetype; (3) the final list is re-sorted by intensity descending. Look up each row's `platform` via `personas_by_id[row.persona_id].platform` (a `ReactionRow` itself has no `platform` field). Never include `persona_id` or `persona_name` in the `QuoteItem` — only `archetype_label` (via `ARCHETYPE_LABELS[row.archetype]`, defaulting to the raw archetype string if somehow not in the dict) identifies who's speaking.
  - `@dataclass(frozen=True) class CategoryDiff: agreed: list[CategoryScore]; ensemble_only: list[CategoryScore]; naive_only: list[CategoryScore]; backlash_agreement: bool`
  - `compare_predictions(naive: ArmPrediction, ensemble: ArmPrediction, taxonomy: Taxonomy) -> CategoryDiff` — `agreed` = categories in both `ranked_categories` lists (order: ensemble's order), each built via `top_categories`-style lookup against the **ensemble** prediction's scores (so agreed categories show real confidence where available). `ensemble_only` = in ensemble's list but not naive's (built from the ensemble prediction, real scores). `naive_only` = in naive's list but not ensemble's (built from the naive prediction, `confidence=None` always). `backlash_agreement = (naive.backlash_predicted == ensemble.backlash_predicted)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_transform.py`:

```python
from __future__ import annotations

import pytest

from src.dashboard import ReactionRow
from src.models import ArmPrediction
from src.personas import Persona
from src.taxonomy import Taxonomy, TaxonomyEntry
from web.backend.transform import (
    CategoryDiff,
    CategoryScore,
    QuoteItem,
    category_label,
    compare_predictions,
    reaction_mix_summary,
    select_curated_quotes,
    top_categories,
)


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy(
        entries=(
            TaxonomyEntry(id="privacy", label="Privacy & data", description="d"),
            TaxonomyEntry(id="overclaim", label="Technical overclaim", description="d"),
            TaxonomyEntry(id="security", label="Security", description="d"),
            TaxonomyEntry(id="hypocrisy", label="Hypocrisy & receipts", description="d"),
            TaxonomyEntry(id="none", label="No meaningful backlash", description="d"),
        )
    )


def make_persona(persona_id: str, archetype: str, platform: str = "x") -> Persona:
    return Persona(
        id=persona_id, name=f"persona_{persona_id}", axis="privacy",
        archetype=archetype, baseline_skepticism=0.5, platform=platform,
        body="body", source_path=None,  # adjust if Persona's real fields differ; verify against src/personas.py
    )


# --- category_label / top_categories ---


def test_category_label_known_id(taxonomy):
    assert category_label(taxonomy, "privacy") == "Privacy & data"


def test_top_categories_zips_scores(taxonomy):
    pred = ArmPrediction(
        arm="B30", event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "security"],
        scores={"privacy": 0.395, "overclaim": 0.24, "security": 0.023},
        backlash_predicted=True,
    )
    result = top_categories(pred, taxonomy)
    assert [c.id for c in result] == ["privacy", "overclaim", "security"]
    assert result[0].label == "Privacy & data"
    assert result[0].confidence == pytest.approx(0.395)


def test_top_categories_naive_arm_has_none_confidence(taxonomy):
    pred = ArmPrediction(
        arm="A", event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "hypocrisy"],
        scores={}, backlash_predicted=True,
    )
    result = top_categories(pred, taxonomy)
    assert all(c.confidence is None for c in result)


# --- reaction_mix_summary ---


def test_reaction_mix_summary_evt_001_real_counts():
    # Real counts from runs/20260730T002314.423Z_2ec30ae6/raw/B30/evt_001
    counts = {"ignore": 11, "mild_concern": 11, "criticize": 8, "outrage": 0}
    summary = reaction_mix_summary(counts)
    assert "8 personas objected" in summary or "8 personas criticized" in summary
    assert "11" in summary  # appears at least for mild_concern and/or ignore
    assert "outrage" not in summary.lower() or "0" not in summary  # zero-count reaction omitted


def test_reaction_mix_summary_singular_counts():
    counts = {"ignore": 1, "mild_concern": 0, "criticize": 1, "outrage": 1}
    summary = reaction_mix_summary(counts)
    assert "1 persona" in summary  # singular, not "1 personas"
    assert "personas" not in summary.split("1 persona")[1][:2]  # crude singular check; adjust as needed


def test_reaction_mix_summary_all_zero_except_ignore():
    counts = {"ignore": 30, "mild_concern": 0, "criticize": 0, "outrage": 0}
    summary = reaction_mix_summary(counts)
    assert "30" in summary
    assert "objected" not in summary
    assert "mild concern" not in summary.lower()


# --- select_curated_quotes ---


def test_select_curated_quotes_diversifies_by_archetype():
    personas_by_id = {
        "001": make_persona("001", "critic"),
        "002": make_persona("002", "sympathetic"),
        "003": make_persona("003", "critic"),
    }
    reacted = [
        ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.9, "q1"),
        ReactionRow("003", "p3", "critic", "criticize", ("privacy",), 0.8, "q3"),
        ReactionRow("002", "p2", "sympathetic", "mild_concern", ("privacy",), 0.5, "q2"),
    ]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    # all 3 fit under the limit; archetype-first pass takes 001 (critic) and 002
    # (sympathetic) before backfilling with 003 (second critic) -- final order
    # re-sorted by intensity descending regardless of pass order.
    assert [q.quote for q in result] == ["q1", "q3", "q2"]
    assert result[0].archetype_label == "Critic"


def test_select_curated_quotes_respects_limit():
    personas_by_id = {str(i).zfill(3): make_persona(str(i).zfill(3), "critic") for i in range(1, 11)}
    reacted = [
        ReactionRow(str(i).zfill(3), f"p{i}", "critic", "criticize", ("privacy",), 1.0 - i * 0.01, f"q{i}")
        for i in range(1, 11)
    ]
    result = select_curated_quotes(reacted, personas_by_id, limit=5)
    assert len(result) == 5
    assert result[0].quote == "q1"  # highest intensity


def test_select_curated_quotes_fewer_than_limit():
    personas_by_id = {"001": make_persona("001", "critic")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert len(result) == 1


def test_select_curated_quotes_looks_up_platform():
    personas_by_id = {"001": make_persona("001", "critic", platform="reddit")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert result[0].platform == "reddit"


def test_select_curated_quotes_never_leaks_persona_id():
    personas_by_id = {"001": make_persona("001", "critic")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert not hasattr(result[0], "persona_id")
    assert not hasattr(result[0], "persona_name")


# --- compare_predictions ---


def test_compare_predictions_evt_001_real_diff(taxonomy):
    # Real evt_001 predictions from runs/20260730T002314.423Z_2ec30ae6
    naive = ArmPrediction(
        arm="A", event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "hypocrisy"],
        scores={}, backlash_predicted=True,
    )
    ensemble = ArmPrediction(
        arm="B30", event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "security"],
        scores={"privacy": 0.395, "overclaim": 0.24, "security": 0.023333},
        backlash_predicted=True,
    )
    diff = compare_predictions(naive, ensemble, taxonomy)
    assert {c.id for c in diff.agreed} == {"privacy", "overclaim"}
    assert [c.id for c in diff.ensemble_only] == ["security"]
    assert [c.id for c in diff.naive_only] == ["hypocrisy"]
    assert diff.backlash_agreement is True
    # agreed categories carry the ensemble's real confidence, not None
    agreed_privacy = next(c for c in diff.agreed if c.id == "privacy")
    assert agreed_privacy.confidence == pytest.approx(0.395)


def test_compare_predictions_backlash_disagreement(taxonomy):
    naive = ArmPrediction(
        arm="A", event_id="evt_002", ranked_categories=["none", "labor", "environment"],
        scores={}, backlash_predicted=False,
    )
    ensemble = ArmPrediction(
        arm="B30", event_id="evt_002", ranked_categories=["labor", "none", "environment"],
        scores={"labor": 0.3}, backlash_predicted=True,
    )
    diff = compare_predictions(naive, ensemble, taxonomy)
    assert diff.backlash_agreement is False
```

**Note to implementer:** verify `Persona`'s exact constructor fields against `src/personas.py` before writing `make_persona` — the sketch above may not match the real dataclass field names/order exactly (e.g. `source_path` type). Read the real file first and adjust the fixture helper accordingly; do not guess. Similarly, verify `ReactionRow`'s exact field order against `src/dashboard.py` (`persona_id, persona_name, archetype, reaction, categories, intensity, quote`).

The two "crude" assertions in `test_reaction_mix_summary_singular_counts` and the first assertion in `test_reaction_mix_summary_evt_001_real_counts` are deliberately loose (an "or") because the exact sentence wording is the implementer's craft to nail down — tighten these assertions once you've written the real sentence, replacing the loose checks with exact string equality against your actual output. Do not leave loose/"or" assertions in the final committed test file — this is a placeholder for you to firm up during TDD, not an acceptable final state.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_transform.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web'`.

- [ ] **Step 3: Implement `web/backend/runs.py` and `web/backend/transform.py`**

Write the implementation to match every interface and test above. Read `src/dashboard.py`, `src/models.py`, `src/personas.py`, and `src/taxonomy.py` first to confirm exact field names before writing any code that touches them.

- [ ] **Step 4: Run tests to verify they pass, tightening any loose assertions**

Run: `uv run pytest tests/test_web_transform.py -v`
Expected: all tests PASS, with the loose "or" assertions replaced by exact checks against your real sentence-building logic.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all previously-passing tests still pass, plus the new ones.

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
git add web/__init__.py web/backend/__init__.py web/backend/runs.py web/backend/transform.py tests/test_web_transform.py
git commit -m "Add web/backend runs.py + transform.py: pure display transforms"
```

---

### Task 2: `web/backend/service.py`

**Files:**
- Create: `web/backend/service.py`
- Test: `tests/test_web_service.py`

**Interfaces:**
- Consumes: Task 1's `runs.get_run_dir`, `transform.*`; `src.events.load_events`, `src.personas.load_personas`, `src.taxonomy.load_taxonomy`, `src.dashboard.load_event_reactions`, `src.dashboard.load_event_predictions`, `src.dashboard.reaction_mix_counts`, `src.dashboard.split_reacted_and_ignored`.
- Produces:
  - `REPO` (reuse `web.backend.runs.REPO`)
  - `class WebDataError(RuntimeError)` (reuse `web.backend.runs.WebDataError`, or re-export it — do not define a second, different exception class for the same concept)
  - `@dataclass(frozen=True) class EventSummary: id: str; company: str; headline: str; date: str; sector: str`
  - `@dataclass(frozen=True) class EventContext: id: str; company: str; headline: str; date: str; sector: str; source_url: str | None; announcement: str` (note: no `expected_null` field — Global Constraint)
  - `@dataclass(frozen=True) class NaiveResult: backlash_predicted: bool; top_categories: list[CategoryScore]`
  - `@dataclass(frozen=True) class EnsembleResult: backlash_predicted: bool; top_categories: list[CategoryScore]; reaction_mix_summary: str; reaction_counts: dict[str, int]; sample_quotes: list[QuoteItem]`
  - `@dataclass(frozen=True) class EventResult: event: EventContext; naive: NaiveResult; ensemble: EnsembleResult; comparison: CategoryDiff`
  - `list_event_summaries(repo: Path = REPO) -> list[EventSummary]` — every event from `inputs/events.txt`, in file order (which is already `evt_001..evt_023` order).
  - `build_event_result(event_id: str, repo: Path = REPO) -> EventResult` — raises `WebDataError` if `event_id` doesn't exist in `inputs/events.txt`, or if either the `A` or `B30` prediction for it is missing from the run's `predictions/` directory (both should always be present for the one real run, but this must fail loudly rather than `None`-out a whole panel — matches this codebase's "malformed/missing data raises" convention, see `src/diagnostics.py`'s `load_v1_rows` docstring for the same rationale applied elsewhere).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_service.py`:

```python
from __future__ import annotations

import pytest

from web.backend.service import WebDataError, build_event_result, list_event_summaries


def test_list_event_summaries_returns_all_23(repo):
    summaries = list_event_summaries(repo=repo)
    assert len(summaries) == 23
    assert summaries[0].id == "evt_001"
    assert summaries[0].company  # non-empty
    ids = [s.id for s in summaries]
    assert ids == sorted(ids)  # evt_001..evt_023 file order


def test_build_event_result_evt_001_matches_real_data(repo):
    result = build_event_result("evt_001", repo=repo)

    assert result.event.id == "evt_001"
    assert result.event.announcement  # non-empty
    # expected_null must never leak into the client-facing dataclass
    assert not hasattr(result.event, "expected_null")

    assert result.naive.backlash_predicted is True
    assert [c.id for c in result.naive.top_categories] == ["privacy", "overclaim", "hypocrisy"]
    assert all(c.confidence is None for c in result.naive.top_categories)

    assert result.ensemble.backlash_predicted is True
    assert [c.id for c in result.ensemble.top_categories] == ["privacy", "overclaim", "security"]
    assert result.ensemble.reaction_counts == {"ignore": 11, "mild_concern": 11, "criticize": 8, "outrage": 0}
    assert len(result.ensemble.sample_quotes) <= 8
    assert result.ensemble.sample_quotes  # non-empty for an event with real reactions

    assert {c.id for c in result.comparison.agreed} == {"privacy", "overclaim"}
    assert [c.id for c in result.comparison.ensemble_only] == ["security"]
    assert [c.id for c in result.comparison.naive_only] == ["hypocrisy"]
    assert result.comparison.backlash_agreement is True


def test_build_event_result_unknown_event_raises(repo):
    with pytest.raises(WebDataError, match="evt_999"):
        build_event_result("evt_999", repo=repo)
```

Note: `repo` here is the session-scoped `tests/conftest.py` fixture, pointed at this worktree's real repo root — and this worktree's `runs/20260730T002314.423Z_2ec30ae6` directory must exist (it does; it was copied in at worktree setup) for `get_run_dir` to find a complete run. If a test run ever reports "no complete run found," that's an environment setup issue to flag, not a code bug to work around.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.backend.service'`.

- [ ] **Step 3: Implement `web/backend/service.py`**

Wire everything together per the interfaces above. `build_event_result` should: load events once (or use a module-level `functools.lru_cache` on the loader calls, since `inputs/events.txt`/`personas/`/`taxonomy.txt` are static per-process — verify this doesn't fight with `test_build_event_result_unknown_event_raises` needing a fresh lookup per test; `lru_cache` on a function keyed by `repo: Path` is fine since tests always pass the same `repo` fixture value), find the target event, load its `A` and `B30` predictions via `load_event_predictions` (filter the returned list by `.arm`), load its reactions via `load_event_reactions`, split via `split_reacted_and_ignored`, and assemble the dataclasses using Task 1's `transform.*` functions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_service.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
git add web/backend/service.py tests/test_web_service.py
git commit -m "Add web/backend/service.py: event listing and result assembly"
```

---

### Task 3: `web/backend/schemas.py` + `web/backend/main.py`

**Files:**
- Create: `web/backend/schemas.py`
- Create: `web/backend/main.py`
- Test: `tests/test_web_api.py`

**Interfaces:**
- Consumes: Task 2's `service.*`.
- Produces (`schemas.py`, pydantic `BaseModel`s mirroring every `service.py`/`transform.py` dataclass field-for-field — `EventSummaryOut`, `EventContextOut`, `CategoryScoreOut`, `NaiveResultOut`, `QuoteItemOut`, `EnsembleResultOut`, `ComparisonOut`, `EventResultResponse`, `EventsListResponse`) plus adapter functions (`to_event_summary_out`, `to_event_result_response`, etc. — name them however's cleanest, but keep the adapter logic here, not inline in `main.py`, so route handlers stay thin).
- Produces (`main.py`):
  - `app = FastAPI(title="crisis-sim demo")`
  - `GET /api/events` → `EventsListResponse`
  - `GET /api/events/{event_id}/result` → `EventResultResponse`, raising `HTTPException(status_code=404, detail=...)` when `service.build_event_result` raises `WebDataError`
  - `FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"` and `app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")` — **declared after both `/api/*` routes above it in the file** (Global Constraint on route order). `web/frontend/` doesn't need to exist yet with real content for this task's tests to pass (Task 4 populates it) — but the directory itself must exist by the end of this task (create it with a placeholder `index.html` containing just `<!doctype html><title>crisis-sim demo</title>` if Task 4 hasn't run yet, so `StaticFiles` doesn't raise on startup for a missing directory).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_api.py`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from web.backend.main import app

client = TestClient(app)


def test_get_events_returns_23_sorted():
    response = client.get("/api/events")
    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 23
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)
    assert set(events[0].keys()) == {"id", "company", "headline", "date", "sector"}


def test_get_event_result_evt_001():
    response = client.get("/api/events/evt_001/result")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"event", "naive", "ensemble", "comparison"}
    assert body["event"]["id"] == "evt_001"
    assert "expected_null" not in body["event"]
    assert [c["id"] for c in body["naive"]["top_categories"]] == ["privacy", "overclaim", "hypocrisy"]
    assert [c["id"] for c in body["ensemble"]["top_categories"]] == ["privacy", "overclaim", "security"]
    assert body["comparison"]["backlash_agreement"] is True
    assert [c["id"] for c in body["comparison"]["ensemble_only"]] == ["security"]


def test_get_event_result_unknown_event_404():
    response = client.get("/api/events/evt_999/result")
    assert response.status_code == 404


def test_root_serves_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


def test_api_routes_not_shadowed_by_static_mount():
    # Regression test for route/mount declaration order: /api/events must
    # resolve to the API handler, never fall through to the static mount.
    response = client.get("/api/events")
    assert response.headers["content-type"].startswith("application/json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.backend.main'`.

- [ ] **Step 3: Implement `web/backend/schemas.py` and `web/backend/main.py`**

Ensure `web/frontend/` exists (create it with a minimal placeholder `index.html` if Task 4 hasn't landed yet) so `StaticFiles` doesn't fail at import time.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_api.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Manual smoke check**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
uv run uvicorn web.backend.main:app --port 8000 &
sleep 1
curl -s localhost:8000/api/events | head -c 300
curl -s localhost:8000/api/events/evt_001/result | head -c 500
kill %1
```
Expected: both return real JSON (not connection errors, not 500s). This requires `fastapi`/`uvicorn` to already be installed — if `uv run uvicorn` fails with a missing-module error, that means Task 5's `pyproject.toml` dependency wiring needs to happen first for this manual check specifically; note it and proceed, `pytest`'s `TestClient` based tests above don't need the `web` extra installed if `fastapi`/`httpx` are already resolvable in the dev environment (check `uv sync --extra dev` first; if FastAPI import fails even for the pytest step, `web` extra must be added now rather than deferred to Task 5 — use your judgment and prefer adding the dependency now over blocking Task 3's own tests).

- [ ] **Step 7: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
git add web/backend/schemas.py web/backend/main.py web/frontend/index.html tests/test_web_api.py pyproject.toml uv.lock
git commit -m "Add web/backend/schemas.py + main.py: FastAPI app and routes"
```

---

### Task 4: Static frontend

**Files:**
- Create: `web/frontend/index.html` (replace Task 3's placeholder)
- Create: `web/frontend/styles.css`
- Create: `web/frontend/js/api.js`
- Create: `web/frontend/js/render.js`
- Create: `web/frontend/js/main.js`

**No automated tests for this task** — verified manually in-browser (see Step 3).

**Design brief:**
- **Layout:** a header/title ("crisis-sim — see it in action" or similar, your call), a control bar with the event `<select>` (options `"{company} — {headline}"`, populated from `GET /api/events` on page load) and a 3-way mode control (radio group or segmented buttons: "Naive AI", "Persona Ensemble", "Side-by-side") plus a "Run Simulation" button, then a results area below.
- **Results area, by mode:**
  - *Naive AI*: event context card (company/headline/date/sector/announcement), a prominent color-coded "Backlash Predicted: Yes/No" badge, the naive top-3 categories as a simple ranked list (no confidence bars since arm A has none).
  - *Persona Ensemble*: same event context + badge, ensemble top-3 categories **with** confidence shown (e.g. a small bar or percentage next to each), the `reaction_mix_summary` sentence prominently, and a "What our personas said" section listing `sample_quotes` as cards — each showing `archetype_label`, a small platform icon/tag, the quote text, and a subtle intensity indicator. No raw persona ids anywhere.
  - *Side-by-side*: naive and ensemble panels shown next to each other (stack vertically on narrow viewports), plus a highlighted callout box below both: "What a single AI call would have missed" listing `comparison.ensemble_only` categories (this is the money shot — make it visually distinct, e.g. a colored border/background), and a smaller note for `comparison.naive_only` if non-empty.
- **Visual tone:** business/sales-friendly, not developer-tooling. System font stack (`system-ui, -apple-system, sans-serif`). Card-based sections with subtle shadows/rounded corners. A clear, confident color scheme — e.g. warm red/orange for "Backlash: Yes", calm green for "Backlash: No". No raw JSON dumps, no monospace tables, no technical jargon in labels (say "Confidence" not "score", "Concern area" not "category" if that reads better — your judgment).
- **Interaction:** on page load, `js/main.js` calls `api.fetchEvents()` and populates the `<select>`. On "Run Simulation" click: check an in-memory `Map` cache keyed by event id; if present, render immediately from cache; if not, call `api.fetchEventResult(eventId)`, store in the cache, then render. Changing the mode selector re-renders from the currently-cached payload for the currently-selected event without any network call (if nothing has been run yet for that event, do nothing or show a subtle "click Run Simulation" prompt — your judgment on the cleanest UX). Optionally add a short `await new Promise(r => setTimeout(r, 400))` before rendering on a fresh fetch, purely for a moment of anticipation — never delay a cache-hit re-render.
- **Module structure:** `js/api.js` exports `fetchEvents()` and `fetchEventResult(eventId)` — the only file that references `/api/...` paths. `js/render.js` exports pure rendering functions (`renderEventContext(payload)`, `renderNaivePanel(payload)`, `renderEnsemblePanel(payload)`, `renderComparisonPanel(payload)`, `setVisibleMode(mode)`) that take data and mutate/return DOM — no `fetch` calls in this file. `js/main.js` is the only file wiring DOM events (`change`, `click`) to the other two modules. Use `<script type="module" src="js/main.js"></script>` in `index.html`; `main.js` does `import { fetchEvents, fetchEventResult } from './api.js'` and `import { ... } from './render.js'`.

- [ ] **Step 1: Implement the four frontend files** per the design brief above.

- [ ] **Step 2: Start the server**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
uv run uvicorn web.backend.main:app --port 8000 --reload
```

- [ ] **Step 3: Manual verification with the Claude Browser tool**

Open `http://127.0.0.1:8000/`. Confirm: the dropdown populates with 23 events; selecting an event and clicking Run shows results within a moment; all three modes render sensibly for at least 3 different events (pick one with a large `missed_by_naive` set like evt_001, one where `expected_null` is true for the underlying event if easily identifiable, and one more at random); switching modes after a run requires no visible reload/flicker of a network fetch; no errors in the browser console (`read_console_messages`); layout looks reasonable at a normal laptop width (no overflow, no unstyled raw text).

- [ ] **Step 4: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
git add web/frontend/
git commit -m "Add web/frontend: dropdown, mode selector, and results panels"
```

---

### Task 5: Wiring, polish, and final pass

**Files:**
- Modify: `pyproject.toml`
- Modify: `web/frontend/js/main.js` (only if the Step 6 delay wasn't already added in Task 4)
- Create: `docs/superpowers/specs/2026-08-01-client-demo-web-app.md` (short spec doc, following the shape of `docs/superpowers/specs/2026-07-29-dashboard-design.md`: Context/Goals/Non-goals/Data sources/API contract/Page structure/Testing/Files)

- [ ] **Step 1: Finalize `pyproject.toml`**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]
web = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
]

[tool.setuptools]
packages = ["src", "web", "web.backend"]
```
(If Task 3 already added some of this to get its own tests running, just confirm it's complete and correct here — don't duplicate entries.) Run `uv sync --extra web --extra dev` and `uv lock` if needed, then `uv run pytest -q` to confirm the full suite (all prior tasks' tests + these) is green.

- [ ] **Step 2: Write the spec doc** at `docs/superpowers/specs/2026-08-01-client-demo-web-app.md`, summarizing what was built, the API contract, and how to run it locally — for a future engineer picking this up, not a sales audience.

- [ ] **Step 3: Final manual pass**

Using the Claude Browser tool against `uv run uvicorn web.backend.main:app --port 8000`: click through at least 5 different events across all 3 modes each. Confirm no console errors, no visual glitches, and that the "Run Simulation" → mode-switch → different-event flow all behaves as designed.

- [ ] **Step 4: Commit**

```bash
cd "/Users/pranayagarwal/Documents/Claude Code/WePulse/crisis-sim/.claude/worktrees/crisis-sim-demo-web"
git add pyproject.toml uv.lock docs/superpowers/specs/2026-08-01-client-demo-web-app.md
git commit -m "Wire up web extras, add demo web app spec doc"
```

---

## Self-Review Notes

- **Spec coverage:** event dropdown (Task 4), simulation-kind selector across naive/ensemble/comparison (Tasks 2-4), instant replay of precomputed data with zero live calls (Global Constraints + Task 2), business-friendly key metrics (backlash badge, categories with confidence, plain-language reaction mix, curated quotes, missed-by-naive callout — Tasks 1-4), UI/backend separation via a documented JSON API (Tasks 2-3 vs. Task 4), modularity for future features (four single-responsibility backend modules, isolated frontend JS modules).
- **Placeholders:** none load-bearing — Task 3's frontend-directory placeholder is explicitly superseded by Task 4 before the plan is done.
- **Known open craft decisions left to the implementer:** exact wording of `reaction_mix_summary`'s sentence, exact visual design choices in Task 4 (colors, iconography, copy) — the plan specifies behavior/requirements, not literal pixel-perfect output, since this is a design-forward feature rather than a pure algorithm port.
