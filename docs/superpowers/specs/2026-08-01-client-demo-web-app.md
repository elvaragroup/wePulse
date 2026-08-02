# Client-facing demo web app — implementation spec

## Context

The crisis-sim validation harness compares a persona-ensemble approach against a naive
single-LLM baseline for predicting public backlash in response to real-world announcements.
The study validates this works better — measured via AUC, recall, and other metrics across
23 real events. However, communicating *why* the ensemble is better (what individual
personas say, how their reactions aggregate, where naive and ensemble disagree on
categories) requires browsing JSON files and raw predictions — tedious for a stakeholder
review or client demo.

This spec adds a read-only, interactive web app that replays precomputed simulation runs
for 23 real events, showing a client audience: which personas say what, where the ensemble
and naive predictions diverge, a quantified backlash risk, reaction sentiment mix, and
curated quotes. All data is static (precomputed during `run_sim.py` + `score.py`); the app
is a pure display layer with no live LLM calls, no database, and no auth.

## Goals

- Let a client see simulation results for 23 real announcements interactively: pick an
  announcement, run the simulation (instant replay of precomputed data), view predictions
  side-by-side, and understand the reaction breakdown (persona sample quotes, sentiment
  mix, backlash agreement/disagreement between ensemble and naive).
- Minimal dependencies, minimal backend (one FastAPI instance serving precomputed JSON
  plus static assets).
- Mobile-responsive, visually polished single-page app (no page reloads within the app; smooth
  mode switching and event selection).
- Zero live inference — all data is 100% precomputed; "run simulation" is instant.
- Clear business communication: backlash risk (bool), reaction sentiment (counts),
  confidence-scored category predictions, direct quotes from personas.

## Non-goals

- Ground truth, precision/recall metrics, or leakage analysis — the study's validation is
  published separately. This app is a demo, not a research report.
- Comparing across multiple runs or events in a single session; editing scenarios or
  re-running with different parameters.
- User accounts, auth, or persistence of selections.
- Deployment guidance, containerization, or cloud hosting — this spec covers local
  development and manual demo scenarios only.
- Live simulation mode (LLM inference on new announcements).

## Data sources

All data is precomputed and static, sourced from a prior `run_sim.py` + `score.py` cycle:

| Source | Loader | Fields used |
|---|---|---|
| `inputs/events.txt` | `src.events.load_events` | `id`, `company`, `headline`, `date`, `sector`, `source_url`, `announcement` |
| `personas/*.md` | `src.personas.load_personas` | `id`, `name`, `archetype`, `axis` |
| `runs/*/raw/B30/<event_id>/<persona_id>.json` | `PersonaReaction.model_validate_json` | `reaction`, `categories`, `intensity`, `quote` |
| `runs/*/predictions/<event_id>__A.json` | `ArmPrediction.model_validate_json` | `backlash_predicted`, `ranked_categories`, `scores` |
| `runs/*/predictions/<event_id>__B30.json` | `ArmPrediction.model_validate_json` | `backlash_predicted`, `ranked_categories`, `scores` |
| `taxonomy.txt` | `src.taxonomy.load_taxonomy` | category metadata (labels, axes) |

The app requires exactly one `run_*/` directory with `"status": "complete"` in its `manifest.json`
(via `web.backend.runs.get_run_dir`), reads predictions and reactions from it, and merges them with
static event/persona definitions. If zero or multiple complete runs exist, the app errors loudly.

## API contract

### `GET /api/events`

**Response:**
```json
{
  "events": [
    {
      "id": "evt_001",
      "company": "Tesla",
      "headline": "Price increase announcement",
      "date": "2021-06-15",
      "sector": "automotive"
    },
    ...
  ]
}
```

All 23 events, sorted by `id` (evt_001 through evt_023), ready to populate the event-select
dropdown. Fields are narrative summaries, not keys for anything (display only).

### `GET /api/events/{event_id}/result`

**Response for `event_id=evt_001`:**
```json
{
  "event": {
    "id": "evt_001",
    "company": "Tesla",
    "headline": "Price increase announcement",
    "date": "2021-06-15",
    "sector": "automotive",
    "source_url": "https://...",
    "announcement": "Full text of the announcement..."
  },
  "naive": {
    "backlash_predicted": true,
    "top_categories": [
      {"id": "price_criticism", "label": "Pricing concerns", "confidence": 0.89},
      {"id": "trust", "label": "Trust erosion", "confidence": 0.72},
      ...
    ]
  },
  "ensemble": {
    "backlash_predicted": true,
    "top_categories": [
      {"id": "price_criticism", "label": "Pricing concerns", "confidence": 0.91},
      ...
    ],
    "reaction_mix_summary": "Personas overwhelmingly criticize pricing; 23 out of 30 predicted backlash.",
    "reaction_counts": {
      "ignore": 2,
      "mild_concern": 5,
      "criticize": 18,
      "outrage": 5
    },
    "sample_quotes": [
      {
        "archetype_label": "Environmentalist",
        "platform": "Twitter",
        "reaction": "criticize",
        "intensity": 0.89,
        "quote": "Another price hike? Tesla's lost its way.",
        "categories": ["price_criticism", "distrust"]
      },
      ...
    ]
  },
  "comparison": {
    "agreed": [
      {"id": "price_criticism", "label": "Pricing concerns", "confidence": 0.9}
    ],
    "ensemble_only": [
      {"id": "environmental_impact", "label": "Environmental concern", "confidence": 0.65}
    ],
    "naive_only": [],
    "backlash_agreement": true
  }
}
```

**Error response for missing event:**
```json
{"detail": "unknown event_id 'evt_999': not found in inputs/events.txt"}
```
HTTP 404.

## Page structure

**Layout:** Single-page app, desktop and mobile responsive.

**Header:** Logo, tagline ("Know how the public will react — before you press publish.").

**Control bar:**
- Event dropdown (populated from `GET /api/events`, sorted by id).
- View mode radio buttons: "Naive AI", "Persona Ensemble", "Side-by-side".
- "Run Simulation" button (disabled until event selected; triggers fetch from
  `GET /api/events/{event_id}/result`, shows loading state, then renders result).

**Loading state:** "Convening the audience…" message with animated pulse (simulated delay).

**Results panel (hidden until simulation runs):**
- Event context card: company, headline, date, sector, announcement text (expandable or
  full-width depending on viewport).
- Mode-dependent panels (only one visible at a time, switched via radio control):
  - **Naive mode:** backlash prediction (bool badge), top 3 categories with confidence scores.
  - **Ensemble mode:** backlash prediction, top 3 categories, reaction sentiment breakdown
    (counts of ignore/mild/criticize/outrage), 3-5 curated persona quotes with archetype,
    platform, intensity, and their reaction.
  - **Comparison mode:** side-by-side cards for naive vs. ensemble; top categories for each;
    intersection (agreed) and diffs (ensemble-only, naive-only); backlash agreement badge.

**Footer:** "crisis-sim · persona-ensemble reaction modelling · results shown are from
previously recorded simulation runs" (all-caps branding, disclaimer).

## Module layout

### Backend (`web/backend/*.py`)

Separation of concerns following the existing codebase's conventions:

- **`main.py`:** FastAPI HTTP entry point. Declares two API routes (`GET /api/events`,
  `GET /api/events/{event_id}/result`), mounts the static frontend at `/`. No business logic.

- **`runs.py`:** Run-directory discovery. Exposes `get_run_dir(repo)` to find the one run directory
  with `"status": "complete"` in its `manifest.json`; raises `WebDataError` if zero or multiple
  complete runs exist, or if the manifest is missing/malformed. Reuses existing study-side `REPO` convention.

- **`service.py`:** Data orchestration. Defines dataclasses for API response shapes
  (`EventSummary`, `EventContext`, `NaiveResult`, `EnsembleResult`, `EventResult`).
  Implements `list_event_summaries()` (every event from `inputs/events.txt`, sorted by id)
  and `build_event_result(event_id)` (assemble naive-vs-ensemble comparison for one event).
  Wires pre-existing loaders (`load_events`, `load_personas`, `load_taxonomy`,
  `load_event_reactions`, `load_event_predictions`) with `transform.py` to produce display
  shapes. All errors raised, never silenced.

- **`schemas.py`:** Pydantic models mirroring `service.py` dataclasses, plus adapter
  functions for FastAPI serialization. One model per dataclass + `to_*_out()` converter.
  No business logic.

- **`transform.py`:** Pure display transforms. Computes display-friendly summaries from
  raw data: `reaction_mix_summary()` (plain-language reaction breakdown),
  `select_curated_quotes()` (pick 3-5 representative personas), `top_categories()`
  (rank categories by confidence), `compare_predictions()` (diff naive vs. ensemble
  predictions). All functions are pure and testable without LLM calls or file I/O.

### Frontend (`web/frontend/`)

Single-page app (no page reloads). Module structure:

- **`index.html`:** Semantic markup. Event dropdown, mode radios, run button, result
  containers (empty, loading, error, result states). No inline scripts; all logic in
  separate modules.

- **`styles.css`:** All page styling. Mobile-first, responsive to desktop. Animations
  for loading state (pulse), mode transitions (fade), mode panel swaps. Color-coded
  badges for backlash risk (bool), sentiment breakdown (bar chart), confidence scores.

- **`js/main.js`:** App entry point and state machine. Initializes event dropdown (fetches
  from `GET /api/events`), wires up mode radio buttons and run button, manages result
  rendering (shows/hides DOM elements, switches visible mode panel). Communicates with
  `api.js` and `render.js`.

- **`js/api.js`:** HTTP client. Two functions: `fetchEvents()` (calls `GET /api/events`,
  returns array), `fetchEventResult(eventId)` (calls `GET /api/events/{event_id}/result`,
  returns parsed JSON). Handles errors gracefully (rejects on HTTP 404, etc.).

- **`js/render.js`:** DOM rendering. Functions to render each panel type (naive, ensemble,
  comparison) from API response data, and render the event context card. Reads from
  response objects, builds HTML strings or clones templates, inserts into page. No
  business logic; purely visual.

## How to run it locally

```bash
cd /path/to/crisis-sim
uv sync --extra web --extra dev
uv run uvicorn web.backend.main:app --reload
```

Then open `http://127.0.0.1:8000/` in a browser. The app loads immediately.

To run against a specific port:
```bash
uv run uvicorn web.backend.main:app --reload --port 8001
```

To run as a production server (no auto-reload):
```bash
uv run uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

The server discovers the required run directory when processing requests to `/api/events/{event_id}/result`
(not at startup). If no runs with `status: "complete"` exist, or if multiple exist, the endpoint will
error with a clear message (by design; the study must be run first, and run to exactly one complete state).

## Testing

Test files:

- `tests/test_web_service.py`: API response shapes and data orchestration. Tests `list_event_summaries()`
  and `build_event_result()` produce correct shapes against real inputs and runs.
- `tests/test_web_transform.py`: Display transforms and run-directory discovery. Tests
  `reaction_mix_summary()`, `select_curated_quotes()`, `top_categories()`, `compare_predictions()`,
  and `get_run_dir()` (with temp `runs/` directories) to verify correct behavior when zero, one,
  or multiple complete runs exist.
- `tests/test_web_api.py`: End-to-end HTTP layer. Makes requests to `/api/events` and
  `/api/events/{event_id}/result` via `TestClient`, asserts status codes and response shapes.

Frontend testing: manual browser verification only (interactive UI, state transitions).
No JavaScript test suite (small surface area, coverage via end-to-end HTTP tests in `test_web_api.py`).

## Files

**Backend:**
- `web/backend/__init__.py` (empty module marker)
- `web/backend/main.py` (FastAPI routes)
- `web/backend/runs.py` (run-directory discovery)
- `web/backend/service.py` (data orchestration)
- `web/backend/schemas.py` (Pydantic models + adapters)
- `web/backend/transform.py` (pure display transforms)

**Frontend:**
- `web/frontend/index.html` (markup)
- `web/frontend/styles.css` (styling)
- `web/frontend/js/main.js` (state machine)
- `web/frontend/js/api.js` (HTTP client)
- `web/frontend/js/render.js` (DOM rendering)

**Tests:**
- `tests/test_web_service.py`
- `tests/test_web_transform.py`
- `tests/test_web_api.py`

**Dependencies:**
- `pyproject.toml` [project.optional-dependencies]
  - `web`: fastapi>=0.115, uvicorn[standard]>=0.32
  - `dev`: pytest>=8.3, pytest-asyncio>=0.24, httpx>=0.27

**Documentation:**
- `docs/superpowers/specs/2026-08-01-client-demo-web-app.md` (this file)
