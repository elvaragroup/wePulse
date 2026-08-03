# Status

Read this first after any session/token reset. Update it before ending a session.

**Last updated:** 2026-08-02

## Now

Repo cleanup complete: the two previously-stranded feature branches (Milestone A
diagnostics, client demo web app) are merged into `main` and pushed to `origin`. No
in-flight plan or worktree at the moment — `main` is the only checkout.

Demo page UI/UX fix pass complete (2026-08-02): ground-truth display mechanism built
end-to-end (`ground_truth: null` today, renders an honest "pending verification" state,
never a fabricated outcome), unexplained concern-bar dashes replaced with "not a top
concern this run", Persona Ensemble card reordered to lead with quotes, reaction colors
unified via shared CSS variables, panel-size label added. Event curation (item 5 of the
original fix list, including the DeepSeek/evt_013 label question) is explicitly deferred —
it depends on real ground truth existing, which it doesn't yet. Full details:
`docs/superpowers/specs/2026-08-02-persona-improvement-methodology.md` and this session's
plan. The moment `label_truth.py` produces real `ground_truth/labeled/<event_id>.json`
files, the UI starts showing resolved outcomes and correct/incorrect marks with zero
further frontend changes needed.

## Next

In priority order (see `../CRISIS_SIM_VALIDATION_SPEC.md` §9 and
`../persona-v2-implementation-brief.md` "Execution order" for full context):

1. Run the diagnostics Milestone A baseline for real (requires a human-provisioned
   `VOYAGE_API_KEY`) against the 23-event run and the 2-event smoke run, and write
   `results/diagnostics_baseline.json`.
2. Collect real ground truth, following `ground_truth/README.md`'s Batch 0 list (5 events:
   evt_002, evt_005, evt_006, evt_007, evt_023) first, then run `label_truth.py` →
   `check_judge.py` (kappa gate) → `score.py`. **This has never been done** — the spec's
   core question ("does B30 beat A?") has no answer yet. Full methodology, dev/holdout
   partition, and version-acceptance criteria:
   `docs/superpowers/specs/2026-08-02-persona-improvement-methodology.md`. Track progress
   in `../VERSIONS.md` (v1.0 entry currently `BLOCKED`).
3. Only after 1–2: resume persona-v2 Phases 1–4 (corpus ingestion → grievance clustering →
   v2 persona generation → staged pipeline), which are fully unstarted. Use
   brainstorming → writing-plans → subagent-driven-development.

## Blocked on human

```
BLOCKED ON HUMAN: add a real VOYAGE_API_KEY to crisis-sim/.env before running
src/diagnostics.py for real (currently only an empty placeholder in .env.example).
Needed for the Homogeneity/Redundancy embedding-based metrics.
```

```
BLOCKED ON HUMAN: ground_truth/raw/ is empty. All 23 events are already 80+ days old, so
this is retroactive research (finding archived reaction threads), not a 72h wait. Follow
ground_truth/README.md's Batch 0 list (5 events) first.
```

## Known, not urgent

- `main` was 10 commits ahead of `origin` and unpushed for several days before 2026-08-02 —
  now resolved, but a good reminder to push regularly (see CLAUDE.md "End-of-session
  rule").
- `get_run_dir()` still requires exactly one local run marked `complete` by design (not
  changed). The `runs/20260729T...` ambiguity is resolved locally by moving it to
  `runs/_archive/` (gitignored, so this doesn't affect other machines) — if you pull a
  fresh clone or add another real run later, you'll need to archive down to one `complete`
  run again, or someone should change `get_run_dir()` to pick the most recent.

## Verified live, 2026-08-02

Ran the merged demo end-to-end in a browser (`uv run uvicorn web.backend.main:app`) for the
first time since both branches merged. Overall UI/UX is genuinely solid — clear narrative
per event card (verdict summary → Naive AI vs Persona Ensemble panels → "what a single AI
call would have missed" callout), good typography and spacing, side-by-side/single-view
toggle all work. Found and fixed one real bug in the process: the "Read the announcement"
`<details>` panel's CSS (`web/frontend/styles.css` `.announcement__body`) applied its
340px `max-height` unconditionally instead of gating it on `.announcement[open]`, so every
card carried a permanent blank 340px gap even when collapsed. Fixed in `d1ac195` and
verified both collapsed and expanded states render correctly. This is the kind of thing
that's easy to miss without actually loading the page — worth doing a quick live check like
this after merging UI work, not just trusting the test suite.
