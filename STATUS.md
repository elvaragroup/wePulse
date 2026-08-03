# Status

Read this first after any session/token reset. Update it before ending a session.

**Last updated:** 2026-08-02

## Now

Repo cleanup complete: the two previously-stranded feature branches (Milestone A
diagnostics, client demo web app) are merged into `main` and pushed to `origin`. No
in-flight plan or worktree at the moment — `main` is the only checkout.

## Next

In priority order (see `../CRISIS_SIM_VALIDATION_SPEC.md` §9 and
`../persona-v2-implementation-brief.md` "Execution order" for full context):

1. Run the diagnostics Milestone A baseline for real (requires a human-provisioned
   `VOYAGE_API_KEY`) against the 23-event run and the 2-event smoke run, and write
   `results/diagnostics_baseline.json`.
2. Collect real ground truth for a handful of events into `ground_truth/raw/`, then run
   `label_truth.py` → `score.py` → `check_judge.py`. **This has never been done** — the
   spec's core question ("does B30 beat A?") has no answer yet.
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
BLOCKED ON HUMAN: ground_truth/raw/ is empty. The v1 study cannot be scored until real
post-announcement reactions are pasted in per CRISIS_SIM_VALIDATION_SPEC.md §2.4 (typically
72h after each event).
```

## Known, not urgent

- `web/backend/runs.py get_run_dir()` requires exactly one local run marked `complete`;
  this machine currently has two (`runs/20260729T...`, `runs/20260730T...`), so
  `test_web_api.py`/`test_web_service.py`'s real-data tests fail locally. Not a code bug —
  see CLAUDE.md "Local environment gotchas." Undecided: archive old runs, or make
  `get_run_dir()` pick the most recent.
- `main` was 10 commits ahead of `origin` and unpushed for several days before 2026-08-02 —
  now resolved, but a good reminder to push regularly (see CLAUDE.md "End-of-session
  rule").
