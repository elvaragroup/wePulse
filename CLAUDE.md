# CLAUDE.md

Guidance for Claude Code when working in this repo. Read [STATUS.md](STATUS.md) first —
it's the fastest way to get oriented after a session/token reset.

## What this is

A validation harness measuring whether an ensemble of LLM personas predicts real-world
backlash to corporate announcements better than a naive LLM call or a human. Two specs
govern the work, both at the WePulse root (one level up):

- `../CRISIS_SIM_VALIDATION_SPEC.md` — the v1 measurement harness build spec.
- `../persona-v2-implementation-brief.md` — the v2 redesign brief (data-derived personas,
  span-grounded reactions, anti-collapse diagnostics).

Read the relevant spec before starting any new phase of work — both are explicit about
what NOT to build; scope creep against them wastes a session.

## Workflow — hard rule

Any nontrivial change goes through:

`superpowers:brainstorming` → `superpowers:writing-plans` → `superpowers:subagent-driven-development`

Do not execute plans by hand (ad-hoc `executing-plans` or freeform edits). SDD's ledger at
`.superpowers/sdd/<plan-basename>/progress.md` is what makes work recoverable across a
session or token-limit reset — it survives compaction and new sessions; conversational
memory does not. Ad-hoc work has no equivalent safety net.

**Trust the ledger and `git log`, not memory.** If you're picking this project back up after
a reset, read `STATUS.md`, then the SDD ledger for whatever plan is in flight, before doing
anything else. Do not re-derive project state from scratch or re-implement something that
may already be done — check first.

## Worktree rule

When starting isolated work (`superpowers:using-git-worktrees` or manually), always branch
from current local `main`, never `origin/main`. Branching from a stale/unpushed origin has
already caused one real incident here (an implementer silently reconstructed a stale
`src/dashboard.py` from an old git blob instead of reporting BLOCKED — see the demo-web
ledger, Task 1). Push `main` after merging finished work so this stays low-risk.

## Blocked-on-human convention

Any ledger entry, commit message, or status note that needs a human action (API key,
product decision, data collection, a merge that needs sign-off) must be written as:

```
BLOCKED ON HUMAN: <exact action needed>
```

This is grep-able (`grep -rn "BLOCKED ON HUMAN" .`) — a resuming session should run that
before assuming anything is stuck for an unknown reason.

## End-of-session rule

Before ending a session:
1. Commit and push.
2. Update `STATUS.md`'s Now / Next / Blocked lines.
3. If a plan is mid-flight, confirm the SDD ledger reflects the true state (don't leave a
   task half-done and unrecorded).

## Local environment gotchas

- `runs/`, `results/`, and `ground_truth/{raw,labeled}/` are all gitignored — reproducible
  or hand-collected data, not tracked. Don't expect them to be consistent across machines.
- `web/backend/runs.py`'s `get_run_dir()` requires **exactly one** local run directory with
  `manifest.json` `"status": "complete"`. If more than one accumulates locally (e.g. after
  running a smoke test twice), `tests/test_web_api.py` and `tests/test_web_service.py`'s
  real-data integration tests will fail with "ambiguous: N complete runs exist" — this is
  not a code regression, it's local run-directory hygiene. Either archive old run dirs or
  fix `get_run_dir()` to pick the most recent (undecided as of 2026-08-02 — flagged, not
  fixed).
- Install both extras for the full test suite: `uv sync --all-extras` (dev + web). Plain
  `uv sync` won't pull `pytest` or `fastapi`.

## Testing

`uv run pytest -q` (after `uv sync --all-extras`). No network required — everything routes
through `FakeLLMClient`/fixtures. CI (`.github/workflows/`) runs the same command with no
API key.
