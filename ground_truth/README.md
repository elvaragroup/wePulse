# Ground-truth collection runbook

This is Part 1 of `docs/superpowers/specs/2026-08-02-persona-improvement-methodology.md` —
read that doc for why this ordering exists. This file is just the checklist: which events,
in what order, in what format.

**Every event in `inputs/events.txt` is already 80+ days old** (2025-11-12 to 2026-05-13,
today is 2026-08-02). This is retroactive research — finding archived reaction threads —
not a 72h wait.

Predictions already exist for all 23 events
(`runs/20260730T002314.423Z_2ec30ae6/predictions/`), so `label_truth.py` is ready the moment
a `raw/<event_id>.txt` file exists. Ground truth is the only missing ingredient.

## Format (fixed by spec §2.4 — do not deviate)

Save to `ground_truth/raw/<event_id>.txt`:

```
=== GROUND TRUTH ===
id: evt_001
observed_at: <ISO timestamp of when you collected it, not the event date>
collection_rule: top 100 replies + quote posts by engagement, plus first-page news coverage
---
[pasted reactions, one per line or block, no editing, no summarizing]
=== END ===
```

Paste raw text only. Category labels are assigned by `label_truth.py`, never by hand — that's
what makes the comparison between predictions and reality fair (same labeling procedure for
both).

## Batch 0 — collect first (5 events)

Highest-findability reaction threads, plus two easy null confirmations. Collecting these
first validates the plumbing (`label_truth.py` → sane labels?) and seeds the sample for the
judge-kappa check below, before investing effort in the harder-to-find remainder.

| event | company | why easy to find |
|---|---|---|
| evt_006 | Astronomer | Coldplay-concert incident — highest-volume public reaction of the set |
| evt_005 | American Eagle Outfitters | "Bad Genes" campaign — widely covered, easy to search |
| evt_023 | The Coca-Cola Company | AI-generated Christmas ad — widely covered |
| evt_002 | Northwind Logistics | Easy **null** confirmation — new jobs announcement, verify by absence of a reaction thread |
| evt_007 | Social Security Administration | Easy **null** confirmation — routine digital-notice shift |

**After Batch 0:** run `label_truth.py`, hand-label 25 items into
`results/human_labels.csv` (draw from Batch 0's raw text plus persona output quotes already
in `runs/20260730T002314.423Z_2ec30ae6/raw/B30/`), run `check_judge.py`. **Do not proceed
past this gate if kappa < 0.6** — per the script's own text, every downstream number is
noise from that point on.

## Dev-backtest pool — spend freely during iteration (16 events, includes Batch 0's 5)

evt_003, evt_004, evt_005, evt_006, evt_007, evt_008, evt_011, evt_012, evt_013, evt_014,
evt_016, evt_017, evt_020, evt_021, evt_022, evt_023

(9 non-null / 7 null ≈ 44% null, matches the overall set's ratio)

## Frozen holdout pool — collect, but do not label/score until a formal checkpoint (7 events)

evt_001, evt_002, evt_009, evt_010, evt_015, evt_018, evt_019

Collecting the raw text is fine any time; what's restricted is *labeling and scoring* these
before a formal version-acceptance checkpoint (see the methodology doc's "Accepting a new
version" section). Log every holdout consultation in `VERSIONS.md`.

This pool deliberately excludes evt_005, evt_006, evt_023 (the three most likely to be
model-memorized, highest-profile events) — a client-facing "B30 beats A" claim shouldn't
rest on events the model may already know the outcome of. Re-verify this against
`probe_leakage.py`'s actual `CONTAMINATED`/`CLEAN` output once ground truth exists for these
seven; the partition is currently based on content inspection, not a leakage run.

## All 23 events, for reference

| id | company | expected_null |
|---|---|---|
| evt_001 | Acme Corp | no |
| evt_002 | Northwind Logistics | yes |
| evt_003 | Contoso Media | no |
| evt_004 | LG Electronics | no |
| evt_005 | American Eagle Outfitters | no |
| evt_006 | Astronomer | no |
| evt_007 | Social Security Administration | yes |
| evt_008 | IonQ | yes |
| evt_009 | xAI / Grok | no |
| evt_010 | Beyond Fossil Fuels | yes |
| evt_011 | Target Corporation | no |
| evt_012 | Kenvue | no |
| evt_013 | DeepSeek | yes |
| evt_014 | OpenAI | yes |
| evt_015 | OpenAI | no |
| evt_016 | Anthropic | no |
| evt_017 | OpenAI | yes |
| evt_018 | PwC | yes |
| evt_019 | Chaotic Good | no |
| evt_020 | AnswerConnect | yes |
| evt_021 | Department of Justice / NVIDIA | no |
| evt_022 | Milieu AI | yes |
| evt_023 | The Coca-Cola Company | no |

`expected_null` is never used in scoring (spec §2.3) — shown here only for collection
planning. Source: `runs/20260730T002314.423Z_2ec30ae6/manifest.json`'s
`expected_null_share: 0.4348` (10/23), cross-checked against each event's own
`expected_null` field in `inputs/events.txt`.
