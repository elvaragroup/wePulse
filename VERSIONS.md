# VERSIONS.md — persona-set version history and validation record

Every entry below is reproducible from its `persona_set_hash` + `run_id`. See
`docs/superpowers/specs/2026-08-02-persona-improvement-methodology.md` for how these numbers
are produced and what "accepted" means. Numbers are pulled from the archived
`results/archive/<version_tag>/report.txt`, not retyped from memory.

---

## v1.0 — status: BLOCKED

- Date: 2026-08-02 (personas unchanged since original hand-authoring)
- persona_set_hash: `2ec30ae6af168481140388d179491bf31328890b33a566a2e007cf467f428c03`
- run_id: `20260730T002314.423Z_2ec30ae6`
- Change: n/a — original 30 hand-written personas (`personas/001_privacy_hawk.md` ...
  `030_measured_academic.md`)
- Track A: pending — blocked on `VOYAGE_API_KEY` (see `STATUS.md`)
- Track B (dev-backtest, n=16): pending — `ground_truth/raw/` is empty, see
  `ground_truth/README.md` for the collection runbook
- Track B (holdout, n=7): not yet spent
- Judge kappa: pending
- Decision: **BLOCKED** — cannot score until ground truth exists (Part 1 of the
  methodology doc)
- Client-facing summary: not yet written. Do not draft marketing language before the
  numbers exist — that's the overclaim failure mode this whole study exists to prevent.

<!--
Template for future entries — copy this block, fill in, and prepend the previous entry
below it (newest first):

## vX.Y — status: accepted / rejected / dev-set-only
- Date:
- persona_set_hash:
- run_id:
- Change: <one persona edited / N personas edited / specific diff, one line>
- Track A: homogeneity Δ=, redundancy Δ=, specificity Δ=, register Δ=, stability=
- Track B (dev-backtest, n=16): caught_dominant_rate=X (95% CI [.,.]) vs prior version's Y;
  false_positive_rate=A (CI) vs prior's B; McNemar p=; Wilcoxon p=
- Track B (holdout, n=7): [only if this version reached a formal checkpoint]
- Judge kappa: (unchanged / re-verified)
- Decision: accepted / rejected / dev-set-only — one-line rationale
- Client-facing summary: <plain-English translation, e.g. "catches N% more real backlash
  categories than a single AI call, with no increase in false alarms">
-->
