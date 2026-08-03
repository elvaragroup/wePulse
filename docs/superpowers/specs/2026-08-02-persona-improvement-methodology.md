# Persona improvement methodology

**Status:** approved, not yet executed (Part 1 ground-truth collection is human work, see
`ground_truth/README.md`).

## Purpose

`crisis-sim`'s value proposition is an empirical claim: the 30-persona ensemble predicts
real backlash better than a single naive LLM call. That claim is currently unproven —
`ground_truth/` is empty, no `score.py` report has ever been generated, and
`results/diagnostics_baseline.json` doesn't exist. The evaluation machinery to prove it is
already fully built and tested (`src/score.py`, `src/diagnostics.py`, `src/check_judge.py`)
but has never been run end-to-end.

This doc defines a repeatable process — reusing that existing tooling exactly as-is, no new
evaluation code — that turns editing personas into a disciplined, evidence-gated loop, and
produces a `../../../VERSIONS.md` track record credible enough to sell to PR/comms
agencies.

## Two things worth knowing before reading further

**Comparing persona versions isn't literally "just different arm names."**
`ArmPrediction.arm` is typed `Literal["A","B3","B8","B15","B30","C"]`
(`src/models.py:38`), so a prediction file can't be written to disk with an arbitrary arm
string like `"B30_v2"` and loaded normally. The real reuse path: call `score_run()` once per
version's `run_id`, relabel the returned `EventScore` objects in memory
(`EventScore.arm` is a plain unconstrained `str`), concatenate, and pass into the existing
`compare_arms()`. Zero edits to `src/` — see the snippet in Track B below.

**The spec's "paste reactions 72h after the event" is moot for the current event set.**
Every event in `inputs/events.txt` is already 80+ days old (2025-11-12 to 2026-05-13; today
is 2026-08-02). Ground-truth collection here is retroactive research — finding archived
reaction threads — not a waiting game. That's why Part 1's batching is ordered by
findability, not by a clock.

---

## Part 0 — Preconditions

1. **`VOYAGE_API_KEY`** (blocked on human, tracked in `STATUS.md`) — needed for Track A's
   embedding-based metrics (homogeneity, redundancy). Without it, Track A still works with
   the four metrics that don't need embeddings (specificity, register_variance, stability,
   partial distribution_match).
2. **Judge validity gate**, once some ground truth exists: hand-label 25 items into
   `results/human_labels.csv` (`item_id,text,human_category`), run
   `uv run python -m src.check_judge`. If Cohen's kappa < 0.6, stop — per the script's own
   text, every downstream number is noise. One-time-ish; re-run only if the judge model or
   `taxonomy.txt` changes.

---

## Part 1 — Ground-truth collection

See `ground_truth/README.md` for the full runbook (batches, exact event-ID lists, pasted
format). Summary: predictions already exist for all 23 events
(`runs/20260730T002314.423Z_2ec30ae6/predictions/`, 138 files = 6 arms × 23 events), so
`label_truth.py`'s prerequisite is already satisfied — ground truth is the only missing
ingredient.

Three pools, defined once and reused for the life of this methodology:

| Pool | Size | Purpose |
|---|---|---|
| Batch 0 | 5 events | Collect first — validates plumbing + judge kappa gate |
| Dev-backtest | 16 events | Spend freely during ordinary iteration |
| Frozen holdout | 7 events | Consult only at formal version-acceptance checkpoints |

Holdout deliberately excludes the three most likely to be model-memorized (Astronomer
concert incident, American Eagle campaign, Coca-Cola AI ad) — a client-facing "B30 beats A"
claim shouldn't rest on events the model may already know the outcome of. Re-verify this
partition against `probe_leakage.py`'s actual `CONTAMINATED`/`CLEAN` output once ground
truth exists; it's currently based on content inspection, not a leakage run.

---

## Part 2 — The iteration loop

### Why not just a clean train/test split

At n=23, splitting further to get a "pure" held-out test set destroys the statistical power
the spec is already candid about (`report.txt`'s mandated minimum-detectable-effect
sentence). The guard here is procedural discipline on top of a modest partition: only
*look at* the holdout pool at formal checkpoints, and log every consultation in
`VERSIONS.md` like a pre-registered analysis. Long-term fix: persona-v2 Phase 1 (corpus
acquisition, see `persona-v2-implementation-brief.md`) should feed newly-scored real events
back into the dev-backtest pool over time, diluting any accumulated overfitting.

### Track A — cheap, fast, no new ground truth needed

Gates whether Track B (which spends scarce ground truth) is even worth running.

```bash
uv run python -m src.run_sim --init
uv run python -m src.run_sim --execute <RUN_ID>
uv run python -m src.diagnostics --run <RUN_ID> --compare-baseline
```

First real baseline (do this as soon as `VOYAGE_API_KEY` exists, against the existing
23-event run):

```bash
uv run python -m src.diagnostics --run 20260730T002314.423Z_2ec30ae6 --write-baseline
```

**Gate — all must hold, else stop, don't spend Track B:**

- `specificity.false_positive_rate` — must not increase
- `homogeneity.mean_pairwise_cosine` — must not increase
- `redundancy.ratio` — must not decrease
- `register_variance.word_count_stdev` — must not decrease
- `stability.category_agreement_rate` — no material drop
- `distribution_match.total_variation_distance` (where measurable) — must not increase

No absolute numeric floors are set here — the first real `--write-baseline` run *becomes*
the floor. Inventing thresholds before that baseline exists is the "looks plausible, isn't"
failure mode spec §7 warns against.

### Track B — ground-truth-backed, spend deliberately, two tiers

```bash
uv run python -m src.label_truth <RUN_ID>   # once per event, reused across all future versions
uv run python -m src.score <RUN_ID>
```

`score.py` scores whatever's labeled in `ground_truth/labeled/` — label only the
dev-backtest pool during ordinary iteration; label the holdout pool only at a formal
checkpoint. **Archive immediately after every run** — `results/` is a single global location
the next run overwrites:

```bash
mkdir -p results/archive/<version_tag>
cp results/scores.csv results/report.txt results/archive/<version_tag>/
```

For a formal paired v1-vs-v2 comparison at a major version boundary, reuse `score_run()` +
`compare_arms()` directly — no `src/` edits:

```python
import dataclasses
from src.score import score_run, compare_arms
from src.config import load_config

config = load_config(REPO / "config.yaml")
v1 = [dataclasses.replace(s, arm="B30_v1") for s in score_run(run_id=V1_RUN_ID) if s.arm == "B30"]
v2 = [dataclasses.replace(s, arm="B30_v2") for s in score_run(run_id=V2_RUN_ID) if s.arm == "B30"]
comparison = compare_arms(v1 + v2, "B30_v2", "B30_v1", resamples=config.bootstrap_resamples)
```

### Accepting a new version — precise criteria, not vibes

1. Track A gate passed.
2. `false_positive_rate` (dev-backtest) did **not increase** vs the last accepted version —
   hard ratchet, zero tolerance, no trade-off allowed against `caught_dominant_rate` gains.
   Any exception needs an explicit written override in `VERSIONS.md` with justification.
3. `caught_dominant_rate` (dev-backtest) did not decrease. Don't require `p < 0.05` on
   McNemar for incremental persona edits — that bar is unreachable at n=16-23 and the spec
   itself says so. Report the McNemar/Wilcoxon p-values anyway, as transparent secondary
   evidence.
4. **Major version boundaries only** (e.g. v1.0 → v2.0, where the brief expects a
   structurally large effect): do require significance — McNemar `p < 0.05` on the headline
   comparison, or a bootstrap CI on `precision_at_3` excluding zero.
5. Holdout-pool check (spent only here): same two ratchets (no FP increase, no
   caught_dominant decrease) confirmed on the 7 held-out events before calling a version
   production-ready for client use.
6. Judge kappa still ≥ 0.6 (unchanged since last check, or re-verified if the judge or
   taxonomy changed).

Write the change and the expected outcome into a draft `VERSIONS.md` entry *before* running
Track B. Pre-registration is the cheapest available defense against unconsciously iterating
until the backtest set looks good — this is the guardrail the persona-v2 brief states
verbatim: *"Do not tune prompts against the backtest set. Freeze the engine, then
evaluate."*

---

## Where this fits the existing workflow

The routine edit/measure loop above (persona edit → Track A → maybe Track B → `VERSIONS.md`
entry) is lightweight, like TDD's red/green loop — it does not need a fresh
`superpowers:subagent-driven-development` ledger every iteration. Reserve
`brainstorming → writing-plans → subagent-driven-development` for structurally larger
changes: persona-v2 Phases 1–4 (corpus ingestion, grievance clustering, v2 persona
generation, staged reaction pipeline) stay in that heavier track per `STATUS.md`'s existing
ordering. This methodology gives that future work something to be measured against, once a
real v1.0 baseline exists.

## Commercial credibility

- **Reproducibility**: `frozen.json`/`manifest.json`'s `persona_set_hash` already gives an
  auditable "this exact text produced this exact number" claim — cite it in every
  `VERSIONS.md` entry so a skeptical agency can, in principle, verify it.
- **Judge validity**: surface the kappa number prominently, not just buried in
  `VERSIONS.md` — it's what makes every downstream metric trustworthy rather than "a model
  grading itself."
- **Honest power statement**: `report.txt`'s mandated MDE sentence is a credibility
  feature for a sophisticated buyer, not a caveat to hide.
- **`defusable_by`** (v2 Phase 4, not yet built) is the real differentiator per the brief —
  converts the deliverable from "here's your risk" to "here's the exact edit that removes
  it." Prioritizing v2 should wait for the first non-`BLOCKED` `VERSIONS.md` entry — there's
  no evidence yet that v1 is worth selling, so v2 investment should follow proof, not
  precede it.
