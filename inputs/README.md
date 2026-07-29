# Authoring `events.txt`

Replace the three fictional placeholders before running the study. Target ~25
real announcements.

## Format

```
=== EVENT ===
id: evt_001
company: Acme Corp
sector: consumer_tech
date: 2026-06-14
headline: One line describing the announcement
source_url: https://example.com/press/whatever
expected_null: false
---
[the full announcement text, pasted verbatim]
--- PRIOR STATEMENTS ---
[optional; see below]
=== END EVENT ===
```

Headers are all required except `source_url`. `id` must match `evt_NNN`. `date`
must be `YYYY-MM-DD`. A malformed block raises rather than being skipped, so a
typo fails loudly instead of quietly shrinking your sample.

## The rules that actually bind

**At least 40% of events must be null events** (spec 0.6) — announcements that
produced no meaningful backlash. Without them, an arm that predicts backlash for
everything scores well on the headline metric and its false-positive rate is
unmeasurable. `run_sim.py` reports the share and warns below 40%.

**`expected_null` is your a priori guess and is never scored.** It exists only
for sampling-balance reporting. It is parsed onto the Event but there is no code
path that carries it into a model prompt, and a test enforces that.

**Paste the announcement, don't summarise it.** The personas see exactly this
text. Summarising it launders your own judgment about what matters into the
input, which is the thing being measured.

**Prefer events you can still collect reactions for.** Ground truth is pasted by
hand 72h after the event into `ground_truth/raw/<event_id>.txt`. An announcement
from three years ago has no recoverable reply thread and is also far more likely
to be in the model's training data — `probe_leakage.py` will flag it.

## `--- PRIOR STATEMENTS ---` (optional)

Anything after this marker is given only to the hypocrisy personas, which cannot
function from the announcement text alone (spec 2.2). Paste dated quotes from the
company's own prior public statements — earnings calls, blog posts, filings — that
the announcement might contradict.

Omit the section when you have nothing. The hypocrisy personas are told that
without receipts they cannot establish hypocrisy and should ignore the
announcement, so an absent section produces a shrug rather than a fabrication.

## After you add events

```bash
uv run python -m src.events_check
```

Parses the file, prints per-event summaries and the null share, and exits
non-zero if anything is malformed.
