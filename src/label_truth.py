"""Labels ground truth from raw pasted reactions (spec 4.4).

The judge prompt carries only reaction text and the taxonomy -- never the
announcement, a prediction, an arm name, or `expected_null` (spec 4.4) -- so
the same procedure that scores predictions cannot be the one that also
generated them. `label_text` is the shared judging primitive: check_judge.py
reuses it verbatim against sampled persona output text, so the reliability
check exercises the exact same call this module makes against ground truth.

Labeling an event before its run has any prediction on disk would let a label
influence a not-yet-written guess; `_require_predictions_exist` makes that
ordering a runtime check rather than a convention.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src import ground_truth
from src.config import Config, load_config
from src.llm import AnthropicLLMClient, LLMClient, LLMRequest, structured_schema
from src.models import GroundTruthLabel, JudgeVerdict
from src.taxonomy import Taxonomy, load_taxonomy

REPO = Path(__file__).resolve().parent.parent

JUDGE_SYSTEM_PROMPT = """\
You are labeling real-world reaction text against a fixed taxonomy of backlash \
categories. You are shown only the reaction text below -- not the announcement \
that prompted it, not any prediction, and not which company or arm this came \
from. Label strictly from what the text itself says.

Decide:
- dominant_category: the single category that best describes the primary theme \
of the reaction. Use 'none' if the reaction is neutral, positive, or negligible.
- present_categories: every category actually present in the text (must include \
dominant_category). Use ['none'] alone when dominant_category is 'none'.
- backlash_occurred: true unless dominant_category is 'none'.
- judge_confidence: 0.0-1.0.

## Taxonomy

{taxonomy}
"""


class LabelError(RuntimeError):
    pass


async def label_text(
    client: LLMClient, config: Config, taxonomy: Taxonomy, text: str
) -> JudgeVerdict:
    """The judging primitive, shared with check_judge.py."""
    request = LLMRequest(
        role="judge",
        model=config.models["judge"],
        system=JUDGE_SYSTEM_PROMPT.format(taxonomy=taxonomy.render()),
        user=text,
        max_tokens=config.max_tokens["judge"],
        temperature=config.temperature_for("judge"),
        thinking=config.thinking_for("judge"),
        schema_name="JudgeVerdict",
        schema=structured_schema(JudgeVerdict),
    )
    return await client.complete(request, JudgeVerdict)


async def label_event(
    client: LLMClient, config: Config, taxonomy: Taxonomy, raw: ground_truth.RawGroundTruth
) -> GroundTruthLabel:
    verdict = await label_text(client, config, taxonomy, raw.reactions)
    try:
        return GroundTruthLabel(
            event_id=raw.event_id,
            dominant_category=verdict.dominant_category,
            present_categories=verdict.present_categories,
            backlash_occurred=verdict.backlash_occurred,
            judge_confidence=verdict.judge_confidence,
        )
    except ValueError as exc:
        raise LabelError(
            f"{raw.event_id}: judge verdict is internally inconsistent ({exc}). The model "
            "returned a valid JudgeVerdict but its fields fail GroundTruthLabel's own "
            "cross-checks (dominant_category must be in present_categories and consistent "
            "with backlash_occurred)."
        ) from exc


def _require_predictions_exist(repo: Path, run_id: str, event_id: str) -> None:
    predictions_dir = repo / "runs" / run_id / "predictions"
    matches = list(predictions_dir.glob(f"{event_id}__*.json"))
    if not matches:
        raise LabelError(
            f"no predictions for {event_id} under {predictions_dir}. label_truth.py runs only "
            "after predictions exist for an event (spec 4.4)."
        )


async def run_label(
    *,
    repo: Path = REPO,
    run_id: str,
    event_ids: list[str] | None = None,
    client: LLMClient | None = None,
) -> list[GroundTruthLabel]:
    config = load_config(repo / "config.yaml")
    taxonomy = load_taxonomy(repo / "taxonomy.txt")
    raw_dir = repo / "ground_truth" / "raw"
    labeled_dir = repo / "ground_truth" / "labeled"

    if event_ids is None:
        event_ids = sorted(p.stem for p in raw_dir.glob("*.txt"))
    if not event_ids:
        raise LabelError(f"no raw ground truth in {raw_dir}")

    for event_id in event_ids:
        _require_predictions_exist(repo, run_id, event_id)

    owns_client = client is None
    if client is None:
        client = AnthropicLLMClient(
            cache_dir=repo / "runs" / run_id / "cache",
            concurrency=config.concurrency,
            max_retries=config.max_retries,
        )

    labels: list[GroundTruthLabel] = []
    try:
        for event_id in event_ids:
            raw = ground_truth.read_raw(raw_dir, event_id)
            label = await label_event(client, config, taxonomy, raw)
            labeled_dir.mkdir(parents=True, exist_ok=True)
            (labeled_dir / f"{event_id}.json").write_text(
                label.model_dump_json(indent=2), encoding="utf-8"
            )
            labels.append(label)
    finally:
        if owns_client and isinstance(client, AnthropicLLMClient):
            await client.aclose()

    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="run whose predictions must already exist for each event")
    parser.add_argument(
        "events", nargs="*", help="event ids to label; defaults to every file in ground_truth/raw"
    )
    args = parser.parse_args(argv)
    try:
        labels = asyncio.run(run_label(run_id=args.run_id, event_ids=args.events or None))
    except LabelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for label in labels:
        print(f"{label.event_id}: dominant={label.dominant_category} backlash={label.backlash_occurred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
