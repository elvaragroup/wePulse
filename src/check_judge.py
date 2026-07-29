"""Judge reliability check against a human sample (spec 4.6).

The user hand-labels a random sample of items -- a mix of ground-truth texts
and persona outputs -- into `results/human_labels.csv` (columns: item_id,
text, human_category; item_id and text are free-form, human_category must be
a taxonomy id). This script re-labels the same `text` with the judge, via the
identical `label_text` primitive label_truth.py uses on ground truth, and
reports Cohen's kappa between the two.

Every number downstream of the judge -- every score in results/scores.csv,
every comparison in results/report.txt -- is only as trustworthy as the judge
itself. Spec 4.6 is explicit that kappa < 0.6 means all of it is noise, so
that check fails loudly here rather than leaving it to be missed in a wall of
report output.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.label_truth import label_text
from src.llm import AnthropicLLMClient, LLMClient
from src.stats import cohens_kappa
from src.taxonomy import load_taxonomy

REPO = Path(__file__).resolve().parent.parent
KAPPA_FLOOR = 0.6
EXPECTED_SAMPLE_SIZE = 25


class JudgeCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class HumanLabel:
    item_id: str
    text: str
    human_category: str


def load_human_labels(path: Path) -> list[HumanLabel]:
    if not path.exists():
        raise JudgeCheckError(f"no human labels at {path}. Hand-label a sample first (spec 4.6).")

    labels: list[HumanLabel] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"item_id", "text", "human_category"} - set(reader.fieldnames or [])
        if missing:
            raise JudgeCheckError(f"{path}: missing column(s) {sorted(missing)}")
        for row in reader:
            labels.append(
                HumanLabel(
                    item_id=row["item_id"],
                    text=row["text"],
                    human_category=row["human_category"].strip(),
                )
            )

    if not labels:
        raise JudgeCheckError(f"{path}: no rows")
    return labels


async def run_check(
    *,
    repo: Path = REPO,
    human_labels_path: Path | None = None,
    client: LLMClient | None = None,
) -> tuple[float, list[HumanLabel], list[str]]:
    config = load_config(repo / "config.yaml")
    taxonomy = load_taxonomy(repo / "taxonomy.txt")
    labels = load_human_labels(human_labels_path or repo / "results" / "human_labels.csv")

    unknown = [lab.item_id for lab in labels if lab.human_category not in taxonomy.ids]
    if unknown:
        raise JudgeCheckError(f"human_category not a taxonomy id for item(s): {unknown}")

    owns_client = client is None
    if client is None:
        client = AnthropicLLMClient(concurrency=config.concurrency, max_retries=config.max_retries)

    try:
        verdicts = await asyncio.gather(
            *(label_text(client, config, taxonomy, lab.text) for lab in labels)
        )
    finally:
        if owns_client and isinstance(client, AnthropicLLMClient):
            await client.aclose()

    judge_categories = [v.dominant_category for v in verdicts]
    human_categories = [lab.human_category for lab in labels]
    kappa = cohens_kappa(human_categories, judge_categories)
    return kappa, labels, judge_categories


def write_reliability_report(
    path: Path, *, kappa: float, labels: list[HumanLabel], judge_categories: list[str]
) -> str:
    lines = [
        f"Judge reliability: kappa={kappa:.3f} over n={len(labels)} item(s) "
        f"(spec 4.6 target sample size: {EXPECTED_SAMPLE_SIZE}).",
        "",
    ]
    if len(labels) != EXPECTED_SAMPLE_SIZE:
        lines.append(
            f"NOTE: sample size is {len(labels)}, not the spec's {EXPECTED_SAMPLE_SIZE}."
        )
        lines.append("")

    if kappa < KAPPA_FLOOR:
        lines.append(
            f"*** STOP: kappa {kappa:.3f} is below the {KAPPA_FLOOR} floor (spec 4.6). ***"
        )
        lines.append(
            "*** Every downstream number -- every row in results/scores.csv, every  ***"
        )
        lines.append(
            "*** comparison in results/report.txt -- is noise until this is fixed.  ***"
        )
        lines.append("")

    lines.append("item_id, human, judge")
    for label, judge_category in zip(labels, judge_categories):
        mark = "" if label.human_category == judge_category else "  <-- disagree"
        lines.append(f"{label.item_id}: {label.human_category} / {judge_category}{mark}")

    report = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-labels", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        kappa, labels, judge_categories = asyncio.run(
            run_check(human_labels_path=args.human_labels)
        )
    except JudgeCheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = write_reliability_report(
        REPO / "results" / "judge_reliability.txt",
        kappa=kappa,
        labels=labels,
        judge_categories=judge_categories,
    )
    print(report)
    return 1 if kappa < KAPPA_FLOOR else 0


if __name__ == "__main__":
    raise SystemExit(main())
