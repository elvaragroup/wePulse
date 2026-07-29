"""Scores one run's predictions against labeled ground truth (spec 4.5).

Refuses to run if any prediction file's mtime is later than its event's raw
ground-truth mtime -- the file-system evidence that predictions were committed
to disk before ground truth could have been read (spec 0.2). This is checked
against `ground_truth/raw/<event_id>.txt`, the file a human pastes reactions
into 72h after an event, per spec 4.5's exact wording; `ground_truth/labeled/`
is what actually carries the categories used below, but it can only exist
*after* the raw file per label_truth.py (spec 4.4), so raw is the tighter and
earlier check.

Single-arm CIs (caught_dominant_rate, precision_at_3, recall, false_positive_rate)
reuse `paired_bootstrap_ci` against an all-zero series: bootstrapping
`values - 0` is exactly a bootstrap of `values`'s own mean, so no separate
one-sample bootstrap routine is needed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.models import K, ArmPrediction, GroundTruthLabel
from src.stats import BootstrapCI, McNemarResult, mcnemar_exact, mde_paired_binary, paired_bootstrap_ci, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parent.parent
BASELINE_ARM = "A"
HEADLINE_COMPARISON = ("B30", BASELINE_ARM)
SECONDARY_COMPARISON = ("B30", "B3")
CSV_FIELDS = (
    "event_id",
    "arm",
    "contaminated",
    "caught_dominant",
    "precision_at_3",
    "recall",
    "false_positive",
    "predicted_top3",
    "actual_dominant",
    "actual_present",
)


class ScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class EventScore:
    event_id: str
    arm: str
    contaminated: bool
    caught_dominant: bool
    precision_at_3: float
    recall: float
    false_positive: bool | None  # None when backlash_occurred is True: not applicable
    predicted_top3: tuple[str, ...]
    actual_dominant: str
    actual_present: tuple[str, ...]

    def to_row(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "arm": self.arm,
            "contaminated": str(self.contaminated),
            "caught_dominant": str(self.caught_dominant),
            "precision_at_3": f"{self.precision_at_3:.6f}",
            "recall": f"{self.recall:.6f}",
            "false_positive": "" if self.false_positive is None else str(self.false_positive),
            "predicted_top3": "|".join(self.predicted_top3),
            "actual_dominant": self.actual_dominant,
            "actual_present": "|".join(self.actual_present),
        }


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    n: int
    caught_dominant_rate: BootstrapCI
    precision_at_3: BootstrapCI
    recall: BootstrapCI
    false_positive_rate: BootstrapCI | None
    false_positive_n: int


@dataclass(frozen=True)
class PairwiseComparison:
    arm: str
    baseline: str
    caught_dominant: McNemarResult
    precision_at_3_wilcoxon_p: float
    precision_at_3_bootstrap: BootstrapCI


# --- loading ---


def _discover_predictions(predictions_dir: Path) -> dict[str, dict[str, Path]]:
    """event_id -> {arm -> path}, parsed from `<event_id>__<arm>.json` filenames."""
    by_event: dict[str, dict[str, Path]] = {}
    for path in sorted(predictions_dir.glob("*.json")):
        stem = path.stem
        if "__" not in stem:
            raise ScoreError(f"{path}: expected '<event_id>__<arm>.json'")
        event_id, arm = stem.split("__", 1)
        by_event.setdefault(event_id, {})[arm] = path
    return by_event


def _load_leakage(run_dir: Path) -> dict[str, bool]:
    """event_id -> contaminated, from runs/<run_id>/leakage.csv if it exists.

    probe_leakage.py (spec 4.1) has not been wired at the time this module was
    written; scoring must not block on its absence, so a missing file just
    means every event is treated as uncontaminated (noted in the report).
    """
    path = run_dir / "leakage.csv"
    if not path.exists():
        return {}
    verdicts: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            event_id = row["event_id"]
            verdict = row["verdict"].strip().upper()
            # Any one of the three probe runs leaking marks the event contaminated
            # (spec 4.1) -- so OR across rows rather than overwrite.
            verdicts[event_id] = verdicts.get(event_id, False) or verdict == "CONTAMINATED"
    return verdicts


def guard_mtimes(by_event: dict[str, dict[str, Path]], raw_ground_truth_dir: Path) -> None:
    for event_id, arms in sorted(by_event.items()):
        raw_path = raw_ground_truth_dir / f"{event_id}.txt"
        if not raw_path.exists():
            raise ScoreError(f"{event_id}: no ground truth at {raw_path}; cannot score yet")
        raw_mtime = raw_path.stat().st_mtime
        for arm, path in sorted(arms.items()):
            if path.stat().st_mtime > raw_mtime:
                raise ScoreError(
                    f"{path}: prediction is newer than its ground truth ({raw_path}). "
                    "Predictions must be written to disk before ground truth is ever read "
                    "(spec 0.2); scoring refuses to run when the mtimes cannot show that."
                )


def load_ground_truth_labels(labeled_dir: Path, event_ids: set[str]) -> dict[str, GroundTruthLabel]:
    labels: dict[str, GroundTruthLabel] = {}
    for event_id in event_ids:
        path = labeled_dir / f"{event_id}.json"
        if not path.exists():
            continue  # not yet labeled -- excluded from this scoring pass, not an error
        labels[event_id] = GroundTruthLabel.model_validate_json(path.read_text(encoding="utf-8"))
    return labels


# --- per-event scoring ---


def score_event_arm(
    prediction: ArmPrediction, truth: GroundTruthLabel, *, contaminated: bool
) -> EventScore:
    top3 = tuple(prediction.ranked_categories)
    present = tuple(truth.present_categories)
    overlap = len(set(top3) & set(present))

    false_positive: bool | None
    if truth.backlash_occurred:
        false_positive = None
    else:
        false_positive = bool(prediction.backlash_predicted)

    return EventScore(
        event_id=prediction.event_id,
        arm=prediction.arm,
        contaminated=contaminated,
        caught_dominant=truth.dominant_category in top3,
        precision_at_3=overlap / K,
        recall=(overlap / len(present)) if present else 0.0,
        false_positive=false_positive,
        predicted_top3=top3,
        actual_dominant=truth.dominant_category,
        actual_present=present,
    )


def score_run(
    *,
    repo: Path = REPO,
    run_id: str,
) -> list[EventScore]:
    run_dir = repo / "runs" / run_id
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        raise ScoreError(f"no predictions at {predictions_dir}")

    by_event = _discover_predictions(predictions_dir)
    if not by_event:
        raise ScoreError(f"{predictions_dir} has no predictions")

    guard_mtimes(by_event, repo / "ground_truth" / "raw")

    labels = load_ground_truth_labels(repo / "ground_truth" / "labeled", set(by_event))
    if not labels:
        raise ScoreError(
            f"no labeled ground truth in {repo / 'ground_truth' / 'labeled'} for any predicted "
            "event; run label_truth.py first"
        )

    contamination = _load_leakage(run_dir)

    scores: list[EventScore] = []
    for event_id, arms in sorted(by_event.items()):
        truth = labels.get(event_id)
        if truth is None:
            continue
        contaminated = contamination.get(event_id, False)
        for arm, path in sorted(arms.items()):
            prediction = ArmPrediction.model_validate_json(path.read_text(encoding="utf-8"))
            scores.append(score_event_arm(prediction, truth, contaminated=contaminated))
    return scores


# --- aggregation ---


def _bootstrap_mean(values: list[float], *, resamples: int) -> BootstrapCI:
    return paired_bootstrap_ci(values, [0.0] * len(values), resamples=resamples)


def arm_metrics(scores: list[EventScore], arm: str, *, resamples: int) -> ArmMetrics:
    rows = [s for s in scores if s.arm == arm]
    if not rows:
        raise ScoreError(f"no scored events for arm {arm!r}")

    caught = [1.0 if r.caught_dominant else 0.0 for r in rows]
    precision = [r.precision_at_3 for r in rows]
    recall = [r.recall for r in rows]
    fp_rows = [r for r in rows if r.false_positive is not None]
    fp = [1.0 if r.false_positive else 0.0 for r in fp_rows]

    return ArmMetrics(
        arm=arm,
        n=len(rows),
        caught_dominant_rate=_bootstrap_mean(caught, resamples=resamples),
        precision_at_3=_bootstrap_mean(precision, resamples=resamples),
        recall=_bootstrap_mean(recall, resamples=resamples),
        false_positive_rate=_bootstrap_mean(fp, resamples=resamples) if fp else None,
        false_positive_n=len(fp_rows),
    )


def _paired(scores: list[EventScore], arm: str, baseline: str, metric: str) -> tuple[list, list]:
    by_arm = {(s.event_id, s.arm): s for s in scores}
    shared = sorted(
        {e for (e, a) in by_arm if a == arm} & {e for (e, a) in by_arm if a == baseline}
    )
    if not shared:
        raise ScoreError(f"no events scored by both {arm!r} and {baseline!r}")
    first = [getattr(by_arm[(e, arm)], metric) for e in shared]
    second = [getattr(by_arm[(e, baseline)], metric) for e in shared]
    return first, second


def compare_arms(scores: list[EventScore], arm: str, baseline: str, *, resamples: int) -> PairwiseComparison:
    caught_arm, caught_base = _paired(scores, arm, baseline, "caught_dominant")
    precision_arm, precision_base = _paired(scores, arm, baseline, "precision_at_3")

    return PairwiseComparison(
        arm=arm,
        baseline=baseline,
        caught_dominant=mcnemar_exact(caught_arm, caught_base),
        precision_at_3_wilcoxon_p=wilcoxon_signed_rank(precision_arm, precision_base),
        precision_at_3_bootstrap=paired_bootstrap_ci(precision_arm, precision_base, resamples=resamples),
    )


# --- output ---


def write_scores_csv(scores: list[EventScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for score in scores:
            writer.writerow(score.to_row())


def _fmt_ci(ci: BootstrapCI) -> str:
    return f"{ci.point:.3f} (95% CI [{ci.low:.3f}, {ci.high:.3f}], n={ci.resamples})"


def build_report(
    *,
    run_id: str,
    scores: list[EventScore],
    arms: list[str],
    resamples: int,
) -> str:
    lines: list[str] = []
    n_events = len({s.event_id for s in scores})
    n_contaminated = len({s.event_id for s in scores if s.contaminated})
    lines.append(f"Crisis-sim scoring report -- run {run_id}")
    lines.append(f"Events scored: {n_events} ({n_contaminated} flagged contaminated)")
    lines.append("")

    for subset_name, subset in (("all events", scores), ("excluding contaminated", [s for s in scores if not s.contaminated])):
        lines.append(f"--- {subset_name} ---")
        if not subset:
            lines.append("(no events)")
            lines.append("")
            continue
        for arm in arms:
            try:
                metrics = arm_metrics(subset, arm, resamples=resamples)
            except ScoreError:
                lines.append(f"{arm}: no events in this subset")
                continue
            fp = (
                f"{_fmt_ci(metrics.false_positive_rate)} (n={metrics.false_positive_n})"
                if metrics.false_positive_rate is not None
                else "n/a (no null events in subset)"
            )
            lines.append(
                f"{arm} (n={metrics.n}): "
                f"caught_dominant_rate={_fmt_ci(metrics.caught_dominant_rate)}; "
                f"precision_at_3={_fmt_ci(metrics.precision_at_3)}; "
                f"recall={_fmt_ci(metrics.recall)} [biased toward larger arms]; "
                f"false_positive_rate={fp}"
            )
        lines.append("")

    lines.append("--- pairwise comparisons (caught_dominant: McNemar exact; precision_at_3: Wilcoxon + bootstrap) ---")
    comparisons: list[tuple[str, str]] = [(arm, BASELINE_ARM) for arm in arms if arm != BASELINE_ARM]
    comparisons.append(SECONDARY_COMPARISON)
    headline_mde: tuple[int, float] | None = None
    for arm, baseline in comparisons:
        try:
            cmp = compare_arms(scores, arm, baseline, resamples=resamples)
        except ScoreError as exc:
            lines.append(f"{arm} vs {baseline}: {exc}")
            continue
        paired_n = len(_paired(scores, arm, baseline, "caught_dominant")[0])
        lines.append(
            f"{arm} vs {baseline} (n={paired_n}): "
            f"caught_dominant McNemar b={cmp.caught_dominant.b} c={cmp.caught_dominant.c} "
            f"p={cmp.caught_dominant.p_value:.4f}; "
            f"precision_at_3 Wilcoxon p={cmp.precision_at_3_wilcoxon_p:.4f}, "
            f"paired diff {_fmt_ci(cmp.precision_at_3_bootstrap)}"
        )
        if (arm, baseline) == HEADLINE_COMPARISON:
            discordance = cmp.caught_dominant.discordant / paired_n if paired_n else 0.0
            headline_mde = (paired_n, discordance)
    lines.append("")

    if headline_mde is not None:
        n, discordance = headline_mde
        mde_pct = mde_paired_binary(n, discordance=discordance) * 100
        lines.append(
            f"At n={n}, this study can only detect large differences between arms. A null "
            f"result means the effect is smaller than {mde_pct:.1f} percentage points, not "
            "that it is zero."
        )
        lines.append("")

    kappa_path = REPO / "results" / "judge_reliability.txt"
    if kappa_path.exists():
        lines.append("--- judge reliability ---")
        lines.append(kappa_path.read_text(encoding="utf-8").strip())
    else:
        lines.append("--- judge reliability ---")
        lines.append(
            "results/judge_reliability.txt not found -- run check_judge.py before trusting "
            "any number above (kappa < 0.6 means every downstream number is noise, spec 4.6)."
        )
    lines.append("")

    return "\n".join(lines)


# --- CLI ---


def run_score(*, repo: Path = REPO, run_id: str) -> None:
    config = load_config(repo / "config.yaml")
    scores = score_run(repo=repo, run_id=run_id)
    arms = sorted({s.arm for s in scores}, key=lambda a: (a != BASELINE_ARM, a))

    write_scores_csv(scores, repo / "results" / "scores.csv")
    report = build_report(run_id=run_id, scores=scores, arms=arms, resamples=config.bootstrap_resamples)
    (repo / "results" / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        run_score(run_id=args.run_id)
    except ScoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
