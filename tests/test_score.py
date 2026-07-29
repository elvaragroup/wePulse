from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import pytest

from src.models import ArmPrediction, GroundTruthLabel
from src.score import (
    ScoreError,
    arm_metrics,
    build_report,
    compare_arms,
    guard_mtimes,
    run_score,
    score_event_arm,
    score_run,
    write_scores_csv,
)


def prediction(*, event_id: str, arm: str, ranked: list[str], backlash: bool) -> ArmPrediction:
    return ArmPrediction(
        arm=arm,
        event_id=event_id,
        ranked_categories=ranked,
        scores={},
        backlash_predicted=backlash,
    )


def truth(*, event_id: str, dominant: str, present: list[str], backlash: bool) -> GroundTruthLabel:
    return GroundTruthLabel(
        event_id=event_id,
        dominant_category=dominant,
        present_categories=present,
        backlash_occurred=backlash,
        judge_confidence=0.9,
    )


# --- spec 6: hand-computed fixture, 2 events x 3 arms, known answer ---

TRUTH_1 = truth(event_id="evt_001", dominant="privacy", present=["privacy", "security"], backlash=True)
TRUTH_2 = truth(event_id="evt_002", dominant="none", present=["none"], backlash=False)


def test_hand_computed_event_scores():
    # Arm A: catches the dominant category but only 1 of 2 present categories.
    a1 = score_event_arm(
        prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
        TRUTH_1,
        contaminated=False,
    )
    assert a1.caught_dominant is True
    assert a1.precision_at_3 == pytest.approx(1 / 3)
    assert a1.recall == pytest.approx(0.5)
    assert a1.false_positive is None  # backlash_occurred True: not applicable

    # Arm B3: misses entirely on the event that had real backlash.
    b3_1 = score_event_arm(
        prediction(event_id="evt_001", arm="B3", ranked=["legal", "pricing", "labor"], backlash=True),
        TRUTH_1,
        contaminated=False,
    )
    assert b3_1.caught_dominant is False
    assert b3_1.precision_at_3 == 0.0
    assert b3_1.recall == 0.0

    # Arm B3 on the null event: a false alarm.
    b3_2 = score_event_arm(
        prediction(event_id="evt_002", arm="B3", ranked=["privacy", "security", "legal"], backlash=True),
        TRUTH_2,
        contaminated=False,
    )
    assert b3_2.caught_dominant is False
    assert b3_2.false_positive is True

    # Arm B30: catches both present categories on evt_001, and correctly says
    # "none" first (though still padded to k=3) on evt_002.
    b30_1 = score_event_arm(
        prediction(event_id="evt_001", arm="B30", ranked=["privacy", "security", "legal"], backlash=True),
        TRUTH_1,
        contaminated=False,
    )
    assert b30_1.precision_at_3 == pytest.approx(2 / 3)
    assert b30_1.recall == pytest.approx(1.0)

    b30_2 = score_event_arm(
        prediction(event_id="evt_002", arm="B30", ranked=["none", "aesthetic", "labor"], backlash=False),
        TRUTH_2,
        contaminated=False,
    )
    assert b30_2.caught_dominant is True
    assert b30_2.precision_at_3 == pytest.approx(1 / 3)
    assert b30_2.recall == pytest.approx(1.0)
    assert b30_2.false_positive is False


def test_hand_computed_arm_aggregates():
    scores = [
        score_event_arm(
            prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
            TRUTH_1,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_002", arm="A", ranked=["none", "aesthetic", "labor"], backlash=False),
            TRUTH_2,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_001", arm="B3", ranked=["legal", "pricing", "labor"], backlash=True),
            TRUTH_1,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_002", arm="B3", ranked=["privacy", "security", "legal"], backlash=True),
            TRUTH_2,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_001", arm="B30", ranked=["privacy", "security", "legal"], backlash=True),
            TRUTH_1,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_002", arm="B30", ranked=["none", "aesthetic", "labor"], backlash=False),
            TRUTH_2,
            contaminated=False,
        ),
    ]

    a = arm_metrics(scores, "A", resamples=500)
    assert a.caught_dominant_rate.point == pytest.approx(1.0)
    assert a.precision_at_3.point == pytest.approx(1 / 3)
    assert a.recall.point == pytest.approx(0.75)
    assert a.false_positive_rate.point == pytest.approx(0.0)
    assert a.false_positive_n == 1

    b3 = arm_metrics(scores, "B3", resamples=500)
    assert b3.caught_dominant_rate.point == pytest.approx(0.0)
    assert b3.precision_at_3.point == pytest.approx(0.0)
    assert b3.false_positive_rate.point == pytest.approx(1.0)

    b30 = arm_metrics(scores, "B30", resamples=500)
    assert b30.caught_dominant_rate.point == pytest.approx(1.0)
    assert b30.precision_at_3.point == pytest.approx(0.5)
    assert b30.recall.point == pytest.approx(1.0)
    assert b30.false_positive_rate.point == pytest.approx(0.0)

    cmp_ = compare_arms(scores, "B30", "A", resamples=500)
    assert cmp_.caught_dominant.p_value == 1.0  # both arms agree on both events
    assert cmp_.precision_at_3_bootstrap.point == pytest.approx(0.5 - 1 / 3)


def test_arm_metrics_raises_for_unknown_arm():
    with pytest.raises(ScoreError, match="no scored events"):
        arm_metrics([], "Z", resamples=100)


def test_compare_arms_requires_shared_events():
    only_a = [
        score_event_arm(
            prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
            TRUTH_1,
            contaminated=False,
        )
    ]
    with pytest.raises(ScoreError, match="no events scored by both"):
        compare_arms(only_a, "B30", "A", resamples=100)


# --- CSV output ---


def test_write_scores_csv(tmp_path):
    scores = [
        score_event_arm(
            prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
            TRUTH_1,
            contaminated=True,
        )
    ]
    out = tmp_path / "results" / "scores.csv"
    write_scores_csv(scores, out)

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "evt_001"
    assert row["arm"] == "A"
    assert row["contaminated"] == "True"
    assert row["caught_dominant"] == "True"
    assert row["false_positive"] == ""  # not applicable, not "None"
    assert row["predicted_top3"] == "privacy|legal|pricing"
    assert row["actual_present"] == "privacy|security"


def test_report_contains_the_required_mde_sentence():
    scores = [
        score_event_arm(
            prediction(event_id="evt_001", arm="A", ranked=["legal", "pricing", "labor"], backlash=True),
            TRUTH_1,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_002", arm="A", ranked=["privacy", "security", "legal"], backlash=True),
            TRUTH_2,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_001", arm="B30", ranked=["privacy", "security", "legal"], backlash=True),
            TRUTH_1,
            contaminated=False,
        ),
        score_event_arm(
            prediction(event_id="evt_002", arm="B30", ranked=["none", "aesthetic", "labor"], backlash=False),
            TRUTH_2,
            contaminated=False,
        ),
    ]
    report = build_report(run_id="test_run", scores=scores, arms=["A", "B30"], resamples=200)
    assert "this study can only detect large differences between arms" in report
    assert "not that it is zero." in report
    assert "judge_reliability.txt not found" in report


# --- mtime guard (spec 0.2 / 4.5) ---


def _write_prediction(path: Path, pred: ArmPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pred.model_dump_json(indent=2), encoding="utf-8")


def test_guard_mtimes_passes_when_prediction_precedes_ground_truth(tmp_path):
    pred_path = tmp_path / "predictions" / "evt_001__A.json"
    _write_prediction(pred_path, prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    time.sleep(0.05)
    (raw_dir / "evt_001.txt").write_text("reactions", encoding="utf-8")

    guard_mtimes({"evt_001": {"A": pred_path}}, raw_dir)  # must not raise


def test_score_raises_if_ground_truth_predates_predictions(tmp_path):
    """spec 6's required test: score.py raises if ground truth predates predictions."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "evt_001.txt").write_text("reactions", encoding="utf-8")

    time.sleep(0.05)
    pred_path = tmp_path / "predictions" / "evt_001__A.json"
    _write_prediction(pred_path, prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True))

    with pytest.raises(ScoreError, match="prediction is newer than its ground truth"):
        guard_mtimes({"evt_001": {"A": pred_path}}, raw_dir)


def test_guard_mtimes_raises_when_ground_truth_missing(tmp_path):
    pred_path = tmp_path / "predictions" / "evt_001__A.json"
    _write_prediction(pred_path, prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pytest.raises(ScoreError, match="no ground truth at"):
        guard_mtimes({"evt_001": {"A": pred_path}}, raw_dir)


# --- end-to-end score_run / run_score against a sandboxed repo layout ---


@pytest.fixture
def sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copy(repo / "config.yaml", root / "config.yaml")
    (root / "ground_truth" / "raw").mkdir(parents=True)
    (root / "ground_truth" / "labeled").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "runs" / "run_001" / "predictions").mkdir(parents=True)
    return root


def _seed_run(root: Path) -> None:
    # Predictions first, then (after a real time gap) ground truth -- the order
    # the mtime guard exists to enforce (spec 0.2).
    predictions_dir = root / "runs" / "run_001" / "predictions"
    _write_prediction(
        predictions_dir / "evt_001__A.json",
        prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
    )
    _write_prediction(
        predictions_dir / "evt_002__A.json",
        prediction(event_id="evt_002", arm="A", ranked=["none", "aesthetic", "labor"], backlash=False),
    )
    _write_prediction(
        predictions_dir / "evt_001__B30.json",
        prediction(event_id="evt_001", arm="B30", ranked=["privacy", "security", "legal"], backlash=True),
    )
    _write_prediction(
        predictions_dir / "evt_002__B30.json",
        prediction(event_id="evt_002", arm="B30", ranked=["none", "aesthetic", "labor"], backlash=False),
    )

    time.sleep(0.02)

    (root / "ground_truth" / "raw" / "evt_001.txt").write_text("reactions", encoding="utf-8")
    (root / "ground_truth" / "raw" / "evt_002.txt").write_text("reactions", encoding="utf-8")

    (root / "ground_truth" / "labeled" / "evt_001.json").write_text(
        TRUTH_1.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "ground_truth" / "labeled" / "evt_002.json").write_text(
        TRUTH_2.model_dump_json(indent=2), encoding="utf-8"
    )


def test_score_run_end_to_end(sandbox):
    _seed_run(sandbox)
    scores = score_run(repo=sandbox, run_id="run_001")
    assert {(s.event_id, s.arm) for s in scores} == {
        ("evt_001", "A"),
        ("evt_002", "A"),
        ("evt_001", "B30"),
        ("evt_002", "B30"),
    }


def test_score_run_raises_without_labeled_ground_truth(sandbox):
    _write_prediction(
        sandbox / "runs" / "run_001" / "predictions" / "evt_001__A.json",
        prediction(event_id="evt_001", arm="A", ranked=["privacy", "legal", "pricing"], backlash=True),
    )
    time.sleep(0.02)
    (sandbox / "ground_truth" / "raw" / "evt_001.txt").write_text("reactions", encoding="utf-8")
    with pytest.raises(ScoreError, match="no labeled ground truth"):
        score_run(repo=sandbox, run_id="run_001")


def test_run_score_writes_csv_and_report(sandbox):
    _seed_run(sandbox)
    run_score(repo=sandbox, run_id="run_001")

    csv_path = sandbox / "results" / "scores.csv"
    report_path = sandbox / "results" / "report.txt"
    assert csv_path.exists()
    assert report_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4

    report = report_path.read_text(encoding="utf-8")
    assert "run_001" in report
    assert "caught_dominant_rate" in report


def test_contamination_read_from_leakage_csv(sandbox):
    _seed_run(sandbox)
    leakage = sandbox / "runs" / "run_001" / "leakage.csv"
    leakage.write_text(
        "event_id,verdict,response\n"
        "evt_001,CONTAMINATED,describes reaction\n"
        "evt_002,CLEAN,NO KNOWLEDGE\n",
        encoding="utf-8",
    )
    scores = score_run(repo=sandbox, run_id="run_001")
    contaminated = {s.event_id for s in scores if s.contaminated}
    assert contaminated == {"evt_001"}
