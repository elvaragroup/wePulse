from __future__ import annotations

import pytest
from pathlib import Path

from src.dashboard import ReactionRow
from src.models import ArmPrediction, GroundTruthLabel
from src.personas import Persona
from src.taxonomy import Taxonomy, TaxonomyEntry
from web.backend.transform import (
    CategoryDiff,
    CategoryScore,
    QuoteItem,
    category_label,
    compare_predictions,
    reaction_mix_summary,
    select_curated_quotes,
    to_ground_truth_display,
    top_categories,
)
from web.backend.runs import WebDataError, get_run_dir


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy(
        entries=(
            TaxonomyEntry(id="privacy", label="Privacy & data", description="d"),
            TaxonomyEntry(id="overclaim", label="Technical overclaim", description="d"),
            TaxonomyEntry(id="security", label="Security", description="d"),
            TaxonomyEntry(id="hypocrisy", label="Hypocrisy & receipts", description="d"),
            TaxonomyEntry(id="labor", label="Labor", description="d"),
            TaxonomyEntry(id="environment", label="Environment", description="d"),
            TaxonomyEntry(id="none", label="No meaningful backlash", description="d"),
        )
    )


def make_persona(persona_id: str, archetype: str, platform: str = "x") -> Persona:
    return Persona(
        id=persona_id,
        name=f"persona_{persona_id}",
        axis="privacy",
        archetype=archetype,
        baseline_skepticism=0.5,
        platform=platform,
        body="body",
        source_path=Path(f"/tmp/{persona_id}.md"),
    )


# --- category_label / top_categories ---


def test_category_label_known_id(taxonomy):
    assert category_label(taxonomy, "privacy") == "Privacy & data"


def test_top_categories_zips_scores(taxonomy):
    pred = ArmPrediction(
        arm="B30",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "security"],
        scores={"privacy": 0.395, "overclaim": 0.24, "security": 0.023},
        backlash_predicted=True,
    )
    result = top_categories(pred, taxonomy)
    assert [c.id for c in result] == ["privacy", "overclaim", "security"]
    assert result[0].label == "Privacy & data"
    assert result[0].confidence == pytest.approx(0.395)


def test_top_categories_naive_arm_has_none_confidence(taxonomy):
    pred = ArmPrediction(
        arm="A",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "hypocrisy"],
        scores={},
        backlash_predicted=True,
    )
    result = top_categories(pred, taxonomy)
    assert all(c.confidence is None for c in result)


# --- reaction_mix_summary ---


def test_reaction_mix_summary_evt_001_real_counts():
    # Real counts from runs/20260730T002314.423Z_2ec30ae6/raw/B30/evt_001
    counts = {"ignore": 11, "mild_concern": 11, "criticize": 8, "outrage": 0}
    summary = reaction_mix_summary(counts)
    assert summary == "8 personas objected and 11 personas expressed mild concern and 11 personas had no reaction."


def test_reaction_mix_summary_singular_counts():
    counts = {"ignore": 1, "mild_concern": 0, "criticize": 1, "outrage": 1}
    summary = reaction_mix_summary(counts)
    assert summary == "2 personas objected and 1 persona had no reaction."


def test_reaction_mix_summary_all_zero_except_ignore():
    counts = {"ignore": 30, "mild_concern": 0, "criticize": 0, "outrage": 0}
    summary = reaction_mix_summary(counts)
    assert "30" in summary
    assert "objected" not in summary and "criticized" not in summary
    assert "mild concern" not in summary.lower()


# --- select_curated_quotes ---


def test_select_curated_quotes_diversifies_by_archetype():
    personas_by_id = {
        "001": make_persona("001", "critic"),
        "002": make_persona("002", "sympathetic"),
        "003": make_persona("003", "critic"),
    }
    reacted = [
        ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.9, "q1"),
        ReactionRow("003", "p3", "critic", "criticize", ("privacy",), 0.8, "q3"),
        ReactionRow("002", "p2", "sympathetic", "mild_concern", ("privacy",), 0.5, "q2"),
    ]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    # all 3 fit under the limit; archetype-first pass takes 001 (critic) and 002
    # (sympathetic) before backfilling with 003 (second critic) -- final order
    # re-sorted by intensity descending regardless of pass order.
    assert [q.quote for q in result] == ["q1", "q3", "q2"]
    assert result[0].archetype_label == "Critic"


def test_select_curated_quotes_respects_limit():
    personas_by_id = {str(i).zfill(3): make_persona(str(i).zfill(3), "critic") for i in range(1, 11)}
    reacted = [
        ReactionRow(str(i).zfill(3), f"p{i}", "critic", "criticize", ("privacy",), 1.0 - i * 0.01, f"q{i}")
        for i in range(1, 11)
    ]
    result = select_curated_quotes(reacted, personas_by_id, limit=5)
    assert len(result) == 5
    assert result[0].quote == "q1"  # highest intensity


def test_select_curated_quotes_fewer_than_limit():
    personas_by_id = {"001": make_persona("001", "critic")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert len(result) == 1


def test_select_curated_quotes_looks_up_platform():
    personas_by_id = {"001": make_persona("001", "critic", platform="reddit")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert result[0].platform == "reddit"


def test_select_curated_quotes_never_leaks_persona_id():
    personas_by_id = {"001": make_persona("001", "critic")}
    reacted = [ReactionRow("001", "p1", "critic", "criticize", ("privacy",), 0.7, "q1")]
    result = select_curated_quotes(reacted, personas_by_id, limit=8)
    assert not hasattr(result[0], "persona_id")
    assert not hasattr(result[0], "persona_name")


# --- compare_predictions ---


def test_compare_predictions_evt_001_real_diff(taxonomy):
    # Real evt_001 predictions from runs/20260730T002314.423Z_2ec30ae6
    naive = ArmPrediction(
        arm="A",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "hypocrisy"],
        scores={},
        backlash_predicted=True,
    )
    ensemble = ArmPrediction(
        arm="B30",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "security"],
        scores={"privacy": 0.395, "overclaim": 0.24, "security": 0.023333},
        backlash_predicted=True,
    )
    diff = compare_predictions(naive, ensemble, taxonomy)
    assert {c.id for c in diff.agreed} == {"privacy", "overclaim"}
    assert [c.id for c in diff.ensemble_only] == ["security"]
    assert [c.id for c in diff.naive_only] == ["hypocrisy"]
    assert diff.backlash_agreement is True
    # agreed categories carry the ensemble's real confidence, not None
    agreed_privacy = next(c for c in diff.agreed if c.id == "privacy")
    assert agreed_privacy.confidence == pytest.approx(0.395)


def test_compare_predictions_backlash_disagreement(taxonomy):
    naive = ArmPrediction(
        arm="A",
        event_id="evt_002",
        ranked_categories=["none", "labor", "environment"],
        scores={},
        backlash_predicted=False,
    )
    ensemble = ArmPrediction(
        arm="B30",
        event_id="evt_002",
        ranked_categories=["labor", "none", "environment"],
        scores={"labor": 0.3},
        backlash_predicted=True,
    )
    diff = compare_predictions(naive, ensemble, taxonomy)
    assert diff.backlash_agreement is False


# --- to_ground_truth_display ---


def test_to_ground_truth_display_none_stays_none(taxonomy):
    # The universal case today: no labeled ground truth exists for any event
    # yet (see ground_truth/README.md). Must pass through as None, never a
    # fabricated result.
    assert to_ground_truth_display(None, taxonomy) is None


def test_to_ground_truth_display_resolves_real_label(taxonomy):
    label = GroundTruthLabel(
        event_id="evt_001",
        dominant_category="privacy",
        present_categories=["privacy", "overclaim"],
        backlash_occurred=True,
        judge_confidence=0.82,
    )
    display = to_ground_truth_display(label, taxonomy)
    assert display is not None
    assert display.dominant_category.id == "privacy"
    assert display.dominant_category.label == "Privacy & data"
    assert display.dominant_category.confidence == pytest.approx(0.82)
    assert [c.id for c in display.present_categories] == ["privacy", "overclaim"]
    assert display.backlash_occurred is True
    assert display.judge_confidence == pytest.approx(0.82)
    assert "Privacy & data" in display.summary
    assert "meaningful backlash" in display.summary


def test_to_ground_truth_display_no_backlash_summary_wording(taxonomy):
    label = GroundTruthLabel(
        event_id="evt_002",
        dominant_category="none",
        present_categories=["none"],
        backlash_occurred=False,
        judge_confidence=0.9,
    )
    display = to_ground_truth_display(label, taxonomy)
    assert display.backlash_occurred is False
    assert "no significant backlash" in display.summary


# --- get_run_dir ---


@pytest.mark.parametrize("bad_manifest", ["[1, 2, 3]", '"just a string"'])
def test_get_run_dir_raises_on_non_dict_manifest(tmp_path, bad_manifest):
    run_subdir = tmp_path / "runs" / "20260101T000000.000Z_deadbeef"
    run_subdir.mkdir(parents=True)
    (run_subdir / "manifest.json").write_text(bad_manifest, encoding="utf-8")

    with pytest.raises(WebDataError):
        get_run_dir(tmp_path)
