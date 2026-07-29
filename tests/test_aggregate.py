from __future__ import annotations

import pytest

from src.aggregate import (
    aggregate_naive_arm,
    aggregate_persona_arm,
    human_arm,
    pad_to_k,
    rank_categories,
    score_categories,
)
from src.models import NaiveBaselineOutput, PersonaReaction

THRESHOLD = 0.25


def names(*categories, intensity=1.0):
    return PersonaReaction(
        reaction="criticize",
        categories=list(categories),
        intensity=intensity,
        quote="q",
        reasoning="r",
    )


def ignores():
    return PersonaReaction(
        reaction="ignore", categories=[], intensity=0.0, quote=None, reasoning="not my beat"
    )


# --- the spec's worked example (spec 6) ---


def test_three_of_eight_naming_privacy_scores_0_375():
    reactions = [names("privacy") for _ in range(3)] + [ignores() for _ in range(5)]
    scores = score_categories(reactions, arm_size=8)
    assert scores["privacy"] == pytest.approx(0.375)


def test_below_threshold_yields_none(taxonomy):
    """1 of 8 = 0.125, under the 0.25 bar, so the arm predicts nothing happened."""
    reactions = [names("privacy")] + [ignores() for _ in range(7)]
    prediction = aggregate_persona_arm(
        arm="B8",
        event_id="evt_001",
        reactions=reactions,
        arm_size=8,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.backlash_predicted is False
    assert prediction.ranked_categories[0] == "none"


def test_above_threshold_predicts_backlash(taxonomy):
    reactions = [names("privacy") for _ in range(3)] + [ignores() for _ in range(5)]
    prediction = aggregate_persona_arm(
        arm="B8",
        event_id="evt_001",
        reactions=reactions,
        arm_size=8,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.backlash_predicted is True
    assert prediction.ranked_categories[0] == "privacy"
    assert "none" not in prediction.ranked_categories


def test_threshold_is_strictly_greater_than(taxonomy):
    """2 of 8 = exactly 0.25, which does not clear a 0.25 bar."""
    reactions = [names("privacy") for _ in range(2)] + [ignores() for _ in range(6)]
    prediction = aggregate_persona_arm(
        arm="B8",
        event_id="evt_001",
        reactions=reactions,
        arm_size=8,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.backlash_predicted is False


# --- intensity weighting ---


def test_intensity_weights_the_score():
    reactions = [names("privacy", intensity=0.5) for _ in range(4)]
    assert score_categories(reactions, arm_size=8)["privacy"] == pytest.approx(0.25)


def test_ignoring_personas_stay_in_the_denominator():
    """Otherwise one loud persona in a quiet arm would score 1.0."""
    reactions = [names("privacy")] + [ignores() for _ in range(29)]
    assert score_categories(reactions, arm_size=30)["privacy"] == pytest.approx(1 / 30)


def test_one_persona_naming_several_categories():
    scores = score_categories([names("privacy", "security", intensity=0.6)], arm_size=2)
    assert scores == pytest.approx({"privacy": 0.3, "security": 0.3})


def test_zero_arm_size_raises():
    with pytest.raises(ValueError, match="arm_size must be positive"):
        score_categories([], arm_size=0)


# --- truncation always emits exactly k=3 (spec 6) ---


@pytest.mark.parametrize("distinct", [0, 1, 2, 3, 5, 13])
def test_truncation_always_emits_three(taxonomy, distinct):
    pool = [c for c in taxonomy.ids if c != "none"]
    reactions = [names(*pool[:distinct])] if distinct else [ignores()]
    prediction = aggregate_persona_arm(
        arm="B30",
        event_id="evt_001",
        reactions=reactions,
        arm_size=1,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert len(prediction.ranked_categories) == 3
    assert len(set(prediction.ranked_categories)) == 3


def test_padding_uses_taxonomy_order(taxonomy):
    prediction = aggregate_persona_arm(
        arm="B3",
        event_id="evt_001",
        reactions=[names("security")],
        arm_size=1,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.ranked_categories == ["security", "privacy", "labor"]


def test_none_case_pads_after_none(taxonomy):
    prediction = aggregate_persona_arm(
        arm="B8",
        event_id="evt_001",
        reactions=[names("security")] + [ignores() for _ in range(7)],
        arm_size=8,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.ranked_categories[0] == "none"
    assert prediction.ranked_categories[1] == "security"


def test_pad_to_k_raises_if_taxonomy_too_small(taxonomy):
    with pytest.raises(ValueError, match="too small to pad"):
        pad_to_k([], taxonomy, k=len(taxonomy.ids) + 1)


# --- ranking and ties ---


def test_ranking_is_descending_by_score(taxonomy):
    scores = {"pricing": 0.1, "privacy": 0.9, "legal": 0.5}
    assert rank_categories(scores, taxonomy) == ["privacy", "legal", "pricing"]


def test_ties_break_by_taxonomy_order(taxonomy):
    """Deterministic tie-breaks matter: without one, two runs on identical cached
    responses could emit different top-3 lists."""
    scores = {"security": 0.4, "privacy": 0.4, "labor": 0.4}
    assert rank_categories(scores, taxonomy) == ["privacy", "labor", "security"]


def test_scores_are_recorded_for_audit(taxonomy):
    prediction = aggregate_persona_arm(
        arm="B8",
        event_id="evt_001",
        reactions=[names("privacy"), names("legal", intensity=0.5)],
        arm_size=8,
        taxonomy=taxonomy,
        threshold=THRESHOLD,
    )
    assert prediction.scores == pytest.approx({"privacy": 0.125, "legal": 0.0625})


# --- arm A ---


def test_naive_arm_takes_its_own_ranking(taxonomy):
    output = NaiveBaselineOutput(
        risks=["privacy", "legal", "pricing", "security", "labor"], rationale={}
    )
    prediction = aggregate_naive_arm(event_id="evt_001", output=output, taxonomy=taxonomy)
    assert prediction.arm == "A"
    assert prediction.ranked_categories == ["privacy", "legal", "pricing"]
    assert prediction.backlash_predicted is True


def test_naive_arm_ranking_none_first_predicts_no_backlash(taxonomy):
    output = NaiveBaselineOutput(risks=["none", "aesthetic"], rationale={})
    prediction = aggregate_naive_arm(event_id="evt_001", output=output, taxonomy=taxonomy)
    assert prediction.backlash_predicted is False
    assert prediction.ranked_categories[0] == "none"


def test_naive_arm_drops_duplicates_and_still_emits_three(taxonomy):
    output = NaiveBaselineOutput(risks=["privacy", "privacy", "privacy"], rationale={})
    prediction = aggregate_naive_arm(event_id="evt_001", output=output, taxonomy=taxonomy)
    assert len(set(prediction.ranked_categories)) == 3
    assert prediction.ranked_categories[0] == "privacy"


def test_naive_arm_drops_none_when_it_is_not_first(taxonomy):
    """A ranking of [privacy, none, legal] claims backlash; carrying 'none' along
    would let one prediction have it both ways."""
    output = NaiveBaselineOutput(risks=["privacy", "none", "legal"], rationale={})
    prediction = aggregate_naive_arm(event_id="evt_001", output=output, taxonomy=taxonomy)
    assert prediction.ranked_categories == ["privacy", "legal", "labor"]


# --- arm C ---


def test_human_arm(taxonomy):
    prediction = human_arm(event_id="evt_001", ranked=["pricing", "hypocrisy", "aesthetic"], taxonomy=taxonomy)
    assert prediction.arm == "C"
    assert prediction.ranked_categories == ["pricing", "hypocrisy", "aesthetic"]
    assert prediction.backlash_predicted is True


def test_human_arm_predicting_nothing(taxonomy):
    prediction = human_arm(event_id="evt_001", ranked=["none"], taxonomy=taxonomy)
    assert prediction.backlash_predicted is False
    assert prediction.ranked_categories[0] == "none"
    assert len(prediction.ranked_categories) == 3
