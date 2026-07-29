from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import ArmPrediction, GroundTruthLabel, NaiveBaselineOutput, PersonaReaction


def reaction(**overrides):
    base = {
        "reaction": "criticize",
        "categories": ["privacy"],
        "intensity": 0.8,
        "quote": "This is an opt-out, not consent.",
        "reasoning": "Defaults users in.",
    }
    return PersonaReaction(**{**base, **overrides})


def test_valid_reaction():
    assert reaction().categories == ["privacy"]


def test_ignore_requires_empty_categories_and_null_quote():
    assert reaction(reaction="ignore", categories=[], quote=None, intensity=0.0).quote is None


def test_ignore_with_categories_is_rejected():
    with pytest.raises(ValidationError, match="must have empty categories"):
        reaction(reaction="ignore", categories=["privacy"], quote=None)


def test_ignore_with_a_quote_is_rejected():
    with pytest.raises(ValidationError, match="must have quote=None"):
        reaction(reaction="ignore", categories=[], quote="I do have thoughts actually")


def test_non_ignore_without_categories_is_rejected():
    with pytest.raises(ValidationError, match="must name at least one category"):
        reaction(categories=[])


def test_non_ignore_without_quote_is_rejected():
    with pytest.raises(ValidationError, match="must have a quote"):
        reaction(quote=None)


def test_intensity_is_bounded():
    with pytest.raises(ValidationError):
        reaction(intensity=1.5)


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        reaction(categories=["vibes"])


def test_naive_output_caps_at_five_risks():
    NaiveBaselineOutput(risks=["privacy"] * 5, rationale={})
    with pytest.raises(ValidationError):
        NaiveBaselineOutput(risks=["privacy"] * 6, rationale={})


def prediction(**overrides):
    base = {
        "arm": "B30",
        "event_id": "evt_001",
        "ranked_categories": ["privacy", "security", "legal"],
        "scores": {"privacy": 0.6},
        "backlash_predicted": True,
    }
    return ArmPrediction(**{**base, **overrides})


def test_prediction_requires_exactly_three():
    assert len(prediction().ranked_categories) == 3
    with pytest.raises(ValidationError, match="exactly 3"):
        prediction(ranked_categories=["privacy", "security"])
    with pytest.raises(ValidationError, match="exactly 3"):
        prediction(ranked_categories=["privacy", "security", "legal", "pricing"])


def test_prediction_rejects_duplicates():
    with pytest.raises(ValidationError, match="must be distinct"):
        prediction(ranked_categories=["privacy", "privacy", "legal"])


def label(**overrides):
    base = {
        "event_id": "evt_001",
        "dominant_category": "privacy",
        "present_categories": ["privacy", "security"],
        "backlash_occurred": True,
        "judge_confidence": 0.9,
    }
    return GroundTruthLabel(**{**base, **overrides})


def test_dominant_must_be_present():
    with pytest.raises(ValidationError, match="missing from present_categories"):
        label(dominant_category="legal")


def test_backlash_flag_and_dominant_none_must_agree():
    label(dominant_category="none", present_categories=["none"], backlash_occurred=False)
    with pytest.raises(ValidationError, match="inconsistent"):
        label(dominant_category="none", present_categories=["none"], backlash_occurred=True)
    with pytest.raises(ValidationError, match="inconsistent"):
        label(backlash_occurred=False)
