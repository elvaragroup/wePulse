from __future__ import annotations

import pytest

from src.diagnostics_metrics import (
    distribution_match_partial,
    register_variance,
    specificity,
)


# --- register_variance ---


def test_register_variance_word_count_stats():
    quotes = ["one two three four", "five six"]
    result = register_variance(quotes)
    assert result.n == 2
    assert result.word_count_mean == pytest.approx(3.0)
    assert result.word_count_stdev == pytest.approx(1.4142135623730951)
    assert result.word_count_min == 2
    assert result.word_count_max == 4


def test_register_variance_sentence_count_stdev():
    quotes = ["One. Two. Three.", "No punctuation here"]
    result = register_variance(quotes)
    # sentence counts: [3, 1] -- three ". "/". "-at-end matches vs the
    # no-punctuation fallback of 1
    assert result.sentence_count_stdev == pytest.approx(1.4142135623730951)


def test_register_variance_exclamation_and_dash_rate():
    quotes = ["Wait!", "No punctuation here", "Great—truly great", "Another one!"]
    result = register_variance(quotes)
    assert result.exclamation_rate == pytest.approx(0.5)
    assert result.em_dash_rate == pytest.approx(0.25)


def test_register_variance_rejects_empty():
    with pytest.raises(ValueError, match="zero quotes"):
        register_variance([])


# --- specificity ---


def test_specificity_no_false_positives():
    predicted = {"evt_001": True, "evt_002": False, "evt_003": False}
    result = specificity(predicted, {"evt_002", "evt_003"})
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.n_null_events == 2
    assert result.n_false_positive == 0


def test_specificity_one_false_positive():
    predicted = {"evt_001": True, "evt_002": True, "evt_003": False}
    result = specificity(predicted, {"evt_002", "evt_003"})
    assert result.false_positive_rate == pytest.approx(0.5)
    assert result.n_false_positive == 1


def test_specificity_missing_event_defaults_to_not_flagged():
    result = specificity({}, {"evt_005"})
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.n_false_positive == 0


def test_specificity_rejects_empty_null_set():
    with pytest.raises(ValueError, match="no null events"):
        specificity({"evt_001": True}, set())


# --- distribution_match_partial ---


def test_distribution_match_identical_mixes_is_zero():
    result = distribution_match_partial({"privacy": 0.6, "none": 0.4}, {"privacy": 0.6, "none": 0.4})
    assert result.status == "measured_partial"
    assert result.total_variation_distance == pytest.approx(0.0)


def test_distribution_match_disjoint_mixes_is_one():
    result = distribution_match_partial({"privacy": 1.0}, {"none": 1.0})
    assert result.total_variation_distance == pytest.approx(1.0)
    assert result.categories_compared == 2


def test_distribution_match_both_empty():
    result = distribution_match_partial({}, {})
    assert result.total_variation_distance is None
    assert result.categories_compared == 0
