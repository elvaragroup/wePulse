from __future__ import annotations

import pytest

from src.diagnostics_metrics import (
    distribution_match_partial,
    homogeneity,
    redundancy,
    register_variance,
    span_dispersion,
    specificity,
    stability,
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


# --- homogeneity ---


def test_homogeneity_mean_pairwise_cosine():
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    result = homogeneity(embeddings)
    # pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=0.0 -> mean 1/3
    assert result.mean_pairwise_cosine == pytest.approx(1 / 3)
    assert result.n_quotes == 3
    assert result.n_pairs == 3


def test_homogeneity_rejects_fewer_than_two():
    with pytest.raises(ValueError, match="at least 2"):
        homogeneity([[1.0, 0.0]])


# --- redundancy ---


def test_redundancy_fewer_than_min_cluster_size_returns_one_cluster_per_point():
    result = redundancy([[1.0, 0.0]], n_reacting_personas=1, min_cluster_size=2)
    assert result.n_clusters == 1
    assert result.ratio == pytest.approx(1.0)
    assert result.n_noise == 0


def test_redundancy_fewer_than_min_cluster_size_computes_real_ratio():
    # Test case where len(embeddings) != n_reacting_personas in the fallback branch
    result = redundancy([[1.0, 0.0]], n_reacting_personas=2, min_cluster_size=2)
    assert result.n_clusters == 1
    assert result.ratio == pytest.approx(0.5)  # 1 embedding / 2 personas
    assert result.n_noise == 0


def test_redundancy_finds_well_separated_clusters():
    embeddings = [
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.01],
        [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.01],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.01],
    ]
    result = redundancy(embeddings, n_reacting_personas=6, min_cluster_size=2)
    assert result.n_clusters == 3
    assert result.n_noise == 0
    assert result.ratio == pytest.approx(0.5)


# --- span_dispersion ---


def test_span_dispersion_all_same_span_has_zero_stdev():
    result = span_dispersion([(0, 10), (0, 10), (0, 10)], announcement_len=100)
    assert result.normalized_position_stdev == pytest.approx(0.0)
    assert result.distinct_span_fraction == pytest.approx(1 / 3)


def test_span_dispersion_spread_across_document():
    result = span_dispersion([(0, 10), (45, 55), (90, 100)], announcement_len=100)
    # midpoints normalized: 0.05, 0.50, 0.95 -> rounded*1000: 50, 500, 950
    # stdev([50, 500, 950]) = 450.0 -> normalized_position_stdev = 0.45
    assert result.normalized_position_stdev == pytest.approx(0.45)
    assert result.distinct_span_fraction == pytest.approx(1.0)


def test_span_dispersion_rejects_empty_citations():
    with pytest.raises(ValueError, match="no citations"):
        span_dispersion([], announcement_len=100)


def test_span_dispersion_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="announcement_len"):
        span_dispersion([(0, 10)], announcement_len=0)


# --- stability ---


def test_stability_perfect_agreement():
    samples = {("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5}
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(1.0)
    assert result.n_pairs_sampled == 1
    assert result.n_reruns == 5


def test_stability_disagreement_lowers_rate():
    samples = {
        ("001", "evt_001"): [
            ("criticize", frozenset({"privacy"})),
            ("ignore", frozenset()),
            ("criticize", frozenset({"privacy"})),
            ("criticize", frozenset({"privacy"})),
            ("criticize", frozenset({"privacy"})),
        ]
    }
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(0.0)


def test_stability_mixed_pairs():
    samples = {
        ("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5,
        ("002", "evt_002"): [("ignore", frozenset())] * 4 + [("criticize", frozenset({"labor"}))],
    }
    result = stability(samples)
    assert result.category_agreement_rate == pytest.approx(0.5)


def test_stability_rejects_empty_samples():
    with pytest.raises(ValueError, match="no samples"):
        stability({})


def test_stability_rejects_mismatched_rerun_counts():
    samples = {
        ("001", "evt_001"): [("criticize", frozenset({"privacy"}))] * 5,
        ("002", "evt_002"): [("ignore", frozenset())] * 3,
    }
    with pytest.raises(ValueError, match="same rerun count"):
        stability(samples)
