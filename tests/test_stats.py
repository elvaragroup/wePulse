from __future__ import annotations

import pytest

from src.stats import mcnemar_exact, mde_paired_binary, paired_bootstrap_ci, wilcoxon_signed_rank


# --- McNemar, against a hand-computed fixture ---


def test_mcnemar_all_concordant_gives_p_one():
    result = mcnemar_exact([True, True, False, False], [True, True, False, False])
    assert result.b == 0
    assert result.c == 0
    assert result.p_value == 1.0


def test_mcnemar_hand_computed_case():
    """4 discordant pairs, all favouring the same arm: b=0, c=4.
    Exact two-sided p = P(X<=0) + P(X>=4) under Binomial(4, 0.5) = 1/16 + 1/16 = 1/8."""
    first = [True, True, True, True, False, False]
    second = [False, False, False, False, False, False]
    # first wins all 4 disagreements; both agree on the last two (False, False).
    result = mcnemar_exact(first, second)
    assert result.b == 4
    assert result.c == 0
    assert result.p_value == pytest.approx(0.125, abs=1e-9)


def test_mcnemar_symmetric_disagreement_is_not_significant():
    first = [True, False, True, False]
    second = [False, True, False, True]
    result = mcnemar_exact(first, second)
    assert result.b == 2
    assert result.c == 2
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_requires_equal_length():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact([True], [True, False])


def test_mcnemar_large_one_sided_disagreement_is_significant():
    first = [True] * 12 + [False] * 8
    second = [False] * 20
    result = mcnemar_exact(first, second)
    assert result.p_value < 0.05


# --- Wilcoxon ---


def test_wilcoxon_all_ties_is_one():
    assert wilcoxon_signed_rank([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 1.0


def test_wilcoxon_detects_a_consistent_shift():
    first = [0.9, 0.85, 0.95, 0.8, 0.88, 0.92]
    second = [0.5, 0.45, 0.55, 0.4, 0.48, 0.52]
    assert wilcoxon_signed_rank(first, second) < 0.05


def test_wilcoxon_requires_equal_length():
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_signed_rank([0.1], [0.1, 0.2])


# --- bootstrap, against a known-answer fixture ---


def test_bootstrap_zero_difference_is_a_point_mass_at_zero():
    first = [0.5, 0.6, 0.7, 0.4]
    ci = paired_bootstrap_ci(first, first, resamples=2000)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high == 0.0


def test_bootstrap_constant_difference_is_a_point_mass():
    """If every pair differs by exactly 0.2, every resample's mean is 0.2 --
    resampling with replacement cannot change a constant."""
    first = [0.7, 0.8, 0.9, 0.6, 0.75]
    second = [0.5, 0.6, 0.7, 0.4, 0.55]
    ci = paired_bootstrap_ci(first, second, resamples=2000)
    assert ci.point == pytest.approx(0.2, abs=1e-9)
    assert ci.low == pytest.approx(0.2, abs=1e-9)
    assert ci.high == pytest.approx(0.2, abs=1e-9)
    assert ci.excludes_zero()


def test_bootstrap_is_reproducible_with_the_same_seed():
    first = [0.9, 0.1, 0.5, 0.7, 0.3, 0.6]
    second = [0.2, 0.8, 0.4, 0.1, 0.6, 0.3]
    a = paired_bootstrap_ci(first, second, resamples=5000, seed=123)
    b = paired_bootstrap_ci(first, second, resamples=5000, seed=123)
    assert a == b


def test_bootstrap_different_seeds_can_differ_slightly():
    first = [0.9, 0.1, 0.5, 0.7, 0.3, 0.6, 0.2, 0.8]
    second = [0.2, 0.8, 0.4, 0.1, 0.6, 0.3, 0.7, 0.5]
    a = paired_bootstrap_ci(first, second, resamples=2000, seed=1)
    b = paired_bootstrap_ci(first, second, resamples=2000, seed=2)
    assert a.point == b.point  # point estimate is not resampled
    assert (a.low, a.high) != (b.low, b.high)


def test_bootstrap_requires_equal_length_and_nonempty():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_ci([], [])


# --- minimum detectable effect ---


def test_mde_shrinks_with_more_events():
    small = mde_paired_binary(10, discordance=0.4)
    large = mde_paired_binary(40, discordance=0.4)
    assert large < small


def test_mde_grows_with_discordance():
    low_disagreement = mde_paired_binary(25, discordance=0.1)
    high_disagreement = mde_paired_binary(25, discordance=0.8)
    assert high_disagreement > low_disagreement


def test_mde_zero_discordance_falls_back_to_worst_case():
    """No disagreement observed yet -- report the most pessimistic MDE (psi=1)
    rather than a flatteringly small number computed from psi=0."""
    assert mde_paired_binary(25, discordance=0.0) == mde_paired_binary(25, discordance=1.0)


def test_mde_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n must be positive"):
        mde_paired_binary(0, discordance=0.5)
    with pytest.raises(ValueError, match="proportion"):
        mde_paired_binary(10, discordance=1.5)
