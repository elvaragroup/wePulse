"""Paired statistics for arm-vs-arm comparisons.

Every comparison in this study is paired: each arm predicts the same events, so
the unit of resampling is the event, not the prediction. Unpaired tests would
throw away that structure and overstate the uncertainty.

The bootstrap uses an explicit seed so a report is reproducible. That seed is
ours, not the API's -- it does not make model sampling deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

BOOTSTRAP_SEED = 20260728
ALPHA = 0.05
POWER = 0.80


@dataclass(frozen=True)
class McNemarResult:
    b: int  # first arm right, second wrong
    c: int  # first arm wrong, second right
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b + self.c


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    resamples: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def mcnemar_exact(first: list[bool], second: list[bool]) -> McNemarResult:
    """Exact (binomial) McNemar test on paired binary outcomes.

    The exact form is used rather than the chi-square approximation because the
    discordant count here will be small -- with n around 25 events, the
    approximation is not trustworthy.
    """
    if len(first) != len(second):
        raise ValueError("paired inputs must be the same length")

    b = sum(1 for x, y in zip(first, second) if x and not y)
    c = sum(1 for x, y in zip(first, second) if y and not x)

    if b + c == 0:
        # The arms never disagreed. No evidence of a difference either way.
        return McNemarResult(b=b, c=c, p_value=1.0)

    result = stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
    return McNemarResult(b=b, c=c, p_value=float(result.pvalue))


def wilcoxon_signed_rank(first: list[float], second: list[float]) -> float:
    """Two-sided p-value. Returns 1.0 when every pair is tied, which scipy
    treats as an error but which simply means no evidence of a difference."""
    if len(first) != len(second):
        raise ValueError("paired inputs must be the same length")

    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if not np.any(differences):
        return 1.0
    return float(stats.wilcoxon(first, second, zero_method="wilcox").pvalue)


def paired_bootstrap_ci(
    first: list[float],
    second: list[float],
    *,
    resamples: int = 10_000,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Percentile CI for the mean paired difference (first - second).

    Events are resampled together across both arms, preserving the pairing.
    """
    if len(first) != len(second):
        raise ValueError("paired inputs must be the same length")
    if not first:
        raise ValueError("cannot bootstrap an empty sample")

    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(differences)

    indices = rng.integers(0, n, size=(resamples, n))
    means = differences[indices].mean(axis=1)

    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        point=float(differences.mean()),
        low=float(low),
        high=float(high),
        resamples=resamples,
    )


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Unweighted Cohen's kappa for two raters over the same categorical items.

    Used by check_judge.py to compare the judge's labels against a human's on the
    same sample (spec 4.6). Implemented by hand rather than pulled from a
    dependency: scipy has no kappa function and this project's only other
    dependencies are anthropic/pydantic/pyyaml/numpy.

    kappa = (po - pe) / (1 - pe), where po is observed agreement and pe is the
    agreement expected by chance from each rater's own marginal label
    frequencies. kappa == 1.0 when the raters agree on every item and both
    marginals are non-degenerate; when they agree on every item using only one
    category (pe == 1), agreement carries no information, so kappa is defined
    as 0.0 rather than dividing by zero.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("paired inputs must be the same length")
    if not rater_a:
        raise ValueError("cannot compute kappa over an empty sample")

    n = len(rater_a)
    categories = sorted(set(rater_a) | set(rater_b))

    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    freq_a = {c: rater_a.count(c) / n for c in categories}
    freq_b = {c: rater_b.count(c) / n for c in categories}
    pe = sum(freq_a[c] * freq_b[c] for c in categories)

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def mde_paired_binary(
    n: int,
    discordance: float,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
) -> float:
    """Smallest difference in rates detectable at this n, as a proportion.

    Normal approximation for McNemar:  delta = (z_alpha/2 + z_beta) * sqrt(psi/n),
    where psi is the proportion of events on which the two arms disagree.
    Discordance is what actually carries the information in a paired binary test:
    two arms that agree everywhere provide no evidence regardless of n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= discordance <= 1.0:
        raise ValueError("discordance must be a proportion")

    # With no observed disagreement there is nothing to estimate psi from. Fall
    # back to the most pessimistic assumption rather than reporting a flatteringly
    # small number.
    psi = discordance if discordance > 0 else 1.0

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * np.sqrt(psi / n))
