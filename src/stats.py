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
