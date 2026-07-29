"""Turning per-persona reactions into one ranked prediction per arm.

Pure functions, no I/O. This is the arithmetic that decides every headline
number, so it is kept small enough to check by hand and is tested against the
spec's worked example (3 of 8 personas naming privacy -> 0.375).
"""

from __future__ import annotations

from src.models import K, ArmPrediction, Category, NaiveBaselineOutput, PersonaReaction
from src.taxonomy import Taxonomy


def score_categories(
    reactions: list[PersonaReaction],
    *,
    arm_size: int,
) -> dict[str, float]:
    """score(c) = sum of naming personas' intensities / arm size (spec 4.3).

    With every naming persona at intensity 1.0 this reduces to the spec's
    "(# personas naming it) / (# personas in arm)". Personas that ignore
    contribute nothing but still count in the denominator -- that is what makes a
    quiet announcement score low rather than merely score on fewer votes.
    """
    if arm_size <= 0:
        raise ValueError("arm_size must be positive")

    totals: dict[str, float] = {}
    for reaction in reactions:
        for category in reaction.categories:
            totals[category] = totals.get(category, 0.0) + reaction.intensity
    return {category: total / arm_size for category, total in totals.items()}


def rank_categories(
    scores: dict[str, float],
    taxonomy: Taxonomy,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Descending by score, ties broken by taxonomy file order."""
    candidates = [c for c in scores if c not in exclude]
    return sorted(candidates, key=lambda c: (-scores[c], taxonomy.rank(c)))


def pad_to_k(
    ranked: list[str],
    taxonomy: Taxonomy,
    *,
    k: int = K,
    exclude: frozenset[str] = frozenset(),
) -> list[Category]:
    """Every arm emits exactly k guesses so precision is comparable (spec 0.3).

    Fillers are drawn in taxonomy file order. They score zero and are unlikely to
    match, and `precision_at_3` divides by k regardless, so padding cannot inflate
    a metric -- it only prevents a short arm from looking artificially precise.
    """
    out = list(ranked[:k])
    if len(out) < k:
        for category in taxonomy.ids:
            if len(out) == k:
                break
            if category not in out and category not in exclude:
                out.append(category)
    if len(out) != k:
        raise ValueError(f"taxonomy too small to pad to k={k}")
    return out  # type: ignore[return-value]


def aggregate_persona_arm(
    *,
    arm: str,
    event_id: str,
    reactions: list[PersonaReaction],
    arm_size: int,
    taxonomy: Taxonomy,
    threshold: float,
    k: int = K,
) -> ArmPrediction:
    scores = score_categories(reactions, arm_size=arm_size)
    backlash_predicted = any(score > threshold for score in scores.values())

    if backlash_predicted:
        ranked = rank_categories(scores, taxonomy, exclude=frozenset({"none"}))
        final = pad_to_k(ranked, taxonomy, k=k, exclude=frozenset({"none"}))
    else:
        # Nothing cleared the bar: the arm's first guess is that nothing happened.
        rest = rank_categories(scores, taxonomy, exclude=frozenset({"none"}))
        final = pad_to_k(["none", *rest], taxonomy, k=k)

    return ArmPrediction(
        arm=arm,  # type: ignore[arg-type]
        event_id=event_id,
        ranked_categories=final,
        scores={c: round(s, 6) for c, s in sorted(scores.items())},
        backlash_predicted=backlash_predicted,
    )


def aggregate_naive_arm(
    *,
    event_id: str,
    output: NaiveBaselineOutput,
    taxonomy: Taxonomy,
    k: int = K,
) -> ArmPrediction:
    """Arm A: take the model's own ranking, truncate to k (spec 4.3).

    `backlash_predicted` is not specified for arm A. The natural reading of a
    ranked list is that its first entry is the arm's actual claim, so arm A
    predicts backlash unless it ranked `none` first.
    """
    seen: list[str] = []
    for category in output.risks:
        if category not in seen:
            seen.append(category)

    backlash_predicted = bool(seen) and seen[0] != "none"
    exclude = frozenset({"none"}) if backlash_predicted else frozenset()
    ranked = [c for c in seen if c not in exclude]

    return ArmPrediction(
        arm="A",
        event_id=event_id,
        ranked_categories=pad_to_k(ranked, taxonomy, k=k, exclude=exclude),
        scores={},
        backlash_predicted=backlash_predicted,
    )


def human_arm(
    *,
    event_id: str,
    ranked: list[str],
    taxonomy: Taxonomy,
    k: int = K,
) -> ArmPrediction:
    """Arm C. Same first-entry reading of `backlash_predicted` as arm A."""
    backlash_predicted = bool(ranked) and ranked[0] != "none"
    exclude = frozenset({"none"}) if backlash_predicted else frozenset()
    return ArmPrediction(
        arm="C",
        event_id=event_id,
        ranked_categories=pad_to_k(
            [c for c in ranked if c not in exclude], taxonomy, k=k, exclude=exclude
        ),
        scores={},
        backlash_predicted=backlash_predicted,
    )
