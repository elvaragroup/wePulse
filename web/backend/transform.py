"""Pure functions to transform loaded data into client-friendly display objects.

No filesystem access, no imports from runs.py or service.py — these are pure
functions only.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.dashboard import ReactionRow
from src.models import ArmPrediction
from src.personas import Persona
from src.taxonomy import Taxonomy

ARCHETYPE_LABELS: dict[str, str] = {
    "critic": "Critic",
    "neutral": "Neutral observer",
    "sympathetic": "Sympathetic voice",
    "insider": "Industry insider",
}


@dataclass(frozen=True)
class CategoryScore:
    """One category with its confidence score."""

    id: str
    label: str
    confidence: float | None


def category_label(taxonomy: Taxonomy, category_id: str) -> str:
    """Look up the label for a category id from the taxonomy.

    Raises KeyError or ValueError if the id isn't found.
    """
    for entry in taxonomy.entries:
        if entry.id == category_id:
            return entry.label
    raise ValueError(f"category id {category_id!r} not found in taxonomy")


def top_categories(prediction: ArmPrediction, taxonomy: Taxonomy) -> list[CategoryScore]:
    """Build CategoryScore list from prediction's ranked_categories and scores.

    Each category gets a label from taxonomy and a confidence from prediction.scores
    (None if not present). Maintains the same order as prediction.ranked_categories.
    """
    result = []
    for category_id in prediction.ranked_categories:
        label = category_label(taxonomy, category_id)
        confidence = prediction.scores.get(category_id)
        result.append(CategoryScore(id=category_id, label=label, confidence=confidence))
    return result


def reaction_mix_summary(counts: dict[str, int]) -> str:
    """Build a single plain-language sentence from reaction counts.

    Order: outrage, criticize, mild_concern, ignore. Omit zero-count reactions.
    Group outrage + criticize as "objected". Use singular/plural correctly.
    """
    # Extract non-zero counts in the specified order
    order = ["outrage", "criticize", "mild_concern", "ignore"]
    non_zero = {}
    for reaction in order:
        count = counts.get(reaction, 0)
        if count > 0:
            non_zero[reaction] = count

    if not non_zero:
        return ""

    parts = []

    # Handle objected (outrage + criticize together)
    outrage_count = non_zero.get("outrage", 0)
    criticize_count = non_zero.get("criticize", 0)
    objected_count = outrage_count + criticize_count

    if objected_count > 0:
        word = "persona" if objected_count == 1 else "personas"
        parts.append(f"{objected_count} {word} objected")

    # Handle mild_concern
    mild_count = non_zero.get("mild_concern", 0)
    if mild_count > 0:
        word = "persona" if mild_count == 1 else "personas"
        parts.append(f"{mild_count} {word} expressed mild concern")

    # Handle ignore
    ignore_count = non_zero.get("ignore", 0)
    if ignore_count > 0:
        word = "persona" if ignore_count == 1 else "personas"
        parts.append(f"{ignore_count} {word} stayed silent")

    return " and ".join(parts) + "."


@dataclass(frozen=True)
class QuoteItem:
    """A curated quote with its metadata."""

    archetype_label: str
    platform: str
    reaction: str
    intensity: float
    quote: str
    categories: tuple[str, ...]


def select_curated_quotes(
    reacted: list[ReactionRow], personas_by_id: dict[str, Persona], limit: int = 8
) -> list[QuoteItem]:
    """Select curated quotes, diversifying by archetype first, then by intensity.

    Algorithm:
    1. Walk reacted (already sorted by intensity desc) and take the first row for
       each distinct archetype not yet taken, until either every archetype is taken
       or limit is reached.
    2. If slots remain, fill with next highest-intensity rows regardless of archetype.
    3. Re-sort final list by intensity descending.

    Never includes persona_id or persona_name in the result, only archetype_label.
    """
    selected_indices = []
    archetypes_seen = set()

    # Phase 1: diversify by archetype
    for i, row in enumerate(reacted):
        if len(selected_indices) >= limit:
            break
        if row.archetype not in archetypes_seen:
            selected_indices.append(i)
            archetypes_seen.add(row.archetype)

    # Phase 2: fill remaining slots by intensity
    if len(selected_indices) < limit:
        selected_set = set(selected_indices)
        for i, row in enumerate(reacted):
            if len(selected_indices) >= limit:
                break
            if i not in selected_set:
                selected_indices.append(i)
                selected_set.add(i)

    # Build QuoteItem objects
    quotes = []
    for i in selected_indices:
        row = reacted[i]
        persona = personas_by_id[row.persona_id]
        archetype_label = ARCHETYPE_LABELS.get(row.archetype, row.archetype)
        quote_item = QuoteItem(
            archetype_label=archetype_label,
            platform=persona.platform,
            reaction=row.reaction,
            intensity=row.intensity,
            quote=row.quote or "",
            categories=row.categories,
        )
        quotes.append(quote_item)

    # Re-sort by intensity descending
    quotes.sort(key=lambda q: -q.intensity)
    return quotes


@dataclass(frozen=True)
class CategoryDiff:
    """Comparison of category predictions between two arms."""

    agreed: list[CategoryScore]
    ensemble_only: list[CategoryScore]
    naive_only: list[CategoryScore]
    backlash_agreement: bool


def compare_predictions(
    naive: ArmPrediction, ensemble: ArmPrediction, taxonomy: Taxonomy
) -> CategoryDiff:
    """Compare predictions from naive and ensemble arms.

    - agreed: categories in both predictions (ensemble's order)
    - ensemble_only: in ensemble but not naive
    - naive_only: in naive but not ensemble
    - backlash_agreement: whether both arms agree on backlash_predicted
    """
    naive_categories = set(naive.ranked_categories)
    ensemble_categories = set(ensemble.ranked_categories)

    # Agreed categories (in both, ensemble's order)
    agreed = []
    for category_id in ensemble.ranked_categories:
        if category_id in naive_categories:
            label = category_label(taxonomy, category_id)
            confidence = ensemble.scores.get(category_id)
            agreed.append(CategoryScore(id=category_id, label=label, confidence=confidence))

    # Ensemble only
    ensemble_only = []
    for category_id in ensemble.ranked_categories:
        if category_id not in naive_categories:
            label = category_label(taxonomy, category_id)
            confidence = ensemble.scores.get(category_id)
            ensemble_only.append(CategoryScore(id=category_id, label=label, confidence=confidence))

    # Naive only
    naive_only = []
    for category_id in naive.ranked_categories:
        if category_id not in ensemble_categories:
            label = category_label(taxonomy, category_id)
            confidence = None  # naive always has no scores
            naive_only.append(CategoryScore(id=category_id, label=label, confidence=confidence))

    backlash_agreement = naive.backlash_predicted == ensemble.backlash_predicted

    return CategoryDiff(
        agreed=agreed,
        ensemble_only=ensemble_only,
        naive_only=naive_only,
        backlash_agreement=backlash_agreement,
    )
