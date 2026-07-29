"""Pydantic schemas. These are the contract for all structured LLM output.

The validators here are the cheapest defence against a silently wrong study: a
persona that claims to ignore something while still naming categories, or an arm
prediction that quietly carries four guesses instead of three, would corrupt
every downstream number without raising anything.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Models the API is asked to emit set extra="forbid", which renders as
# `additionalProperties: false` -- required by structured outputs.
WIRE = ConfigDict(extra="forbid")

Category = Literal[
    "privacy",
    "labor",
    "financial",
    "safety",
    "environment",
    "hypocrisy",
    "legal",
    "pricing",
    "overclaim",
    "community",
    "accessibility",
    "security",
    "aesthetic",
    "none",
]

Reaction = Literal["ignore", "mild_concern", "criticize", "outrage"]

Arm = Literal["A", "B3", "B8", "B15", "B30", "C"]

K = 3


class PersonaReaction(BaseModel):
    """One persona's response to one announcement."""

    model_config = WIRE

    reaction: Reaction
    categories: list[Category]
    intensity: float = Field(ge=0.0, le=1.0)
    quote: str | None
    reasoning: str

    @model_validator(mode="after")
    def _ignore_is_consistent(self) -> PersonaReaction:
        ignoring = self.reaction == "ignore"
        if ignoring and self.categories:
            raise ValueError("reaction 'ignore' must have empty categories")
        if not ignoring and not self.categories:
            raise ValueError(f"reaction '{self.reaction}' must name at least one category")
        if ignoring and self.quote is not None:
            raise ValueError("reaction 'ignore' must have quote=None")
        if not ignoring and self.quote is None:
            raise ValueError(f"reaction '{self.reaction}' must have a quote")
        return self


class RankedRisk(BaseModel):
    """One entry of the naive baseline's ranked list.

    DEVIATION from spec 3, which types `rationale` as `dict[str, str]`. Structured
    outputs reject any object whose `additionalProperties` is not `false`, so a
    free-keyed dict cannot be requested from the API. The wire shape is a list;
    NaiveBaselineOutput.rationale below rebuilds the spec's dict for storage.
    """

    model_config = WIRE

    category: Category
    rationale: str


class NaiveBaselineWire(BaseModel):
    """What arm A is asked to emit."""

    model_config = WIRE

    risks: list[RankedRisk]


class NaiveBaselineOutput(BaseModel):
    """Arm A, in the spec's shape. Built from NaiveBaselineWire."""

    risks: list[Category] = Field(max_length=5)
    rationale: dict[str, str]

    @classmethod
    def from_wire(cls, wire: NaiveBaselineWire) -> NaiveBaselineOutput:
        ranked: list[Category] = []
        for item in wire.risks:
            if item.category not in ranked:
                ranked.append(item.category)
        return cls(
            risks=ranked[:5],
            rationale={item.category: item.rationale for item in wire.risks},
        )


class ArmPrediction(BaseModel):
    """An arm's committed guess for one event. Written before ground truth exists."""

    arm: Arm
    event_id: str
    ranked_categories: list[Category]
    scores: dict[str, float]
    backlash_predicted: bool

    @model_validator(mode="after")
    def _exactly_k(self) -> ArmPrediction:
        if len(self.ranked_categories) != K:
            raise ValueError(
                f"ranked_categories must hold exactly {K} entries, got {len(self.ranked_categories)}"
            )
        if len(set(self.ranked_categories)) != K:
            raise ValueError(f"ranked_categories must be distinct, got {self.ranked_categories}")
        return self


class GroundTruthLabel(BaseModel):
    """What actually happened, as labeled by the blind judge."""

    event_id: str
    dominant_category: Category
    present_categories: list[Category]
    backlash_occurred: bool
    judge_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _dominant_is_present(self) -> GroundTruthLabel:
        if self.dominant_category not in self.present_categories:
            raise ValueError(
                f"dominant_category {self.dominant_category!r} missing from present_categories"
            )
        if self.backlash_occurred and self.dominant_category == "none":
            raise ValueError("backlash_occurred=True is inconsistent with dominant_category 'none'")
        if not self.backlash_occurred and self.dominant_category != "none":
            raise ValueError(
                f"backlash_occurred=False is inconsistent with dominant_category "
                f"{self.dominant_category!r}"
            )
        return self


class JudgeVerdict(BaseModel):
    """What the judge model is asked to emit. Lacks event_id, which the judge
    never sees -- label_truth.py attaches it when building the GroundTruthLabel."""

    model_config = WIRE

    dominant_category: Category
    present_categories: list[Category]
    backlash_occurred: bool
    judge_confidence: float = Field(ge=0.0, le=1.0)


class LeakageVerdict(BaseModel):
    """One leakage probe classification (spec 4.1)."""

    model_config = WIRE

    describes_reaction: bool
    response: str
