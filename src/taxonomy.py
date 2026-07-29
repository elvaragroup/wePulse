"""Parses taxonomy.txt and keeps it in lockstep with the Category literal.

The taxonomy is frozen before any run. If the file and models.Category ever
drift, every category the judge emits could silently fail validation, so the
cross-check runs at import of the loaded taxonomy rather than at first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from src.models import Category


@dataclass(frozen=True)
class TaxonomyEntry:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Taxonomy:
    entries: tuple[TaxonomyEntry, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.entries)

    def rank(self, category: str) -> int:
        """Position in the file. Used as the deterministic tie-break when padding
        an arm's ranked_categories out to k=3."""
        return self.ids.index(category)

    def render(self) -> str:
        """The block handed to personas and to the judge."""
        return "\n".join(f"{e.id}: {e.label} -- {e.description}" for e in self.entries)


class TaxonomyError(ValueError):
    pass


def load_taxonomy(path: Path) -> Taxonomy:
    entries: list[TaxonomyEntry] = []
    seen: set[str] = set()

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise TaxonomyError(
                f"{path}:{lineno}: expected 'id | label | description', got {len(parts)} field(s)"
            )
        cid, label, description = parts
        if not cid or not label or not description:
            raise TaxonomyError(f"{path}:{lineno}: empty field in {line!r}")
        if cid in seen:
            raise TaxonomyError(f"{path}:{lineno}: duplicate category id {cid!r}")
        seen.add(cid)
        entries.append(TaxonomyEntry(id=cid, label=label, description=description))

    declared = set(get_args(Category))
    if seen != declared:
        missing = sorted(declared - seen)
        extra = sorted(seen - declared)
        raise TaxonomyError(
            f"{path} is out of sync with models.Category. "
            f"Missing from file: {missing}. Not in Category: {extra}."
        )
    if "none" not in seen:
        raise TaxonomyError(f"{path}: 'none' is a real category and must be present (spec 2.1)")

    return Taxonomy(entries=tuple(entries))
