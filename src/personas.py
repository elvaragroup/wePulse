"""Loads, validates, hashes, and renders persona files.

`What you scroll past` is mandatory and the loader rejects any file that omits
it. That section is the mechanism that makes `ignore` a live option (spec 0.1);
without it a persona will manufacture outrage about a puppy adoption
announcement, and false positives become undetectable.

The persona set is hashed so a run can prove which text produced it. Any edit to
any file changes the hash, which changes the run_id, which prevents pooling with
earlier runs (spec 0.5).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from src.taxonomy import Taxonomy

Archetype = Literal["critic", "neutral", "sympathetic", "insider"]

REQUIRED_SECTIONS = ("Who you are", "What you react to", "What you scroll past", "Voice")
REQUIRED_FRONTMATTER = ("id", "name", "axis", "archetype", "baseline_skepticism", "platform")
VALID_ARCHETYPES = ("critic", "neutral", "sympathetic", "insider")
VALID_PLATFORMS = ("x", "linkedin", "reddit", "press")

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_PERSONA_ID = re.compile(r"^\d{3}$")


class PersonaError(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    axis: str
    archetype: Archetype
    baseline_skepticism: float
    platform: str
    body: str
    source_path: Path

    @property
    def wants_prior_statements(self) -> bool:
        return self.axis == "hypocrisy"


def _parse_persona(text: str, path: Path) -> Persona:
    match = _FRONTMATTER.match(text)
    if not match:
        raise PersonaError(f"{path.name}: missing YAML frontmatter delimited by '---' lines")

    raw_front, body = match.group(1), match.group(2).strip()

    try:
        front = yaml.safe_load(raw_front)
    except yaml.YAMLError as exc:
        raise PersonaError(f"{path.name}: frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(front, dict):
        raise PersonaError(f"{path.name}: frontmatter must be a mapping")

    missing = [k for k in REQUIRED_FRONTMATTER if k not in front]
    if missing:
        raise PersonaError(f"{path.name}: missing frontmatter key(s) {missing}")

    persona_id = str(front["id"])
    if not _PERSONA_ID.match(persona_id):
        raise PersonaError(f"{path.name}: id must be a zero-padded 3-digit string, got {persona_id!r}")
    if not path.name.startswith(f"{persona_id}_"):
        raise PersonaError(f"{path.name}: filename must start with the persona id {persona_id!r}")

    archetype = str(front["archetype"])
    if archetype not in VALID_ARCHETYPES:
        raise PersonaError(
            f"{path.name}: archetype must be one of {VALID_ARCHETYPES}, got {archetype!r}"
        )

    platform = str(front["platform"])
    if platform not in VALID_PLATFORMS:
        raise PersonaError(f"{path.name}: platform must be one of {VALID_PLATFORMS}, got {platform!r}")

    skepticism = front["baseline_skepticism"]
    if not isinstance(skepticism, (int, float)) or not 0.0 <= float(skepticism) <= 1.0:
        raise PersonaError(f"{path.name}: baseline_skepticism must be a number in [0.0, 1.0]")

    found = {m.group(1) for m in _SECTION.finditer(body)}
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in found]
    if missing_sections:
        raise PersonaError(
            f"{path.name}: missing required section(s) {missing_sections}. "
            f"'What you scroll past' in particular is load-bearing -- it is what makes "
            f"'ignore' a live option (spec 0.1)."
        )

    return Persona(
        id=persona_id,
        name=str(front["name"]),
        axis=str(front["axis"]),
        archetype=archetype,  # type: ignore[arg-type]
        baseline_skepticism=float(skepticism),
        platform=platform,
        body=body,
        source_path=path,
    )


def load_persona_file(path: Path) -> Persona:
    return _parse_persona(path.read_text(encoding="utf-8"), path)


def load_personas(directory: Path) -> list[Persona]:
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise PersonaError(f"no persona files found in {directory}")

    personas = [load_persona_file(p) for p in paths]

    ids = [p.id for p in personas]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PersonaError(f"duplicate persona id(s): {duplicates}")

    return personas


def persona_set_hash(directory: Path) -> str:
    """sha256 over sorted (filename, bytes). Any edit anywhere changes this."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.md"), key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def archetype_counts(personas: list[Persona]) -> dict[str, int]:
    counts = dict.fromkeys(VALID_ARCHETYPES, 0)
    for p in personas:
        counts[p.archetype] += 1
    return counts


def validate_axes(personas: list[Persona], taxonomy: Taxonomy) -> None:
    """Every persona's axis must be a real taxonomy category or 'neutral' (spec 2.2).

    Kept separate from load_personas so the loader stays usable without a
    taxonomy; run_sim.py calls this before any request goes out.
    """
    allowed = set(taxonomy.ids) | {"neutral"}
    bad = [(p.id, p.axis) for p in personas if p.axis not in allowed]
    if bad:
        raise PersonaError(
            f"persona axis must be a taxonomy category or 'neutral'; offending: {bad}"
        )


OUTPUT_CONTRACT = """\
## How to respond

You are shown one corporate announcement. Decide, in character, whether you would \
post anything at all about it.

Choose exactly one `reaction`:
- `ignore` -- you scroll past. This is the correct answer whenever the announcement \
falls outside your beat or simply does not bother you. Ignoring is a normal, \
frequent outcome, not a failure to engage.
- `mild_concern` -- worth a caveat, not a fight.
- `criticize` -- you post a pointed objection.
- `outrage` -- you post something angry that you expect others to amplify.

Then:
- `categories`: which of the taxonomy categories below your objection falls under. \
Leave this empty if and only if you chose `ignore`. Do not list categories you would \
not actually post about.
- `intensity`: 0.0 to 1.0, how strongly you feel. Use 0.0 when ignoring.
- `quote`: the post you would actually write, in your own voice. Use null if and \
only if you chose `ignore`.
- `reasoning`: one sentence, for audit only.

Do not manufacture an objection to stay relevant. If nothing here is yours, ignore it.

## Taxonomy

{taxonomy}
"""

PRIOR_STATEMENTS_BLOCK = """\
## The company's prior public statements

Use these to judge whether the announcement contradicts what they have said before. \
If nothing here is contradicted, say so by ignoring the announcement rather than \
straining to find a contradiction.

{prior_statements}
"""

NO_PRIOR_STATEMENTS_BLOCK = """\
## The company's prior public statements

None are available for this company. Without receipts you cannot establish \
hypocrisy, so unless the announcement contradicts itself on its face, ignore it.
"""


def render_system_prompt(
    persona: Persona,
    taxonomy: Taxonomy,
    *,
    prior_statements: str | None = None,
) -> str:
    parts = [persona.body]

    if persona.wants_prior_statements:
        if prior_statements:
            parts.append(PRIOR_STATEMENTS_BLOCK.format(prior_statements=prior_statements.strip()))
        else:
            parts.append(NO_PRIOR_STATEMENTS_BLOCK)

    parts.append(OUTPUT_CONTRACT.format(taxonomy=taxonomy.render()))
    return "\n\n".join(parts)


NAIVE_SYSTEM_PROMPT = """\
You are analysing a corporate announcement for likely public reaction.

List the 5 most likely sources of public backlash, ranked from most to least \
likely, drawing only on the categories in the taxonomy below. If you believe the \
announcement will draw no meaningful backlash, rank `none` first.

For each category you list, give a one-sentence rationale keyed by the category id.

## Taxonomy

{taxonomy}
"""


def render_naive_system_prompt(taxonomy: Taxonomy) -> str:
    return NAIVE_SYSTEM_PROMPT.format(taxonomy=taxonomy.render())
