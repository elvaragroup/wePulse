from __future__ import annotations

import pytest

from src.personas import (
    PersonaError,
    load_persona_file,
    load_personas,
    persona_set_hash,
    render_naive_system_prompt,
    render_system_prompt,
)
from tests.conftest import VALID_PERSONA, write_persona


def test_valid_persona_loads(tmp_path):
    path = write_persona(tmp_path, "001_privacy_hawk.md")
    persona = load_persona_file(path)
    assert persona.id == "001"
    assert persona.archetype == "critic"
    assert persona.baseline_skepticism == 0.7
    assert persona.platform == "x"


# --- the load-bearing rejection (spec 2.2) ---


def test_missing_what_you_scroll_past_is_rejected(tmp_path):
    text = VALID_PERSONA.replace(
        "## What you scroll past\n\nProduct announcements with no data dimension.\n\n", ""
    )
    path = write_persona(tmp_path, "001_privacy_hawk.md", text)
    with pytest.raises(PersonaError, match="What you scroll past"):
        load_persona_file(path)


@pytest.mark.parametrize("section", ["Who you are", "What you react to", "Voice"])
def test_other_required_sections_are_enforced(tmp_path, section):
    lines = VALID_PERSONA.splitlines(keepends=True)
    start = lines.index(f"## {section}\n")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    path = write_persona(tmp_path, "001_privacy_hawk.md", "".join(lines[:start] + lines[end:]))
    with pytest.raises(PersonaError, match=section):
        load_persona_file(path)


# --- frontmatter validation ---


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda t: t.replace("archetype: critic", "archetype: villain"), "archetype must be one of"),
        (lambda t: t.replace("platform: x", "platform: myspace"), "platform must be one of"),
        (lambda t: t.replace("baseline_skepticism: 0.7", "baseline_skepticism: 4"), "in \\[0.0, 1.0\\]"),
        (lambda t: t.replace('id: "001"', 'id: "1"'), "zero-padded 3-digit"),
        (lambda t: t.replace("axis: privacy\n", ""), "missing frontmatter key"),
    ],
)
def test_bad_frontmatter_raises(tmp_path, mutation, match):
    path = write_persona(tmp_path, "001_privacy_hawk.md", mutation(VALID_PERSONA))
    with pytest.raises(PersonaError, match=match):
        load_persona_file(path)


def test_missing_frontmatter_raises(tmp_path):
    path = write_persona(tmp_path, "001_privacy_hawk.md", "## Who you are\n\nSomeone.\n")
    with pytest.raises(PersonaError, match="missing YAML frontmatter"):
        load_persona_file(path)


def test_filename_must_match_id(tmp_path):
    path = write_persona(tmp_path, "007_privacy_hawk.md")
    with pytest.raises(PersonaError, match="filename must start with the persona id"):
        load_persona_file(path)


def test_duplicate_ids_across_files_raise(tmp_path):
    write_persona(tmp_path, "001_a.md")
    write_persona(tmp_path, "001_b.md")
    with pytest.raises(PersonaError, match="duplicate persona id"):
        load_personas(tmp_path)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(PersonaError, match="no persona files found"):
        load_personas(tmp_path)


# --- hashing (spec 0.5) ---


def test_hash_is_stable_and_edit_sensitive(tmp_path):
    write_persona(tmp_path, "001_privacy_hawk.md")
    first = persona_set_hash(tmp_path)
    assert persona_set_hash(tmp_path) == first

    write_persona(tmp_path, "001_privacy_hawk.md", VALID_PERSONA.replace("Terse", "Verbose"))
    assert persona_set_hash(tmp_path) != first


def test_hash_changes_when_a_persona_is_added(tmp_path):
    write_persona(tmp_path, "001_privacy_hawk.md")
    before = persona_set_hash(tmp_path)
    write_persona(tmp_path, "002_other.md", VALID_PERSONA.replace('id: "001"', 'id: "002"'))
    assert persona_set_hash(tmp_path) != before


# --- prompt rendering ---


def test_system_prompt_carries_body_taxonomy_and_contract(tmp_path, taxonomy):
    persona = load_persona_file(write_persona(tmp_path, "001_privacy_hawk.md"))
    prompt = render_system_prompt(persona, taxonomy)
    assert "A security and privacy researcher" in prompt
    assert "accessibility" in prompt
    assert "`ignore`" in prompt
    assert "Ignoring is a normal, frequent outcome" in prompt


def test_non_hypocrisy_persona_gets_no_prior_statements_block(tmp_path, taxonomy):
    persona = load_persona_file(write_persona(tmp_path, "001_privacy_hawk.md"))
    prompt = render_system_prompt(persona, taxonomy, prior_statements="They said otherwise.")
    assert "prior public statements" not in prompt
    assert "They said otherwise." not in prompt


def test_hypocrisy_persona_receives_prior_statements(tmp_path, taxonomy):
    text = VALID_PERSONA.replace("axis: privacy", "axis: hypocrisy")
    persona = load_persona_file(write_persona(tmp_path, "001_receipts.md", text))
    assert persona.wants_prior_statements

    with_receipts = render_system_prompt(persona, taxonomy, prior_statements="They said otherwise.")
    assert "They said otherwise." in with_receipts

    without = render_system_prompt(persona, taxonomy)
    assert "None are available" in without
    assert "ignore it" in without


def test_naive_prompt_asks_for_five_ranked(taxonomy):
    prompt = render_naive_system_prompt(taxonomy)
    assert "5 most likely sources of public backlash" in prompt
    assert "ranked" in prompt
    assert "privacy" in prompt
