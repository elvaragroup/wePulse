"""Checks the committed persona set against spec 2.2 and 0.1.

These run against personas/ itself rather than fixtures. If someone edits the
set, these tests say whether it is still the instrument the spec describes.
"""

from __future__ import annotations

import pytest

from src.personas import (
    REQUIRED_SECTIONS,
    archetype_counts,
    load_personas,
    persona_set_hash,
    render_system_prompt,
    validate_axes,
)

EXPECTED_ARCHETYPES = {"critic": 18, "insider": 4, "neutral": 5, "sympathetic": 3}
AXIS_CRITIC_IDS = [f"{i:03d}" for i in range(1, 14)]
SECOND_VARIANT_AXES = {"privacy", "financial", "hypocrisy", "overclaim", "pricing"}


@pytest.fixture(scope="module")
def personas(repo):
    return load_personas(repo / "personas")


def test_thirty_personas(personas):
    assert len(personas) == 30


def test_ids_are_contiguous_001_to_030(personas):
    assert [p.id for p in personas] == [f"{i:03d}" for i in range(1, 31)]


def test_archetype_distribution_matches_spec(personas):
    assert archetype_counts(personas) == EXPECTED_ARCHETYPES


def test_axes_are_valid(personas, taxonomy):
    validate_axes(personas, taxonomy)


def test_one_critic_per_taxonomy_axis(personas, taxonomy):
    """13 critics, one for each non-'none' taxonomy axis (spec 2.2)."""
    by_id = {p.id: p for p in personas}
    axes = [by_id[pid].axis for pid in AXIS_CRITIC_IDS]
    expected = [cid for cid in taxonomy.ids if cid != "none"]
    assert sorted(axes) == sorted(expected)
    assert len(set(axes)) == 13


def test_second_variants_cover_the_high_frequency_axes(personas):
    by_id = {p.id: p for p in personas}
    variant_ids = [f"{i:03d}" for i in range(18, 23)]
    assert {by_id[pid].axis for pid in variant_ids} == SECOND_VARIANT_AXES
    assert all(by_id[pid].archetype == "critic" for pid in variant_ids)


def test_second_variants_differ_in_voice_or_platform(personas):
    """A second variant that shares its sibling's platform is a duplicate, not a
    variant, and would inflate an axis without adding perspective."""
    by_axis: dict[str, list] = {}
    for p in personas:
        if p.archetype == "critic":
            by_axis.setdefault(p.axis, []).append(p)

    for axis, group in by_axis.items():
        if len(group) > 1:
            platforms = [p.platform for p in group]
            assert len(set(platforms)) == len(platforms), f"axis {axis} reuses a platform"


def test_insiders_are_the_four_the_spec_names(personas):
    by_id = {p.id: p for p in personas}
    ids = [f"{i:03d}" for i in range(14, 18)]
    assert all(by_id[pid].archetype == "insider" for pid in ids)
    assert {by_id[pid].name for pid in ids} == {
        "competitor_engineer",
        "industry_analyst",
        "ex_employee",
        "domain_regulator",
    }


def test_neutrals_have_distinct_thresholds(personas):
    """Spec 2.2: neutrals differ by threshold. Identical skepticism across all
    five would make them one persona counted five times."""
    neutrals = [p for p in personas if p.archetype == "neutral"]
    assert len(neutrals) == 5
    assert len(set(p.baseline_skepticism for p in neutrals)) >= 3


def test_sympathetics_are_less_skeptical_than_critics(personas):
    critics = [p.baseline_skepticism for p in personas if p.archetype == "critic"]
    sympathetics = [p.baseline_skepticism for p in personas if p.archetype == "sympathetic"]
    assert max(sympathetics) < min(critics)


# --- spec 0.1: every persona must be able to shrug ---


def test_every_persona_has_all_required_sections(personas):
    for p in personas:
        for section in REQUIRED_SECTIONS:
            assert f"## {section}" in p.body, f"{p.id} missing {section}"


def test_scroll_past_sections_are_substantive(personas):
    """The loader only checks the heading exists. A one-word section would pass
    the loader and still fail to make 'ignore' a live option."""
    for p in personas:
        _, _, after = p.body.partition("## What you scroll past")
        section = after.partition("\n## ")[0].strip()
        assert len(section.split()) >= 25, f"{p.id} has a thin 'What you scroll past' section"


def test_every_rendered_prompt_licenses_ignoring(personas, taxonomy):
    for p in personas:
        prompt = render_system_prompt(p, taxonomy)
        assert "Ignoring is a normal, frequent outcome" in prompt
        assert "Do not manufacture an objection" in prompt


def test_only_hypocrisy_personas_request_prior_statements(personas):
    wanting = {p.id for p in personas if p.wants_prior_statements}
    assert wanting == {"006", "020"}


# --- spec 0.5: the set is hashable and the config subsets resolve ---


def test_hash_is_deterministic(repo):
    assert persona_set_hash(repo / "personas") == persona_set_hash(repo / "personas")


def test_config_subsets_reference_real_personas(personas, config):
    known = {p.id for p in personas}
    for arm, members in config.subsets.items():
        missing = set(members) - known
        assert not missing, f"subsets.{arm} names unknown persona(s) {sorted(missing)}"


def test_subsets_are_stratified_not_sequential(personas, config):
    """Spec 4.2: B3 must not be three critics of the same axis."""
    by_id = {p.id: p for p in personas}
    for arm in ("B3", "B8", "B15"):
        members = [by_id[m] for m in config.subsets[arm]]
        assert len(set(p.archetype for p in members)) >= 3, f"{arm} is not archetype-stratified"
        critic_axes = [p.axis for p in members if p.archetype == "critic"]
        assert len(set(critic_axes)) == len(critic_axes), f"{arm} repeats a critic axis"
