from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.dashboard import (
    ARM_ORDER,
    REACTION_KEYS,
    DashboardError,
    ReactionRow,
    build_dashboard,
    load_event_predictions,
    load_event_reactions,
    main,
    reaction_mix_counts,
    render_event_card,
    render_missing_event_card,
    render_page,
    split_reacted_and_ignored,
)
from src.events import Event
from src.models import ArmPrediction, PersonaReaction


def write_reaction(path: Path, **kwargs) -> None:
    defaults = dict(reaction="ignore", categories=[], intensity=0.0, quote=None, reasoning="r")
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PersonaReaction(**defaults).model_dump_json(), encoding="utf-8")


def write_prediction(path: Path, **kwargs) -> None:
    defaults = dict(
        arm="A",
        event_id="evt_001",
        ranked_categories=["privacy", "legal", "pricing"],
        scores={},
        backlash_predicted=True,
    )
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ArmPrediction(**defaults).model_dump_json(), encoding="utf-8")


def test_load_event_reactions_maps_persona_metadata(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(
        tmp_path / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize",
        categories=["privacy"],
        intensity=0.8,
        quote="q",
    )
    rows = load_event_reactions(tmp_path, "evt_001", personas_by_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.persona_id == "001"
    assert row.persona_name == "privacy_hawk"
    assert row.archetype == "critic"
    assert row.reaction == "criticize"
    assert row.categories == ("privacy",)
    assert row.intensity == 0.8
    assert row.quote == "q"


def test_load_event_reactions_falls_back_for_unknown_persona_id(tmp_path):
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "999.json")
    rows = load_event_reactions(tmp_path, "evt_001", {})
    assert len(rows) == 1
    assert rows[0].persona_id == "999"
    assert "not found" in rows[0].persona_name


def test_load_event_reactions_empty_when_directory_missing(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    rows = load_event_reactions(tmp_path, "evt_absent", personas_by_id)
    assert rows == []


def test_load_event_reactions_sorted_by_persona_id(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "002.json")
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    rows = load_event_reactions(tmp_path, "evt_001", personas_by_id)
    assert [r.persona_id for r in rows] == ["001", "002"]


def test_load_event_predictions_sorted_by_arm_order(tmp_path):
    write_prediction(
        tmp_path / "predictions" / "evt_001__C.json",
        arm="C",
        ranked_categories=["none", "labor", "environment"],
        backlash_predicted=False,
    )
    write_prediction(tmp_path / "predictions" / "evt_001__A.json", arm="A")
    write_prediction(tmp_path / "predictions" / "evt_001__B30.json", arm="B30")
    preds = load_event_predictions(tmp_path, "evt_001")
    assert [p.arm for p in preds] == ["A", "B30", "C"]
    assert ARM_ORDER == ("A", "B3", "B8", "B15", "B30", "C")


def test_load_event_predictions_empty_when_no_matches(tmp_path):
    assert load_event_predictions(tmp_path, "evt_absent") == []


def test_load_event_predictions_ignores_other_events(tmp_path):
    write_prediction(tmp_path / "predictions" / "evt_002__A.json", event_id="evt_002")
    write_prediction(tmp_path / "predictions" / "evt_001__A.json", event_id="evt_001")
    preds = load_event_predictions(tmp_path, "evt_001")
    assert len(preds) == 1
    assert preds[0].event_id == "evt_001"


def make_row(persona_id: str, reaction: str, intensity: float = 0.0) -> ReactionRow:
    return ReactionRow(
        persona_id=persona_id,
        persona_name=f"name{persona_id}",
        archetype="critic",
        reaction=reaction,
        categories=() if reaction == "ignore" else ("privacy",),
        intensity=intensity,
        quote=None if reaction == "ignore" else "q",
    )


def test_reaction_mix_counts_all_four_keys_always_present():
    counts = reaction_mix_counts([make_row("001", "criticize", 0.5)])
    assert counts == {"ignore": 0, "mild_concern": 0, "criticize": 1, "outrage": 0}


def test_reaction_mix_counts_tallies_across_rows():
    rows = [
        make_row("001", "ignore"),
        make_row("002", "ignore"),
        make_row("003", "outrage", 1.0),
        make_row("004", "mild_concern", 0.3),
    ]
    assert reaction_mix_counts(rows) == {"ignore": 2, "mild_concern": 1, "criticize": 0, "outrage": 1}


def test_reaction_mix_counts_key_order_matches_reaction_keys():
    assert list(reaction_mix_counts([]).keys()) == list(REACTION_KEYS)


def test_split_separates_ignored_from_reacted():
    rows = [make_row("001", "ignore"), make_row("002", "criticize", 0.5)]
    reacted, ignored = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["002"]
    assert [r.persona_id for r in ignored] == ["001"]


def test_split_reacted_sorted_by_intensity_descending():
    rows = [make_row("001", "criticize", 0.3), make_row("002", "outrage", 0.9), make_row("003", "mild_concern", 0.5)]
    reacted, _ = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["002", "003", "001"]


def test_split_ties_broken_by_persona_id():
    rows = [make_row("003", "criticize", 0.5), make_row("001", "criticize", 0.5)]
    reacted, _ = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in reacted] == ["001", "003"]


def test_split_ignored_sorted_by_persona_id():
    rows = [make_row("003", "ignore"), make_row("001", "ignore")]
    _, ignored = split_reacted_and_ignored(rows)
    assert [r.persona_id for r in ignored] == ["001", "003"]


def make_event(**overrides) -> Event:
    defaults = dict(
        id="evt_001",
        company="Acme Corp",
        sector="consumer_tech",
        date="2026-06-14",
        headline="Acme introduces Acme Assist",
        source_url=None,
        expected_null=False,
        illustrative=False,
        announcement="Acme Corp today announced Acme Assist.",
        prior_statements=None,
    )
    defaults.update(overrides)
    return Event(**defaults)


def test_render_event_card_includes_headline_and_arm_predictions():
    event = make_event()
    prediction = ArmPrediction(
        arm="B30",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "financial"],
        scores={},
        backlash_predicted=True,
    )
    reacted = [make_row("001", "criticize", 0.8)]
    html_out = render_event_card(event, reacted, [], [prediction], reaction_mix_counts(reacted))
    assert "evt_001" in html_out
    assert "Acme introduces Acme Assist" in html_out
    assert "B30" in html_out
    assert "privacy" in html_out and "overclaim" in html_out


def test_render_event_card_predictions_render_inside_summary_when_collapsed():
    """The design spec puts arm predictions in the 'Collapsed (always visible)'
    section. HTML <details> only renders <summary>'s content when collapsed
    (no `open` attribute, no JS toggling), so the predictions div must appear
    before </summary> in the raw markup -- not after it (finding #1)."""
    event = make_event()
    prediction = ArmPrediction(
        arm="B30",
        event_id="evt_001",
        ranked_categories=["privacy", "overclaim", "financial"],
        scores={},
        backlash_predicted=True,
    )
    html_out = render_event_card(event, [], [], [prediction], reaction_mix_counts([]))
    assert "<details" in html_out
    assert " open" not in html_out and "<details open" not in html_out  # no `open` attribute anywhere
    predictions_index = html_out.index('<div class="predictions">')
    summary_close_index = html_out.index("</summary>")
    assert predictions_index < summary_close_index, (
        "predictions div must appear before </summary> so it is visible while the card is collapsed"
    )


def test_render_event_card_escapes_html_in_announcement_and_quotes():
    event = make_event(announcement="Uses <script>alert(1)</script> data.")
    reacted = [
        ReactionRow(
            persona_id="001",
            persona_name="privacy_hawk",
            archetype="critic",
            reaction="criticize",
            categories=("privacy",),
            intensity=0.8,
            quote="<b>quote</b>",
        )
    ]
    html_out = render_event_card(event, reacted, [], [], reaction_mix_counts(reacted))
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<b>quote</b>" not in html_out


HOSTILE = '<script>alert(1)</script>"><svg onload=alert(1)>'


def test_render_event_card_escapes_hostile_markup_in_all_free_text_fields():
    event = make_event(id=HOSTILE, company=HOSTILE, headline=HOSTILE, announcement=HOSTILE)
    reacted = [
        ReactionRow(
            persona_id=HOSTILE,
            persona_name=HOSTILE,
            archetype=HOSTILE,
            reaction="criticize",
            categories=("privacy",),
            intensity=0.8,
            quote=HOSTILE,
        )
    ]
    ignored = [
        ReactionRow(
            persona_id=HOSTILE,
            persona_name=HOSTILE,
            archetype=HOSTILE,
            reaction="ignore",
            categories=(),
            intensity=0.0,
            quote=None,
        )
    ]
    html_out = render_event_card(event, reacted, ignored, [], reaction_mix_counts(reacted))
    assert "<script" not in html_out
    assert "<svg" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "&lt;svg" in html_out


def test_render_missing_event_card_escapes_hostile_markup():
    event = make_event(id=HOSTILE, company=HOSTILE, headline=HOSTILE)
    html_out = render_missing_event_card(event)
    assert "<script" not in html_out
    assert "<svg" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "&lt;svg" in html_out


def test_render_page_escapes_hostile_run_id():
    page = render_page(HOSTILE, [])
    assert "<script" not in page
    assert "<svg" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;svg" in page


def test_render_event_card_lists_ignored_personas_compactly():
    event = make_event()
    ignored = [
        ReactionRow(
            persona_id="002",
            persona_name="labor_advocate",
            archetype="critic",
            reaction="ignore",
            categories=(),
            intensity=0.0,
            quote=None,
        )
    ]
    html_out = render_event_card(event, [], ignored, [], reaction_mix_counts([]))
    assert "labor_advocate" in html_out


def test_render_missing_event_card_notes_not_yet_run():
    event = make_event(id="evt_003")
    html_out = render_missing_event_card(event)
    assert "evt_003" in html_out
    assert "not yet run" in html_out.lower()


def test_render_page_wraps_cards_with_run_id():
    page = render_page("run_001", ["<details>card A</details>", "<details>card B</details>"])
    assert "run_001" in page
    assert "card A" in page and "card B" in page
    assert "<html" in page


@pytest.fixture
def sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copytree(repo / "personas", root / "personas")
    (root / "inputs").mkdir()
    (root / "inputs" / "events.txt").write_text(
        "=== EVENT ===\n"
        "id: evt_001\n"
        "company: Acme Corp\n"
        "sector: consumer_tech\n"
        "date: 2026-06-14\n"
        "headline: Acme launches an AI layer\n"
        "expected_null: false\n"
        "---\n"
        "Acme Corp today announced an AI layer.\n"
        "=== END EVENT ===\n"
        "\n"
        "=== EVENT ===\n"
        "id: evt_002\n"
        "company: Northwind\n"
        "sector: industrial\n"
        "date: 2026-06-20\n"
        "headline: Northwind opens a warehouse\n"
        "expected_null: true\n"
        "---\n"
        "Northwind opened a distribution centre.\n"
        "=== END EVENT ===\n",
        encoding="utf-8",
    )
    return root


def test_build_dashboard_raises_without_manifest(sandbox):
    with pytest.raises(DashboardError, match="no run at"):
        build_dashboard(repo=sandbox, run_id="run_001")


def test_build_dashboard_renders_missing_card_for_unrun_event(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    page = build_dashboard(repo=sandbox, run_id="run_001")
    assert "evt_001" in page
    assert "evt_002" in page
    assert "not yet run" in page.lower()


def test_build_dashboard_renders_full_card_when_data_present(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize",
        categories=["privacy"],
        intensity=0.8,
        quote="q",
    )
    write_prediction(run_dir / "predictions" / "evt_001__A.json", event_id="evt_001", arm="A")

    page = build_dashboard(repo=sandbox, run_id="run_001")
    assert "privacy_hawk" in page
    assert "not yet run" in page.lower()  # evt_002 still has no data


def test_main_writes_dashboard_html_to_default_path(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    exit_code = main(["run_001", "--repo", str(sandbox)])
    assert exit_code == 0
    out_path = run_dir / "dashboard.html"
    assert out_path.exists()
    assert "evt_001" in out_path.read_text(encoding="utf-8")


def test_main_returns_nonzero_for_missing_run(sandbox, capsys):
    exit_code = main(["does_not_exist", "--repo", str(sandbox)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_main_creates_missing_parent_directories_for_out(sandbox):
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    out_path = sandbox / "nested" / "deep" / "dashboard.html"

    exit_code = main(["run_001", "--repo", str(sandbox), "--out", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()


def test_main_returns_nonzero_for_malformed_events_file(sandbox, capsys):
    """A corrupt inputs/events.txt raises EventParseError -- main must catch it
    and print a clean error instead of a raw traceback (finding #3)."""
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (sandbox / "inputs" / "events.txt").write_text("not a valid events file at all", encoding="utf-8")

    exit_code = main(["run_001", "--repo", str(sandbox)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_main_returns_nonzero_for_corrupt_reaction_json(sandbox, capsys):
    """A truncated/corrupt persona reaction JSON file (e.g. a run killed
    mid-write) raises a pydantic ValidationError -- main must catch it too
    (finding #3)."""
    run_dir = sandbox / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    reaction_path = run_dir / "raw" / "B30" / "evt_001" / "001.json"
    reaction_path.parent.mkdir(parents=True, exist_ok=True)
    reaction_path.write_text('{"reaction": "criticize"}', encoding="utf-8")  # missing required fields

    with pytest.raises(ValidationError):
        PersonaReaction.model_validate_json(reaction_path.read_text(encoding="utf-8"))

    exit_code = main(["run_001", "--repo", str(sandbox)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err
