from __future__ import annotations

from pathlib import Path

import pytest

from src.dashboard import ARM_ORDER, ReactionRow, load_event_predictions, load_event_reactions
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
