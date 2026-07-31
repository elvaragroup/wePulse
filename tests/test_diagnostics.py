from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.diagnostics import DiagnosticsError, load_rows, load_v1_rows


def write_reaction(path: Path, **kwargs) -> None:
    defaults = dict(reaction="ignore", categories=[], intensity=0.0, quote=None, reasoning="r")
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults), encoding="utf-8")


def test_load_v1_rows_reads_every_persona_reaction(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(
        tmp_path / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize", categories=["privacy"], intensity=0.8, quote="q",
    )
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "002.json")

    rows = load_v1_rows(tmp_path, personas_by_id)
    assert len(rows) == 2
    by_id = {r.persona_id: r for r in rows}
    assert by_id["001"].event_id == "evt_001"
    assert by_id["001"].reaction == "criticize"
    assert by_id["001"].text == "q"
    assert by_id["001"].category == "privacy"
    assert by_id["001"].char_start is None
    assert by_id["002"].reaction == "ignore"
    assert by_id["002"].text is None
    assert by_id["002"].category is None


def test_load_v1_rows_across_multiple_events(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    write_reaction(tmp_path / "raw" / "B30" / "evt_002" / "001.json")
    rows = load_v1_rows(tmp_path, personas_by_id)
    assert {r.event_id for r in rows} == {"evt_001", "evt_002"}


def test_load_v1_rows_raises_when_b30_missing(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    with pytest.raises(DiagnosticsError, match="no raw/B30"):
        load_v1_rows(tmp_path, personas_by_id)


def test_load_v1_rows_raises_when_no_reactions_present(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    (tmp_path / "raw" / "B30").mkdir(parents=True)
    with pytest.raises(DiagnosticsError, match="no reactions found"):
        load_v1_rows(tmp_path, personas_by_id)


def test_load_v1_rows_raises_for_unknown_persona_id(tmp_path):
    """personas_by_id is a real validation, not a decorative parameter: a
    persona_id in raw/B30/ that isn't in the current roster (persona set
    changed since this run happened) must fail loudly rather than silently
    skew the diagnostics numbers with data from a persona nobody can audit
    anymore."""
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "999.json")
    with pytest.raises(DiagnosticsError, match="999.*not found in the current persona set"):
        load_v1_rows(tmp_path, personas_by_id={})


def test_load_rows_dispatches_to_v1(tmp_path, personas):
    personas_by_id = {p.id: p for p in personas}
    write_reaction(tmp_path / "raw" / "B30" / "evt_001" / "001.json")
    rows = load_rows(tmp_path, personas_by_id=personas_by_id)
    assert len(rows) == 1
