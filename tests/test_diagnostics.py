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


import asyncio
import shutil

from src.diagnostics import DiagnosticsReport, run_diagnostics, write_report
from src.embeddings import FakeEmbeddingClient
from src.llm import FakeLLMClient, LLMRequest


def write_prediction(path: Path, **kwargs) -> None:
    defaults = dict(
        arm="B30", event_id="evt_001", ranked_categories=["privacy", "legal", "pricing"],
        scores={}, backlash_predicted=True,
    )
    defaults.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults), encoding="utf-8")


EVENTS_TEXT = (
    "=== EVENT ===\n"
    "id: evt_001\n"
    "company: Acme Corp\n"
    "sector: consumer_tech\n"
    "date: 2026-06-14\n"
    "headline: Acme launches an AI layer\n"
    "expected_null: false\n"
    "---\n"
    "Acme Corp today announced an AI layer, enabled by default.\n"
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
    "Northwind opened a distribution centre, adding 400 jobs.\n"
    "=== END EVENT ===\n"
)


@pytest.fixture
def diagnostics_sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copytree(repo / "personas", root / "personas")
    shutil.copy(repo / "config.yaml", root / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", root / "taxonomy.txt")
    (root / "inputs").mkdir()
    (root / "inputs" / "events.txt").write_text(EVENTS_TEXT, encoding="utf-8")

    run_dir = root / "runs" / "run_001"
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "001.json",
        reaction="criticize", categories=["privacy"], intensity=0.8, quote="This is a real complaint about defaults.",
    )
    write_reaction(
        run_dir / "raw" / "B30" / "evt_001" / "002.json",
        reaction="mild_concern", categories=["overclaim"], intensity=0.5, quote="Slightly overstated marketing copy here.",
    )
    write_reaction(run_dir / "raw" / "B30" / "evt_002" / "001.json")
    write_reaction(run_dir / "raw" / "B30" / "evt_002" / "002.json")
    write_prediction(run_dir / "predictions" / "evt_001__B30.json", event_id="evt_001", backlash_predicted=True)
    write_prediction(run_dir / "predictions" / "evt_002__B30.json", event_id="evt_002", backlash_predicted=False, ranked_categories=["none", "labor", "environment"])

    return root


def fake_embedding_responder(text: str) -> list[float]:
    # Deterministic, distinguishable-by-length embedding so homogeneity/
    # redundancy produce non-degenerate but reproducible numbers.
    return [float(len(text) % 7), float(len(text) % 5), 1.0]


def test_run_diagnostics_computes_report_without_stability(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    assert report.run_id == "run_001"
    assert report.n_rows == 4
    assert report.register_variance.n == 2  # only the 2 non-ignore rows have quotes
    assert report.homogeneity.n_quotes == 2
    assert report.specificity.n_null_events == 1  # evt_002 is the only expected_null event
    assert report.specificity.false_positive_rate == pytest.approx(0.0)  # B30 correctly said no backlash
    assert report.stability is None
    assert report.span_dispersion is None


def test_run_diagnostics_raises_without_manifest_dir(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    with pytest.raises(DiagnosticsError, match="no run at"):
        asyncio.run(
            run_diagnostics(repo=diagnostics_sandbox, run_id="does_not_exist", embedding_client=embedding_client)
        )


def test_run_diagnostics_runs_stability_when_llm_client_given(diagnostics_sandbox):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)

    def responder(request: LLMRequest, attempt: int) -> str:
        return json.dumps(
            {"reaction": "criticize", "categories": ["privacy"], "intensity": 0.8, "quote": "q", "reasoning": "r"}
        )

    llm_client = FakeLLMClient(responder)
    report = asyncio.run(
        run_diagnostics(
            repo=diagnostics_sandbox, run_id="run_001",
            embedding_client=embedding_client, llm_client=llm_client, stability_sample=1,
        )
    )
    assert report.stability is not None
    assert report.stability.n_reruns == 5
    assert report.stability.category_agreement_rate == pytest.approx(1.0)


def test_write_report_creates_json_and_txt(diagnostics_sandbox, tmp_path):
    embedding_client = FakeEmbeddingClient(fake_embedding_responder)
    report = asyncio.run(
        run_diagnostics(repo=diagnostics_sandbox, run_id="run_001", embedding_client=embedding_client)
    )
    run_dir = diagnostics_sandbox / "runs" / "run_001"
    write_report(report, run_dir)
    assert (run_dir / "diagnostics_report.json").exists()
    assert (run_dir / "diagnostics_report.txt").exists()
    text = (run_dir / "diagnostics_report.txt").read_text(encoding="utf-8")
    assert "Homogeneity" in text
    assert "Stability: not measured" in text
