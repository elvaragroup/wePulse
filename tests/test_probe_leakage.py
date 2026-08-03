from __future__ import annotations

import asyncio
import csv
import json
import shutil

import pytest

from src.events import Event
from src.llm import FakeLLMClient, LLMRequest
from src.probe_leakage import LeakageError, PROBES_PER_EVENT, probe_event, run_probe, write_leakage_csv

EVENT = Event(
    id="evt_001",
    company="Acme Corp",
    sector="consumer_tech",
    date="2026-06-14",
    headline="Acme launches an AI layer",
    source_url=None,
    expected_null=False,
    illustrative=False,
    announcement="Acme Corp today announced an AI layer.",
    prior_statements=None,
)


def verdict(describes_reaction: bool, response: str = "some text") -> str:
    return json.dumps({"describes_reaction": describes_reaction, "response": response})


def test_probe_event_asks_the_verbatim_probe_question(config):
    client = FakeLLMClient(lambda request, attempt: verdict(False))
    asyncio.run(probe_event(client, config, EVENT))
    assert all(r.user == EVENT.probe_question() for r in client.requests)


def test_probe_event_runs_three_times(config):
    client = FakeLLMClient(lambda request, attempt: verdict(False))
    result = asyncio.run(probe_event(client, config, EVENT))
    assert len(client.requests) == PROBES_PER_EVENT
    assert len(result.verdicts) == PROBES_PER_EVENT


def test_probe_event_all_clean_is_not_contaminated(config):
    client = FakeLLMClient(lambda request, attempt: verdict(False))
    result = asyncio.run(probe_event(client, config, EVENT))
    assert result.contaminated is False


def test_probe_event_any_leak_marks_contaminated(config):
    calls = {"n": 0}

    def responder(request: LLMRequest, attempt: int) -> str:
        calls["n"] += 1
        return verdict(calls["n"] == 3)  # only the third attempt leaks

    client = FakeLLMClient(responder)
    result = asyncio.run(probe_event(client, config, EVENT))
    assert result.contaminated is True
    assert sum(v.describes_reaction for v in result.verdicts) == 1


def test_each_attempt_gets_a_distinct_cache_key(config, tmp_path):
    """Three identical-looking probes must not collapse into one cached call --
    the per-attempt system-prompt marker exists precisely to prevent that."""
    seen_keys = set()

    def responder(request: LLMRequest, attempt: int) -> str:
        seen_keys.add(request.cache_key())
        return verdict(False)

    client = FakeLLMClient(responder, cache_dir=tmp_path / "cache")
    asyncio.run(probe_event(client, config, EVENT))
    assert len(seen_keys) == PROBES_PER_EVENT
    assert client.cache_hits == 0


def test_write_leakage_csv_one_row_per_attempt(tmp_path):
    from src.probe_leakage import EventLeakage
    from src.models import LeakageVerdict

    result = EventLeakage(
        event_id="evt_001",
        contaminated=True,
        verdicts=(
            LeakageVerdict(describes_reaction=False, response="NO KNOWLEDGE"),
            LeakageVerdict(describes_reaction=False, response="NO KNOWLEDGE"),
            LeakageVerdict(describes_reaction=True, response="It went badly."),
        ),
    )
    path = tmp_path / "leakage.csv"
    write_leakage_csv([result], path)

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert [r["verdict"] for r in rows] == ["CLEAN", "CLEAN", "CONTAMINATED"]
    assert all(r["event_id"] == "evt_001" for r in rows)


# --- run_probe orchestration ---


@pytest.fixture
def sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copy(repo / "config.yaml", root / "config.yaml")
    (root / "inputs").mkdir()
    (root / "runs" / "run_001").mkdir(parents=True)
    return root


EVENTS_TEXT = (
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
)


def test_run_probe_requires_an_initialised_run(sandbox):
    (sandbox / "inputs" / "events.txt").write_text(EVENTS_TEXT, encoding="utf-8")
    client = FakeLLMClient(lambda request, attempt: verdict(False))
    with pytest.raises(LeakageError, match="no run at"):
        asyncio.run(run_probe(repo=sandbox, run_id="does_not_exist", client=client))


def test_run_probe_writes_leakage_csv(sandbox):
    (sandbox / "inputs" / "events.txt").write_text(EVENTS_TEXT, encoding="utf-8")
    client = FakeLLMClient(lambda request, attempt: verdict(False))
    results = asyncio.run(run_probe(repo=sandbox, run_id="run_001", client=client))

    assert len(results) == 1
    csv_path = sandbox / "runs" / "run_001" / "leakage.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == PROBES_PER_EVENT
