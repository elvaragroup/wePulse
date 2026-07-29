from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src import ground_truth
from src.label_truth import LabelError, label_event, label_text, run_label
from src.llm import FakeLLMClient, LLMRequest

VERDICT_PRIVACY = json.dumps(
    {
        "dominant_category": "privacy",
        "present_categories": ["privacy", "security"],
        "backlash_occurred": True,
        "judge_confidence": 0.8,
    }
)
VERDICT_NONE = json.dumps(
    {
        "dominant_category": "none",
        "present_categories": ["none"],
        "backlash_occurred": False,
        "judge_confidence": 0.9,
    }
)
VERDICT_INCONSISTENT = json.dumps(
    {
        "dominant_category": "none",
        "present_categories": ["privacy"],  # missing 'none' -- fails the cross-check
        "backlash_occurred": True,
        "judge_confidence": 0.5,
    }
)

RAW_TEXT = (
    "=== GROUND TRUTH ===\n"
    "id: evt_001\n"
    "observed_at: 2026-06-17T09:00:00Z\n"
    "collection_rule: top 100 replies\n"
    "---\n"
    "This is an outrage, opt-out is not consent.\n"
    "=== END ===\n"
)


@pytest.fixture(autouse=True)
def _reset_reads():
    ground_truth.reset_reads()
    yield
    ground_truth.reset_reads()


def responder_factory(text: str):
    def responder(request: LLMRequest, _attempt: int) -> str:
        assert request.role == "judge"
        # The event id and arm names must never reach the judge (spec 4.4).
        assert "evt_001" not in request.system
        assert not any(name in request.system for name in ("B3", "B8", "B15", "B30"))
        return text

    return responder


def test_label_text_never_sees_event_metadata(taxonomy, config):
    client = FakeLLMClient(responder_factory(VERDICT_PRIVACY))
    import asyncio

    verdict = asyncio.run(label_text(client, config, taxonomy, "some reaction text"))
    assert verdict.dominant_category == "privacy"
    assert client.requests[0].user == "some reaction text"


def test_label_event_builds_ground_truth_label(taxonomy, config):
    import asyncio

    client = FakeLLMClient(responder_factory(VERDICT_NONE))
    raw = ground_truth.parse_raw(
        RAW_TEXT.replace("This is an outrage, opt-out is not consent.", "Nothing much happened."),
        path=Path("evt_001.txt"),
    )
    label = asyncio.run(label_event(client, config, taxonomy, raw))
    assert label.event_id == "evt_001"
    assert label.dominant_category == "none"
    assert label.backlash_occurred is False


def test_label_event_raises_on_internally_inconsistent_verdict(taxonomy, config):
    import asyncio

    client = FakeLLMClient(responder_factory(VERDICT_INCONSISTENT))
    raw = ground_truth.parse_raw(RAW_TEXT, path=Path("evt_001.txt"))
    with pytest.raises(LabelError, match="internally inconsistent"):
        asyncio.run(label_event(client, config, taxonomy, raw))


# --- run_label end-to-end ---


@pytest.fixture
def sandbox(tmp_path, repo):
    root = tmp_path / "crisis-sim"
    root.mkdir()
    shutil.copy(repo / "config.yaml", root / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", root / "taxonomy.txt")
    (root / "ground_truth" / "raw").mkdir(parents=True)
    (root / "ground_truth" / "labeled").mkdir(parents=True)
    (root / "runs" / "run_001" / "predictions").mkdir(parents=True)
    return root


def test_run_label_requires_predictions_first(sandbox):
    import asyncio

    (sandbox / "ground_truth" / "raw" / "evt_001.txt").write_text(RAW_TEXT, encoding="utf-8")
    client = FakeLLMClient(responder_factory(VERDICT_PRIVACY))
    with pytest.raises(LabelError, match="no predictions for evt_001"):
        asyncio.run(run_label(repo=sandbox, run_id="run_001", client=client))


def test_run_label_writes_labeled_json(sandbox):
    import asyncio

    (sandbox / "ground_truth" / "raw" / "evt_001.txt").write_text(RAW_TEXT, encoding="utf-8")
    (sandbox / "runs" / "run_001" / "predictions" / "evt_001__A.json").write_text("{}", encoding="utf-8")

    client = FakeLLMClient(responder_factory(VERDICT_PRIVACY))
    labels = asyncio.run(run_label(repo=sandbox, run_id="run_001", client=client))

    assert len(labels) == 1
    assert labels[0].event_id == "evt_001"
    out_path = sandbox / "ground_truth" / "labeled" / "evt_001.json"
    assert out_path.exists()
    assert json.loads(out_path.read_text())["dominant_category"] == "privacy"


def test_run_label_raises_with_no_raw_ground_truth(sandbox):
    import asyncio

    client = FakeLLMClient(responder_factory(VERDICT_PRIVACY))
    with pytest.raises(LabelError, match="no raw ground truth"):
        asyncio.run(run_label(repo=sandbox, run_id="run_001", client=client))
