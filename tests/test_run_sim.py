from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src import ground_truth
from src.llm import FakeLLMClient, LLMRequest
from src.run_sim import RunError, RunPaths, execute_run, init_run, parse_human_file

EVENTS = """\
=== EVENT ===
id: evt_001
company: Acme Corp
sector: consumer_tech
date: 2026-06-14
headline: Acme launches an AI layer
expected_null: false
---
Acme Corp today announced an AI layer, enabled by default, drawing on partner data.
=== END EVENT ===

=== EVENT ===
id: evt_002
company: Northwind
sector: industrial
date: 2026-06-20
headline: Northwind opens a warehouse
expected_null: true
---
Northwind opened a distribution centre, adding 400 jobs at existing pay grades.
=== END EVENT ===
"""

REACTION_CRITICIZE = json.dumps(
    {
        "reaction": "criticize",
        "categories": ["privacy"],
        "intensity": 1.0,
        "quote": "Opt-out is not consent.",
        "reasoning": "Defaults users in.",
    }
)
REACTION_IGNORE = json.dumps(
    {"reaction": "ignore", "categories": [], "intensity": 0.0, "quote": None, "reasoning": "not mine"}
)
NAIVE = json.dumps(
    {
        "risks": [
            {"category": "privacy", "rationale": "Data sharing."},
            {"category": "security", "rationale": "New surface."},
            {"category": "legal", "rationale": "Regulators."},
        ]
    }
)


@pytest.fixture
def sandbox(tmp_path, repo):
    """A throwaway copy of the repo so runs never touch the real runs/ directory."""
    root = tmp_path / "crisis-sim"
    root.mkdir()
    for item in ("config.yaml", "taxonomy.txt"):
        shutil.copy(repo / item, root / item)
    shutil.copytree(repo / "personas", root / "personas")
    (root / "inputs").mkdir()
    (root / "inputs" / "events.txt").write_text(EVENTS, encoding="utf-8")
    (root / "ground_truth" / "raw").mkdir(parents=True)
    ground_truth.reset_reads()
    yield root
    ground_truth.reset_reads()


def responder(request: LLMRequest, _attempt: int) -> str:
    if request.role == "naive":
        return NAIVE
    # Only the privacy-adjacent personas react, so scores stay hand-checkable.
    return REACTION_CRITICIZE if "privacy" in request.system[:400].lower() else REACTION_IGNORE


def fill_human(sandbox: Path, run_id: str, ranked=("privacy", "security", "legal")) -> None:
    human = RunPaths(sandbox / "runs" / run_id).human
    for stub in human.glob("*.txt"):
        stub.write_text("\n".join(ranked) + "\n", encoding="utf-8")


async def run_all(sandbox: Path, **kwargs):
    run_id = init_run(repo=sandbox)
    fill_human(sandbox, run_id)
    client = FakeLLMClient(responder, cache_dir=sandbox / "runs" / run_id / "cache", backoff_base=0)
    predictions = await execute_run(run_id, repo=sandbox, client=client, **kwargs)
    return run_id, predictions, client


# --- init ---


def test_init_creates_the_run_skeleton(sandbox):
    run_id = init_run(repo=sandbox)
    paths = RunPaths(sandbox / "runs" / run_id)
    assert paths.manifest.exists()
    assert sorted(p.name for p in paths.human.glob("*.txt")) == ["evt_001.txt", "evt_002.txt"]
    assert paths.cache.is_dir()


def test_run_id_carries_the_persona_hash(sandbox):
    from src.personas import persona_set_hash

    run_id = init_run(repo=sandbox)
    assert run_id.endswith(persona_set_hash(sandbox / "personas")[:8])


def test_init_writes_frozen_json_once(sandbox):
    init_run(repo=sandbox)
    frozen = json.loads((sandbox / "frozen.json").read_text(encoding="utf-8"))
    first = frozen["persona_set_hash"]
    init_run(repo=sandbox)
    assert json.loads((sandbox / "frozen.json").read_text(encoding="utf-8"))["persona_set_hash"] == first


def test_edited_personas_mark_a_run_dev_set(sandbox):
    init_run(repo=sandbox)
    target = sandbox / "personas" / "001_privacy_hawk.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nAn extra line.\n", encoding="utf-8")

    run_id = init_run(repo=sandbox)
    manifest = json.loads(RunPaths(sandbox / "runs" / run_id).manifest.read_text(encoding="utf-8"))
    assert manifest["dev_set"] is True


def test_manifest_records_what_a_rerun_needs(sandbox):
    run_id = init_run(repo=sandbox)
    manifest = json.loads(RunPaths(sandbox / "runs" / run_id).manifest.read_text(encoding="utf-8"))

    assert manifest["persona_set_hash"]
    assert manifest["taxonomy_hash"]
    assert manifest["config"]["models"]["persona"] == "claude-haiku-4-5-20251001"
    assert set(manifest["library_versions"]) == {"python", "anthropic", "pydantic"}
    assert len(manifest["prompt_hashes"]) == 31  # 30 personas + naive
    assert manifest["deviations"]
    assert [e["id"] for e in manifest["events"]] == ["evt_001", "evt_002"]


# --- arm C gate (spec 4.2) ---


async def test_execute_refuses_while_human_stubs_are_empty(sandbox):
    run_id = init_run(repo=sandbox)
    client = FakeLLMClient(responder, backoff_base=0)
    with pytest.raises(RunError, match="no prediction in it"):
        await execute_run(run_id, repo=sandbox, client=client)
    assert client.calls_made == 0


async def test_execute_refuses_on_a_partially_filled_set(sandbox):
    run_id = init_run(repo=sandbox)
    human = RunPaths(sandbox / "runs" / run_id).human
    (human / "evt_001.txt").write_text("privacy\nsecurity\nlegal\n", encoding="utf-8")
    client = FakeLLMClient(responder, backoff_base=0)
    with pytest.raises(RunError, match="evt_002"):
        await execute_run(run_id, repo=sandbox, client=client)


def test_human_stub_lists_valid_ids(sandbox, taxonomy):
    run_id = init_run(repo=sandbox)
    stub = (RunPaths(sandbox / "runs" / run_id).human / "evt_001.txt").read_text(encoding="utf-8")
    for cid in taxonomy.ids:
        assert cid in stub
    assert "BEFORE you look at" in stub
    # The stub itself must not parse as a prediction -- it is all comments.
    with pytest.raises(RunError, match="no prediction in it"):
        parse_human_file(RunPaths(sandbox / "runs" / run_id).human / "evt_001.txt", taxonomy)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("privacy\nsecurity\n", "expected exactly 3"),
        ("privacy\nsecurity\nlegal\npricing\n", "expected exactly 3"),
        ("privacy\nvibes\nlegal\n", "unknown category"),
        ("privacy\nprivacy\nlegal\n", "must be distinct"),
        ("privacy\nnone\nlegal\n", "may only appear first"),
    ],
)
def test_bad_human_files_are_rejected(tmp_path, taxonomy, content, match):
    path = tmp_path / "evt_001.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RunError, match=match):
        parse_human_file(path, taxonomy)


def test_human_file_ignores_comments(tmp_path, taxonomy):
    path = tmp_path / "evt_001.txt"
    path.write_text("# a note\nprivacy\n# another\nsecurity\nlegal\n", encoding="utf-8")
    assert parse_human_file(path, taxonomy) == ["privacy", "security", "legal"]


# --- execution ---


async def test_execute_writes_one_prediction_per_event_per_arm(sandbox):
    run_id, predictions, _ = await run_all(sandbox)
    paths = RunPaths(sandbox / "runs" / run_id)

    files = sorted(p.name for p in paths.predictions.glob("*.json"))
    assert len(files) == 12  # 2 events x 6 arms
    for arm in ("A", "B3", "B8", "B15", "B30", "C"):
        assert f"evt_001__{arm}.json" in files

    for event_predictions in predictions.values():
        assert {p.arm for p in event_predictions} == {"A", "B3", "B8", "B15", "B30", "C"}
        for prediction in event_predictions:
            assert len(prediction.ranked_categories) == 3


async def test_personas_are_sampled_once_and_shared_across_arms(sandbox):
    """30 personas + 1 naive call per event. If arms resampled, this would be far higher."""
    _, _, client = await run_all(sandbox)
    assert client.calls_made == 2 * 31


async def test_rerun_is_served_entirely_from_cache(sandbox):
    run_id, _, first = await run_all(sandbox)

    def explode(_request, _attempt):
        raise AssertionError("a cached run must make no calls")

    second = FakeLLMClient(explode, cache_dir=sandbox / "runs" / run_id / "cache", backoff_base=0)
    await execute_run(run_id, repo=sandbox, client=second)
    assert second.calls_made == 0
    assert second.cache_hits == first.calls_made


async def test_raw_reactions_are_written_for_audit(sandbox):
    run_id, _, _ = await run_all(sandbox)
    paths = RunPaths(sandbox / "runs" / run_id)
    assert len(list((paths.raw / "B30" / "evt_001").glob("*.json"))) == 30
    assert (paths.raw / "A" / "evt_001" / "naive.json").exists()


async def test_manifest_is_completed(sandbox):
    run_id, _, client = await run_all(sandbox)
    manifest = json.loads(RunPaths(sandbox / "runs" / run_id).manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["started_at"] and manifest["completed_at"]
    assert manifest["api_calls"] == client.calls_made
    assert manifest["subset_members"]["B30"]


async def test_editing_personas_after_init_blocks_execution(sandbox):
    run_id = init_run(repo=sandbox)
    fill_human(sandbox, run_id)
    target = sandbox / "personas" / "001_privacy_hawk.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nEdited mid-run.\n", encoding="utf-8")

    client = FakeLLMClient(responder, backoff_base=0)
    with pytest.raises(RunError, match="frozen once a run begins"):
        await execute_run(run_id, repo=sandbox, client=client)


# --- spec 0.2: no ground truth may be read during a run ---


async def test_execute_refuses_if_ground_truth_was_read(sandbox):
    run_id = init_run(repo=sandbox)
    fill_human(sandbox, run_id)

    raw = sandbox / "ground_truth" / "raw"
    (raw / "evt_001.txt").write_text(
        "=== GROUND TRUTH ===\n"
        "id: evt_001\n"
        "observed_at: 2026-06-17T09:00:00Z\n"
        "collection_rule: top 100 replies\n"
        "---\n"
        "people were annoyed\n"
        "=== END ===\n",
        encoding="utf-8",
    )
    ground_truth.read_raw(raw, "evt_001")

    client = FakeLLMClient(responder, backoff_base=0)
    with pytest.raises(ground_truth.GroundTruthError, match="spec 0.2"):
        await execute_run(run_id, repo=sandbox, client=client)
    assert client.calls_made == 0


async def test_a_clean_run_reads_no_ground_truth(sandbox):
    await run_all(sandbox)
    assert ground_truth.reads_so_far() == ()


# --- expected_null containment (spec 2.3) ---


async def test_expected_null_never_reaches_a_prompt(sandbox):
    _, _, client = await run_all(sandbox)
    assert client.requests
    for request in client.requests:
        assert "expected_null" not in request.system
        assert "expected_null" not in request.user


async def test_event_metadata_never_reaches_a_prompt(sandbox):
    """Personas see the announcement text only -- not the headline, id, or date,
    any of which could cue a model that has seen coverage of the event."""
    _, _, client = await run_all(sandbox)
    for request in client.requests:
        assert "evt_001" not in request.user
        assert "Acme launches an AI layer" not in request.user
        assert "2026-06-14" not in request.user


async def test_prior_statements_reach_only_hypocrisy_personas(sandbox):
    events = EVENTS.replace(
        "=== END EVENT ===\n\n=== EVENT ===\nid: evt_002",
        "--- PRIOR STATEMENTS ---\n2024-01-01: \"We will never share data.\"\n"
        "=== END EVENT ===\n\n=== EVENT ===\nid: evt_002",
        1,
    )
    (sandbox / "inputs" / "events.txt").write_text(events, encoding="utf-8")

    _, _, client = await run_all(sandbox)
    carrying = [r for r in client.requests if "We will never share data." in r.system]
    assert carrying, "hypocrisy personas should have received the receipts"
    assert all("hypocrisy" in r.system.lower() for r in carrying)
    assert len(carrying) == 2  # personas 006 and 020, one event
