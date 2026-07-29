from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from src.check_judge import (
    JudgeCheckError,
    KAPPA_FLOOR,
    load_human_labels,
    run_check,
    write_reliability_report,
)
from src.llm import FakeLLMClient, LLMRequest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "text", "human_category"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_human_labels_missing_file_raises(tmp_path):
    with pytest.raises(JudgeCheckError, match="no human labels"):
        load_human_labels(tmp_path / "human_labels.csv")


def test_load_human_labels_missing_column_raises(tmp_path):
    path = tmp_path / "human_labels.csv"
    path.write_text("item_id,text\nitem1,some text\n", encoding="utf-8")
    with pytest.raises(JudgeCheckError, match="missing column"):
        load_human_labels(path)


def test_load_human_labels_empty_raises(tmp_path):
    path = tmp_path / "human_labels.csv"
    write_csv(path, [])
    with pytest.raises(JudgeCheckError, match="no rows"):
        load_human_labels(path)


def test_load_human_labels_parses_rows(tmp_path):
    path = tmp_path / "human_labels.csv"
    write_csv(path, [{"item_id": "item1", "text": "opt-out is not consent", "human_category": "privacy"}])
    labels = load_human_labels(path)
    assert labels[0].item_id == "item1"
    assert labels[0].human_category == "privacy"


# --- run_check ---


def judge_response(category: str) -> str:
    return json.dumps(
        {
            "dominant_category": category,
            "present_categories": [category],
            "backlash_occurred": category != "none",
            "judge_confidence": 0.9,
        }
    )


def test_run_check_rejects_unknown_human_category(tmp_path, repo):
    import shutil

    sandbox = tmp_path / "crisis-sim"
    sandbox.mkdir()
    shutil.copy(repo / "config.yaml", sandbox / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", sandbox / "taxonomy.txt")
    write_csv(
        sandbox / "results" / "human_labels.csv",
        [{"item_id": "item1", "text": "x", "human_category": "not_a_real_category"}],
    )
    client = FakeLLMClient(lambda request, attempt: judge_response("privacy"))
    with pytest.raises(JudgeCheckError, match="not a taxonomy id"):
        asyncio.run(run_check(repo=sandbox, client=client))


def test_run_check_computes_kappa_against_judge(tmp_path, repo):
    import shutil

    sandbox = tmp_path / "crisis-sim"
    sandbox.mkdir()
    shutil.copy(repo / "config.yaml", sandbox / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", sandbox / "taxonomy.txt")
    write_csv(
        sandbox / "results" / "human_labels.csv",
        [
            {"item_id": "item1", "text": "opt-out is not consent", "human_category": "privacy"},
            {"item_id": "item2", "text": "nothing happened", "human_category": "none"},
        ],
    )

    # The judge agrees on both items -> perfect agreement -> kappa == 1.0.
    order = ["privacy", "none"]
    calls = {"n": 0}

    def responder(request: LLMRequest, attempt: int) -> str:
        category = order[calls["n"]]
        calls["n"] += 1
        return judge_response(category)

    client = FakeLLMClient(responder)
    kappa, labels, judge_categories = asyncio.run(run_check(repo=sandbox, client=client))
    assert kappa == pytest.approx(1.0)
    assert judge_categories == ["privacy", "none"]
    assert [lab.item_id for lab in labels] == ["item1", "item2"]


def test_run_check_never_leaks_human_category_to_the_judge(tmp_path, repo):
    import shutil

    sandbox = tmp_path / "crisis-sim"
    sandbox.mkdir()
    shutil.copy(repo / "config.yaml", sandbox / "config.yaml")
    shutil.copy(repo / "taxonomy.txt", sandbox / "taxonomy.txt")
    write_csv(
        sandbox / "results" / "human_labels.csv",
        [{"item_id": "item1", "text": "opt-out is not consent", "human_category": "privacy"}],
    )

    def responder(request: LLMRequest, attempt: int) -> str:
        assert request.user == "opt-out is not consent"
        # Only the raw text goes to the judge -- not the item id or the human's label.
        assert "item1" not in request.system
        return judge_response("security")

    client = FakeLLMClient(responder)
    asyncio.run(run_check(repo=sandbox, client=client))


# --- report ---


def test_report_warns_loudly_below_kappa_floor(tmp_path):
    from src.check_judge import HumanLabel

    labels = [HumanLabel(item_id="item1", text="x", human_category="privacy")]
    report = write_reliability_report(
        tmp_path / "judge_reliability.txt",
        kappa=0.1,
        labels=labels,
        judge_categories=["security"],
    )
    assert "STOP" in report
    assert f"below the {KAPPA_FLOOR}" in report
    assert "item1: privacy / security  <-- disagree" in report


def test_report_notes_sample_size_mismatch(tmp_path):
    from src.check_judge import HumanLabel

    labels = [HumanLabel(item_id="item1", text="x", human_category="privacy")]
    report = write_reliability_report(
        tmp_path / "judge_reliability.txt", kappa=0.9, labels=labels, judge_categories=["privacy"]
    )
    assert "sample size is 1, not the spec's 25" in report
    assert "STOP" not in report
