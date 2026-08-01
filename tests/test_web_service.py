from __future__ import annotations

import pytest

from web.backend.service import WebDataError, build_event_result, list_event_summaries


def test_list_event_summaries_returns_all_23(repo):
    summaries = list_event_summaries(repo=repo)
    assert len(summaries) == 23
    assert summaries[0].id == "evt_001"
    assert summaries[0].company  # non-empty
    ids = [s.id for s in summaries]
    assert ids == sorted(ids)  # evt_001..evt_023 file order


def test_build_event_result_evt_001_matches_real_data(repo):
    result = build_event_result("evt_001", repo=repo)

    assert result.event.id == "evt_001"
    assert result.event.announcement  # non-empty
    # expected_null must never leak into the client-facing dataclass
    assert not hasattr(result.event, "expected_null")

    assert result.naive.backlash_predicted is True
    assert [c.id for c in result.naive.top_categories] == ["privacy", "overclaim", "hypocrisy"]
    assert all(c.confidence is None for c in result.naive.top_categories)

    assert result.ensemble.backlash_predicted is True
    assert [c.id for c in result.ensemble.top_categories] == ["privacy", "overclaim", "security"]
    assert result.ensemble.reaction_counts == {"ignore": 11, "mild_concern": 11, "criticize": 8, "outrage": 0}
    assert len(result.ensemble.sample_quotes) <= 8
    assert result.ensemble.sample_quotes  # non-empty for an event with real reactions

    assert {c.id for c in result.comparison.agreed} == {"privacy", "overclaim"}
    assert [c.id for c in result.comparison.ensemble_only] == ["security"]
    assert [c.id for c in result.comparison.naive_only] == ["hypocrisy"]
    assert result.comparison.backlash_agreement is True


def test_build_event_result_unknown_event_raises(repo):
    with pytest.raises(WebDataError, match="evt_999"):
        build_event_result("evt_999", repo=repo)
