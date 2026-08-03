from __future__ import annotations

import pytest

from src.events import EventParseError, load_events, null_share, parse_events

TWO_EVENTS = """\
=== EVENT ===
id: evt_001
company: Acme Corp
sector: consumer_tech
date: 2026-06-14
headline: Acme launches something
source_url: https://example.invalid/a
expected_null: false
---
FOR IMMEDIATE RELEASE

Acme Corp today announced a thing.
=== END EVENT ===

=== EVENT ===
id: evt_002
company: Northwind
sector: industrial
date: 2026-06-20
headline: Northwind opens a warehouse
expected_null: true
---
Northwind opened a warehouse.
--- PRIOR STATEMENTS ---
2024-01-01, blog: "We will never open a warehouse."
=== END EVENT ===
"""


def test_round_trips_a_multi_event_file():
    events = parse_events(TWO_EVENTS)
    assert [e.id for e in events] == ["evt_001", "evt_002"]

    first, second = events
    assert first.company == "Acme Corp"
    assert first.source_url == "https://example.invalid/a"
    assert first.expected_null is False
    assert first.prior_statements is None
    assert first.announcement.startswith("FOR IMMEDIATE RELEASE")
    assert first.announcement.endswith("announced a thing.")

    assert second.source_url is None
    assert second.expected_null is True
    assert second.prior_statements == '2024-01-01, blog: "We will never open a warehouse."'
    assert "PRIOR STATEMENTS" not in second.announcement


def test_illustrative_defaults_false_when_header_absent():
    events = parse_events(TWO_EVENTS)
    assert all(e.illustrative is False for e in events)


def test_illustrative_true_when_header_present():
    text = TWO_EVENTS.replace(
        "expected_null: true\n---",
        "expected_null: true\nillustrative: true\n---",
    )
    events = parse_events(text)
    assert events[1].illustrative is True
    assert events[0].illustrative is False


def test_illustrative_never_appears_in_a_prompt():
    text = TWO_EVENTS.replace(
        "expected_null: true\n---",
        "expected_null: true\nillustrative: true\n---",
    )
    for event in parse_events(text):
        prompt = event.to_prompt()
        assert "illustrative" not in prompt


def test_repo_events_file_parses(repo):
    events = load_events(repo / "inputs" / "events.txt")
    assert len(events) == 24
    assert events[2].prior_statements is not None
    assert sum(1 for e in events if e.illustrative) == 1


def test_null_share():
    assert null_share(parse_events(TWO_EVENTS)) == 0.5
    assert null_share([]) == 0.0


# --- expected_null must never reach a model (spec 2.3) ---


def test_expected_null_never_appears_in_a_prompt():
    for event in parse_events(TWO_EVENTS):
        prompt = event.to_prompt()
        assert "expected_null" not in prompt
        assert prompt == event.announcement
        # The literal values must not leak either, in a file where the body
        # genuinely does not contain them.
        assert "true" not in prompt.lower().split()
        assert "false" not in prompt.lower().split()


def test_to_prompt_excludes_metadata():
    event = parse_events(TWO_EVENTS)[0]
    prompt = event.to_prompt()
    assert event.headline not in prompt
    assert event.source_url not in prompt
    assert event.id not in prompt


# --- malformed blocks must raise, never be skipped silently (spec 6) ---


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda t: t.replace("=== END EVENT ===", "", 1), "missing"),
        (lambda t: t.replace("id: evt_001", "id: 1"), "must look like"),
        (lambda t: t.replace("date: 2026-06-14", "date: 14 June 2026"), "YYYY-MM-DD"),
        (lambda t: t.replace("expected_null: false\n", ""), "missing required header"),
        (lambda t: t.replace("company: Acme Corp", "company:"), "company is empty"),
        (lambda t: t.replace("expected_null: false", "expected_null: maybe"), "must be true or false"),
        (lambda t: t.replace("headline: Acme launches something", "headline Acme"), "expected 'key: value'"),
        (lambda t: t.replace("id: evt_002", "id: evt_001"), "duplicate event id"),
        (lambda t: t.replace("sector: consumer_tech", "sector: consumer_tech\nflavour: vanilla"), "unknown header"),
        (lambda t: t.replace("id: evt_001", "id: evt_001\nid: evt_009"), "duplicate header"),
    ],
)
def test_malformed_block_raises(mutation, match):
    with pytest.raises(EventParseError, match=match):
        parse_events(mutation(TWO_EVENTS))


def test_empty_body_raises():
    text = TWO_EVENTS.replace("FOR IMMEDIATE RELEASE\n\nAcme Corp today announced a thing.\n", "")
    with pytest.raises(EventParseError, match="announcement body is empty"):
        parse_events(text)


def test_no_event_marker_raises():
    with pytest.raises(EventParseError, match="no '=== EVENT ==='"):
        parse_events("just some prose")


def test_stray_text_after_end_marker_raises():
    text = TWO_EVENTS.replace("=== END EVENT ===\n\n=== EVENT ===", "=== END EVENT ===\noops\n\n=== EVENT ===", 1)
    with pytest.raises(EventParseError, match="unexpected text after"):
        parse_events(text)


def test_leading_comments_are_allowed():
    events = parse_events("# a note\n# another\n\n" + TWO_EVENTS)
    assert len(events) == 2


def test_leading_prose_raises():
    with pytest.raises(EventParseError, match="must be blank or '#' comments"):
        parse_events("some stray prose\n" + TWO_EVENTS)
