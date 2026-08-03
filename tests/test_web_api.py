from __future__ import annotations

from fastapi.testclient import TestClient

from web.backend.main import app

client = TestClient(app)


def test_get_events_returns_24_sorted():
    response = client.get("/api/events")
    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 24
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)
    assert set(events[0].keys()) == {"id", "company", "headline", "date", "sector", "illustrative"}
    # evt_025 is the one illustrative (fictional, clearly-labeled) demo scenario
    illustrative_ids = [e["id"] for e in events if e["illustrative"]]
    assert illustrative_ids == ["evt_025"]


def test_get_event_result_evt_001():
    response = client.get("/api/events/evt_001/result")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"event", "naive", "ensemble", "comparison", "ground_truth"}
    assert body["event"]["id"] == "evt_001"
    assert "expected_null" not in body["event"]
    assert [c["id"] for c in body["naive"]["top_categories"]] == ["privacy", "overclaim", "hypocrisy"]
    assert [c["id"] for c in body["ensemble"]["top_categories"]] == ["privacy", "overclaim", "security"]
    assert body["comparison"]["backlash_agreement"] is True
    assert [c["id"] for c in body["comparison"]["ensemble_only"]] == ["security"]
    # No real ground truth exists for any event yet (see ground_truth/README.md)
    assert body["ground_truth"] is None


def test_get_event_result_unknown_event_404():
    response = client.get("/api/events/evt_999/result")
    assert response.status_code == 404


def test_root_serves_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


def test_api_routes_not_shadowed_by_static_mount():
    # Regression test for route/mount declaration order: /api/events must
    # resolve to the API handler, never fall through to the static mount.
    # If the mount were declared before the route, /api/events would return 404
    # from StaticFiles rather than 200 with the real events list.
    response = client.get("/api/events")
    assert response.status_code == 200
    body = response.json()
    assert "events" in body
    assert len(body["events"]) == 24
