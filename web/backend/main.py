"""FastAPI HTTP layer for the client-facing crisis-sim demo web app.

Route/mount declaration order matters here: FastAPI/Starlette matches routes
and mounts in declaration order, so the `/api/*` routes MUST be declared
before the catch-all static-file mount at `/` -- otherwise the mount would
swallow every request (including `/api/*`) as a static-file lookup.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from web.backend import service
from web.backend.runs import WebDataError
from web.backend.schemas import (
    EventResultResponse,
    EventsListResponse,
    to_event_result_response,
    to_events_list_response,
)

app = FastAPI(title="crisis-sim demo")


@app.get("/api/events")
def get_events() -> EventsListResponse:
    summaries = service.list_event_summaries()
    return to_events_list_response(summaries)


@app.get("/api/events/{event_id}/result")
def get_event_result(event_id: str) -> EventResultResponse:
    try:
        result = service.build_event_result(event_id)
    except WebDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return to_event_result_response(result)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
