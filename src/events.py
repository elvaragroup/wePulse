"""Parses inputs/events.txt.

Two properties matter here beyond ordinary parsing:

1. Malformed blocks raise. Silently skipping one would shrink n without anyone
   noticing, and n is already the binding constraint on what this study can
   detect.
2. `expected_null` is the user's a priori guess and must never reach a model
   (spec 2.3). It is parsed onto the Event but `to_prompt()` returns only the
   announcement text, so there is no code path that carries it into a request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BEGIN = "=== EVENT ==="
END = "=== END EVENT ==="
HEADER_SEP = "---"
PRIOR_SEP = "--- PRIOR STATEMENTS ---"

REQUIRED_HEADERS = ("id", "company", "sector", "date", "headline", "expected_null")
OPTIONAL_HEADERS = ("source_url",)

_EVENT_ID = re.compile(r"^evt_\d{3,}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class EventParseError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    id: str
    company: str
    sector: str
    date: str
    headline: str
    source_url: str | None
    expected_null: bool
    announcement: str
    prior_statements: str | None

    def to_prompt(self) -> str:
        """The only text a persona or the naive baseline ever sees."""
        return self.announcement

    def probe_question(self) -> str:
        """Spec 4.1, verbatim. Used only by probe_leakage.py."""
        return (
            f"What do you know about {self.company}'s announcement of "
            f"{self.headline} on {self.date}? Describe what happened afterward, "
            f"including any public reaction. If you have no knowledge of this "
            f"event, say exactly: NO KNOWLEDGE."
        )


def _parse_bool(value: str, *, event_id: str, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    raise EventParseError(f"{event_id}: {field} must be true or false, got {value!r}")


def _parse_block(block: str, *, index: int) -> Event:
    where = f"event block #{index}"

    if PRIOR_SEP in block:
        head_and_body, _, prior_raw = block.partition(PRIOR_SEP)
        prior_statements: str | None = prior_raw.strip() or None
    else:
        head_and_body, prior_statements = block, None

    if HEADER_SEP not in head_and_body:
        raise EventParseError(f"{where}: missing '{HEADER_SEP}' separator between headers and body")

    header_raw, _, body_raw = head_and_body.partition(f"\n{HEADER_SEP}")
    headers: dict[str, str] = {}
    for lineno, raw in enumerate(header_raw.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise EventParseError(f"{where} line {lineno}: expected 'key: value', got {line!r}")
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in headers:
            raise EventParseError(f"{where}: duplicate header {key!r}")
        headers[key] = value.strip()

    missing = [h for h in REQUIRED_HEADERS if h not in headers]
    if missing:
        raise EventParseError(f"{where}: missing required header(s) {missing}")
    unknown = sorted(set(headers) - set(REQUIRED_HEADERS) - set(OPTIONAL_HEADERS))
    if unknown:
        raise EventParseError(f"{where}: unknown header(s) {unknown}")

    event_id = headers["id"]
    if not _EVENT_ID.match(event_id):
        raise EventParseError(f"{where}: id must look like 'evt_001', got {event_id!r}")
    if not _DATE.match(headers["date"]):
        raise EventParseError(f"{event_id}: date must be YYYY-MM-DD, got {headers['date']!r}")

    announcement = body_raw.strip()
    if not announcement:
        raise EventParseError(f"{event_id}: announcement body is empty")

    for field in ("company", "sector", "headline"):
        if not headers[field]:
            raise EventParseError(f"{event_id}: {field} is empty")

    return Event(
        id=event_id,
        company=headers["company"],
        sector=headers["sector"],
        date=headers["date"],
        headline=headers["headline"],
        source_url=headers.get("source_url") or None,
        expected_null=_parse_bool(headers["expected_null"], event_id=event_id, field="expected_null"),
        announcement=announcement,
        prior_statements=prior_statements,
    )


def parse_events(text: str) -> list[Event]:
    if BEGIN not in text:
        raise EventParseError(f"no {BEGIN!r} marker found")

    preamble, _, rest = text.partition(BEGIN)
    if preamble.strip() and not all(
        line.strip().startswith("#") for line in preamble.strip().splitlines()
    ):
        raise EventParseError("text before the first event block must be blank or '#' comments")

    events: list[Event] = []
    for index, chunk in enumerate(rest.split(BEGIN), start=1):
        if END not in chunk:
            raise EventParseError(f"event block #{index}: missing {END!r} marker")
        block, _, trailing = chunk.partition(END)
        if trailing.strip():
            raise EventParseError(
                f"event block #{index}: unexpected text after {END!r}: {trailing.strip()[:60]!r}"
            )
        events.append(_parse_block(block, index=index))

    ids = [e.id for e in events]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EventParseError(f"duplicate event id(s): {duplicates}")

    return events


def load_events(path: Path) -> list[Event]:
    return parse_events(path.read_text(encoding="utf-8"))


def null_share(events: list[Event]) -> float:
    """Spec 0.6 requires >=40% expected-null events. Reported, never scored."""
    if not events:
        return 0.0
    return sum(1 for e in events if e.expected_null) / len(events)
