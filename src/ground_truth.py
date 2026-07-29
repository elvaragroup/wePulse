"""The only module permitted to read ground truth.

Spec 0.2: predictions must be written to disk before ground truth is ever read.
Making that a comment in run_sim.py would be a promise; routing every read
through here makes it checkable. `reads_so_far()` is asserted empty by run_sim
before and after it runs, so a future edit that reaches for ground truth mid-run
fails loudly instead of quietly invalidating the study.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_READS: list[str] = []

HEADER_BEGIN = "=== GROUND TRUTH ==="
HEADER_END = "=== END ==="
SEP = "---"

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class GroundTruthError(ValueError):
    pass


@dataclass(frozen=True)
class RawGroundTruth:
    event_id: str
    observed_at: str
    collection_rule: str
    reactions: str
    path: Path


def reads_so_far() -> tuple[str, ...]:
    return tuple(_READS)


def reset_reads() -> None:
    """Test-only. Production code never clears the log."""
    _READS.clear()


def assert_untouched(context: str) -> None:
    if _READS:
        raise GroundTruthError(
            f"{context}: ground truth was read during this process "
            f"({sorted(set(_READS))}). Predictions must be committed to disk before "
            f"any ground truth is read (spec 0.2)."
        )


def parse_raw(text: str, *, path: Path) -> RawGroundTruth:
    body = text.strip()
    if not body.startswith(HEADER_BEGIN):
        raise GroundTruthError(f"{path}: must begin with {HEADER_BEGIN!r}")
    if not body.endswith(HEADER_END):
        raise GroundTruthError(f"{path}: must end with {HEADER_END!r}")

    inner = body[len(HEADER_BEGIN) : -len(HEADER_END)]
    if f"\n{SEP}" not in inner:
        raise GroundTruthError(f"{path}: missing {SEP!r} separator between header and reactions")

    header_raw, _, reactions = inner.partition(f"\n{SEP}")

    headers: dict[str, str] = {}
    for raw in header_raw.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise GroundTruthError(f"{path}: expected 'key: value', got {line!r}")
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()

    for required in ("id", "observed_at", "collection_rule"):
        if required not in headers:
            raise GroundTruthError(f"{path}: missing required header {required!r}")

    if not _ISO.match(headers["observed_at"]):
        raise GroundTruthError(
            f"{path}: observed_at must be ISO 8601 UTC like 2026-06-17T09:00:00Z, "
            f"got {headers['observed_at']!r}"
        )

    if not reactions.strip():
        raise GroundTruthError(f"{path}: no reaction text found")

    return RawGroundTruth(
        event_id=headers["id"],
        observed_at=headers["observed_at"],
        collection_rule=headers["collection_rule"],
        reactions=reactions.strip(),
        path=path,
    )


def read_raw(directory: Path, event_id: str) -> RawGroundTruth:
    path = directory / f"{event_id}.txt"
    if not path.exists():
        raise GroundTruthError(f"no ground truth at {path}")
    _READS.append(str(path))
    parsed = parse_raw(path.read_text(encoding="utf-8"), path=path)
    if parsed.event_id != event_id:
        raise GroundTruthError(
            f"{path}: header id is {parsed.event_id!r} but the filename says {event_id!r}"
        )
    return parsed
