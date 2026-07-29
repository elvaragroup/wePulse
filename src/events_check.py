"""Validates inputs/events.txt without touching the API. `uv run python -m src.events_check`"""

from __future__ import annotations

import sys
from pathlib import Path

from src.events import EventParseError, load_events, null_share

REPO = Path(__file__).resolve().parent.parent
MIN_NULL_SHARE = 0.40


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else REPO / "inputs" / "events.txt"

    try:
        events = load_events(path)
    except EventParseError as exc:
        print(f"FAILED to parse {path}\n  {exc}", file=sys.stderr)
        return 1

    print(f"{path}: {len(events)} event(s)\n")
    for e in events:
        prior = "yes" if e.prior_statements else "no"
        print(
            f"  {e.id}  {e.company:<24} {e.date}  "
            f"expected_null={str(e.expected_null):<5} prior_statements={prior}"
        )
        print(f"         {e.headline}")

    share = null_share(events)
    print(f"\nexpected-null share: {share:.0%} ({sum(e.expected_null for e in events)}/{len(events)})")

    if share < MIN_NULL_SHARE:
        print(
            f"\nWARNING: spec 0.6 requires at least {MIN_NULL_SHARE:.0%} null events.\n"
            "Without them, an arm that cries wolf on everything looks good on the headline\n"
            "metric and its false-positive rate cannot be measured at all.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
