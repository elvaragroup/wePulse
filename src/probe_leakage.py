"""Pre-simulation leakage probe (spec 4.1).

Before any simulation, each event is probed in a fresh context carrying no
persona and no other instructions: it is asked what it knows about the
announcement and its aftermath, and asked to self-classify whether its own
answer describes a post-announcement reaction. Contaminated events are kept,
not dropped -- score.py reports metrics both with and without them, because
silently dropping one would shrink n without anyone deciding to (spec 4.1).

Each event is probed three times so one lucky or unlucky sample can't decide
the verdict; contaminated if *any* of the three leaks. LLMClient's cache key
does not vary across identical repeated requests -- run three identical calls
through it unmodified and the cache collapses them into one sample. A harmless
per-attempt marker is appended to the *system* prompt purely to give each
attempt its own cache key; the user turn stays exactly the spec's verbatim
question.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, load_config
from src.events import Event, load_events
from src.llm import AnthropicLLMClient, LLMClient, LLMRequest, structured_schema
from src.models import LeakageVerdict

REPO = Path(__file__).resolve().parent.parent
PROBES_PER_EVENT = 3

PROBE_SYSTEM_PROMPT = """\
Answer the question below plainly and honestly.

Set describes_reaction to true if and only if your answer describes any public \
reaction, backlash, controversy, or aftermath that happened after the \
announcement -- not merely the announcement itself. If you have no knowledge of \
the event, respond with exactly "NO KNOWLEDGE" and set describes_reaction to \
false.
"""


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class EventLeakage:
    event_id: str
    contaminated: bool
    verdicts: tuple[LeakageVerdict, ...]


async def probe_event(client: LLMClient, config: Config, event: Event) -> EventLeakage:
    verdicts: list[LeakageVerdict] = []
    for i in range(1, PROBES_PER_EVENT + 1):
        request = LLMRequest(
            role="probe",
            model=config.models["probe"],
            system=f"{PROBE_SYSTEM_PROMPT}\n<!-- probe attempt {i}/{PROBES_PER_EVENT} -->",
            user=event.probe_question(),
            max_tokens=config.max_tokens["probe"],
            temperature=config.temperature_for("probe"),
            thinking=config.thinking_for("probe"),
            schema_name="LeakageVerdict",
            schema=structured_schema(LeakageVerdict),
        )
        verdicts.append(await client.complete(request, LeakageVerdict))

    contaminated = any(v.describes_reaction for v in verdicts)
    return EventLeakage(event_id=event.id, contaminated=contaminated, verdicts=tuple(verdicts))


def write_leakage_csv(results: list[EventLeakage], path: Path) -> None:
    """One row per probe attempt. score.py's loader ORs verdicts by event_id, so
    repeated rows per event are exactly what it expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_id", "verdict", "response"])
        for result in results:
            for verdict in result.verdicts:
                row_verdict = "CONTAMINATED" if verdict.describes_reaction else "CLEAN"
                writer.writerow([result.event_id, row_verdict, verdict.response])


async def run_probe(
    *,
    repo: Path = REPO,
    run_id: str,
    events_path: Path | None = None,
    client: LLMClient | None = None,
) -> list[EventLeakage]:
    config = load_config(repo / "config.yaml")
    events = load_events(events_path or repo / "inputs" / "events.txt")
    if not events:
        raise LeakageError("no events to probe")

    run_dir = repo / "runs" / run_id
    if not run_dir.exists():
        raise LeakageError(f"no run at {run_dir}; run `run_sim.py --init` first")

    owns_client = client is None
    if client is None:
        client = AnthropicLLMClient(
            cache_dir=run_dir / "cache",
            concurrency=config.concurrency,
            max_retries=config.max_retries,
        )

    try:
        results = list(await asyncio.gather(*(probe_event(client, config, e) for e in events)))
    finally:
        if owns_client and isinstance(client, AnthropicLLMClient):
            await client.aclose()

    write_leakage_csv(results, run_dir / "leakage.csv")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--events", type=Path, default=None, help="override inputs/events.txt")
    args = parser.parse_args(argv)

    try:
        results = asyncio.run(run_probe(run_id=args.run_id, events_path=args.events))
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.event_id}: {'CONTAMINATED' if result.contaminated else 'CLEAN'}")
    n_contaminated = sum(r.contaminated for r in results)
    if n_contaminated:
        print(
            f"\n{n_contaminated}/{len(results)} event(s) flagged CONTAMINATED. They are kept, "
            "not dropped -- score.py reports metrics with and without them (spec 4.1)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
