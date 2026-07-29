"""Runs the simulation. Two phases, because arm C has to go first.

    python -m src.run_sim --init
    # fill in runs/<run_id>/human/evt_XXX.txt
    python -m src.run_sim --execute <run_id>

Spec 4.2 requires the human's predictions to be written before any other arm
runs, but run_id is only minted when a run starts. `--init` mints the id and
leaves empty stubs; `--execute` refuses to start until every stub is filled. The
resulting file mtimes are real evidence of the ordering, not an assertion about it.

Personas are sampled once per (persona, event) and the sample is shared across
every arm containing that persona. The response cache would force this anyway --
identical request, identical key -- and it is the better design regardless: arms
then differ only by composition, not by resampling noise, which is what makes the
paired tests in score.py meaningful.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import ground_truth
from src.aggregate import aggregate_naive_arm, aggregate_persona_arm, human_arm
from src.config import Config, load_config
from src.events import Event, load_events, null_share
from src.llm import AnthropicLLMClient, LLMClient, LLMRequest, structured_schema
from src.models import ArmPrediction, NaiveBaselineOutput, NaiveBaselineWire, PersonaReaction
from src.personas import (
    Persona,
    load_personas,
    persona_set_hash,
    render_naive_system_prompt,
    render_system_prompt,
    validate_axes,
)
from src.taxonomy import Taxonomy, load_taxonomy

REPO = Path(__file__).resolve().parent.parent
PERSONA_ARMS = ("B3", "B8", "B15", "B30")
MIN_NULL_SHARE = 0.40

DEVIATIONS = [
    "config.temperature.judge is null: claude-sonnet-5 rejects non-default sampling "
    "parameters with a 400 (spec 5 asked for 0.0).",
    "No seed is sent: the Anthropic API exposes no seed parameter (spec 4.3). "
    "Reproducibility is delivered by the response cache alone.",
    "Cache key is sha256 over canonical JSON of the whole request rather than "
    "model+system+user+temperature+seed (spec 4.3).",
    "NaiveBaselineOutput.rationale is requested from the API as a list and rebuilt "
    "into the spec's dict: structured outputs rejects free-keyed objects (spec 3).",
    "Predictions are written to predictions/<event_id>__<arm>.json per spec 4.3; "
    "spec 1's layout diagram shows a different filename.",
]


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def human(self) -> Path:
        return self.root / "human"

    @property
    def cache(self) -> Path:
        return self.root / "cache"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _library_versions() -> dict[str, str]:
    import anthropic
    import pydantic

    return {
        "python": platform.python_version(),
        "anthropic": anthropic.__version__,
        "pydantic": pydantic.VERSION,
    }


# --- arm C stubs ---


def human_stub(event: Event, taxonomy: Taxonomy) -> str:
    return (
        f"# Arm C -- your own prediction for {event.id}, written BEFORE you look at\n"
        f"# any reaction data. You get 15 minutes and the announcement text only.\n"
        f"#\n"
        f"# Company:  {event.company}\n"
        f"# Headline: {event.headline}\n"
        f"#\n"
        f"# Write exactly 3 category ids below, one per line, most likely first.\n"
        f"# If you think nothing will happen, put 'none' first and follow it with\n"
        f"# your two next-best guesses, so arm C stays comparable to the others.\n"
        f"#\n"
        f"# Valid ids: {', '.join(taxonomy.ids)}\n"
        f"#\n"
        f"# Lines starting with # are ignored.\n"
    )


def parse_human_file(path: Path, taxonomy: Taxonomy) -> list[str]:
    if not path.exists():
        raise RunError(f"missing arm C prediction: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise RunError(
            f"{path} has no prediction in it. Arm C must be written before any other "
            f"arm runs (spec 4.2), so the run cannot start until this is filled in."
        )
    if len(lines) != 3:
        raise RunError(f"{path}: expected exactly 3 category ids, found {len(lines)}: {lines}")

    unknown = [line for line in lines if line not in taxonomy.ids]
    if unknown:
        raise RunError(f"{path}: unknown category id(s) {unknown}. Valid: {', '.join(taxonomy.ids)}")
    if len(set(lines)) != 3:
        raise RunError(f"{path}: category ids must be distinct, got {lines}")
    if "none" in lines and lines[0] != "none":
        raise RunError(f"{path}: 'none' may only appear first; got {lines}")

    return lines


# --- init ---


def init_run(
    *,
    repo: Path = REPO,
    events_path: Path | None = None,
) -> str:
    config = load_config(repo / "config.yaml")
    taxonomy = load_taxonomy(repo / "taxonomy.txt")
    events = load_events(events_path or repo / "inputs" / "events.txt")
    personas = load_personas(repo / "personas")
    validate_axes(personas, taxonomy)

    if not events:
        raise RunError("no events to run")

    known = {p.id for p in personas}
    for arm, members in config.subsets.items():
        missing = sorted(set(members) - known)
        if missing:
            raise RunError(f"config subsets.{arm} names unknown persona(s) {missing}")

    set_hash = persona_set_hash(repo / "personas")
    # Milliseconds, so two inits in the same second do not collide.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    run_id = f"{stamp}_{set_hash[:8]}"

    paths = RunPaths(repo / "runs" / run_id)
    if paths.root.exists():
        raise RunError(f"run {run_id} already exists")
    for directory in (paths.raw, paths.predictions, paths.human, paths.cache):
        directory.mkdir(parents=True)

    for event in events:
        (paths.human / f"{event.id}.txt").write_text(human_stub(event, taxonomy), encoding="utf-8")

    frozen_path = repo / "frozen.json"
    if frozen_path.exists():
        frozen_hash = json.loads(frozen_path.read_text(encoding="utf-8"))["persona_set_hash"]
    else:
        frozen_hash = set_hash
        frozen_path.write_text(
            json.dumps(
                {
                    "persona_set_hash": set_hash,
                    "frozen_at": _now(),
                    "note": (
                        "The persona set as first run. Runs whose hash differs are marked "
                        "dev_set and cannot be pooled with this one (spec 0.5)."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    prompt_hashes = {p.id: _sha(render_system_prompt(p, taxonomy)) for p in personas}
    prompt_hashes["naive"] = _sha(render_naive_system_prompt(taxonomy))

    manifest = {
        "run_id": run_id,
        "status": "initialized",
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "persona_set_hash": set_hash,
        "frozen_persona_set_hash": frozen_hash,
        "dev_set": set_hash != frozen_hash,
        "taxonomy_hash": _sha((repo / "taxonomy.txt").read_text(encoding="utf-8")),
        "events_hash": _sha((events_path or repo / "inputs" / "events.txt").read_text("utf-8")),
        "events": [{"id": e.id, "announcement_sha256": _sha(e.announcement)} for e in events],
        "expected_null_share": round(null_share(events), 4),
        "prompt_hashes": prompt_hashes,
        "config": config.snapshot(),
        "library_versions": _library_versions(),
        "deviations": DEVIATIONS,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"\nWrote {len(events)} arm C stub(s) to {paths.human}")
    print("Fill each one in -- 3 category ids, most likely first -- then run:")
    print(f"\n  uv run python -m src.run_sim --execute {run_id}\n")
    if manifest["dev_set"]:
        print(
            "NOTE: the persona set has changed since frozen.json was written. This run "
            "is marked dev_set and score.py will refuse to pool it with earlier runs."
        )
    if null_share(events) < MIN_NULL_SHARE:
        print(
            f"WARNING: only {null_share(events):.0%} of events are expected-null; "
            f"spec 0.6 requires at least {MIN_NULL_SHARE:.0%}."
        )
    return run_id


# --- execute ---


async def _persona_reaction(
    client: LLMClient,
    config: Config,
    taxonomy: Taxonomy,
    persona: Persona,
    event: Event,
) -> PersonaReaction:
    request = LLMRequest(
        role="persona",
        model=config.models["persona"],
        system=render_system_prompt(persona, taxonomy, prior_statements=event.prior_statements),
        user=event.to_prompt(),
        max_tokens=config.max_tokens["persona"],
        temperature=config.temperature_for("persona"),
        thinking=config.thinking_for("persona"),
        schema_name="PersonaReaction",
        schema=structured_schema(PersonaReaction),
    )
    return await client.complete(request, PersonaReaction)


async def _naive_output(
    client: LLMClient,
    config: Config,
    taxonomy: Taxonomy,
    event: Event,
) -> NaiveBaselineOutput:
    request = LLMRequest(
        role="naive",
        model=config.models["naive"],
        system=render_naive_system_prompt(taxonomy),
        user=event.to_prompt(),
        max_tokens=config.max_tokens["naive"],
        temperature=config.temperature_for("naive"),
        thinking=config.thinking_for("naive"),
        schema_name="NaiveBaselineWire",
        schema=structured_schema(NaiveBaselineWire),
    )
    wire = await client.complete(request, NaiveBaselineWire)
    return NaiveBaselineOutput.from_wire(wire)


async def execute_run(
    run_id: str,
    *,
    repo: Path = REPO,
    client: LLMClient | None = None,
    events_path: Path | None = None,
) -> dict[str, list[ArmPrediction]]:
    paths = RunPaths(repo / "runs" / run_id)
    if not paths.manifest.exists():
        raise RunError(f"no run at {paths.root}. Run --init first.")

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    config = load_config(repo / "config.yaml")
    taxonomy = load_taxonomy(repo / "taxonomy.txt")
    events = load_events(events_path or repo / "inputs" / "events.txt")
    personas = load_personas(repo / "personas")
    validate_axes(personas, taxonomy)

    current_hash = persona_set_hash(repo / "personas")
    if current_hash != manifest["persona_set_hash"]:
        raise RunError(
            f"personas changed since --init (manifest {manifest['persona_set_hash'][:8]}, "
            f"now {current_hash[:8]}). Personas are frozen once a run begins (spec 0.5). "
            f"Start a new run with --init."
        )

    # Spec 0.2: nothing in this process may have read ground truth.
    ground_truth.assert_untouched("run_sim.execute_run (before)")

    human_rankings = {e.id: parse_human_file(paths.human / f"{e.id}.txt", taxonomy) for e in events}

    by_id = {p.id: p for p in personas}
    subsets = {**config.subsets, "B30": [p.id for p in personas]}
    union = sorted({pid for arm in PERSONA_ARMS for pid in subsets[arm]})

    owns_client = client is None
    if client is None:
        client = AnthropicLLMClient(
            cache_dir=paths.cache,
            concurrency=config.concurrency,
            max_retries=config.max_retries,
        )

    manifest["started_at"] = _now()
    manifest["status"] = "running"
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    predictions: dict[str, list[ArmPrediction]] = {}

    try:
        for event in events:
            persona_tasks = [
                _persona_reaction(client, config, taxonomy, by_id[pid], event) for pid in union
            ]
            reactions_list, naive = await asyncio.gather(
                asyncio.gather(*persona_tasks),
                _naive_output(client, config, taxonomy, event),
            )
            reactions = dict(zip(union, reactions_list))

            for pid, reaction in reactions.items():
                target = paths.raw / "B30" / event.id
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{pid}.json").write_text(
                    reaction.model_dump_json(indent=2), encoding="utf-8"
                )

            naive_dir = paths.raw / "A" / event.id
            naive_dir.mkdir(parents=True, exist_ok=True)
            (naive_dir / "naive.json").write_text(naive.model_dump_json(indent=2), encoding="utf-8")

            event_predictions = [
                aggregate_naive_arm(event_id=event.id, output=naive, taxonomy=taxonomy),
                human_arm(event_id=event.id, ranked=human_rankings[event.id], taxonomy=taxonomy),
            ]
            for arm in PERSONA_ARMS:
                members = subsets[arm]
                event_predictions.append(
                    aggregate_persona_arm(
                        arm=arm,
                        event_id=event.id,
                        reactions=[reactions[pid] for pid in members],
                        arm_size=len(members),
                        taxonomy=taxonomy,
                        threshold=config.backlash_threshold,
                    )
                )

            for prediction in event_predictions:
                (paths.predictions / f"{event.id}__{prediction.arm}.json").write_text(
                    prediction.model_dump_json(indent=2), encoding="utf-8"
                )
            predictions[event.id] = event_predictions
    finally:
        if owns_client and isinstance(client, AnthropicLLMClient):
            await client.aclose()

    ground_truth.assert_untouched("run_sim.execute_run (after)")

    manifest["completed_at"] = _now()
    manifest["status"] = "complete"
    manifest["arms"] = ["A", *PERSONA_ARMS, "C"]
    manifest["subset_members"] = subsets
    manifest["api_calls"] = client.calls_made
    manifest["cache_hits"] = client.cache_hits
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"{run_id}: {len(events)} event(s) x {len(PERSONA_ARMS) + 2} arms -> "
        f"{sum(len(v) for v in predictions.values())} predictions "
        f"({client.calls_made} API call(s), {client.cache_hits} cache hit(s))"
    )
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true", help="mint a run id and write arm C stubs")
    group.add_argument("--execute", metavar="RUN_ID", help="run arms A and B* for an initialised run")
    parser.add_argument("--events", type=Path, default=None, help="override inputs/events.txt")
    args = parser.parse_args(argv)

    try:
        if args.init:
            init_run(events_path=args.events)
        else:
            asyncio.run(execute_run(args.execute, events_path=args.events))
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
