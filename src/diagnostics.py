"""Anti-collapse diagnostics: turns a run's raw output into the seven
metrics from the v2 persona-engine brief's Phase 5 table, comparable across
engine changes. See
docs/superpowers/plans/2026-07-30-diagnostics-milestone-a.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.embeddings import EmbeddingClient, EmbeddingError
from src.llm import LLMClient
from src.personas import Persona

REPO = Path(__file__).resolve().parent.parent


class DiagnosticsError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiagnosticsRow:
    """Run-shape-agnostic view every metric function consumes. char_start/
    char_end are always None for v1 rows (no spans exist); populated once a
    future Milestone B extends load_rows() to detect pipeline-shaped runs."""

    event_id: str
    persona_id: str
    reaction: str
    text: str | None       # the quote, None iff reaction == "ignore"
    category: str | None   # first category if any, else None
    char_start: int | None
    char_end: int | None


def load_v1_rows(run_dir: Path, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]:
    """Reads every runs/<id>/raw/B30/<event_id>/<persona_id>.json -- B30 holds
    every persona's reaction regardless of which arms ran (same read pattern
    as src/dashboard.py's load_event_reactions). Every persona_id found must
    exist in personas_by_id: silently including a reaction from a persona no
    longer in the current roster (persona set changed since this run) would
    skew every downstream metric without anyone noticing -- fail loudly
    instead, matching this codebase's "malformed data raises, never skips
    silently" convention (see src/events.py's parser)."""
    b30_dir = run_dir / "raw" / "B30"
    if not b30_dir.exists():
        raise DiagnosticsError(f"no raw/B30 reactions at {b30_dir}")

    rows: list[DiagnosticsRow] = []
    for event_dir in sorted(b30_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        for path in sorted(event_dir.glob("*.json")):
            persona_id = path.stem
            if persona_id not in personas_by_id:
                raise DiagnosticsError(
                    f"persona {persona_id!r} not found in the current persona set "
                    f"(reaction at {path}) -- diagnostics refuses to silently analyze "
                    "reactions from personas outside the current roster"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            categories = data.get("categories") or []
            rows.append(
                DiagnosticsRow(
                    event_id=event_dir.name,
                    persona_id=path.stem,
                    reaction=data["reaction"],
                    text=data.get("quote"),
                    category=categories[0] if categories else None,
                    char_start=None,
                    char_end=None,
                )
            )
    if not rows:
        raise DiagnosticsError(f"no reactions found under {b30_dir}")
    return rows


def load_rows(run_dir: Path, *, personas_by_id: dict[str, Persona]) -> list[DiagnosticsRow]:
    """Today this always loads the v1 shape (raw/B30/...) -- it's the only
    shape any run produces yet. A future Milestone B plan extends this to
    shape-detect and dispatch to a pipeline-shaped loader once
    runs/<id>/pipeline/ exists."""
    return load_v1_rows(run_dir, personas_by_id)


from src.diagnostics_metrics import (
    DistributionMatchResult,
    HomogeneityResult,
    RedundancyResult,
    RegisterVarianceResult,
    SpanDispersionResult,
    SpecificityResult,
    StabilityResult,
    distribution_match_partial,
    homogeneity,
    redundancy,
    register_variance,
    specificity,
    stability,
)
from src.personas import load_personas

DEFAULT_STABILITY_SAMPLE = 25
DEFAULT_EMBEDDING_MODEL = "voyage-3-lite"  # verify against Voyage's current model list before real use
STABILITY_RERUNS = 5
STABILITY_SAMPLE_SEED = 20260730


@dataclass(frozen=True)
class DiagnosticsReport:
    run_id: str
    n_rows: int
    register_variance: RegisterVarianceResult
    homogeneity: HomogeneityResult
    redundancy: RedundancyResult
    span_dispersion: SpanDispersionResult | None
    stability: StabilityResult | None
    specificity: SpecificityResult
    distribution_match: DistributionMatchResult | None

    def to_json_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_rows": self.n_rows,
            "register_variance": asdict(self.register_variance),
            "homogeneity": asdict(self.homogeneity),
            "redundancy": asdict(self.redundancy),
            "span_dispersion": asdict(self.span_dispersion) if self.span_dispersion is not None else None,
            "stability": asdict(self.stability) if self.stability is not None else None,
            "specificity": asdict(self.specificity),
            "distribution_match": (
                asdict(self.distribution_match) if self.distribution_match is not None else None
            ),
        }


def _backlash_predicted_by_event(repo: Path, run_id: str, event_ids: set[str]) -> dict[str, bool]:
    predictions_dir = repo / "runs" / run_id / "predictions"
    result: dict[str, bool] = {}
    for event_id in event_ids:
        path = predictions_dir / f"{event_id}__B30.json"
        if not path.exists():
            raise DiagnosticsError(f"missing prediction file for event {event_id!r}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        result[event_id] = bool(data["backlash_predicted"])
    return result


def _ground_truth_mix(repo: Path, event_ids: set[str]) -> dict[str, float]:
    labeled_dir = repo / "ground_truth" / "labeled"
    counts: dict[str, int] = {}
    total = 0
    for event_id in event_ids:
        path = labeled_dir / f"{event_id}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for category in data["present_categories"]:
            counts[category] = counts.get(category, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {category: count / total for category, count in counts.items()}


def _simulated_mix(rows: list[DiagnosticsRow]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        if row.category is None:
            continue
        counts[row.category] = counts.get(row.category, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {category: count / total for category, count in counts.items()}


async def _run_stability_check(
    *, client, config, taxonomy, personas_by_id, events_by_id, rows: list[DiagnosticsRow], sample_size: int
) -> StabilityResult | None:
    import random

    from src.llm import LLMRequest, structured_schema
    from src.models import PersonaReaction
    from src.personas import render_system_prompt

    candidates = sorted({(r.persona_id, r.event_id) for r in rows if r.reaction != "ignore"})
    if not candidates:
        return None

    rng = random.Random(STABILITY_SAMPLE_SEED)
    sample = rng.sample(candidates, k=min(sample_size, len(candidates)))

    samples: dict[tuple[str, str], list[tuple[str, frozenset]]] = {}
    for persona_id, event_id in sample:
        persona = personas_by_id[persona_id]
        event = events_by_id[event_id]
        reruns: list[tuple[str, frozenset]] = []
        for attempt in range(1, STABILITY_RERUNS + 1):
            system = render_system_prompt(persona, taxonomy, prior_statements=event.prior_statements)
            system = f"{system}\n<!-- diagnostics stability attempt {attempt}/{STABILITY_RERUNS} -->"
            request = LLMRequest(
                role="persona",
                model=config.models["persona"],
                system=system,
                user=event.to_prompt(),
                max_tokens=config.max_tokens["persona"],
                temperature=config.temperature_for("persona"),
                thinking=config.thinking_for("persona"),
                schema_name="PersonaReaction",
                schema=structured_schema(PersonaReaction),
            )
            reaction = await client.complete(request, PersonaReaction)
            reruns.append((reaction.reaction, frozenset(reaction.categories)))
        samples[(persona_id, event_id)] = reruns

    return stability(samples)


async def run_diagnostics(
    *,
    repo: Path = REPO,
    run_id: str,
    embedding_client: EmbeddingClient,
    llm_client: LLMClient | None = None,
    stability_sample: int = DEFAULT_STABILITY_SAMPLE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> DiagnosticsReport:
    from src.config import load_config
    from src.events import load_events
    from src.taxonomy import load_taxonomy

    run_dir = repo / "runs" / run_id
    if not run_dir.exists():
        raise DiagnosticsError(f"no run at {run_dir}")

    personas_by_id = {p.id: p for p in load_personas(repo / "personas")}
    events = load_events(repo / "inputs" / "events.txt")
    events_by_id = {e.id: e for e in events}
    null_event_ids = {e.id for e in events if e.expected_null}

    rows = load_rows(run_dir, personas_by_id=personas_by_id)
    non_ignore = [r for r in rows if r.reaction != "ignore" and r.text]
    if not non_ignore:
        raise DiagnosticsError(f"{run_dir}: every reaction was 'ignore' -- nothing to measure")

    quotes = [r.text for r in non_ignore]  # type: ignore[misc]
    register = register_variance(quotes)

    embeddings = await embedding_client.embed(quotes, model=embedding_model)
    homog = homogeneity(embeddings)
    redund = redundancy(embeddings, n_reacting_personas=len({r.persona_id for r in non_ignore}))

    event_ids = {r.event_id for r in rows}
    backlash_by_event = _backlash_predicted_by_event(repo, run_id, event_ids)
    relevant_null_ids = null_event_ids & event_ids
    if not relevant_null_ids:
        raise DiagnosticsError(
            f"{run_dir}: no expected_null events overlap this run's events -- specificity cannot "
            "be measured (spec 0.6 requires >=40% null events in the study design)"
        )
    spec_result = specificity(backlash_by_event, relevant_null_ids)

    gt_mix = _ground_truth_mix(repo, event_ids)
    dist_match = distribution_match_partial(_simulated_mix(rows), gt_mix) if gt_mix else None

    stability_result = None
    if llm_client is not None:
        config = load_config(repo / "config.yaml")
        taxonomy = load_taxonomy(repo / "taxonomy.txt")
        stability_result = await _run_stability_check(
            client=llm_client, config=config, taxonomy=taxonomy,
            personas_by_id=personas_by_id, events_by_id=events_by_id,
            rows=rows, sample_size=stability_sample,
        )

    return DiagnosticsReport(
        run_id=run_id, n_rows=len(rows), register_variance=register,
        homogeneity=homog, redundancy=redund, span_dispersion=None,
        stability=stability_result, specificity=spec_result, distribution_match=dist_match,
    )


def write_report(report: DiagnosticsReport, run_dir: Path) -> None:
    (run_dir / "diagnostics_report.json").write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"Diagnostics -- run {report.run_id} ({report.n_rows} rows)",
        "",
        f"Register variance: word_count mean={report.register_variance.word_count_mean:.1f} "
        f"stdev={report.register_variance.word_count_stdev:.2f} "
        f"min={report.register_variance.word_count_min} max={report.register_variance.word_count_max}; "
        f"exclamation_rate={report.register_variance.exclamation_rate:.2%} "
        f"em_dash_rate={report.register_variance.em_dash_rate:.2%}",
        f"Homogeneity: mean_pairwise_cosine={report.homogeneity.mean_pairwise_cosine:.4f} "
        f"(n={report.homogeneity.n_quotes})",
        f"Redundancy: {report.redundancy.n_clusters} clusters / "
        f"{report.redundancy.n_reacting_personas} reacting personas "
        f"(ratio={report.redundancy.ratio:.3f}, noise={report.redundancy.n_noise})",
        "(Homogeneity/Redundancy are computed over quotes pooled across all of this run's "
        "events -- see src/diagnostics_metrics.py docstrings for the topic-vs-voice caveat.)",
        f"Specificity: false_positive_rate={report.specificity.false_positive_rate:.2%} "
        f"({report.specificity.n_false_positive}/{report.specificity.n_null_events} null events)",
    ]
    if report.stability is not None:
        lines.append(
            f"Stability: category_agreement_rate={report.stability.category_agreement_rate:.2%} "
            f"over {report.stability.n_pairs_sampled} pair(s) x {report.stability.n_reruns} reruns"
        )
    else:
        lines.append("Stability: not measured (no llm_client passed)")
    if report.distribution_match is not None:
        lines.append(
            f"Distribution match (partial, vs ground_truth/labeled): "
            f"tvd={report.distribution_match.total_variation_distance:.4f} "
            f"over {report.distribution_match.categories_compared} categor(y/ies)"
        )
    else:
        lines.append("Distribution match: not measured (no ground truth labeled for this run's events)")
    lines.append("Span dispersion: not measured (v1 runs have no span citations)")

    (run_dir / "diagnostics_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline(repo: Path) -> dict | None:
    path = repo / "results" / "diagnostics_baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(report: DiagnosticsReport, repo: Path) -> None:
    path = repo / "results" / "diagnostics_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def compare_to_baseline(report: DiagnosticsReport, baseline: dict) -> str:
    lines = [f"Comparison against results/diagnostics_baseline.json (run {baseline['run_id']}):", ""]
    lines.append(
        f"Homogeneity: {report.homogeneity.mean_pairwise_cosine:.4f} vs "
        f"baseline {baseline['homogeneity']['mean_pairwise_cosine']:.4f}"
    )
    lines.append(
        f"Redundancy ratio: {report.redundancy.ratio:.3f} vs baseline {baseline['redundancy']['ratio']:.3f}"
    )
    lines.append(
        f"Specificity false_positive_rate: {report.specificity.false_positive_rate:.2%} vs "
        f"baseline {baseline['specificity']['false_positive_rate']:.2%}"
    )
    lines.append(
        f"Register variance word_count_stdev: {report.register_variance.word_count_stdev:.2f} vs "
        f"baseline {baseline['register_variance']['word_count_stdev']:.2f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, dest="run_id")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--stability-sample", type=int, default=DEFAULT_STABILITY_SAMPLE)
    parser.add_argument("--skip-stability", action="store_true", help="skip the extra LLM calls Stability needs")
    args = parser.parse_args(argv)

    run_dir = REPO / "runs" / args.run_id

    async def _run() -> DiagnosticsReport:
        from src.embeddings import VoyageEmbeddingClient
        from src.llm import AnthropicLLMClient

        if not run_dir.exists():
            raise DiagnosticsError(f"no run at {run_dir}")

        embedding_client = VoyageEmbeddingClient(cache_dir=run_dir / "cache" / "embeddings")
        llm_client = None if args.skip_stability else AnthropicLLMClient(cache_dir=run_dir / "cache")
        try:
            return await run_diagnostics(
                run_id=args.run_id, embedding_client=embedding_client, llm_client=llm_client,
                stability_sample=args.stability_sample,
            )
        finally:
            await embedding_client.aclose()
            if llm_client is not None:
                await llm_client.aclose()

    try:
        report = asyncio.run(_run())
    except (DiagnosticsError, EmbeddingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_report(report, run_dir)
    print((run_dir / "diagnostics_report.txt").read_text(encoding="utf-8"))

    if args.write_baseline:
        write_baseline(report, REPO)
        print("\nwrote baseline to results/diagnostics_baseline.json")

    if args.compare_baseline:
        baseline = load_baseline(REPO)
        if baseline is None:
            print(
                "\nerror: no baseline at results/diagnostics_baseline.json -- run --write-baseline first",
                file=sys.stderr,
            )
            return 1
        print()
        print(compare_to_baseline(report, baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
