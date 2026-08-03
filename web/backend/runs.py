"""Find which run directory to serve for the web app."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class WebDataError(RuntimeError):
    """Raised when the run directory cannot be determined uniquely or is missing."""

    pass


def get_run_dir(repo: Path = REPO) -> Path:
    """Find the run directory with status='complete'.

    Scans repo/runs/*/manifest.json and returns the directory of the one with
    "status": "complete". Raises WebDataError if zero or multiple complete runs exist.
    """
    runs_dir = repo / "runs"
    if not runs_dir.exists():
        raise WebDataError(f"no runs directory at {runs_dir}")

    complete_runs = []
    for run_subdir in sorted(runs_dir.iterdir()):
        if not run_subdir.is_dir():
            continue

        manifest_path = run_subdir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise WebDataError(f"failed to read {manifest_path}: {exc}") from exc

        if not isinstance(manifest, dict):
            raise WebDataError(f"{manifest_path} did not parse to a JSON object (got {type(manifest).__name__})")

        if manifest.get("status") == "complete":
            complete_runs.append(run_subdir)

    if len(complete_runs) == 0:
        raise WebDataError("no runs with status='complete' found")
    if len(complete_runs) > 1:
        raise WebDataError(f"ambiguous: {len(complete_runs)} complete runs exist (expected exactly 1)")

    return complete_runs[0]
