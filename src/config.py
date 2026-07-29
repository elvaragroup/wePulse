"""Loads and validates config.yaml.

The one check worth singling out: models.persona must equal models.naive. If they
differ, arm A vs arm B compares architecture *and* model capability at once,
which is the single thing this study exists to separate (spec 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROLES = ("persona", "naive", "judge", "probe")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    models: dict[str, str]
    temperature: dict[str, float | None]
    thinking: dict[str, str]
    max_tokens: dict[str, int]
    k: int
    backlash_threshold: float
    concurrency: int
    max_retries: int
    bootstrap_resamples: int
    subsets: dict[str, list[str]]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def temperature_for(self, role: str) -> float | None:
        return self.temperature.get(role)

    def thinking_for(self, role: str) -> str | None:
        return self.thinking.get(role)

    def snapshot(self) -> dict[str, Any]:
        """Verbatim config as loaded, for manifest.json."""
        return self.raw


def _require(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in mapping:
        raise ConfigError(f"config.yaml: missing required key {key!r}")
    value = mapping[key]
    if not isinstance(value, kind):
        raise ConfigError(f"config.yaml: {key!r} must be {kind}, got {type(value).__name__}")
    return value


def parse_config(data: dict[str, Any]) -> Config:
    models = _require(data, "models", dict)
    missing = [r for r in ROLES if r not in models]
    if missing:
        raise ConfigError(f"config.yaml: models is missing role(s) {missing}")

    if models["persona"] != models["naive"]:
        raise ConfigError(
            "config.yaml: models.persona and models.naive must be the same model "
            f"(got {models['persona']!r} and {models['naive']!r}). Otherwise the arm A vs "
            "arm B comparison confounds architecture with model capability, which is the "
            "one thing this study exists to separate (spec 5)."
        )
    if models["judge"] == models["persona"]:
        raise ConfigError(
            "config.yaml: models.judge must differ from the generator model "
            f"(both are {models['judge']!r}). A judge that shares the generator's "
            "blind spots cannot label ground truth independently (spec 0.4)."
        )

    temperature = dict(_require(data, "temperature", dict))
    for role in ROLES:
        if role not in temperature:
            raise ConfigError(f"config.yaml: temperature is missing role {role!r}")
        value = temperature[role]
        if value is not None and not isinstance(value, (int, float)):
            raise ConfigError(f"config.yaml: temperature.{role} must be a number or null")

    max_tokens = dict(_require(data, "max_tokens", dict))
    for role in ROLES:
        if role not in max_tokens:
            raise ConfigError(f"config.yaml: max_tokens is missing role {role!r}")

    thinking = dict(data.get("thinking") or {})
    for role, mode in thinking.items():
        if mode not in ("disabled", "adaptive"):
            raise ConfigError(f"config.yaml: thinking.{role} must be 'disabled' or 'adaptive'")

    k = _require(data, "k", int)
    if k != 3:
        raise ConfigError(
            f"config.yaml: k must be 3 (got {k}). More guesses trivially raise recall; "
            "the spec fixes k so precision stays comparable across arms (spec 0.3)."
        )

    threshold = _require(data, "backlash_threshold", (int, float))
    if not 0.0 < float(threshold) < 1.0:
        raise ConfigError(f"config.yaml: backlash_threshold must be in (0.0, 1.0), got {threshold}")

    concurrency = _require(data, "concurrency", int)
    if concurrency < 1:
        raise ConfigError("config.yaml: concurrency must be >= 1")

    max_retries = _require(data, "max_retries", int)
    if max_retries < 1:
        raise ConfigError("config.yaml: max_retries must be >= 1")

    resamples = int(data.get("bootstrap_resamples", 10000))
    if resamples < 1000:
        raise ConfigError("config.yaml: bootstrap_resamples must be >= 1000")

    subsets_raw = _require(data, "subsets", dict)
    subsets: dict[str, list[str]] = {}
    for arm, members in subsets_raw.items():
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise ConfigError(f"config.yaml: subsets.{arm} must be a list of persona id strings")
        if len(set(members)) != len(members):
            raise ConfigError(f"config.yaml: subsets.{arm} contains duplicate persona ids")
        expected = int(arm.lstrip("B")) if arm.startswith("B") and arm[1:].isdigit() else None
        if expected is not None and len(members) != expected:
            raise ConfigError(
                f"config.yaml: subsets.{arm} names {len(members)} personas but the arm is "
                f"called {arm}"
            )
        subsets[arm] = list(members)

    for required_arm in ("B3", "B8", "B15"):
        if required_arm not in subsets:
            raise ConfigError(f"config.yaml: subsets is missing {required_arm!r}")

    return Config(
        models=dict(models),
        temperature=temperature,
        thinking=thinking,
        max_tokens={r: int(max_tokens[r]) for r in ROLES},
        k=k,
        backlash_threshold=float(threshold),
        concurrency=concurrency,
        max_retries=max_retries,
        bootstrap_resamples=resamples,
        subsets=subsets,
        raw=data,
    )


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return parse_config(data)
