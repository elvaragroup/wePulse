from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.taxonomy import load_taxonomy

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy(REPO / "taxonomy.txt")


@pytest.fixture(scope="session")
def config():
    return load_config(REPO / "config.yaml")


VALID_PERSONA = """\
---
id: "001"
name: privacy_hawk
axis: privacy
archetype: critic
baseline_skepticism: 0.7
platform: x
---

## Who you are

A security and privacy researcher with a mid-sized following.

## What you react to

- Any expansion of data collection

## What you scroll past

Product announcements with no data dimension.

## Voice

Terse, specific, cites the exact clause.
"""


def write_persona(directory: Path, filename: str, text: str = VALID_PERSONA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    return path
