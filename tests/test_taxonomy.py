from __future__ import annotations

from typing import get_args

import pytest

from src.models import Category
from src.taxonomy import TaxonomyError, load_taxonomy


def test_repo_taxonomy_matches_category_literal(taxonomy):
    assert set(taxonomy.ids) == set(get_args(Category))


def test_none_is_a_real_category(taxonomy):
    assert "none" in taxonomy.ids


def test_ragged_pipe_spacing_is_tolerated(taxonomy):
    """taxonomy.txt has no space before the pipe on the 'accessibility' and
    'none' lines. Parsing must not depend on the column alignment."""
    entry = next(e for e in taxonomy.entries if e.id == "accessibility")
    assert entry.label == "Accessibility"
    none = next(e for e in taxonomy.entries if e.id == "none")
    assert none.label == "No meaningful backlash"


def test_rank_follows_file_order(taxonomy):
    assert taxonomy.rank("privacy") == 0
    assert taxonomy.rank("none") == len(taxonomy.ids) - 1


def test_render_includes_every_category(taxonomy):
    rendered = taxonomy.render()
    for cid in taxonomy.ids:
        assert cid in rendered


def test_missing_category_raises(tmp_path):
    path = tmp_path / "taxonomy.txt"
    path.write_text("privacy | Privacy | Data stuff\nnone | None | Nothing\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="out of sync"):
        load_taxonomy(path)


def test_duplicate_id_raises(tmp_path, repo):
    original = (repo / "taxonomy.txt").read_text(encoding="utf-8")
    path = tmp_path / "taxonomy.txt"
    path.write_text(original + "privacy | Privacy again | Duplicate\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="duplicate"):
        load_taxonomy(path)


def test_wrong_field_count_raises(tmp_path):
    path = tmp_path / "taxonomy.txt"
    path.write_text("privacy | Privacy\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="expected 'id | label | description'"):
        load_taxonomy(path)
