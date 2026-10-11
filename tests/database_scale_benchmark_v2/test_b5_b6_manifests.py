"""Tests for B5/B6 acquisition manifest construction."""

from scripts.database_scale_benchmark_v2.prepare_b5_b6_manifests import (
    LOCKED_ATLAS_CANDIDATES,
    atlas_rows,
)


def test_locked_atlas_panel_covers_registered_size_strata():
    lengths = [row[1] for row in LOCKED_ATLAS_CANDIDATES]
    assert min(lengths) <= 200
    assert any(300 <= value <= 500 for value in lengths)
    assert any(800 <= value <= 1500 for value in lengths)
    assert max(lengths) >= 2000


def test_atlas_candidates_are_not_marked_downloaded_or_audited():
    rows = atlas_rows()
    assert len(rows) == 8
    assert all(row["availability"] == "remote_candidate_not_downloaded" for row in rows)
    assert all(row["coordinate_unit"] == "verify_after_download" for row in rows)
    assert all(row["stitched"] is False for row in rows)
