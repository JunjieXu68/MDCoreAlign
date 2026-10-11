"""Regression tests for the opt-in guarded fast-global cascade."""

from __future__ import annotations

import numpy as np

from mdcorealign.guarded import (
    GuardedBatchWorker,
    solve_guarded_fast_global_alignment,
)


def _toy_trajectory(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reference = np.cumsum(rng.normal(size=(36, 3)), axis=0)
    coords = np.repeat(reference[None, :, :], 24, axis=0)
    coords += rng.normal(scale=0.015, size=coords.shape)
    coords[:, 15:21] += rng.normal(scale=0.15, size=(24, 6, 3))
    return coords


def test_guarded_result_has_explicit_level_and_audit_fields() -> None:
    result = solve_guarded_fast_global_alignment(
        _toy_trajectory(),
        level1_representative_count=16,
        level2_representative_count=24,
        exact_representative_count=24,
        chunk_size=24,
        store_aligned=False,
    )

    assert result.accepted_level in {1, 3}
    assert result.rotations.shape == (24, 3, 3)
    assert result.translations.shape == (24, 3)
    assert result.rmsd.shape == (24,)
    assert result.frozen_weights.shape == (36,)
    assert result.aligned_coords is None
    assert "route_to_level2" in result.risk_signals
    assert result.final_quality_status
    assert result.metadata["public_default_changed"] is False


def test_guarded_materialization_matches_returned_transforms() -> None:
    coords = _toy_trajectory(19)
    result = solve_guarded_fast_global_alignment(
        coords,
        level1_representative_count=16,
        level2_representative_count=24,
        exact_representative_count=24,
        chunk_size=24,
        store_aligned=True,
    )

    expected = (
        coords @ np.swapaxes(result.rotations, 1, 2)
        + result.translations[:, None, :]
    )
    assert result.aligned_coords is not None
    np.testing.assert_allclose(result.aligned_coords, expected, atol=1e-12)


def test_persistent_worker_preserves_guarded_outputs() -> None:
    coords = _toy_trajectory(23)
    direct = solve_guarded_fast_global_alignment(
        coords,
        level1_representative_count=16,
        level2_representative_count=24,
        exact_representative_count=24,
        chunk_size=2000,
        store_aligned=False,
    )
    worker = GuardedBatchWorker.create(
        24,
        36,
        level1_representative_count=16,
        level2_representative_count=24,
        exact_representative_count=24,
    )
    persistent = worker.align(coords)

    assert persistent.accepted_level == direct.accepted_level
    np.testing.assert_allclose(
        persistent.frozen_weights,
        direct.frozen_weights,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        persistent.rotations,
        direct.rotations,
        atol=1e-12,
    )
    assert worker.cache_summary["zero_allocation_claimed"] is False
