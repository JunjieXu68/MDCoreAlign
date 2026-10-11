"""Correctness contracts for bounded window-preconditioned refinement."""

from __future__ import annotations

import numpy as np

from mdsuperpose.window_refine import (
    InitialWeightResult,
    blend_soft_prior,
    bounded_global_weight_refinement,
    conservative_exclusion_prior,
    solve_window_preconditioned_refinement,
    temporally_stratified_indices,
)


def _trajectory() -> np.ndarray:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(24, 3))
    coords = np.repeat(reference[None], 20, axis=0)
    coords[:, 16:20] += rng.normal(scale=0.3, size=(20, 4, 3))
    return coords


def test_temporal_stratification_is_deterministic_and_bounded() -> None:
    first = temporally_stratified_indices(20, 32)
    second = temporally_stratified_indices(20, 32)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.arange(20))


def test_soft_prior_never_hard_zeros_for_fixed_alpha_grid() -> None:
    prior = np.linspace(0.0, 1.0, 24)
    for alpha in (0.25, 0.5, 0.75):
        weights = blend_soft_prior(prior, alpha)
        assert np.min(weights) >= 1.0 - alpha
        assert np.max(weights) == 1.0


def test_conservative_exclusion_retains_minimum_weight() -> None:
    movement = np.arange(20, dtype=float)
    weights = conservative_exclusion_prior(
        movement,
        top_mobility_fraction=0.2,
        minimum_retained_weight=0.25,
    )
    assert np.count_nonzero(weights == 0.25) == 4
    assert np.min(weights) == 0.25
    assert not np.any(weights == 0.0)


def test_fixed_budget_executes_all_rounds_without_early_stop() -> None:
    coords = _trajectory()
    weights, changes, counts, _ = bounded_global_weight_refinement(
        coords[:8],
        coords[0],
        np.ones(coords.shape[1]),
        update_rounds=4,
    )
    assert changes.shape == (4,)
    assert counts.shape == (4,)
    assert np.isfinite(weights).all()


def test_equal_budget_initializers_use_identical_representatives() -> None:
    coords = _trajectory()
    uniform = InitialWeightResult(
        np.ones(coords.shape[1]),
        "uniform",
        0.0,
        {},
    )
    soft = InitialWeightResult(
        blend_soft_prior(np.linspace(0.0, 1.0, coords.shape[1]), 0.5),
        "soft",
        0.01,
        {},
    )
    left = solve_window_preconditioned_refinement(
        coords,
        uniform,
        representative_count=8,
        update_rounds=2,
    )
    right = solve_window_preconditioned_refinement(
        coords,
        soft,
        representative_count=8,
        update_rounds=2,
    )
    np.testing.assert_array_equal(
        left.representative_indices,
        right.representative_indices,
    )
    assert left.update_rounds == right.update_rounds == 2


def test_explicit_final_transforms_reproduce_reported_rmsd() -> None:
    coords = _trajectory()
    initial = InitialWeightResult(
        np.ones(coords.shape[1]),
        "uniform",
        0.0,
        {},
    )
    result = solve_window_preconditioned_refinement(
        coords,
        initial,
        representative_count=8,
        update_rounds=1,
    )
    aligned = (
        np.einsum("tni,tji->tnj", coords, result.fit.rotations)
        + result.fit.translations[:, None]
    )
    observed = np.sqrt(np.mean(np.sum((aligned - coords[0]) ** 2, axis=2), axis=1))
    np.testing.assert_allclose(observed, result.fit.rmsd, rtol=1e-12, atol=1e-12)
    assert result.metadata["early_stopping"] is False
