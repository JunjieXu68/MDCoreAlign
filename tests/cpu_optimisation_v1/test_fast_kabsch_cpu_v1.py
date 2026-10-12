"""Focused parity tests for the CPU optimisation Kabsch kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose.fast_kabsch import (
    evaluate_weighted_kabsch_batch,
    evaluate_weighted_kabsch_variable_weights_batch,
    weighted_kabsch_batch,
    weighted_kabsch_single,
    weighted_kabsch_transform_batch,
    weighted_kabsch_variable_weights_transform_batch,
)


def _problem(seed: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(31, 3))
    coords = np.stack(
        [reference + rng.normal(scale=0.08, size=reference.shape) for _ in range(9)]
    )
    fixed = rng.uniform(0.05, 1.5, size=reference.shape[0])
    variable = rng.uniform(0.05, 1.5, size=coords.shape[:2])
    return coords, reference, fixed, variable


def test_transform_only_matches_compatibility_result_without_aligned_storage() -> None:
    coords, reference, weights, _ = _problem()
    compatibility = weighted_kabsch_batch(coords, reference, weights)
    transform_only = weighted_kabsch_transform_batch(coords, reference, weights)

    np.testing.assert_allclose(transform_only.rotations, compatibility.rotations, atol=1e-12)
    np.testing.assert_allclose(transform_only.translations, compatibility.translations, atol=1e-12)
    assert not hasattr(transform_only, "aligned_coords")


def test_metric_controls_are_independent_and_preserve_compatibility_contract() -> None:
    coords, reference, weights, _ = _problem()
    transform_only = evaluate_weighted_kabsch_batch(
        coords,
        reference,
        weights,
        return_aligned=False,
        compute_rmsd=False,
        compute_weighted_rmsd=False,
        compute_residuals=False,
    )
    assert transform_only.aligned_coords is None
    assert transform_only.rmsd is None
    assert transform_only.weighted_rmsd is None
    assert transform_only.residuals is None

    compatibility = weighted_kabsch_batch(
        coords,
        reference,
        weights,
        compute_metrics=False,
    )
    assert compatibility.aligned_coords.shape == coords.shape
    assert np.isnan(compatibility.rmsd).all()
    assert np.isnan(compatibility.weighted_rmsd).all()


def test_fixed_and_variable_evaluators_match_single_frame_reference() -> None:
    coords, reference, fixed, variable = _problem()
    fixed_result = evaluate_weighted_kabsch_batch(
        coords,
        reference,
        fixed,
        return_aligned=True,
        compute_residuals=True,
    )
    variable_result = evaluate_weighted_kabsch_variable_weights_batch(
        coords,
        reference,
        variable,
        return_aligned=True,
        compute_residuals=True,
    )
    fixed_single = [weighted_kabsch_single(frame, reference, fixed) for frame in coords]
    variable_single = [
        weighted_kabsch_single(frame, reference, frame_weights)
        for frame, frame_weights in zip(coords, variable, strict=True)
    ]

    np.testing.assert_allclose(
        fixed_result.aligned_coords,
        np.stack([result.aligned_coords for result in fixed_single]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        variable_result.weighted_rmsd,
        np.asarray([result.weighted_rmsd for result in variable_single]),
        atol=1e-12,
    )
    assert fixed_result.residuals is not None
    np.testing.assert_allclose(
        fixed_result.residuals,
        np.linalg.norm(fixed_result.aligned_coords - reference[None, :, :], axis=2),
        atol=1e-12,
    )


@pytest.mark.parametrize("geometry", ["collinear", "coplanar", "reflection"])
def test_rank_deficient_and_reflection_cases_are_finite_and_proper(geometry: str) -> None:
    axis = np.linspace(-2.0, 2.0, 12)
    if geometry == "collinear":
        reference = np.column_stack((axis, np.zeros_like(axis), np.zeros_like(axis)))
    else:
        reference = np.column_stack((axis, axis * axis, np.zeros_like(axis)))
    coords = np.stack((reference, reference + np.array([0.3, -0.4, 0.2])))
    if geometry == "reflection":
        coords = coords.copy()
        coords[:, :, 0] *= -1.0
    weights = np.linspace(0.2, 1.2, reference.shape[0])

    result = weighted_kabsch_transform_batch(coords, reference, weights)

    assert np.isfinite(result.rotations).all()
    assert np.isfinite(result.translations).all()
    np.testing.assert_allclose(np.linalg.det(result.rotations), 1.0, atol=1e-10)


def test_variable_transform_does_not_mutate_input_arrays() -> None:
    coords, reference, _, weights = _problem()
    coords_before = coords.copy()
    reference_before = reference.copy()
    weights_before = weights.copy()

    weighted_kabsch_variable_weights_transform_batch(coords, reference, weights)

    np.testing.assert_array_equal(coords, coords_before)
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(weights, weights_before)


@pytest.mark.parametrize(
    "bad_weights, message",
    [
        (lambda shape: np.zeros(shape), "positive total"),
        (lambda shape: -np.ones(shape), "non-negative"),
        (lambda shape: np.full(shape, np.nan), "finite"),
    ],
)
def test_invalid_fixed_weights_remain_rejected(bad_weights, message: str) -> None:
    coords, reference, _, _ = _problem()
    with pytest.raises(ValueError, match=message):
        weighted_kabsch_transform_batch(
            coords,
            reference,
            bad_weights(reference.shape[0]),
        )
