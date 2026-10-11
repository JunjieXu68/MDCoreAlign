"""Unit checks for the independent Phase 7A metric evaluator."""

from __future__ import annotations

import numpy as np

from scripts.scaling_optimization_v4.run_phase7a_rmsd_audit import (
    apply_transforms_independent,
    coordinate_scale_audit,
    fit_transforms,
    hinge_mask,
    regression,
    rmsd_from_aligned,
    rmsf_from_aligned,
    weighted_objective_from_aligned,
)


def _trajectory() -> np.ndarray:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(24, 3))
    frames = np.repeat(reference[None], 8, axis=0)
    frames[:, 16:20] += rng.normal(scale=0.2, size=(8, 4, 3))
    return frames


def test_independent_transform_application_reproduces_fitted_rmsd() -> None:
    coords = _trajectory()
    weights = np.linspace(0.25, 1.0, coords.shape[1])
    rotations, translations = fit_transforms(coords, coords[0], weights)
    aligned = apply_transforms_independent(coords, rotations, translations)
    observed = rmsd_from_aligned(aligned, coords[0])
    assert observed.shape == (coords.shape[0],)
    assert np.isfinite(observed).all()
    assert observed[0] < 1e-12


def test_weighted_objective_is_invariant_to_global_weight_scale() -> None:
    coords = _trajectory()
    weights = np.linspace(0.25, 1.0, coords.shape[1])
    rotations, translations = fit_transforms(coords, coords[0], weights)
    aligned = apply_transforms_independent(coords, rotations, translations)
    first = weighted_objective_from_aligned(aligned, coords[0], weights)
    second = weighted_objective_from_aligned(
        aligned, coords[0], 7.5 * weights
    )
    np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-13)


def test_rmsd_selection_and_rmsf_shapes_are_explicit() -> None:
    coords = _trajectory()
    mask = np.zeros(coords.shape[1], dtype=bool)
    mask[:12] = True
    full = rmsd_from_aligned(coords, coords[0])
    selected = rmsd_from_aligned(coords, coords[0], mask)
    rmsf = rmsf_from_aligned(coords)
    assert full.shape == selected.shape == (coords.shape[0],)
    assert rmsf.shape == (coords.shape[1],)


def test_regression_recovers_known_calibration() -> None:
    predictor = np.linspace(0.0, 3.0, 20)
    response = 1.25 * predictor + 0.4
    correlation, slope, intercept = regression(predictor, response)
    np.testing.assert_allclose(
        (correlation, slope, intercept),
        (1.0, 1.25, 0.4),
        rtol=1e-12,
        atol=1e-12,
    )


def test_hinge_annotation_is_only_exposed_for_known_toy() -> None:
    mask = hinge_mask("toy_hinge", 24)
    assert mask is not None
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(10, 14))
    assert hinge_mask("formal_adk", 214) is None


def test_coordinate_scale_audit_detects_mixed_scale_frames() -> None:
    reference = np.zeros((12, 3), dtype=np.float64)
    reference[:, 0] = 3.8 * np.arange(12)
    coords = np.repeat(reference[None], 8, axis=0)
    coords[:4] *= 10.0
    result = coordinate_scale_audit(
        "test_case", "real_replicate", coords, None
    )
    assert result["coordinate_scale_status"] == "mixed_scale_contaminated"
    assert result["angstrom_like_frame_count"] == 4
    assert result["tenfold_like_frame_count"] == 4
    assert np.isnan(result["physical_scale_factor_to_angstrom"])


def test_coordinate_scale_audit_accepts_angstrom_ca_geometry() -> None:
    reference = np.zeros((12, 3), dtype=np.float64)
    reference[:, 0] = 3.8 * np.arange(12)
    coords = np.repeat(reference[None], 6, axis=0)
    result = coordinate_scale_audit(
        "test_case", "real_replicate", coords, None
    )
    assert result["coordinate_scale_status"] == "angstrom_consistent"
    assert result["physical_scale_factor_to_angstrom"] == 1.0
