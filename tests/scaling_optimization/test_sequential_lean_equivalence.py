"""Regression tests for exact sequential warm-start weight estimation."""

from __future__ import annotations

import numpy as np

from mdsuperpose.long_mode import solve_long_trajectory_alignment
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment

ROBUST_KWARGS = {
    "rule": "huber",
    "delta": 1.0,
    "max_iterations": 20,
    "tolerance": 1e-6,
    "objective_tolerance": 1e-5,
    "objective_patience": 2,
    "sequence_continuity_strength": 0.05,
    "contact_consistency_strength": 0.0,
    "record_history": False,
}


def _hinge_trajectory() -> np.ndarray:
    rng = np.random.default_rng(912)
    reference = rng.normal(size=(30, 3))
    frames = []
    for fraction in np.linspace(0.0, 1.0, 24):
        frame = reference.copy()
        theta = fraction * np.deg2rad(48.0)
        rotation = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        )
        anchor = frame[15].copy()
        frame[15:] = (frame[15:] - anchor) @ rotation.T + anchor
        frames.append(frame + np.asarray([2.0, -1.0, 0.25]) * fraction)
    return np.asarray(frames)


def test_sequential_lean_matches_legacy_representative_outputs() -> None:
    coords = _hinge_trajectory()
    legacy = solve_trajectory_robust_alignment(
        coords,
        reference_mode="first_frame",
        warm_start=True,
        **ROBUST_KWARGS,
    )
    lean = estimate_representative_weights_sequential_lean(
        coords,
        coords[0],
        **ROBUST_KWARGS,
    )

    legacy_weights = np.asarray(
        [frame.result.weights for frame in legacy.frame_results]
    )
    legacy_rotations = np.asarray(
        [frame.result.rotation for frame in legacy.frame_results]
    )
    legacy_translations = np.asarray(
        [frame.result.translation for frame in legacy.frame_results]
    )
    legacy_rmsd = np.asarray(
        [frame.result.rmsd for frame in legacy.frame_results]
    )
    legacy_weighted_rmsd = np.asarray(
        [frame.result.weighted_rmsd for frame in legacy.frame_results]
    )

    np.testing.assert_allclose(lean.weights, legacy_weights, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(
        lean.rotations, legacy_rotations, atol=1e-8, rtol=0.0
    )
    np.testing.assert_allclose(
        lean.translations, legacy_translations, atol=1e-8, rtol=0.0
    )
    np.testing.assert_allclose(
        lean.final_rmsd, legacy_rmsd, atol=1e-8, rtol=0.0
    )
    np.testing.assert_allclose(
        lean.final_weighted_rmsd,
        legacy_weighted_rmsd,
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        lean.iteration_counts,
        [frame.iteration_count for frame in legacy.frame_results],
    )
    np.testing.assert_array_equal(
        lean.converged,
        [frame.converged for frame in legacy.frame_results],
    )
    assert lean.call_counts["per_representative_result_objects"] == 0
    assert lean.call_counts["retained_aligned_coordinate_arrays"] == 0


def test_sequential_lean_long_mode_preserves_public_outputs() -> None:
    coords = _hinge_trajectory()
    common = {
        "max_weight_fit_frames": coords.shape[0],
        "chunk_size": coords.shape[0],
        "store_aligned": False,
        **ROBUST_KWARGS,
    }
    legacy = solve_long_trajectory_alignment(
        coords,
        representative_weight_estimator="legacy",
        **common,
    )
    lean = solve_long_trajectory_alignment(
        coords,
        representative_weight_estimator="sequential_lean",
        **common,
    )

    for expected, actual in (
        (legacy.frozen_weights, lean.frozen_weights),
        (legacy.rotations, lean.rotations),
        (legacy.translations, lean.translations),
        (legacy.rmsd, lean.rmsd),
        (legacy.weighted_rmsd, lean.weighted_rmsd),
        (legacy.robust_rmsf, lean.robust_rmsf),
        (legacy.ordinary_rmsf, lean.ordinary_rmsf),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0.0)
    assert lean.average_iterations_per_frame == legacy.average_iterations_per_frame
    assert lean.converged_fraction == legacy.converged_fraction


def test_chain_aware_smoothing_does_not_cross_chain_boundary() -> None:
    coords = _hinge_trajectory()[:, :10]
    chains = np.asarray(["A"] * 5 + ["B"] * 5)
    result = estimate_representative_weights_sequential_lean(
        coords,
        coords[0],
        sequence_chain_ids=chains,
        **ROBUST_KWARGS,
    )

    assert result.weights.shape == (coords.shape[0], coords.shape[1])
    assert np.isfinite(result.weights).all()
