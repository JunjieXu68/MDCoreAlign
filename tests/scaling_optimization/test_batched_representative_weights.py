"""Correctness gates for the Phase 3 batched representative estimator."""

from __future__ import annotations

import numpy as np

from mdsuperpose.fast_kabsch import (
    weighted_kabsch_single,
    weighted_kabsch_variable_weights_batch,
)
from mdsuperpose.long_mode import solve_long_trajectory_alignment
from mdsuperpose.representative_weights import (
    apply_sequence_continuity_regularization_batch,
    estimate_representative_weights,
)
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment
from mdsuperpose.weights import apply_sequence_continuity_regularization


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(30, 3))
    coords = np.stack(
        [
            reference + rng.normal(scale=0.05, size=reference.shape)
            for _ in range(8)
        ]
    )
    coords[:, 20:] += rng.normal(scale=0.5, size=(8, 10, 3))
    return coords, reference


def _kwargs() -> dict[str, object]:
    return {
        "rule": "huber",
        "delta": 0.25,
        "max_iterations": 20,
        "tolerance": 1e-8,
        "sequence_continuity_strength": 0.2,
        "sequence_continuity_passes": 2,
        "objective_tolerance": None,
        "objective_patience": 0,
        "record_history": False,
    }


def test_variable_weight_batch_matches_individual_kabsch() -> None:
    coords, reference = _coordinates()
    rng = np.random.default_rng(9)
    weights = rng.uniform(0.1, 1.0, size=coords.shape[:2])
    batch = weighted_kabsch_variable_weights_batch(coords, reference, weights)
    singles = [
        weighted_kabsch_single(coords[index], reference, weights[index])
        for index in range(coords.shape[0])
    ]

    np.testing.assert_allclose(
        batch.rotations,
        np.asarray([result.rotation for result in singles]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        batch.translations,
        np.asarray([result.translation for result in singles]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        batch.rmsd,
        np.asarray([result.rmsd for result in singles]),
        atol=1e-12,
        rtol=0.0,
    )


def test_batched_sequence_rule_matches_scalar_and_respects_chains() -> None:
    rng = np.random.default_rng(11)
    weights = rng.uniform(size=(5, 12))
    batch = apply_sequence_continuity_regularization_batch(
        weights,
        strength=0.35,
        passes=3,
    )
    scalar = np.stack(
        [
            apply_sequence_continuity_regularization(
                row,
                strength=0.35,
                passes=3,
            )
            for row in weights
        ]
    )
    np.testing.assert_allclose(batch, scalar, atol=0.0, rtol=0.0)

    boundary = np.zeros((2, 6), dtype=float)
    boundary[:, 3:] = 1.0
    chain_ids = np.asarray(["A", "A", "A", "B", "B", "B"])
    smoothed = apply_sequence_continuity_regularization_batch(
        boundary,
        strength=1.0,
        passes=2,
        chain_ids=chain_ids,
    )
    np.testing.assert_array_equal(smoothed[:, :3], 0.0)
    np.testing.assert_array_equal(smoothed[:, 3:], 1.0)


def test_batched_estimator_matches_scalar_independent_irls() -> None:
    coords, reference = _coordinates()
    batched = estimate_representative_weights(
        coords,
        reference,
        batch_size=coords.shape[0],
        **_kwargs(),
    )
    trajectory = np.concatenate((reference[None, :, :], coords), axis=0)
    scalar = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode="first_frame",
        warm_start=False,
        **_kwargs(),
    )
    scalar_frames = scalar.frame_results[1:]

    np.testing.assert_allclose(
        batched.weights,
        np.asarray([frame.result.weights for frame in scalar_frames]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        batched.rotations,
        np.asarray([frame.result.rotation for frame in scalar_frames]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        batched.translations,
        np.asarray([frame.result.translation for frame in scalar_frames]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        batched.iteration_counts,
        np.asarray([frame.iteration_count for frame in scalar_frames]),
    )
    assert batched.call_counts["per_frame_result_objects"] == 0
    assert batched.call_counts["per_frame_kabsch_calls"] == 0


def test_chunk_size_and_input_precision_preserve_batched_result() -> None:
    coords, reference = _coordinates()
    full = estimate_representative_weights(
        coords,
        reference,
        batch_size=coords.shape[0],
        **_kwargs(),
    )
    chunked = estimate_representative_weights(
        coords,
        reference,
        batch_size=3,
        **_kwargs(),
    )
    float32 = estimate_representative_weights(
        coords.astype(np.float32),
        reference.astype(np.float32),
        batch_size=coords.shape[0],
        **_kwargs(),
    )
    float32_as_float64 = estimate_representative_weights(
        coords.astype(np.float32).astype(np.float64),
        reference.astype(np.float32).astype(np.float64),
        batch_size=coords.shape[0],
        **_kwargs(),
    )

    np.testing.assert_allclose(chunked.weights, full.weights, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        float32.weights,
        float32_as_float64.weights,
        atol=0.0,
        rtol=0.0,
    )
    identity = np.eye(3)
    for rotation in full.rotations:
        np.testing.assert_allclose(rotation @ rotation.T, identity, atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_shifted_refinement_is_chunk_independent() -> None:
    coords, reference = _coordinates()
    full = estimate_representative_weights(
        coords,
        reference,
        batch_size=coords.shape[0],
        warm_start_refinement_passes=2,
        **_kwargs(),
    )
    chunked = estimate_representative_weights(
        coords,
        reference,
        batch_size=3,
        warm_start_refinement_passes=2,
        **_kwargs(),
    )
    np.testing.assert_allclose(chunked.weights, full.weights, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        chunked.rotations,
        full.rotations,
        atol=1e-12,
        rtol=0.0,
    )


def test_long_mode_keeps_legacy_default_and_batched_is_explicit() -> None:
    coords, _ = _coordinates()
    kwargs = _kwargs()
    implicit = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=8,
        chunk_size=8,
        **kwargs,
    )
    explicit = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=8,
        chunk_size=8,
        representative_weight_estimator="legacy",
        **kwargs,
    )
    candidate = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=8,
        chunk_size=8,
        representative_weight_estimator="batched",
        representative_batch_size=4,
        **kwargs,
    )

    np.testing.assert_array_equal(
        implicit.frozen_weights,
        explicit.frozen_weights,
    )
    np.testing.assert_array_equal(implicit.rmsd, explicit.rmsd)
    assert implicit.representative_weight_estimator == "legacy"
    assert candidate.representative_weight_estimator == "batched"
    assert candidate.subset_result.call_counts["per_frame_result_objects"] == 0
