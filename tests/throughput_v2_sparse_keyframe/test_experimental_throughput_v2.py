"""Focused correctness tests for isolated Throughput-v2 research candidates."""

from __future__ import annotations

import numpy as np

from mdcorealign.experimental_throughput_v2 import (
    audit_against_full_fits,
    build_frame_signature,
    deterministic_audit_indices,
    fit_keyframe_alignment,
    fit_sampled_profile,
    fit_sparse_all_frames,
    interpolate_transforms,
    select_keyframes,
    select_weighted_landmarks,
    temporal_sample_indices,
)
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights


def trajectory(seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(24, 3))
    coords = np.empty((80, 24, 3), dtype=np.float64)
    for frame in range(coords.shape[0]):
        angle = 0.12 * np.sin(frame / 17.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        coords[frame] = reference @ rotation.T + np.asarray([0.01 * frame, -0.1, 0.2])
        coords[frame, 18:] += rng.normal(scale=0.03, size=(6, 3))
    weights = np.linspace(1.0, 0.3, reference.shape[0])
    return coords, reference, weights


def test_landmarks_are_deterministic_and_conditioned() -> None:
    _, reference, weights = trajectory()
    first = select_weighted_landmarks(
        reference,
        weights,
        count=16,
        method="conditioning_aware",
        pool_fraction=0.8,
    )
    second = select_weighted_landmarks(
        reference,
        weights,
        count=16,
        method="conditioning_aware",
        pool_fraction=0.8,
    )
    assert first.valid
    assert first.covariance_rank >= 2
    np.testing.assert_array_equal(first.indices, second.indices)


def test_sparse_all_residue_landmarks_match_full_engine() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(
        reference,
        weights,
        count=reference.shape[0],
        method="top_stable",
        pool_fraction=1.0,
        fallback_counts=(reference.shape[0],),
    )
    candidate = fit_sparse_all_frames(coords, reference, weights, landmarks)
    baseline = fit_all_frames_fixed_weights(
        coords,
        reference,
        weights,
        compute_ordinary_rmsf=False,
    )
    np.testing.assert_allclose(candidate.rotations, baseline.rotations, atol=1e-12)
    np.testing.assert_allclose(candidate.translations, baseline.translations, atol=1e-12)
    np.testing.assert_allclose(candidate.rmsd, baseline.rmsd, atol=1e-12)
    np.testing.assert_allclose(candidate.robust_rmsf, baseline.robust_rmsf, atol=1e-12)


def test_keyframe_selection_and_interpolation_are_valid() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(
        reference,
        weights,
        count=16,
        method="spatial_farthest",
    )
    signature, _ = build_frame_signature(
        coords,
        landmarks.indices,
        method="pair_distances",
    )
    keys, metadata = select_keyframes(
        signature,
        method="max_gap_novelty",
        fraction=0.1,
        maximum_gap=12,
    )
    assert keys[0] == 0 and keys[-1] == coords.shape[0] - 1
    assert metadata["maximum_gap"] <= 12
    result = fit_keyframe_alignment(
        coords,
        reference,
        weights,
        landmarks,
        keys,
    )
    determinants = np.linalg.det(result.rotations)
    np.testing.assert_allclose(determinants, 1.0, atol=1e-10)
    assert np.isfinite(result.rmsd).all()


def test_piecewise_and_se3_interpolation_preserve_keyframes() -> None:
    keys = np.asarray([0, 5, 10])
    rotations = RotationFixture.rotations()
    translations = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    for method in ("piecewise_constant", "slerp_linear", "se3"):
        out_rotation, out_translation, _ = interpolate_transforms(
            11,
            keys,
            rotations,
            translations,
            method=method,
        )
        np.testing.assert_allclose(out_rotation[keys], rotations, atol=1e-12)
        np.testing.assert_allclose(out_translation[keys], translations, atol=1e-12)


class RotationFixture:
    @staticmethod
    def rotations() -> np.ndarray:
        output = []
        for angle in (0.0, 0.2, 0.4):
            output.append(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        return np.asarray(output)


def test_sampled_profile_has_finite_uncertainty_and_schema() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(
        reference,
        weights,
        count=16,
        method="chain_balanced",
    )
    sample = temporal_sample_indices(
        coords.shape[0],
        fraction=0.2,
        strategy="stratified_blocks",
    )
    result = fit_sampled_profile(coords, reference, weights, landmarks, sample)
    assert result.contract == "v2_profile"
    assert result.robust_rmsf.shape == (coords.shape[1],)
    assert np.isfinite(result.rmsf_ci_lower).all()
    assert np.all(result.rmsf_ci_upper >= result.rmsf_ci_lower)
    assert result.effective_sample_size > 0.0


def test_audit_detects_perturbed_transform() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(
        reference,
        weights,
        count=reference.shape[0],
        method="top_stable",
        pool_fraction=1.0,
        fallback_counts=(reference.shape[0],),
    )
    result = fit_sparse_all_frames(coords, reference, weights, landmarks)
    audit_indices = deterministic_audit_indices(coords.shape[0])
    clean = audit_against_full_fits(coords, reference, weights, result, audit_indices)
    assert clean.flagged_indices.size == 0
    result.rotations[audit_indices[0]] = np.eye(3)
    result.translations[audit_indices[0]] += 3.0
    perturbed = audit_against_full_fits(coords, reference, weights, result, audit_indices)
    assert audit_indices[0] in perturbed.flagged_indices


def test_chain_balanced_landmarks_cover_each_chain() -> None:
    _, reference, weights = trajectory()
    chains = np.asarray(["A"] * 8 + ["B"] * 8 + ["C"] * 8)
    selected = select_weighted_landmarks(
        reference,
        weights,
        count=12,
        method="chain_balanced",
        chain_ids=chains,
        pool_fraction=1.0,
    )
    assert selected.valid
    assert set(chains[selected.indices]) == {"A", "B", "C"}


def test_collinear_landmarks_fail_safely() -> None:
    reference = np.column_stack((np.arange(12, dtype=float), np.zeros(12), np.zeros(12)))
    selected = select_weighted_landmarks(
        reference,
        np.ones(12),
        count=8,
        method="conditioning_aware",
        fallback_counts=(8, 12),
    )
    assert not selected.valid
    assert selected.warning is not None


def test_registered_keyframe_selectors_are_deterministic_and_bounded() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(reference, weights, count=16, method="spatial_farthest")
    signature, _ = build_frame_signature(coords, landmarks.indices, method="pair_distances")
    for method in ("uniform", "novelty", "hierarchical", "max_gap_novelty"):
        first, metadata = select_keyframes(signature, method=method, fraction=0.1, maximum_gap=15)
        second, _ = select_keyframes(signature, method=method, fraction=0.1, maximum_gap=15)
        np.testing.assert_array_equal(first, second)
        assert first[0] == 0 and first[-1] == coords.shape[0] - 1
        assert metadata["maximum_gap"] <= 15


def test_temporal_sampling_is_deterministic_and_in_range() -> None:
    scores = np.linspace(0.0, 1.0, 80)
    for strategy in ("uniform", "stratified_blocks", "motion_weighted"):
        kwargs = {"transition_scores": scores} if strategy == "motion_weighted" else {}
        first = temporal_sample_indices(80, fraction=0.1, strategy=strategy, **kwargs)
        second = temporal_sample_indices(80, fraction=0.1, strategy=strategy, **kwargs)
        np.testing.assert_array_equal(first, second)
        assert np.all(np.diff(first) > 0)
        assert first.min() >= 0 and first.max() < 80


def test_profile_uncertainty_covers_its_point_estimate() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(reference, weights, count=16, method="spatial_farthest")
    sample = temporal_sample_indices(80, fraction=0.2, strategy="stratified_blocks")
    result = fit_sampled_profile(coords, reference, weights, landmarks, sample)
    assert np.all(result.rmsf_ci_lower <= result.robust_rmsf)
    assert np.all(result.robust_rmsf <= result.rmsf_ci_upper)


def test_sparse_output_contract_does_not_materialise_coordinates() -> None:
    coords, reference, weights = trajectory()
    landmarks = select_weighted_landmarks(reference, weights, count=16, method="spatial_farthest")
    result = fit_sparse_all_frames(coords, reference, weights, landmarks)
    assert result.contract == "v2_sparse_align"
    assert result.metadata["explicit_transform_every_frame"] is True
    assert result.metadata["aligned_coordinates_materialised"] is False
    assert result.metadata["ordinary_rmsf_computed"] is False
    assert result.rotations.shape == (coords.shape[0], 3, 3)


def test_audit_flags_a_rapid_transition_interval() -> None:
    coords, reference, weights = trajectory()
    coords[40:, :12] += np.asarray([4.0, -3.0, 2.0])
    landmarks = select_weighted_landmarks(reference, weights, count=16, method="spatial_farthest")
    signature, _ = build_frame_signature(coords, landmarks.indices, method="centered_coordinates")
    keys, _ = select_keyframes(signature, method="uniform", fraction=0.05, maximum_gap=25)
    candidate = fit_keyframe_alignment(coords, reference, weights, landmarks, keys)
    audit = deterministic_audit_indices(coords.shape[0], keyframes=keys)
    result = audit_against_full_fits(coords, reference, weights, candidate, audit)
    assert result.flagged_indices.size > 0
