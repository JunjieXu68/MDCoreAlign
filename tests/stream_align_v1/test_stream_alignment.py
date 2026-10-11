"""Focused tests for the isolated stream-align research implementation."""

from __future__ import annotations

import numpy as np
import pytest

from mdcorealign.experimental_stream import (
    build_landmark_evidence,
    fit_sparse_landmarks,
    interpolate_keyframe_transforms,
    run_stream_contract,
    select_adaptive_keyframes,
    select_sparse_landmarks,
    temporally_stratified_indices,
)
from mdcorealign.gpu_backend import gpu_available


def _toy_trajectory(n_frames: int = 120, n_residues: int = 48) -> np.ndarray:
    rng = np.random.default_rng(173)
    reference = rng.normal(size=(n_residues, 3)) * np.asarray([8.0, 5.0, 3.0])
    frames = []
    for frame in range(n_frames):
        angle = 0.35 * frame / max(1, n_frames - 1)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.asarray([0.8 * np.sin(angle), -0.3 * angle, 0.2 * angle])
        mobile = (reference - translation) @ rotation
        mobile[18:24] += rng.normal(scale=0.08 + 0.04 * np.sin(frame / 7.0) ** 2, size=(6, 3))
        frames.append(mobile)
    return np.asarray(frames)


def test_temporal_stratification_is_deterministic_and_excludes() -> None:
    first = temporally_stratified_indices(101, 17, exclude=np.asarray([0, 50, 100]))
    second = temporally_stratified_indices(101, 17, exclude=np.asarray([0, 50, 100]))
    assert np.array_equal(first, second)
    assert first.size == 17
    assert not np.isin(first, [0, 50, 100]).any()


def test_all_landmark_selectors_are_unique_dispersed_and_non_collinear() -> None:
    coords = _toy_trajectory()
    chains = np.asarray(["A"] * 24 + ["B"] * 24)
    evidence = build_landmark_evidence(
        coords,
        chain_ids=chains,
        representative_count=64,
    )
    methods = (
        "lowest_deformation_only",
        "lowest_deformation_sequence_spacing",
        "lowest_deformation_3d_farthest",
        "chain_balanced_spatial",
    )
    for method in methods:
        selected = select_sparse_landmarks(
            coords[0],
            evidence,
            chain_ids=chains,
            landmark_count=16,
            method=method,
        )
        assert selected.indices.size == 16
        assert np.unique(selected.indices).size == 16
        assert selected.metadata["spatial_rank"] >= 2
        assert np.isfinite(selected.confidence).all()
    balanced = select_sparse_landmarks(
        coords[0],
        evidence,
        chain_ids=chains,
        landmark_count=16,
        method="chain_balanced_spatial",
    )
    assert set(chains[balanced.indices]) == {"A", "B"}


def test_sparse_fit_recovers_rigid_transforms_from_landmarks() -> None:
    coords = _toy_trajectory(n_frames=60)
    rigid_residues = np.setdiff1d(np.arange(coords.shape[1]), np.arange(18, 24))
    landmarks = rigid_residues[::2]
    fit = fit_sparse_landmarks(
        coords,
        coords[0],
        landmarks,
        backend="cpu",
    )
    identity = np.eye(3)[None, :, :]
    orthogonality = fit.rotations @ np.swapaxes(fit.rotations, 1, 2)
    assert np.max(np.abs(orthogonality - identity)) < 1e-10
    assert np.max(np.abs(np.linalg.det(fit.rotations) - 1.0)) < 1e-10
    assert np.median(fit.landmark_rmsd) < 0.05


def test_keyframes_include_endpoints_and_obey_gap_bound() -> None:
    x = np.linspace(0.0, 8.0, 501)
    signature = np.column_stack((np.sin(x), np.cos(x), np.sin(2.0 * x)))
    keyframes, metadata = select_adaptive_keyframes(signature, target_fraction=0.05)
    assert keyframes[0] == 0
    assert keyframes[-1] == 500
    assert np.max(np.diff(keyframes)) <= metadata["maximum_allowed_gap"]


def test_sign_continuous_slerp_produces_valid_smooth_rotations() -> None:
    angles = np.deg2rad([0.0, 170.0, 190.0, 350.0])
    rotations = np.asarray(
        [
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
            for a in angles
        ]
    )
    keys = np.asarray([0, 10, 20, 30])
    interpolated, translations = interpolate_keyframe_transforms(
        31,
        keys,
        rotations,
        np.zeros((4, 3)),
    )
    assert interpolated.shape == (31, 3, 3)
    assert translations.shape == (31, 3)
    assert np.max(np.abs(np.linalg.det(interpolated) - 1.0)) < 1e-12
    steps = interpolated[1:] @ np.swapaxes(interpolated[:-1], 1, 2)
    cosine = np.clip((np.trace(steps, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    assert np.max(np.degrees(np.arccos(cosine))) <= 18.1


def test_three_stream_contracts_remain_semantically_separate() -> None:
    coords = _toy_trajectory()
    common = {
        "representative_count": 64,
        "landmark_count": 16,
        "keyframe_fraction": 0.1,
        "backend": "cpu",
    }
    scan = run_stream_contract(coords, mode="stream-scan", **common)
    profile = run_stream_contract(coords, mode="stream-profile", **common)
    align = run_stream_contract(
        coords,
        mode="stream-align",
        maximum_refinement_rounds=1,
        **common,
    )
    assert scan.rotations is None and scan.approximate_rmsf is None
    assert profile.rotations is None and profile.approximate_rmsf.shape == (48,)
    assert align.rotations.shape == (120, 3, 3)
    assert align.translations.shape == (120, 3)
    assert align.approximate_rmsd.shape == (120,)
    for result in (scan, profile, align):
        for field in (
            "mode",
            "landmark_count",
            "keyframe_count",
            "frame_sampling_fraction",
            "estimated_error_A",
            "validation_frame_count",
            "refined_interval_count",
            "unresolved_interval_count",
        ):
            assert field in result.metadata
        assert result.metadata["approximate_RMSD_not_exact"] is True
        assert result.metadata["approximate_RMSF_not_exact"] is True
        assert result.metadata["full_aligned_trajectory_materialised"] is False


def test_stream_alignment_never_returns_nonfinite_or_improper_transforms() -> None:
    result = run_stream_contract(
        _toy_trajectory(n_frames=180),
        mode="stream-align",
        representative_count=64,
        landmark_count=32,
        keyframe_fraction=0.05,
        maximum_refinement_rounds=2,
        backend="cpu",
    )
    assert np.isfinite(result.rotations).all()
    assert np.isfinite(result.translations).all()
    assert np.isfinite(result.approximate_rmsd).all()
    assert np.isfinite(result.approximate_rmsf).all()
    assert np.max(np.abs(np.linalg.det(result.rotations) - 1.0)) < 1e-10
    assert result.refined_intervals.ndim == 2
    assert result.unresolved_intervals.ndim == 2


@pytest.mark.skipif(not gpu_available(), reason="optional CUDA backend unavailable")
def test_sparse_gpu_horn_matches_cpu_sparse_fit() -> None:
    coords = _toy_trajectory(n_frames=80)
    landmarks = np.setdiff1d(np.arange(coords.shape[1]), np.arange(18, 24))[::2]
    cpu = fit_sparse_landmarks(coords, coords[0], landmarks, backend="cpu")
    gpu = fit_sparse_landmarks(coords, coords[0], landmarks, backend="gpu")
    assert gpu.backend == "gpu_horn_float32"
    assert np.max(np.abs(cpu.rotations - gpu.rotations)) < 2e-5
    assert np.max(np.abs(cpu.translations - gpu.translations)) < 2e-4
    assert np.max(np.abs(cpu.landmark_rmsd - gpu.landmark_rmsd)) < 2e-4
