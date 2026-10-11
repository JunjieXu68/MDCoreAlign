"""Tests for descriptor-guided representative-frame selection."""

from __future__ import annotations

import numpy as np

from mdsuperpose.long_mode import solve_long_trajectory_alignment
from mdsuperpose.representative_selection import (
    distance_sketch_descriptor,
    sample_distance_sketch_pairs,
    select_representative_frames,
)


def _toy_coords(n_frames: int = 18, n_atoms: int = 10) -> np.ndarray:
    """Build a small trajectory with rigid motion and one flexible region."""

    rng = np.random.default_rng(7)
    base = np.cumsum(rng.normal(size=(n_atoms, 3)), axis=0)
    frames = []
    for frame_index in range(n_frames):
        angle = 0.02 * frame_index
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        coords = base @ rotation.T + np.array([0.1 * frame_index, -0.2, 0.05])
        coords[-2:] += np.array([0.0, 0.15 * frame_index, 0.0])
        frames.append(coords)
    return np.asarray(frames, dtype=float)


def test_distance_sketch_is_rigid_transform_invariant() -> None:
    """Pairwise-distance sketches should not depend on raw XYZ orientation."""

    coords = _toy_coords()
    pairs, metadata = sample_distance_sketch_pairs(coords, max_pairs=12, seed=1)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = coords @ rotation.T + np.array([10.0, -3.0, 2.0])
    first = distance_sketch_descriptor(coords, pairs)
    second = distance_sketch_descriptor(moved, pairs)
    assert metadata["pair_sampling_mode"] in {"all_pairs", "contact_change_plus_random_long_range"}
    assert np.allclose(first, second, atol=1e-5)


def test_descriptor_selectors_are_deterministic_and_include_first_frame() -> None:
    """Descriptor-guided selectors should be deterministic for a fixed seed."""

    coords = _toy_coords(n_frames=30, n_atoms=12)
    first = select_representative_frames(
        coords,
        max_frames=6,
        strategy="distance_sketch_farthest_first",
        seed=42,
        raw_descriptor_size=16,
        compressed_dim=6,
        candidate_pool_size=12,
    )
    second = select_representative_frames(
        coords,
        max_frames=6,
        strategy="distance_sketch_farthest_first",
        seed=42,
        raw_descriptor_size=16,
        compressed_dim=6,
        candidate_pool_size=12,
    )
    assert first.indices[0] == 0
    assert np.array_equal(first.indices, second.indices)
    assert first.metadata["uses_raw_xyz_descriptor"] is False
    assert first.metadata["selected_frames"] == 6


def test_pca_kmedoids_selector_runs_on_small_trajectory() -> None:
    """The PCA-kmedoids prototype should return a valid subset."""

    coords = _toy_coords(n_frames=24, n_atoms=11)
    result = select_representative_frames(
        coords,
        max_frames=5,
        strategy="pca_kmedoids",
        seed=3,
        raw_descriptor_size=14,
        compressed_dim=5,
        candidate_pool_size=10,
    )
    assert result.indices.shape == (5,)
    assert result.metadata["compression"] == "pca"


def test_simple_selector_respects_boundary_exclusion() -> None:
    """Simple selectors should filter excluded frames and refill from allowed frames."""

    coords = _toy_coords(n_frames=21, n_atoms=10)
    result = select_representative_frames(
        coords,
        max_frames=5,
        strategy="uniform",
        excluded_frame_indices=np.asarray([10, 15]),
    )
    assert result.indices.shape == (5,)
    assert 10 not in result.indices
    assert 15 not in result.indices
    assert result.metadata["boundary_exclusion_enabled"] is True
    assert result.metadata["n_excluded_frames"] == 2


def test_descriptor_selector_respects_boundary_exclusion() -> None:
    """Descriptor selectors should not select excluded stitched-boundary frames."""

    coords = _toy_coords(n_frames=28, n_atoms=12)
    excluded = np.asarray([6, 7, 8, 14, 15, 16])
    result = select_representative_frames(
        coords,
        max_frames=8,
        strategy="distance_sketch_farthest_first",
        seed=5,
        raw_descriptor_size=16,
        compressed_dim=6,
        candidate_pool_size=28,
        excluded_frame_indices=excluded,
    )
    assert not np.intersect1d(result.indices, excluded).size
    assert result.metadata["boundary_exclusion_enabled"] is True
    assert result.metadata["n_excluded_frames"] == excluded.shape[0]


def test_long_mode_accepts_descriptor_guided_selector() -> None:
    """Long-mode should use descriptor-guided frame selection when requested."""

    coords = _toy_coords(n_frames=20, n_atoms=10)
    result = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=5,
        frame_selection="distance_sketch_farthest_first",
        raw_descriptor_size=12,
        compressed_descriptor_dim=5,
        candidate_pool_size=10,
        chunk_size=8,
        max_iterations=4,
        delta=0.5,
    )
    assert result.selection_strategy == "distance_sketch_farthest_first"
    assert result.selection_metadata["uses_raw_xyz_descriptor"] is False
    assert result.selected_frame_indices.shape[0] == 5
