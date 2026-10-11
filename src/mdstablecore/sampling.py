"""Representative frame selection utilities."""

from __future__ import annotations

import numpy as np


def choose_coarse_frame_count(
    n_frames: int,
    coarse_fraction: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """Choose the number of frames for coarse analysis."""

    estimated = int(round(n_frames * coarse_fraction))
    return int(np.clip(estimated, min_frames, min(max_frames, n_frames)))


def uniform_frame_indices(n_frames: int, target_count: int) -> np.ndarray:
    """Return evenly spaced frame indices."""

    if target_count >= n_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, num=target_count, dtype=int))


def representative_frame_indices(
    residue_coords: np.ndarray,
    target_count: int,
    random_seed: int = 7,
) -> np.ndarray:
    """Select representative frames with a greedy farthest-point heuristic."""

    n_frames = residue_coords.shape[0]
    if target_count >= n_frames:
        return np.arange(n_frames, dtype=int)

    flattened = residue_coords.reshape(n_frames, -1)
    center = flattened.mean(axis=0, keepdims=True)
    distances_to_center = np.sum((flattened - center) ** 2, axis=1)
    first = int(np.argmin(distances_to_center))

    rng = np.random.default_rng(random_seed)
    selected = [first]
    min_distances = np.sum((flattened - flattened[first]) ** 2, axis=1)

    while len(selected) < target_count:
        candidate_pool = np.flatnonzero(min_distances == min_distances.max())
        next_index = int(rng.choice(candidate_pool))
        selected.append(next_index)
        current = np.sum((flattened - flattened[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, current)

    return np.array(sorted(set(selected)), dtype=int)


def adaptive_sample_frames(
    residue_coords: np.ndarray,
    target_count: int,
    random_seed: int = 7,
) -> np.ndarray:
    """Blend uniform temporal coverage with structural diversity."""

    n_frames = residue_coords.shape[0]
    if target_count >= n_frames:
        return np.arange(n_frames, dtype=int)

    uniform_count = max(2, target_count // 2)
    diversity_count = max(2, target_count - uniform_count)
    uniform = uniform_frame_indices(n_frames, uniform_count)
    diversity = representative_frame_indices(residue_coords, diversity_count, random_seed=random_seed)
    merged = np.unique(np.concatenate([uniform, diversity]))

    if merged.size > target_count:
        positions = uniform_frame_indices(merged.size, target_count)
        merged = merged[positions]

    return merged.astype(int)
