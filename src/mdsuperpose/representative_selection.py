"""Descriptor-guided representative-frame selection for long-mode fitting.

The selectors in this module only choose frames used to estimate long-mode
stable-core weights. They do not replace exact weighted Kabsch fitting.
Descriptors are based on C-alpha pairwise distance sketches, not raw XYZ
coordinates, so they are invariant to rigid-body rotation and translation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import time
from typing import Any

import numpy as np

from .utils import ensure_xyz_frames


DESCRIPTOR_STRATEGIES = {
    "distance_sketch_farthest_first",
    "pca_kmedoids",
    "annoy",
    "faiss",
}
SIMPLE_STRATEGIES = {"uniform", "random", "stratified_blocks"}
SUPPORTED_REPRESENTATIVE_STRATEGIES = SIMPLE_STRATEGIES | DESCRIPTOR_STRATEGIES


@dataclass(frozen=True, slots=True)
class RepresentativeSelectionResult:
    """Selected representative frame indices and selector metadata."""

    indices: np.ndarray
    strategy: str
    metadata: dict[str, Any]


def select_representative_frame_indices(
    n_frames: int,
    *,
    max_frames: int = 2000,
    strategy: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Select frame indices using non-coordinate strategies."""

    n_frames = int(n_frames)
    max_frames = int(max_frames)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if max_frames >= n_frames:
        return np.arange(n_frames, dtype=int)
    if strategy == "uniform":
        indices = np.linspace(0, n_frames - 1, num=max_frames, dtype=int)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        tail = rng.choice(np.arange(1, n_frames, dtype=int), size=max_frames - 1, replace=False)
        indices = np.concatenate(([0], np.sort(tail)))
    elif strategy == "stratified_blocks":
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, n_frames, num=max_frames + 1, dtype=int)
        chosen = [0]
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            if start == 0 or end <= start:
                continue
            chosen.append(int(rng.integers(start, end)))
        indices = np.asarray(chosen[:max_frames], dtype=int)
    else:
        raise ValueError("strategy must be one of: 'uniform', 'random', 'stratified_blocks'")
    return _sanitize_indices(indices, n_frames=n_frames, max_frames=max_frames)


def select_representative_frames(
    coords: np.ndarray,
    *,
    max_frames: int = 2000,
    strategy: str = "uniform",
    seed: int = 42,
    raw_descriptor_size: int = 2048,
    compressed_dim: int = 128,
    candidate_pool_size: int = 5000,
    contact_cutoff: float = 8.0,
    min_sequence_separation: int = 4,
    excluded_frame_indices: np.ndarray | None = None,
) -> RepresentativeSelectionResult:
    """Select representative frames for long-mode stable-core weight fitting."""

    start = time.perf_counter()
    coords = ensure_xyz_frames(coords)
    n_frames, n_atoms, _ = coords.shape
    max_frames = int(max_frames)
    strategy = str(strategy)
    excluded = _excluded_mask(n_frames, excluded_frame_indices)
    if strategy not in SUPPORTED_REPRESENTATIVE_STRATEGIES:
        raise ValueError(f"unsupported representative-frame strategy: {strategy}")
    if strategy in SIMPLE_STRATEGIES:
        indices = select_representative_frame_indices(
            n_frames,
            max_frames=max_frames,
            strategy=strategy,
            seed=seed,
        )
        indices = _apply_exclusion_to_indices(indices, excluded, n_frames=n_frames, max_frames=max_frames)
        return RepresentativeSelectionResult(
            indices=indices,
            strategy=strategy,
            metadata={
                "strategy": strategy,
                "selector_kind": "simple",
                "n_frames": n_frames,
                "n_atoms": n_atoms,
                "requested_frames": max_frames,
                "selected_frames": int(indices.shape[0]),
                "selection_runtime_seconds": time.perf_counter() - start,
                "boundary_exclusion_enabled": bool(np.any(excluded)),
                "n_excluded_frames": int(np.sum(excluded)),
            },
        )

    if strategy == "annoy" and importlib.util.find_spec("annoy") is None:
        raise ImportError("annoy is not installed; install it to use strategy='annoy'")
    if strategy == "faiss" and importlib.util.find_spec("faiss") is None and importlib.util.find_spec("faiss_cpu") is None:
        raise ImportError("faiss is not installed; install faiss-cpu to use strategy='faiss'")

    candidates = _candidate_frame_indices(
        n_frames,
        max_frames=max_frames,
        candidate_pool_size=candidate_pool_size,
        excluded=excluded,
    )
    pairs, pair_metadata = sample_distance_sketch_pairs(
        coords,
        max_pairs=raw_descriptor_size,
        seed=seed,
        contact_cutoff=contact_cutoff,
        min_sequence_separation=min_sequence_separation,
    )
    descriptor_start = time.perf_counter()
    raw = distance_sketch_descriptor(coords, pairs, frame_indices=candidates)
    descriptor_seconds = time.perf_counter() - descriptor_start

    compression = "pca" if strategy == "pca_kmedoids" else "random_projection"
    compressed, compression_metadata = compress_descriptors(
        raw,
        compressed_dim=compressed_dim,
        method=compression,
        seed=seed,
    )
    if strategy in {"distance_sketch_farthest_first", "annoy", "faiss"}:
        local_selected = _farthest_first_indices(compressed, min(max_frames, compressed.shape[0]))
    elif strategy == "pca_kmedoids":
        local_selected = _approximate_kmedoids_indices(compressed, min(max_frames, compressed.shape[0]), seed=seed)
    else:  # pragma: no cover - exhaustive guard
        raise ValueError(strategy)

    indices = _sanitize_indices(candidates[local_selected], n_frames=n_frames, max_frames=max_frames)
    metadata = {
        "strategy": strategy,
        "selector_kind": "distance_sketch",
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "requested_frames": max_frames,
        "selected_frames": int(indices.shape[0]),
        "candidate_pool_size": int(candidates.shape[0]),
        "raw_descriptor_size": int(raw.shape[1]),
        "compressed_dim": int(compressed.shape[1]),
        "compression": compression,
        "descriptor_runtime_seconds": descriptor_seconds,
        "selection_runtime_seconds": time.perf_counter() - start,
        "contact_cutoff": float(contact_cutoff),
        "min_sequence_separation": int(min_sequence_separation),
        "uses_raw_xyz_descriptor": False,
        "optional_ann_backend": strategy in {"annoy", "faiss"},
        "boundary_exclusion_enabled": bool(np.any(excluded)),
        "n_excluded_frames": int(np.sum(excluded)),
    }
    metadata.update(pair_metadata)
    metadata.update(compression_metadata)
    return RepresentativeSelectionResult(indices=indices, strategy=strategy, metadata=metadata)


def sample_distance_sketch_pairs(
    coords: np.ndarray,
    *,
    max_pairs: int = 2048,
    seed: int = 42,
    contact_cutoff: float = 8.0,
    min_sequence_separation: int = 4,
    pilot_frames: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample long-range CA pairs and prioritize contact-changing pairs."""

    coords = ensure_xyz_frames(coords)
    n_frames, n_atoms, _ = coords.shape
    if n_atoms < min_sequence_separation + 2:
        raise ValueError("not enough atoms for long-range distance sketch pairs")
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n_atoms, k=min_sequence_separation)
    all_pairs = np.column_stack([i, j]).astype(int)
    if all_pairs.shape[0] <= max_pairs:
        return all_pairs, {
            "n_possible_long_range_pairs": int(all_pairs.shape[0]),
            "n_contact_changing_pairs": int(all_pairs.shape[0]),
            "pair_sampling_mode": "all_pairs",
        }

    candidate_size = min(all_pairs.shape[0], max(max_pairs * 6, max_pairs))
    candidate_indices = rng.choice(all_pairs.shape[0], size=candidate_size, replace=False)
    candidates = all_pairs[candidate_indices]
    frame_indices = np.linspace(0, n_frames - 1, num=min(pilot_frames, n_frames), dtype=int)
    sketch = distance_sketch_descriptor(coords, candidates, frame_indices=frame_indices)
    contact_state = sketch < float(contact_cutoff)
    contact_change = np.var(contact_state.astype(np.float32), axis=0)
    n_contact = min(max_pairs // 2, int(np.sum(contact_change > 0.0)))
    if n_contact > 0:
        contact_order = np.argsort(-contact_change, kind="mergesort")[:n_contact]
        contact_pairs = candidates[contact_order]
    else:
        contact_pairs = np.empty((0, 2), dtype=int)

    remaining = max_pairs - contact_pairs.shape[0]
    selected_keys = {tuple(pair) for pair in contact_pairs.tolist()}
    pool = np.asarray([pair for pair in all_pairs.tolist() if tuple(pair) not in selected_keys], dtype=int)
    random_pairs = pool[rng.choice(pool.shape[0], size=remaining, replace=False)]
    pairs = np.vstack([contact_pairs, random_pairs]).astype(int)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    return pairs, {
        "n_possible_long_range_pairs": int(all_pairs.shape[0]),
        "n_contact_changing_pairs": int(contact_pairs.shape[0]),
        "pair_sampling_mode": "contact_change_plus_random_long_range",
    }


def distance_sketch_descriptor(
    coords: np.ndarray,
    pairs: np.ndarray,
    *,
    frame_indices: np.ndarray | None = None,
    pair_batch_size: int = 256,
) -> np.ndarray:
    """Return a pairwise CA-distance sketch descriptor matrix."""

    coords = ensure_xyz_frames(coords)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_pairs, 2)")
    if frame_indices is None:
        frame_indices = np.arange(coords.shape[0], dtype=int)
    frame_indices = np.asarray(frame_indices, dtype=int)
    selected = coords[frame_indices].astype(np.float32, copy=False)
    descriptor = np.empty((frame_indices.shape[0], pairs.shape[0]), dtype=np.float32)
    for start in range(0, pairs.shape[0], pair_batch_size):
        stop = min(start + pair_batch_size, pairs.shape[0])
        batch = pairs[start:stop]
        diff = selected[:, batch[:, 0], :] - selected[:, batch[:, 1], :]
        descriptor[:, start:stop] = np.sqrt(np.sum(diff * diff, axis=2))
    return descriptor


def compress_descriptors(
    descriptors: np.ndarray,
    *,
    compressed_dim: int = 128,
    method: str = "random_projection",
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize and compress descriptors with random projection or PCA."""

    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be two-dimensional")
    mean = np.mean(descriptors, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(descriptors, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    centered = (descriptors - mean[None, :]) / std[None, :]
    dim = max(1, min(int(compressed_dim), centered.shape[1], centered.shape[0]))
    if method == "random_projection":
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(centered.shape[1], dim)).astype(np.float32)
        projection /= np.sqrt(float(dim))
        compressed = centered @ projection
    elif method == "pca":
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        compressed = centered @ vt[:dim].T.astype(np.float32, copy=False)
    else:
        raise ValueError("method must be 'random_projection' or 'pca'")
    return compressed.astype(np.float32, copy=False), {
        "compression_method": method,
        "descriptor_standardized": True,
    }


def _candidate_frame_indices(
    n_frames: int,
    *,
    max_frames: int,
    candidate_pool_size: int,
    excluded: np.ndarray,
) -> np.ndarray:
    """Return candidate frames considered by descriptor selectors."""

    pool = max(int(candidate_pool_size), int(max_frames))
    allowed = np.flatnonzero(~excluded)
    if allowed.size == 0:
        raise ValueError("all frames were excluded from representative-frame candidates")
    if pool >= n_frames:
        return allowed.astype(int)
    positions = np.linspace(0, allowed.shape[0] - 1, num=min(pool, allowed.shape[0]), dtype=int)
    return allowed[positions].astype(int)


def _excluded_mask(n_frames: int, excluded_frame_indices: np.ndarray | None) -> np.ndarray:
    """Return a boolean excluded-frame mask."""

    mask = np.zeros(int(n_frames), dtype=bool)
    if excluded_frame_indices is None:
        return mask
    indices = np.asarray(excluded_frame_indices, dtype=int)
    indices = indices[(indices >= 0) & (indices < n_frames)]
    mask[indices] = True
    mask[0] = False
    return mask


def _apply_exclusion_to_indices(
    indices: np.ndarray,
    excluded: np.ndarray,
    *,
    n_frames: int,
    max_frames: int,
) -> np.ndarray:
    """Filter selected indices and refill from allowed frames when possible."""

    if not np.any(excluded):
        return _sanitize_indices(indices, n_frames=n_frames, max_frames=max_frames)
    allowed = np.flatnonzero(~excluded)
    filtered = np.asarray([idx for idx in np.asarray(indices, dtype=int) if not excluded[idx]], dtype=int)
    if filtered.shape[0] < min(max_frames, allowed.shape[0]):
        missing = np.setdiff1d(allowed, filtered, assume_unique=False)
        need = min(max_frames, allowed.shape[0]) - filtered.shape[0]
        if need > 0:
            positions = np.linspace(0, missing.shape[0] - 1, num=need, dtype=int)
            filtered = np.concatenate([filtered, missing[positions]])
    return _sanitize_indices(filtered, n_frames=n_frames, max_frames=max_frames)


def _sanitize_indices(indices: np.ndarray, *, n_frames: int, max_frames: int) -> np.ndarray:
    """Sort, uniquify, include frame zero, and enforce requested count."""

    indices = np.unique(np.asarray(indices, dtype=int))
    indices = indices[(indices >= 0) & (indices < n_frames)]
    if indices.size == 0 or indices[0] != 0:
        indices = np.concatenate(([0], indices))
    indices = np.unique(indices)
    if indices.shape[0] > max_frames:
        keep = np.concatenate(([0], indices[1:max_frames]))
        indices = np.unique(keep)
    return np.sort(indices).astype(int)


def _farthest_first_indices(descriptors: np.ndarray, n_select: int) -> np.ndarray:
    """Greedy farthest-first traversal in descriptor space."""

    descriptors = np.asarray(descriptors, dtype=np.float32)
    n_items = descriptors.shape[0]
    n_select = min(int(n_select), n_items)
    if n_select >= n_items:
        return np.arange(n_items, dtype=int)
    selected = np.empty(n_select, dtype=int)
    selected[0] = 0
    diff = descriptors - descriptors[0]
    min_dist = np.einsum("ij,ij->i", diff, diff)
    for pos in range(1, n_select):
        next_index = int(np.argmax(min_dist))
        selected[pos] = next_index
        diff = descriptors - descriptors[next_index]
        dist = np.einsum("ij,ij->i", diff, diff)
        min_dist = np.minimum(min_dist, dist)
    return selected


def _approximate_kmedoids_indices(descriptors: np.ndarray, n_select: int, *, seed: int) -> np.ndarray:
    """Approximate k-medoids by farthest initialization and one medoid update."""

    descriptors = np.asarray(descriptors, dtype=np.float32)
    n_items = descriptors.shape[0]
    n_select = min(int(n_select), n_items)
    if n_select >= n_items:
        return np.arange(n_items, dtype=int)
    initial = _farthest_first_indices(descriptors, n_select)
    centers = descriptors[initial]
    labels = _assign_to_centers(descriptors, centers)
    medoids = []
    rng = np.random.default_rng(seed)
    for cluster_index in range(n_select):
        members = np.flatnonzero(labels == cluster_index)
        if members.size == 0:
            medoids.append(int(initial[cluster_index]))
            continue
        centroid = np.mean(descriptors[members], axis=0)
        diff = descriptors[members] - centroid
        medoids.append(int(members[int(np.argmin(np.einsum("ij,ij->i", diff, diff)))]))
    medoids = np.unique(np.asarray(medoids, dtype=int))
    if medoids.shape[0] < n_select:
        missing = np.setdiff1d(np.arange(n_items, dtype=int), medoids, assume_unique=False)
        extra = rng.choice(missing, size=n_select - medoids.shape[0], replace=False)
        medoids = np.concatenate([medoids, extra])
    return medoids[:n_select]


def _assign_to_centers(descriptors: np.ndarray, centers: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    """Assign each descriptor row to its nearest center."""

    labels = np.empty(descriptors.shape[0], dtype=int)
    center_norm = np.sum(centers * centers, axis=1)
    for start in range(0, descriptors.shape[0], chunk_size):
        stop = min(start + chunk_size, descriptors.shape[0])
        chunk = descriptors[start:stop]
        distances = np.sum(chunk * chunk, axis=1, keepdims=True) + center_norm[None, :] - 2.0 * chunk @ centers.T
        labels[start:stop] = np.argmin(distances, axis=1)
    return labels
