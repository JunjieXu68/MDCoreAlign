"""Isolated sparse/keyframe candidates for the Throughput-v2 study.

The functions in this module are internal research contracts. They consume a
weight vector produced by the frozen current-throughput estimator and never
alter exact, throughput, GPU, W5, or public CLI behavior.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from mdsuperpose.fast_kabsch import (
    apply_transforms_batch,
    weighted_kabsch_transform_batch,
)

LANDMARK_COUNTS = (16, 32, 64, 96, 128, 192, 256)
LandmarkMethod = Literal[
    "top_stable",
    "sequence_spaced",
    "spatial_farthest",
    "chain_balanced",
    "conditioning_aware",
    "w5_low_deformation",
]
SignatureMethod = Literal[
    "centered_coordinates",
    "pair_distances",
    "anchor_distances",
    "pca_centered",
    "hybrid",
]
KeyframeMethod = Literal[
    "uniform",
    "novelty",
    "hierarchical",
    "max_gap_novelty",
]
InterpolationMethod = Literal["piecewise_constant", "slerp_linear", "se3"]


@dataclass(frozen=True, slots=True)
class LandmarkSelection:
    """One deterministic, conditioned landmark selection."""

    indices: np.ndarray
    requested_count: int
    actual_count: int
    fallback_count: int | None
    method: str
    pool_fraction: float
    covariance_rank: int
    smallest_singular_value: float
    condition_number: float
    spatial_radius: float
    spatial_coverage: float
    chain_coverage: float
    sequence_coverage: float
    valid: bool
    warning: str | None
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class V2AlignmentResult:
    """Matched compact outputs for sparse or keyframe alignment."""

    contract: str
    weights: np.ndarray
    landmarks: LandmarkSelection
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    robust_rmsf: np.ndarray
    mean_residual: np.ndarray
    keyframe_indices: np.ndarray
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class KeyframeTransformResult:
    """Keyframe-fitted and interpolated transforms before full evaluation."""

    rotations: np.ndarray
    translations: np.ndarray
    keyframe_indices: np.ndarray
    timing: dict[str, float]


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Approximate sampled profile with uncertainty under contract C."""

    contract: str
    weights: np.ndarray
    landmarks: LandmarkSelection
    sample_indices: np.ndarray
    sampling_weights: np.ndarray
    approximate_rmsd: np.ndarray
    robust_rmsf: np.ndarray
    rmsf_ci_lower: np.ndarray
    rmsf_ci_upper: np.ndarray
    effective_sample_size: float
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Deterministic audit-frame comparison against full-residue fits."""

    audit_indices: np.ndarray
    rotation_geodesic_degrees: np.ndarray
    translation_difference: np.ndarray
    aligned_coordinate_rmsd: np.ndarray
    rmsd_absolute_difference: np.ndarray
    flagged_indices: np.ndarray
    unresolved_rate: float
    timing: dict[str, float]
    metadata: dict[str, object]


class _StreamingMoments:
    """Chunk-combinable coordinate moments for full or weighted samples."""

    def __init__(self, n_residues: int) -> None:
        self.mass = 0.0
        self.mass_squared = 0.0
        self.mean = np.zeros((n_residues, 3), dtype=np.float64)
        self.m2 = np.zeros((n_residues, 3), dtype=np.float64)

    def update(self, coords: np.ndarray, sample_weights: np.ndarray | None = None) -> None:
        values = np.asarray(coords, dtype=np.float64)
        if values.shape[0] == 0:
            return
        weights = (
            np.ones(values.shape[0], dtype=np.float64)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float64)
        )
        chunk_mass = float(np.sum(weights))
        if chunk_mass <= 0.0:
            return
        chunk_mean = np.einsum("f,fni->ni", weights, values) / chunk_mass
        centered = values - chunk_mean[None, :, :]
        chunk_m2 = np.einsum("f,fni->ni", weights, centered * centered)
        if self.mass == 0.0:
            self.mean[...] = chunk_mean
            self.m2[...] = chunk_m2
        else:
            total = self.mass + chunk_mass
            delta = chunk_mean - self.mean
            self.mean += delta * (chunk_mass / total)
            self.m2 += chunk_m2 + delta * delta * (
                self.mass * chunk_mass / total
            )
        self.mass += chunk_mass
        self.mass_squared += float(np.sum(weights * weights))

    def rmsf(self) -> np.ndarray:
        if self.mass <= 0.0:
            raise RuntimeError("no samples accumulated")
        return np.sqrt(np.sum(self.m2 / self.mass, axis=1))

    @property
    def effective_sample_size(self) -> float:
        return self.mass * self.mass / max(self.mass_squared, 1e-15)


def deterministic_core_mask(weights: np.ndarray, fraction: float = 0.60) -> np.ndarray:
    """Return the frozen floor-fraction deterministic core convention."""

    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(float(fraction) * values.size)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _candidate_pool(weights: np.ndarray, count: int, fraction: float) -> np.ndarray:
    pool_count = min(weights.size, max(count, int(math.ceil(weights.size * fraction))))
    order = np.lexsort((np.arange(weights.size), -weights))
    return order[:pool_count]


def _sequence_spaced(
    pool: np.ndarray,
    weights: np.ndarray,
    chains: np.ndarray,
    count: int,
    spacing: int,
) -> np.ndarray:
    chosen: list[int] = []
    order = pool[np.lexsort((pool, -weights[pool]))]
    for candidate in order:
        index = int(candidate)
        if all(
            chains[index] != chains[other]
            or abs(index - other) >= spacing
            for other in chosen
        ):
            chosen.append(index)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        chosen.extend(
            int(index)
            for index in order
            if int(index) not in chosen
        )
    return np.asarray(chosen[:count], dtype=np.int64)


def _spatial_farthest(
    pool: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    count: int,
    initial: list[int] | None = None,
) -> np.ndarray:
    chosen = [] if initial is None else list(dict.fromkeys(initial))
    available = np.asarray(pool, dtype=np.int64)
    if not chosen:
        chosen.append(int(available[np.argmax(weights[available])]))
    minimum_squared = np.full(available.size, np.inf, dtype=np.float64)
    for index in chosen:
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((reference[available] - reference[index]) ** 2, axis=1),
        )
    weight_span = float(np.ptp(weights[available]))
    scaled_weight = (
        np.ones(available.size)
        if weight_span <= 1e-15
        else (weights[available] - np.min(weights[available])) / weight_span
    )
    while len(chosen) < count:
        scale = max(float(np.max(minimum_squared[np.isfinite(minimum_squared)])), 1e-15)
        objective = 0.75 * minimum_squared / scale + 0.25 * scaled_weight
        objective[np.isin(available, chosen)] = -np.inf
        selected = int(available[int(np.argmax(objective))])
        chosen.append(selected)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((reference[available] - reference[selected]) ** 2, axis=1),
        )
    return np.asarray(chosen[:count], dtype=np.int64)


def _chain_balanced(
    pool: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    chains: np.ndarray,
    count: int,
) -> np.ndarray:
    unique = np.unique(chains)
    initial: list[int] = []
    for chain in unique:
        local = pool[chains[pool] == chain]
        if local.size:
            initial.append(int(local[np.argmax(weights[local])]))
    if len(initial) > count:
        initial = sorted(initial, key=lambda index: (-weights[index], index))[:count]
    return _spatial_farthest(pool, reference, weights, count, initial)


def _conditioning_aware(
    pool: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    chains: np.ndarray,
    count: int,
) -> np.ndarray:
    chosen: list[int] = [int(pool[np.argmax(weights[pool])])]
    unique_chains = np.unique(chains[pool])
    while len(chosen) < count:
        remaining = pool[~np.isin(pool, chosen)]
        selected_coords = reference[np.asarray(chosen)]
        minimum_distance = np.min(
            np.linalg.norm(
                remaining[:, None, None] * 0.0
                + reference[remaining, None, :]
                - selected_coords[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        distance_scale = max(float(np.max(minimum_distance)), 1e-15)
        weight_scale = max(float(np.ptp(weights[pool])), 1e-15)
        scaled_weight = (weights[remaining] - np.min(weights[pool])) / weight_scale
        missing_chain = ~np.isin(chains[remaining], chains[np.asarray(chosen)])
        objective = (
            0.45 * scaled_weight
            + 0.45 * minimum_distance / distance_scale
            + 0.10 * missing_chain.astype(np.float64)
        )
        best_order = np.lexsort((remaining, -objective))
        chosen.append(int(remaining[best_order[0]]))
        if len(chosen) >= len(unique_chains) and len(chosen) >= count:
            break
    return np.asarray(chosen[:count], dtype=np.int64)


def _conditioning_metrics(
    reference: np.ndarray,
    indices: np.ndarray,
    chains: np.ndarray,
) -> dict[str, float | int | bool]:
    selected = reference[indices]
    centered = selected - np.mean(selected, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > tolerance))
    smallest = float(singular[-1])
    condition = float(singular[0] / smallest) if smallest > tolerance else float("inf")
    center = np.mean(reference, axis=0)
    full_radius = max(float(np.max(np.linalg.norm(reference - center, axis=1))), 1e-15)
    selected_radius = float(np.max(np.linalg.norm(selected - center, axis=1)))
    nearest = np.min(
        np.linalg.norm(reference[:, None, :] - selected[None, :, :], axis=2),
        axis=1,
    )
    spatial_coverage = 1.0 - min(1.0, float(np.max(nearest)) / full_radius)
    unique_all = np.unique(chains)
    chain_coverage = float(np.unique(chains[indices]).size / max(1, unique_all.size))
    sequence_coverages: list[float] = []
    for chain in unique_all:
        positions = np.where(chains == chain)[0]
        selected_positions = indices[chains[indices] == chain]
        if positions.size <= 1 or selected_positions.size == 0:
            sequence_coverages.append(float(selected_positions.size > 0))
        else:
            sequence_coverages.append(
                float((selected_positions.max() - selected_positions.min()) / (positions.max() - positions.min()))
            )
    return {
        "covariance_rank": rank,
        "smallest_singular_value": smallest,
        "condition_number": condition,
        "spatial_radius": selected_radius,
        "spatial_coverage": spatial_coverage,
        "chain_coverage": chain_coverage,
        "sequence_coverage": float(np.mean(sequence_coverages)),
        "valid": rank >= 2 and np.isfinite(condition),
    }


def select_weighted_landmarks(
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    count: int,
    method: LandmarkMethod,
    pool_fraction: float = 0.60,
    chain_ids: np.ndarray | None = None,
    minimum_sequence_spacing: int = 4,
    representative_w5: np.ndarray | None = None,
    fallback_counts: tuple[int, ...] = LANDMARK_COUNTS,
) -> LandmarkSelection:
    """Select a stable, spatially conditioned landmark set with safe fallback."""

    started = time.perf_counter()
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or stable.shape != (target.shape[0],):
        raise ValueError("reference/weights shape mismatch")
    if not 0.0 < pool_fraction <= 1.0:
        raise ValueError("pool_fraction must be in (0, 1]")
    if not 3 <= int(count) <= target.shape[0]:
        raise ValueError("count must be between 3 and n_residues")
    chains = (
        np.full(target.shape[0], "A", dtype="U1")
        if chain_ids is None
        else np.asarray(chain_ids).astype(str)
    )
    if chains.shape != (target.shape[0],):
        raise ValueError("chain_ids shape mismatch")
    candidates = [value for value in fallback_counts if count <= value <= target.shape[0]]
    if count not in candidates:
        candidates.insert(0, count)
    warning: str | None = None
    selected = np.empty(0, dtype=np.int64)
    metrics: dict[str, float | int | bool] = {}
    actual_count = count
    for candidate_count in candidates:
        pool = _candidate_pool(stable, candidate_count, pool_fraction)
        ranking_weights = stable
        if method == "w5_low_deformation":
            if representative_w5 is None:
                raise ValueError("representative_w5 is required for w5_low_deformation")
            w5 = np.asarray(representative_w5, dtype=np.float64)
            if w5.shape != stable.shape:
                raise ValueError("representative_w5 shape mismatch")
            ranking_weights = 0.75 * stable + 0.25 * (1.0 - w5)
            pool = _candidate_pool(ranking_weights, candidate_count, pool_fraction)
            selected = _spatial_farthest(pool, target, ranking_weights, candidate_count)
        elif method == "top_stable":
            selected = pool[:candidate_count]
        elif method == "sequence_spaced":
            selected = _sequence_spaced(
                pool,
                stable,
                chains,
                candidate_count,
                int(minimum_sequence_spacing),
            )
        elif method == "spatial_farthest":
            selected = _spatial_farthest(pool, target, stable, candidate_count)
        elif method == "chain_balanced":
            selected = _chain_balanced(pool, target, stable, chains, candidate_count)
        elif method == "conditioning_aware":
            selected = _conditioning_aware(pool, target, stable, chains, candidate_count)
        else:
            raise ValueError(f"unknown landmark method: {method}")
        selected = np.sort(np.unique(selected))
        if selected.size != candidate_count:
            warning = "selection_cardinality_mismatch"
            continue
        metrics = _conditioning_metrics(target, selected, chains)
        actual_count = candidate_count
        if bool(metrics["valid"]):
            break
        warning = "ill_conditioned_selection_fell_forward"
    valid = bool(metrics.get("valid", False))
    if not valid and warning is None:
        warning = "no_valid_registered_landmark_count"
    return LandmarkSelection(
        indices=selected,
        requested_count=int(count),
        actual_count=int(actual_count),
        fallback_count=(int(actual_count) if actual_count != int(count) else None),
        method=method,
        pool_fraction=float(pool_fraction),
        covariance_rank=int(metrics.get("covariance_rank", 0)),
        smallest_singular_value=float(metrics.get("smallest_singular_value", 0.0)),
        condition_number=float(metrics.get("condition_number", float("inf"))),
        spatial_radius=float(metrics.get("spatial_radius", 0.0)),
        spatial_coverage=float(metrics.get("spatial_coverage", 0.0)),
        chain_coverage=float(metrics.get("chain_coverage", 0.0)),
        sequence_coverage=float(metrics.get("sequence_coverage", 0.0)),
        valid=valid,
        warning=warning,
        runtime_seconds=time.perf_counter() - started,
    )


def _evaluate_transforms(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_frames, n_residues, _ = coords.shape
    rmsd = np.empty(n_frames, dtype=np.float64)
    weighted_rmsd = np.empty(n_frames, dtype=np.float64)
    residual_sum = np.zeros(n_residues, dtype=np.float64)
    moments = _StreamingMoments(n_residues)
    weight_sum = float(np.sum(weights))
    timing = {"transform_application": 0.0, "RMSD": 0.0, "RMSF": 0.0}
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        apply_start = time.perf_counter()
        aligned = apply_transforms_batch(
            coords[start:stop],
            rotations[start:stop],
            translations[start:stop],
        )
        timing["transform_application"] += time.perf_counter() - apply_start
        metric_start = time.perf_counter()
        delta = aligned - reference[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        rmsd[start:stop] = np.sqrt(np.mean(squared, axis=1))
        weighted_rmsd[start:stop] = np.sqrt((squared @ weights) / weight_sum)
        np.sqrt(squared, out=squared)
        residual_sum += np.sum(squared, axis=0)
        timing["RMSD"] += time.perf_counter() - metric_start
        rmsf_start = time.perf_counter()
        moments.update(aligned)
        timing["RMSF"] += time.perf_counter() - rmsf_start
    return rmsd, weighted_rmsd, moments.rmsf(), residual_sum / n_frames, timing


def fit_sparse_all_frames(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    landmarks: LandmarkSelection,
    *,
    chunk_size: int = 2000,
) -> V2AlignmentResult:
    """Fit every frame explicitly from sparse weighted landmarks."""

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    if not landmarks.valid:
        raise ValueError("landmark selection is invalid")
    n_frames = values.shape[0]
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    fit_seconds = 0.0
    indices = landmarks.indices
    landmark_weights = stable[indices]
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        fit_start = time.perf_counter()
        transform = weighted_kabsch_transform_batch(
            values[start:stop, indices],
            target[indices],
            landmark_weights,
            validate_inputs=False,
        )
        fit_seconds += time.perf_counter() - fit_start
        rotations[start:stop] = transform.rotations
        translations[start:stop] = transform.translations
    rmsd, weighted_rmsd, robust_rmsf, mean_residual, evaluation_timing = (
        _evaluate_transforms(
            values,
            target,
            stable,
            rotations,
            translations,
            chunk_size=chunk_size,
        )
    )
    total = time.perf_counter() - total_start
    return V2AlignmentResult(
        contract="v2_sparse_align",
        weights=stable,
        landmarks=landmarks,
        rotations=rotations,
        translations=translations,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        robust_rmsf=robust_rmsf,
        mean_residual=mean_residual,
        keyframe_indices=np.arange(n_frames, dtype=np.int64),
        timing={
            "landmark_selection": landmarks.runtime_seconds,
            "explicit_frame_fits": fit_seconds,
            **evaluation_timing,
            "total_excluding_weight_estimation": total,
        },
        metadata={
            "explicit_transform_every_frame": True,
            "ordinary_rmsf_computed": False,
            "aligned_coordinates_materialised": False,
            "approximate_weight_estimation": False,
            "transform_fit_residue_count": int(indices.size),
        },
    )


def build_frame_signature(
    coords: np.ndarray,
    landmarks: np.ndarray,
    *,
    method: SignatureMethod,
    maximum_pairs: int = 128,
    pca_dimensions: int = 16,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build one deterministic landmark-only temporal signature."""

    started = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)[:, np.asarray(landmarks), :]
    centered = values - np.mean(values, axis=1, keepdims=True)
    metadata: dict[str, object] = {"method": method}
    if method == "centered_coordinates":
        signature = centered.reshape(centered.shape[0], -1)
    elif method in {"pair_distances", "hybrid"}:
        left, right = np.triu_indices(values.shape[1], k=1)
        separation = np.abs(np.asarray(landmarks)[right] - np.asarray(landmarks)[left])
        order = np.lexsort((right, left, -separation))
        if order.size > maximum_pairs:
            positions = np.rint(np.linspace(0, order.size - 1, maximum_pairs)).astype(int)
            order = order[positions]
        left = left[order]
        right = right[order]
        distances = np.linalg.norm(values[:, left] - values[:, right], axis=2)
        scale = np.maximum(distances[0], 1e-12)
        distance_signature = distances / scale[None, :] - 1.0
        metadata["pair_count"] = int(left.size)
        if method == "pair_distances":
            signature = distance_signature
        else:
            coordinate = centered.reshape(centered.shape[0], -1)
            coordinate /= max(float(np.std(coordinate)), 1e-12)
            signature = np.concatenate((distance_signature, coordinate), axis=1)
    elif method == "anchor_distances":
        center = np.mean(values[0], axis=0)
        radii = np.linalg.norm(values[0] - center, axis=1)
        anchors = np.argsort(radii, kind="stable")[-min(8, values.shape[1]) :]
        distances = np.linalg.norm(values[:, :, None, :] - values[:, None, anchors, :], axis=3)
        scale = np.maximum(distances[0], 1e-12)
        signature = (distances / scale[None, :, :] - 1.0).reshape(values.shape[0], -1)
        metadata["anchor_count"] = int(anchors.size)
    elif method == "pca_centered":
        matrix = centered.reshape(centered.shape[0], -1)
        fit_indices = np.unique(
            np.rint(np.linspace(0, matrix.shape[0] - 1, min(2000, matrix.shape[0]))).astype(int)
        )
        fit = matrix[fit_indices]
        mean = np.mean(fit, axis=0)
        _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
        dimensions = min(int(pca_dimensions), vt.shape[0])
        signature = (matrix - mean) @ vt[:dimensions].T
        metadata["pca_dimensions"] = dimensions
    else:
        raise ValueError(f"unknown signature method: {method}")
    feature_scale = np.std(signature, axis=0)
    usable = feature_scale > 1e-12
    if np.any(usable):
        signature = signature[:, usable] / feature_scale[usable]
    else:
        signature = np.zeros((values.shape[0], 1), dtype=np.float64)
    metadata.update(
        {
            "feature_count": int(signature.shape[1]),
            "construction_seconds": time.perf_counter() - started,
        }
    )
    return np.asarray(signature, dtype=np.float64), metadata


def _fill_maximum_gaps(indices: np.ndarray, maximum_gap: int, n_frames: int) -> np.ndarray:
    selected = set(int(value) for value in indices)
    selected.update((0, n_frames - 1))
    while True:
        ordered = np.asarray(sorted(selected), dtype=np.int64)
        gaps = np.diff(ordered)
        if gaps.size == 0 or int(np.max(gaps)) <= int(maximum_gap):
            return ordered
        position = int(np.argmax(gaps))
        selected.add(int((ordered[position] + ordered[position + 1]) // 2))


def select_keyframes(
    signature: np.ndarray,
    *,
    method: KeyframeMethod,
    fraction: float,
    maximum_gap: int,
    required_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select deterministic keyframes under a budget and hard gap bound."""

    started = time.perf_counter()
    values = np.asarray(signature, dtype=np.float64)
    n_frames = values.shape[0]
    if values.ndim != 2 or n_frames < 2:
        raise ValueError("signature must have shape (n_frames, n_features)")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    target = max(2, int(math.ceil(n_frames * fraction)))
    required = np.asarray([0, n_frames - 1], dtype=np.int64)
    if required_indices is not None:
        required = np.unique(np.concatenate((required, np.asarray(required_indices, dtype=np.int64))))
    if method == "uniform":
        selected = np.rint(np.linspace(0, n_frames - 1, target)).astype(np.int64)
    elif method in {"novelty", "max_gap_novelty"}:
        increments = np.linalg.norm(np.diff(values, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        if cumulative[-1] <= 1e-15:
            selected = np.rint(np.linspace(0, n_frames - 1, target)).astype(np.int64)
        else:
            selected = np.searchsorted(
                cumulative,
                np.linspace(0.0, cumulative[-1], target),
                side="left",
            ).astype(np.int64)
    elif method == "hierarchical":
        selected_set = {0, n_frames - 1}
        heap: list[tuple[float, int, int, int]] = []

        def push(left: int, right: int) -> None:
            if right - left <= 1:
                return
            frames = np.arange(left + 1, right)
            alpha = ((frames - left) / (right - left))[:, None]
            prediction = (1.0 - alpha) * values[left] + alpha * values[right]
            error = np.linalg.norm(values[frames] - prediction, axis=1)
            offset = int(np.argmax(error))
            heapq.heappush(heap, (-float(error[offset]), left, right, int(frames[offset])))

        push(0, n_frames - 1)
        while len(selected_set) < target and heap:
            _, left, right, split = heapq.heappop(heap)
            if split in selected_set:
                continue
            selected_set.add(split)
            push(left, split)
            push(split, right)
        selected = np.asarray(sorted(selected_set), dtype=np.int64)
    else:
        raise ValueError(f"unknown keyframe method: {method}")
    selected = np.unique(np.concatenate((selected, required)))
    if method == "max_gap_novelty" or maximum_gap > 0:
        selected = _fill_maximum_gaps(selected, int(maximum_gap), n_frames)
    return selected, {
        "method": method,
        "target_fraction": float(fraction),
        "actual_fraction": float(selected.size / n_frames),
        "keyframe_count": int(selected.size),
        "maximum_gap": int(np.max(np.diff(selected))),
        "selection_seconds": time.perf_counter() - started,
    }


def _continuous_quaternions(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, quaternions.shape[0]):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def _slerp_linear(
    n_frames: int,
    keys: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quaternions = _continuous_quaternions(rotations)
    frame_indices = np.arange(n_frames)
    interval = np.clip(np.searchsorted(keys, frame_indices, side="right") - 1, 0, keys.size - 2)
    left = keys[interval]
    right = keys[interval + 1]
    alpha = (frame_indices - left) / np.maximum(right - left, 1)
    first = quaternions[interval]
    second = quaternions[interval + 1]
    dot = np.sum(first * second, axis=1)
    flip = dot < 0.0
    second = second.copy()
    second[flip] *= -1.0
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    output = np.empty_like(first)
    linear = dot > 0.9995
    output[linear] = first[linear] + alpha[linear, None] * (second[linear] - first[linear])
    nonlinear = ~linear
    angle = np.arccos(dot[nonlinear])
    sine = np.sin(angle)
    output[nonlinear] = (
        np.sin((1.0 - alpha[nonlinear]) * angle)[:, None] / sine[:, None] * first[nonlinear]
        + np.sin(alpha[nonlinear] * angle)[:, None] / sine[:, None] * second[nonlinear]
    )
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    shifts = (1.0 - alpha)[:, None] * translations[interval] + alpha[:, None] * translations[interval + 1]
    return Rotation.from_quat(output).as_matrix(), shifts


def _se3_interpolate(
    n_frames: int,
    keys: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output_rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    output_translations = np.empty((n_frames, 3), dtype=np.float64)
    for interval, (left, right) in enumerate(zip(keys[:-1], keys[1:], strict=True)):
        alpha = np.linspace(0.0, 1.0, right - left + 1)
        r0 = rotations[interval]
        relative = r0.T @ rotations[interval + 1]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        relative_rotations = Rotation.from_rotvec(alpha[:, None] * rotvec[None, :]).as_matrix()
        output_rotations[left : right + 1] = r0[None, :, :] @ relative_rotations
        # Translation is kept linear. This is the decoupled SE(3) geodesic
        # convention registered for this study, not a screw-motion claim.
        output_translations[left : right + 1] = (
            (1.0 - alpha)[:, None] * translations[interval]
            + alpha[:, None] * translations[interval + 1]
        )
    return output_rotations, output_translations


def interpolate_transforms(
    n_frames: int,
    keyframe_indices: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    method: InterpolationMethod,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate keyframe transforms using a registered method."""

    started = time.perf_counter()
    keys = np.asarray(keyframe_indices, dtype=np.int64)
    if keys[0] != 0 or keys[-1] != n_frames - 1 or np.any(np.diff(keys) <= 0):
        raise ValueError("keyframes must be sorted and include trajectory endpoints")
    if method == "piecewise_constant":
        frames = np.arange(n_frames)
        interval = np.clip(np.searchsorted(keys, frames, side="right") - 1, 0, keys.size - 1)
        output_rotations = rotations[interval].copy()
        output_translations = translations[interval].copy()
    elif method == "slerp_linear":
        output_rotations, output_translations = _slerp_linear(
            n_frames, keys, rotations, translations
        )
    elif method == "se3":
        output_rotations, output_translations = _se3_interpolate(
            n_frames, keys, rotations, translations
        )
    else:
        raise ValueError(f"unknown interpolation method: {method}")
    return output_rotations, output_translations, time.perf_counter() - started


def fit_keyframe_alignment(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    landmarks: LandmarkSelection,
    keyframe_indices: np.ndarray,
    *,
    interpolation: InterpolationMethod = "slerp_linear",
    chunk_size: int = 2000,
) -> V2AlignmentResult:
    """Fit sparse keyframes, interpolate transforms, then evaluate all frames."""

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    keys = np.asarray(keyframe_indices, dtype=np.int64)
    transformed = fit_keyframe_transforms(
        values,
        target,
        stable,
        landmarks,
        keys,
        interpolation=interpolation,
    )
    rotations = transformed.rotations
    translations = transformed.translations
    key_fit_seconds = transformed.timing["explicit_keyframe_fits"]
    interpolation_seconds = transformed.timing["transform_interpolation"]
    rmsd, weighted_rmsd, robust_rmsf, mean_residual, evaluation_timing = (
        _evaluate_transforms(
            values,
            target,
            stable,
            rotations,
            translations,
            chunk_size=chunk_size,
        )
    )
    total = time.perf_counter() - total_start
    return V2AlignmentResult(
        contract="v2_keyframe_align",
        weights=stable,
        landmarks=landmarks,
        rotations=rotations,
        translations=translations,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        robust_rmsf=robust_rmsf,
        mean_residual=mean_residual,
        keyframe_indices=keys,
        timing={
            "landmark_selection": landmarks.runtime_seconds,
            "explicit_keyframe_fits": key_fit_seconds,
            "transform_interpolation": interpolation_seconds,
            **evaluation_timing,
            "total_excluding_weight_estimation": total,
        },
        metadata={
            "explicit_transform_every_frame": False,
            "interpolation": interpolation,
            "keyframe_fraction": float(keys.size / values.shape[0]),
            "ordinary_rmsf_computed": False,
            "aligned_coordinates_materialised": False,
        },
    )


def fit_keyframe_transforms(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    landmarks: LandmarkSelection,
    keyframe_indices: np.ndarray,
    *,
    interpolation: InterpolationMethod = "slerp_linear",
) -> KeyframeTransformResult:
    """Fit and interpolate keyframe transforms without all-frame metrics."""

    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    keys = np.asarray(keyframe_indices, dtype=np.int64)
    fit_start = time.perf_counter()
    key_fit = weighted_kabsch_transform_batch(
        values[keys][:, landmarks.indices],
        target[landmarks.indices],
        stable[landmarks.indices],
        validate_inputs=False,
    )
    key_fit_seconds = time.perf_counter() - fit_start
    rotations, translations, interpolation_seconds = interpolate_transforms(
        values.shape[0],
        keys,
        key_fit.rotations,
        key_fit.translations,
        method=interpolation,
    )
    return KeyframeTransformResult(
        rotations=rotations,
        translations=translations,
        keyframe_indices=keys,
        timing={
            "explicit_keyframe_fits": key_fit_seconds,
            "transform_interpolation": interpolation_seconds,
            "total_transform_seconds": key_fit_seconds + interpolation_seconds,
        },
    )


def temporal_sample_indices(
    n_frames: int,
    *,
    fraction: float,
    strategy: str,
    keyframes: np.ndarray | None = None,
    transition_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic profile samples for one registered strategy."""

    count = max(4, min(n_frames, int(math.ceil(n_frames * fraction))))
    uniform = np.unique(np.rint(np.linspace(0, n_frames - 1, count)).astype(np.int64))
    if strategy == "uniform":
        return uniform
    if strategy == "stratified_blocks":
        blocks = min(16, count)
        pieces = []
        for block in range(blocks):
            left = int(math.floor(block * n_frames / blocks))
            right = int(math.floor((block + 1) * n_frames / blocks)) - 1
            local_count = max(1, count // blocks + int(block < count % blocks))
            pieces.append(np.rint(np.linspace(left, max(left, right), local_count)).astype(np.int64))
        return np.unique(np.concatenate(pieces))[:count]
    if strategy == "keyframe_interval":
        if keyframes is None:
            raise ValueError("keyframes required for keyframe_interval sampling")
        selected = set(int(value) for value in np.asarray(keyframes, dtype=np.int64))
        for left, right in zip(keyframes[:-1], keyframes[1:], strict=True):
            if len(selected) >= count:
                break
            selected.add(int((left + right) // 2))
        selected.update(int(value) for value in uniform)
        return np.asarray(sorted(selected), dtype=np.int64)[:count]
    if strategy in {"transition_oversampled", "motion_weighted", "adaptive_variance"}:
        if transition_scores is None:
            raise ValueError("transition_scores required for adaptive sampling")
        scores = np.asarray(transition_scores, dtype=np.float64)
        high_count = max(1, count // 2)
        high = np.lexsort((np.arange(n_frames), -scores))[:high_count]
        selected = np.unique(np.concatenate((high, uniform)))
        if selected.size > count:
            selected = selected[np.rint(np.linspace(0, selected.size - 1, count)).astype(int)]
        return np.sort(selected)
    raise ValueError(f"unknown sampling strategy: {strategy}")


def _voronoi_temporal_weights(indices: np.ndarray, n_frames: int) -> np.ndarray:
    sample = np.asarray(indices, dtype=np.int64)
    boundaries = np.empty(sample.size + 1, dtype=np.float64)
    boundaries[0] = -0.5
    boundaries[-1] = n_frames - 0.5
    boundaries[1:-1] = 0.5 * (sample[:-1] + sample[1:])
    return np.diff(boundaries)


def fit_sampled_profile(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    landmarks: LandmarkSelection,
    sample_indices: np.ndarray,
    *,
    block_count: int = 8,
    chunk_size: int = 512,
) -> ProfileResult:
    """Fit sampled frames and estimate weighted RMSF with block uncertainty."""

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    sample = np.asarray(sample_indices, dtype=np.int64)
    temporal_weights = _voronoi_temporal_weights(sample, values.shape[0])
    moments = _StreamingMoments(values.shape[1])
    sampled_rmsd = np.empty(sample.size, dtype=np.float64)
    block_moments = [_StreamingMoments(values.shape[1]) for _ in range(block_count)]
    fit_seconds = 0.0
    profile_seconds = 0.0
    for start in range(0, sample.size, int(chunk_size)):
        stop = min(start + int(chunk_size), sample.size)
        indices = sample[start:stop]
        fit_start = time.perf_counter()
        transform = weighted_kabsch_transform_batch(
            values[indices][:, landmarks.indices],
            target[landmarks.indices],
            stable[landmarks.indices],
            validate_inputs=False,
        )
        fit_seconds += time.perf_counter() - fit_start
        profile_start = time.perf_counter()
        aligned = apply_transforms_batch(values[indices], transform.rotations, transform.translations)
        delta = aligned - target[None, :, :]
        sampled_rmsd[start:stop] = np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))
        moments.update(aligned, temporal_weights[start:stop])
        block_ids = np.minimum(
            block_count - 1,
            (indices * block_count // values.shape[0]).astype(int),
        )
        for block in np.unique(block_ids):
            mask = block_ids == block
            block_moments[int(block)].update(aligned[mask], temporal_weights[start:stop][mask])
        profile_seconds += time.perf_counter() - profile_start
    robust_rmsf = moments.rmsf()
    block_profiles = np.asarray(
        [item.rmsf() for item in block_moments if item.mass > 0.0],
        dtype=np.float64,
    )
    if block_profiles.shape[0] >= 2:
        standard_error = np.std(block_profiles, axis=0, ddof=1) / math.sqrt(block_profiles.shape[0])
    else:
        standard_error = np.full(values.shape[1], np.nan, dtype=np.float64)
    total = time.perf_counter() - total_start
    return ProfileResult(
        contract="v2_profile",
        weights=stable,
        landmarks=landmarks,
        sample_indices=sample,
        sampling_weights=temporal_weights,
        approximate_rmsd=sampled_rmsd,
        robust_rmsf=robust_rmsf,
        rmsf_ci_lower=np.maximum(0.0, robust_rmsf - 1.96 * standard_error),
        rmsf_ci_upper=robust_rmsf + 1.96 * standard_error,
        effective_sample_size=moments.effective_sample_size,
        timing={
            "landmark_selection": landmarks.runtime_seconds,
            "sampled_frame_fits": fit_seconds,
            "sampled_profile_statistics": profile_seconds,
            "total_excluding_weight_estimation": total,
        },
        metadata={
            "sample_fraction": float(sample.size / values.shape[0]),
            "block_count": int(block_profiles.shape[0]),
            "ordinary_rmsf_computed": False,
            "explicit_all_frame_transforms": False,
            "uncertainty_method": "temporal_block_standard_error_normal_95pct",
        },
    )


def deterministic_audit_indices(
    n_frames: int,
    *,
    keyframes: np.ndarray | None = None,
    curvature: np.ndarray | None = None,
) -> np.ndarray:
    """Reserve max(64, 1%) fixed audit frames, excluding keyframes where possible."""

    count = min(n_frames, max(64, int(math.ceil(0.01 * n_frames))))
    excluded = set() if keyframes is None else set(int(value) for value in keyframes)
    available = np.asarray([index for index in range(n_frames) if index not in excluded], dtype=np.int64)
    if available.size <= count:
        return available
    stratified = available[np.rint(np.linspace(0, available.size - 1, count // 2)).astype(int)]
    if curvature is None:
        extra = available[np.rint(np.linspace(0, available.size - 1, count - stratified.size)).astype(int)]
    else:
        scores = np.asarray(curvature, dtype=np.float64)[available]
        order = np.lexsort((available, -scores))
        extra = available[order[: count - stratified.size]]
    selected = np.unique(np.concatenate((stratified, extra)))
    if selected.size < count:
        remaining = available[~np.isin(available, selected)]
        selected = np.concatenate((selected, remaining[: count - selected.size]))
    return np.sort(selected[:count])


def audit_against_full_fits(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    candidate: V2AlignmentResult,
    audit_indices: np.ndarray,
    *,
    rotation_limit_degrees: float = 1.0,
    coordinate_limit_A: float = 0.25,
    rmsd_limit_A: float = 0.25,
) -> AuditResult:
    """Compare candidate transforms with untimed full fits on audit frames."""

    started = time.perf_counter()
    indices = np.asarray(audit_indices, dtype=np.int64)
    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    stable = np.asarray(weights, dtype=np.float64)
    full = weighted_kabsch_transform_batch(
        values[indices], target, stable, validate_inputs=False
    )
    relative = candidate.rotations[indices] @ np.swapaxes(full.rotations, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation = np.degrees(np.arccos(cosine))
    translation = np.linalg.norm(candidate.translations[indices] - full.translations, axis=1)
    candidate_aligned = apply_transforms_batch(
        values[indices], candidate.rotations[indices], candidate.translations[indices]
    )
    full_aligned = apply_transforms_batch(values[indices], full.rotations, full.translations)
    coordinate = np.sqrt(
        np.mean(np.sum((candidate_aligned - full_aligned) ** 2, axis=2), axis=1)
    )
    full_rmsd = np.sqrt(
        np.mean(np.sum((full_aligned - target[None, :, :]) ** 2, axis=2), axis=1)
    )
    rmsd_difference = np.abs(candidate.rmsd[indices] - full_rmsd)
    flagged = (
        (rotation > rotation_limit_degrees)
        | (coordinate > coordinate_limit_A)
        | (rmsd_difference > rmsd_limit_A)
    )
    return AuditResult(
        audit_indices=indices,
        rotation_geodesic_degrees=rotation,
        translation_difference=translation,
        aligned_coordinate_rmsd=coordinate,
        rmsd_absolute_difference=rmsd_difference,
        flagged_indices=indices[flagged],
        unresolved_rate=float(np.mean(flagged)) if flagged.size else 0.0,
        timing={"audit_seconds": time.perf_counter() - started},
        metadata={
            "rotation_limit_degrees": rotation_limit_degrees,
            "coordinate_limit_A": coordinate_limit_A,
            "rmsd_limit_A": rmsd_limit_A,
            "whole_trajectory_baseline_run": False,
        },
    )
