"""Experimental sparse-landmark streaming alignment primitives.

This module is intentionally isolated from the public exact and throughput
paths.  It never estimates a full binary core and it never labels sampled RMSD
or RMSF as exact.  Stable landmarks are estimated from representative frames,
then sparse rigid fits and optional keyframe interpolation reduce the residue
and time dimensions independently.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_scan import percentile_ranks

LandmarkMethod = Literal[
    "lowest_deformation_only",
    "lowest_deformation_sequence_spacing",
    "lowest_deformation_3d_farthest",
    "chain_balanced_spatial",
]


@dataclass(frozen=True, slots=True)
class LandmarkSet:
    """One deterministic stable-landmark set for a trajectory."""

    indices: np.ndarray
    confidence: np.ndarray
    representative_indices: np.ndarray
    local_deformation_index: np.ndarray
    representative_residual: np.ndarray
    combined_score: np.ndarray
    method: str
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class LandmarkEvidence:
    """Reusable representative-frame evidence for landmark ablations."""

    representative_indices: np.ndarray
    local_deformation_index: np.ndarray
    local_rigidity_score: np.ndarray
    local_confidence: np.ndarray
    representative_residual: np.ndarray
    combined_score: np.ndarray
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class SparseFitResult:
    """Rigid transforms fitted from a sparse landmark set."""

    rotations: np.ndarray
    translations: np.ndarray
    landmark_rmsd: np.ndarray
    runtime_seconds: float
    timing: dict[str, float]
    peak_vram_bytes: int
    backend: str


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Output shared by the three experimental stream contracts."""

    mode: str
    landmark_set: LandmarkSet
    keyframe_indices: np.ndarray
    rotations: np.ndarray | None
    translations: np.ndarray | None
    approximate_rmsd: np.ndarray | None
    approximate_rmsf: np.ndarray | None
    profile_frame_indices: np.ndarray
    validation_frame_indices: np.ndarray
    refined_intervals: np.ndarray
    unresolved_intervals: np.ndarray
    interval_error_estimates: np.ndarray
    runtime_seconds: float
    timing: dict[str, float]
    peak_ram_mb: float
    peak_vram_bytes: int
    metadata: dict[str, object]


def temporally_stratified_indices(
    n_frames: int,
    count: int,
    *,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic temporal strata without duplicate indices."""

    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    target = min(int(count), int(n_frames))
    if target < 1:
        raise ValueError("count must be positive")
    excluded = set()
    if exclude is not None:
        excluded = {int(value) for value in np.asarray(exclude, dtype=int)}
    available = np.asarray(
        [index for index in range(n_frames) if index not in excluded],
        dtype=int,
    )
    if available.size == 0:
        return np.empty(0, dtype=int)
    target = min(target, available.size)
    positions = np.linspace(0, available.size - 1, target)
    return available[np.rint(positions).astype(int)]


def _percentile_low(values: np.ndarray) -> np.ndarray:
    return 1.0 - percentile_ranks(np.asarray(values, dtype=np.float64))


def _greedy_sequence_spaced(
    order: np.ndarray,
    chains: np.ndarray,
    count: int,
    minimum_spacing: int,
) -> np.ndarray:
    chosen: list[int] = []
    for candidate in order:
        index = int(candidate)
        if all(
            chains[index] != chains[other]
            or abs(index - other) >= minimum_spacing
            for other in chosen
        ):
            chosen.append(index)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        for candidate in order:
            index = int(candidate)
            if index not in chosen:
                chosen.append(index)
                if len(chosen) == count:
                    break
    return np.asarray(chosen, dtype=int)


def _farthest_spatial(
    candidates: np.ndarray,
    reference: np.ndarray,
    score: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select a scored, spatially dispersed deterministic landmark subset."""

    pool = np.asarray(candidates, dtype=int)
    if pool.size <= count:
        return pool.copy()
    chosen = [int(pool[np.argmax(score[pool])])]
    minimum_squared = np.sum(
        (reference[pool] - reference[chosen[0]]) ** 2,
        axis=1,
    )
    score_scale = float(np.ptp(score[pool]))
    score_scaled = (
        np.ones(pool.size)
        if score_scale <= np.finfo(float).eps
        else (score[pool] - np.min(score[pool])) / score_scale
    )
    while len(chosen) < count:
        dispersion = minimum_squared / max(
            float(np.max(minimum_squared)),
            np.finfo(float).eps,
        )
        objective = 0.75 * dispersion + 0.25 * score_scaled
        objective[np.isin(pool, chosen)] = -np.inf
        selected = int(pool[int(np.argmax(objective))])
        chosen.append(selected)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((reference[pool] - reference[selected]) ** 2, axis=1),
        )
    return np.asarray(chosen, dtype=int)


def _chain_balanced_spatial(
    candidates: np.ndarray,
    reference: np.ndarray,
    score: np.ndarray,
    chains: np.ndarray,
    count: int,
) -> np.ndarray:
    unique, sizes = np.unique(chains, return_counts=True)
    raw = count * sizes / np.sum(sizes)
    allocations = np.floor(raw).astype(int)
    allocations[(allocations == 0) & (sizes > 0)] = 1
    while allocations.sum() > count:
        eligible = np.where(allocations > 1)[0]
        allocations[eligible[np.argmax(allocations[eligible] - raw[eligible])]] -= 1
    while allocations.sum() < count:
        capacity = sizes - allocations
        eligible = np.where(capacity > 0)[0]
        allocations[eligible[np.argmax(raw[eligible] - allocations[eligible])]] += 1

    chosen: list[int] = []
    for chain, allocation in zip(unique, allocations, strict=True):
        local = candidates[chains[candidates] == chain]
        if local.size < allocation:
            local = np.where(chains == chain)[0]
        chosen.extend(
            _farthest_spatial(
                local,
                reference,
                score,
                min(int(allocation), local.size),
            ).tolist()
        )
    if len(chosen) < count:
        remaining = candidates[~np.isin(candidates, chosen)]
        chosen.extend(
            _farthest_spatial(
                remaining,
                reference,
                score,
                min(count - len(chosen), remaining.size),
            ).tolist()
        )
    return np.asarray(chosen[:count], dtype=int)


def build_landmark_evidence(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    evaluate_temporal_blocks: bool = False,
) -> LandmarkEvidence:
    """Compute representative-frame evidence once for many subset choices."""

    started = time.perf_counter()
    values = np.asarray(coords)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    n_frames, n_residues, _ = values.shape
    chains = (
        np.asarray(["A"] * n_residues, dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    if chains.shape != (n_residues,):
        raise ValueError("chain_ids must have shape (n_residues,)")
    representatives = temporally_stratified_indices(n_frames, representative_count)
    if not np.isfinite(values[representatives]).all():
        raise ValueError("representative coordinates contain non-finite values")

    local_start = time.perf_counter()
    local = scan_local_kinematics(
        values[representatives],
        chain_ids=chains,
        evaluate_blocks=evaluate_temporal_blocks,
    )
    local_seconds = time.perf_counter() - local_start

    residual_start = time.perf_counter()
    representative_fit = weighted_kabsch_batch(
        values[representatives],
        values[0],
        np.ones(n_residues, dtype=np.float64),
        validate_inputs=False,
        compute_metrics=False,
    )
    residual = np.mean(
        np.linalg.norm(
            representative_fit.aligned_coords - values[0][None, :, :],
            axis=2,
        ),
        axis=0,
    )
    residual_seconds = time.perf_counter() - residual_start

    rigidity = np.asarray(local.local_rigidity_score, dtype=np.float64)
    combined = 0.65 * rigidity + 0.35 * _percentile_low(residual)
    total = time.perf_counter() - started
    return LandmarkEvidence(
        representative_indices=representatives,
        local_deformation_index=np.asarray(
            local.local_deformation_index,
            dtype=np.float64,
        ),
        local_rigidity_score=rigidity,
        local_confidence=np.asarray(local.confidence, dtype=np.float64),
        representative_residual=np.asarray(residual, dtype=np.float64),
        combined_score=np.asarray(combined, dtype=np.float64),
        timing={
            "representative_W5_seconds": local_seconds,
            "representative_residual_seconds": residual_seconds,
            "total_evidence_seconds": total,
        },
        metadata={
            "representative_count": int(representatives.size),
            "frame_sampling_fraction": float(representatives.size / n_frames),
            "temporal_block_reproducibility": (
                float(local.temporal_block_reproducibility)
                if evaluate_temporal_blocks
                else "not_evaluated_in_landmark_fast_path"
            ),
        },
    )


def select_sparse_landmarks(
    reference: np.ndarray,
    evidence: LandmarkEvidence,
    *,
    chain_ids: np.ndarray | None = None,
    landmark_count: int = 64,
    method: LandmarkMethod = "chain_balanced_spatial",
) -> LandmarkSet:
    """Select one landmark set from cached representative-frame evidence."""

    started = time.perf_counter()
    target = np.asarray(reference)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("reference must have shape (n_residues, 3)")
    n_residues = target.shape[0]
    if not 3 <= int(landmark_count) <= n_residues:
        raise ValueError("landmark_count must be between 3 and n_residues")
    chains = (
        np.asarray(["A"] * n_residues, dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    if chains.shape != (n_residues,):
        raise ValueError("chain_ids must have shape (n_residues,)")
    combined = evidence.combined_score
    residual = evidence.representative_residual
    order = np.lexsort((np.arange(n_residues), residual, -combined))
    pool_count = min(n_residues, max(landmark_count * 4, 64))
    pool = order[:pool_count]
    selection_start = time.perf_counter()
    if method == "lowest_deformation_only":
        selected = order[:landmark_count]
    elif method == "lowest_deformation_sequence_spacing":
        spacing = max(2, n_residues // max(2 * landmark_count, 1))
        selected = _greedy_sequence_spaced(
            order,
            chains,
            landmark_count,
            spacing,
        )
    elif method == "lowest_deformation_3d_farthest":
        selected = _farthest_spatial(
            pool,
            target,
            combined,
            landmark_count,
        )
    elif method == "chain_balanced_spatial":
        selected = _chain_balanced_spatial(
            pool,
            target,
            combined,
            chains,
            landmark_count,
        )
    else:
        raise ValueError(f"unsupported landmark method: {method}")
    selected = np.asarray(selected, dtype=int)
    if np.unique(selected).size != landmark_count:
        raise RuntimeError("landmark selection did not return unique indices")
    centred = target[selected] - np.mean(target[selected], axis=0)
    spatial_rank = int(np.linalg.matrix_rank(centred, tol=1e-8))
    if spatial_rank < 2:
        raise RuntimeError("selected landmarks do not contain three non-collinear points")
    selection_seconds = time.perf_counter() - selection_start
    total = time.perf_counter() - started
    return LandmarkSet(
        indices=np.sort(selected),
        confidence=np.asarray(combined[selected], dtype=np.float64),
        representative_indices=evidence.representative_indices,
        local_deformation_index=evidence.local_deformation_index,
        representative_residual=np.asarray(residual, dtype=np.float64),
        combined_score=np.asarray(combined, dtype=np.float64),
        method=method,
        timing={
            **evidence.timing,
            "landmark_selection_seconds": selection_seconds,
            "total_landmark_estimation_seconds": evidence.timing["total_evidence_seconds"] + total,
        },
        metadata={
            "representative_count": int(evidence.representative_indices.size),
            "landmark_count": int(selected.size),
            "frame_sampling_fraction": evidence.metadata["frame_sampling_fraction"],
            "chain_count": int(np.unique(chains[selected]).size),
            "spatial_rank": spatial_rank,
            "temporal_block_reproducibility": evidence.metadata[
                "temporal_block_reproducibility"
            ],
            "score_semantics": "combined local rigidity and representative-frame residual stability",
        },
    )


def estimate_sparse_landmarks(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    landmark_count: int = 64,
    method: LandmarkMethod = "chain_balanced_spatial",
) -> LandmarkSet:
    """Estimate stable, spatially dispersed landmarks from sampled frames."""

    values = np.asarray(coords)
    evidence = build_landmark_evidence(
        values,
        chain_ids=chain_ids,
        representative_count=representative_count,
    )
    return select_sparse_landmarks(
        values[0],
        evidence,
        chain_ids=chain_ids,
        landmark_count=landmark_count,
        method=method,
    )


def _fit_sparse_cpu(
    coords: np.ndarray,
    reference: np.ndarray,
    landmarks: np.ndarray,
    chunk_size: int,
) -> SparseFitResult:
    started = time.perf_counter()
    n_frames = coords.shape[0]
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    residual = np.empty(n_frames, dtype=np.float64)
    selected_reference = reference[landmarks]
    weights = np.ones(landmarks.size, dtype=np.float64)
    fit_seconds = 0.0
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        fit_start = time.perf_counter()
        result = weighted_kabsch_batch(
            coords[start:stop, landmarks],
            selected_reference,
            weights,
            validate_inputs=False,
            compute_metrics=True,
        )
        fit_seconds += time.perf_counter() - fit_start
        rotations[start:stop] = result.rotations
        translations[start:stop] = result.translations
        residual[start:stop] = result.rmsd
    total = time.perf_counter() - started
    return SparseFitResult(
        rotations=rotations,
        translations=translations,
        landmark_rmsd=residual,
        runtime_seconds=total,
        timing={"sparse_fit_compute_seconds": fit_seconds, "total_seconds": total},
        peak_vram_bytes=0,
        backend="cpu_svd_float64",
    )


def _fit_sparse_gpu(
    coords: np.ndarray,
    reference: np.ndarray,
    landmarks: np.ndarray,
    chunk_size: int,
) -> SparseFitResult:
    from mdcorealign.gpu_backend import (  # Local import keeps CUDA optional.
        _cupy,
        _event_seconds,
        _weighted_kabsch_gpu,
    )

    cp = _cupy()
    started = time.perf_counter()
    cp.cuda.Device().synchronize()
    transfer_start = time.perf_counter()
    target = cp.asarray(reference[landmarks], dtype=cp.float32)
    weights = cp.ones(landmarks.size, dtype=cp.float32)
    cp.cuda.Device().synchronize()
    h2d = time.perf_counter() - transfer_start
    n_frames = coords.shape[0]
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    residual = np.empty(n_frames, dtype=np.float64)
    compute_seconds = 0.0
    d2h_seconds = 0.0
    peak_vram = int(cp.get_default_memory_pool().used_bytes())
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        cp.cuda.Device().synchronize()
        transfer_start = time.perf_counter()
        mobile = cp.asarray(coords[start:stop, landmarks], dtype=cp.float32)
        cp.cuda.Device().synchronize()
        h2d += time.perf_counter() - transfer_start

        def compute(
            mobile=mobile,
            target=target,
            weights=weights,
        ):
            rotation, translation, aligned = _weighted_kabsch_gpu(
                cp,
                mobile,
                target,
                weights,
                rotation_solver="horn",
            )
            delta = aligned - target[None, :, :]
            rmsd = cp.sqrt(cp.mean(cp.sum(delta * delta, axis=2), axis=1))
            return rotation, translation, rmsd

        output, elapsed = _event_seconds(cp, compute)
        compute_seconds += elapsed
        peak_vram = max(peak_vram, int(cp.get_default_memory_pool().used_bytes()))
        cp.cuda.Device().synchronize()
        transfer_start = time.perf_counter()
        host = [cp.asnumpy(value) for value in output]
        cp.cuda.Device().synchronize()
        d2h_seconds += time.perf_counter() - transfer_start
        rotations[start:stop] = host[0]
        translations[start:stop] = host[1]
        residual[start:stop] = host[2]
        del mobile, output
    total = time.perf_counter() - started
    return SparseFitResult(
        rotations=rotations,
        translations=translations,
        landmark_rmsd=residual,
        runtime_seconds=total,
        timing={
            "H2D_seconds": h2d,
            "GPU_compute_seconds": compute_seconds,
            "D2H_seconds": d2h_seconds,
            "total_seconds": total,
        },
        peak_vram_bytes=peak_vram,
        backend="gpu_horn_float32",
    )


def fit_sparse_landmarks(
    coords: np.ndarray,
    reference: np.ndarray,
    landmark_indices: np.ndarray,
    *,
    backend: Literal["cpu", "gpu"] = "cpu",
    chunk_size: int = 4096,
) -> SparseFitResult:
    """Fit frames using only selected landmarks without full materialisation."""

    values = np.asarray(coords)
    target = np.asarray(reference)
    landmarks = np.asarray(landmark_indices, dtype=int)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if target.shape != values.shape[1:]:
        raise ValueError("reference has incompatible shape")
    if landmarks.ndim != 1 or landmarks.size < 3:
        raise ValueError("at least three landmark indices are required")
    if np.any(landmarks < 0) or np.any(landmarks >= values.shape[1]):
        raise ValueError("landmark index out of range")
    if backend == "cpu":
        return _fit_sparse_cpu(values, target, landmarks, chunk_size)
    if backend == "gpu":
        return _fit_sparse_gpu(values, target, landmarks, chunk_size)
    raise ValueError("backend must be 'cpu' or 'gpu'")


def build_landmark_signature(
    coords: np.ndarray,
    landmark_indices: np.ndarray,
    *,
    maximum_pairs: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a rigid-invariant sparse inter-landmark distance signature."""

    values = np.asarray(coords)
    landmarks = np.asarray(landmark_indices, dtype=int)
    left, right = np.triu_indices(landmarks.size, k=1)
    sequence_separation = np.abs(landmarks[right] - landmarks[left])
    order = np.lexsort((right, left, -sequence_separation))
    if order.size > maximum_pairs:
        positions = np.linspace(0, order.size - 1, maximum_pairs)
        order = order[np.rint(positions).astype(int)]
    pairs = np.column_stack((landmarks[left[order]], landmarks[right[order]])).astype(int)
    delta = values[:, pairs[:, 0], :] - values[:, pairs[:, 1], :]
    distances = np.linalg.norm(delta, axis=2)
    reference_distance = np.maximum(distances[0], np.finfo(float).eps)
    signature = distances / reference_distance[None, :] - 1.0
    return np.asarray(signature, dtype=np.float64), pairs


def select_adaptive_keyframes(
    signature: np.ndarray,
    *,
    target_fraction: float,
    maximum_gap_factor: float = 1.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select deterministic novelty-aware keyframes with a hard gap bound."""

    values = np.asarray(signature, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("signature must have at least two frames")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    n_frames = values.shape[0]
    target = max(2, int(math.ceil(n_frames * target_fraction)))
    increments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if cumulative[-1] <= np.finfo(float).eps:
        selected = temporally_stratified_indices(n_frames, target)
    else:
        desired = np.linspace(0.0, cumulative[-1], target)
        selected = np.unique(np.searchsorted(cumulative, desired, side="left"))
        selected = np.clip(selected, 0, n_frames - 1)
    selected = np.unique(np.concatenate(([0], selected, [n_frames - 1]))).astype(int)
    curvature = np.zeros(n_frames, dtype=np.float64)
    if n_frames > 2:
        curvature[1:-1] = np.linalg.norm(
            values[2:] - 2.0 * values[1:-1] + values[:-2],
            axis=1,
        )
    if selected.size < target:
        candidate_order = np.lexsort((np.arange(n_frames), -curvature))
        available = candidate_order[~np.isin(candidate_order, selected)]
        selected = np.sort(
            np.concatenate((selected, available[: target - selected.size]))
        )
    maximum_gap = max(1, int(math.ceil(maximum_gap_factor / target_fraction)))
    while np.max(np.diff(selected)) > maximum_gap:
        gaps = np.diff(selected)
        gap_index = int(np.argmax(gaps))
        midpoint = int((selected[gap_index] + selected[gap_index + 1]) // 2)
        selected = np.sort(np.append(selected, midpoint))

    interpolation_error = np.zeros(n_frames, dtype=np.float64)
    for left_index, right_index in zip(selected[:-1], selected[1:], strict=True):
        if right_index - left_index <= 1:
            continue
        alpha = np.linspace(0.0, 1.0, right_index - left_index + 1)[:, None]
        predicted = (1.0 - alpha) * values[left_index] + alpha * values[right_index]
        interpolation_error[left_index : right_index + 1] = np.linalg.norm(
            values[left_index : right_index + 1] - predicted,
            axis=1,
        )
    return selected, {
        "target_keyframe_fraction": float(target_fraction),
        "actual_keyframe_fraction": float(selected.size / n_frames),
        "maximum_keyframe_gap": int(np.max(np.diff(selected))),
        "maximum_allowed_gap": maximum_gap,
        "signature_coverage_P95": float(np.quantile(interpolation_error, 0.95)),
        "signature_coverage_max": float(np.max(interpolation_error)),
    }


def _continuous_quaternions(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat()
    for index in range(1, quaternions.shape[0]):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def interpolate_keyframe_transforms(
    n_frames: int,
    keyframe_indices: np.ndarray,
    keyframe_rotations: np.ndarray,
    keyframe_translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate keyframe transforms with sign-continuous SLERP and linear shifts."""

    keys = np.asarray(keyframe_indices, dtype=int)
    if keys.ndim != 1 or keys.size < 2 or keys[0] != 0 or keys[-1] != n_frames - 1:
        raise ValueError("keyframes must be sorted and include first and last frames")
    quaternions = _continuous_quaternions(keyframe_rotations)
    output_quaternions = np.empty((n_frames, 4), dtype=np.float64)
    output_translations = np.empty((n_frames, 3), dtype=np.float64)
    for interval, (left, right) in enumerate(zip(keys[:-1], keys[1:], strict=True)):
        alpha = np.linspace(0.0, 1.0, right - left + 1)
        first = quaternions[interval]
        second = quaternions[interval + 1]
        dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
        if dot < 0.0:
            second = -second
            dot = -dot
        if dot > 0.9995:
            interpolated = first[None, :] + alpha[:, None] * (second - first)[None, :]
            interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)
        else:
            angle = math.acos(dot)
            sine = math.sin(angle)
            interpolated = (
                np.sin((1.0 - alpha) * angle)[:, None] / sine * first[None, :]
                + np.sin(alpha * angle)[:, None] / sine * second[None, :]
            )
        output_quaternions[left : right + 1] = interpolated
        output_translations[left : right + 1] = (
            (1.0 - alpha)[:, None] * keyframe_translations[interval]
            + alpha[:, None] * keyframe_translations[interval + 1]
        )
    return Rotation.from_quat(output_quaternions).as_matrix(), output_translations


def apply_sampled_transforms(
    coords: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Apply transforms only to an explicitly sampled set of frames."""

    sample = np.asarray(indices, dtype=int)
    return (
        np.einsum(
            "fni,fji->fnj",
            np.asarray(coords)[sample],
            np.asarray(rotations)[sample],
        )
        + np.asarray(translations)[sample, None, :]
    )


def sampled_rmsf(aligned_sample: np.ndarray) -> np.ndarray:
    """Return sample-based approximate RMSF; this is not a full exact profile."""

    values = np.asarray(aligned_sample, dtype=np.float64)
    centre = np.mean(values, axis=0)
    return np.sqrt(np.mean(np.sum((values - centre[None, :, :]) ** 2, axis=2), axis=0))


def _landmark_rmsd_from_transforms(
    coords: np.ndarray,
    reference: np.ndarray,
    landmarks: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    aligned = (
        np.einsum("fni,fji->fnj", coords[:, landmarks], rotations)
        + translations[:, None, :]
    )
    delta = aligned - reference[landmarks][None, :, :]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))


def _rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.asarray(left) @ np.swapaxes(np.asarray(right), 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _refine_keyframes(
    coords: np.ndarray,
    reference: np.ndarray,
    landmarks: np.ndarray,
    signature: np.ndarray,
    keyframes: np.ndarray,
    *,
    backend: Literal["cpu", "gpu"],
    maximum_rounds: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    int,
]:
    """Refine risky intervals using direct sparse fits on held-out audit frames."""

    timing = {"audit_seconds": 0.0, "refinement_fit_seconds": 0.0, "interpolation_seconds": 0.0}
    refined: set[tuple[int, int]] = set()
    audit_count = max(32, int(math.ceil(coords.shape[0] * 0.0075)))
    audit_count = min(audit_count, 512)
    audit = temporally_stratified_indices(coords.shape[0], audit_count, exclude=keyframes)
    current = np.asarray(keyframes, dtype=int)
    unresolved = np.empty((0, 2), dtype=int)
    estimates = np.empty((0, 6), dtype=np.float64)
    peak_vram = 0

    for round_index in range(maximum_rounds + 1):
        fit_start = time.perf_counter()
        key_fit = fit_sparse_landmarks(
            coords[current],
            reference,
            landmarks,
            backend=backend,
        )
        timing["refinement_fit_seconds"] += time.perf_counter() - fit_start
        peak_vram = max(peak_vram, key_fit.peak_vram_bytes)
        interpolation_start = time.perf_counter()
        rotations, translations = interpolate_keyframe_transforms(
            coords.shape[0],
            current,
            key_fit.rotations,
            key_fit.translations,
        )
        timing["interpolation_seconds"] += time.perf_counter() - interpolation_start

        audit_start = time.perf_counter()
        direct = fit_sparse_landmarks(
            coords[audit],
            reference,
            landmarks,
            backend=backend,
        )
        peak_vram = max(peak_vram, direct.peak_vram_bytes)
        rotation_error = _rotation_error_degrees(rotations[audit], direct.rotations)
        translation_error = np.linalg.norm(translations[audit] - direct.translations, axis=1)
        propagated = (
            np.einsum("fni,fji->fnj", coords[audit][:, landmarks], rotations[audit])
            + translations[audit, None, :]
        )
        direct_aligned = (
            np.einsum("fni,fji->fnj", coords[audit][:, landmarks], direct.rotations)
            + direct.translations[:, None, :]
        )
        coordinate_error = np.sqrt(np.mean(np.sum((propagated - direct_aligned) ** 2, axis=2), axis=1))
        landmark_residual = _landmark_rmsd_from_transforms(
            coords[audit],
            reference,
            landmarks,
            rotations[audit],
            translations[audit],
        )
        curvature = np.zeros(coords.shape[0], dtype=np.float64)
        if coords.shape[0] > 2:
            curvature[1:-1] = np.linalg.norm(
                signature[2:] - 2.0 * signature[1:-1] + signature[:-2],
                axis=1,
            )
        residual_limit = float(np.median(landmark_residual) + 2.5 * 1.4826 * np.median(np.abs(landmark_residual - np.median(landmark_residual))))
        curvature_limit = float(np.quantile(curvature[audit], 0.95))
        risk = np.maximum.reduce(
            [
                rotation_error / 0.75,
                translation_error / 0.15,
                coordinate_error / 0.15,
                landmark_residual / max(residual_limit, 1e-8),
                curvature[audit] / max(curvature_limit, 1e-8),
            ]
        )
        timing["audit_seconds"] += time.perf_counter() - audit_start
        high = risk > 1.0
        interval_rows: list[list[float]] = []
        additions: list[int] = []
        for left, right in zip(current[:-1], current[1:], strict=True):
            members = np.where((audit > left) & (audit < right))[0]
            if members.size == 0:
                continue
            local_max = members[int(np.argmax(risk[members]))]
            interval_rows.append(
                [
                    float(left),
                    float(right),
                    float(np.max(rotation_error[members])),
                    float(np.max(translation_error[members])),
                    float(np.max(coordinate_error[members])),
                    float(np.max(risk[members])),
                ]
            )
            if np.any(high[members]):
                additions.append(int(audit[local_max]))
        estimates = np.asarray(interval_rows, dtype=np.float64).reshape(-1, 6)
        unresolved = (
            estimates[estimates[:, 5] > 1.0, :2].astype(int)
            if estimates.size
            else np.empty((0, 2), dtype=int)
        )
        if round_index >= maximum_rounds or not additions:
            break
        for value in additions:
            location = int(np.searchsorted(current, value))
            refined.add((int(current[location - 1]), int(current[location])))
        current = np.unique(np.concatenate((current, additions))).astype(int)

    refined_array = np.asarray(sorted(refined), dtype=int).reshape(-1, 2)
    return current, rotations, translations, refined_array, unresolved, estimates, timing, peak_vram


def run_stream_contract(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    mode: Literal["stream-scan", "stream-profile", "stream-align"] = "stream-align",
    representative_count: int = 256,
    landmark_count: int = 64,
    landmark_method: LandmarkMethod = "chain_balanced_spatial",
    keyframe_fraction: float = 0.05,
    maximum_refinement_rounds: int = 1,
    backend: Literal["cpu", "gpu"] = "cpu",
    landmark_set: LandmarkSet | None = None,
) -> StreamResult:
    """Execute one isolated experimental streaming output contract."""

    started = time.perf_counter()
    values = np.asarray(coords)
    reference = np.asarray(values[0])
    landmark_started = time.perf_counter()
    landmarks = landmark_set or estimate_sparse_landmarks(
        values,
        chain_ids=chain_ids,
        representative_count=representative_count,
        landmark_count=landmark_count,
        method=landmark_method,
    )
    landmark_seconds = time.perf_counter() - landmark_started
    signature_started = time.perf_counter()
    signature, signature_pairs = build_landmark_signature(values, landmarks.indices)
    keyframes, keyframe_metadata = select_adaptive_keyframes(
        signature,
        target_fraction=keyframe_fraction,
    )
    signature_seconds = time.perf_counter() - signature_started
    profile_indices = temporally_stratified_indices(values.shape[0], min(512, values.shape[0]))
    validation_count = min(512, max(32, int(math.ceil(values.shape[0] * 0.0075))))
    validation_indices = temporally_stratified_indices(
        values.shape[0],
        validation_count,
        exclude=keyframes,
    )
    rotations = translations = approximate_rmsd = approximate_rmsf = None
    refined = np.empty((0, 2), dtype=int)
    unresolved = np.empty((0, 2), dtype=int)
    estimates = np.empty((0, 6), dtype=np.float64)
    fit_seconds = interpolation_seconds = profile_seconds = audit_seconds = 0.0
    peak_vram = 0

    if mode == "stream-profile":
        fit_started = time.perf_counter()
        sampled = fit_sparse_landmarks(
            values[profile_indices],
            reference,
            landmarks.indices,
            backend=backend,
        )
        fit_seconds = time.perf_counter() - fit_started
        peak_vram = sampled.peak_vram_bytes
        profile_started = time.perf_counter()
        aligned = (
            np.einsum(
                "fni,fji->fnj",
                values[profile_indices],
                sampled.rotations,
            )
            + sampled.translations[:, None, :]
        )
        approximate_rmsf = sampled_rmsf(aligned)
        approximate_rmsd = np.sqrt(
            np.mean(np.sum((aligned - reference[None, :, :]) ** 2, axis=2), axis=1)
        )
        profile_seconds = time.perf_counter() - profile_started
    elif mode == "stream-align":
        refine_started = time.perf_counter()
        (
            keyframes,
            rotations,
            translations,
            refined,
            unresolved,
            estimates,
            refine_timing,
            peak_vram,
        ) = _refine_keyframes(
            values,
            reference,
            landmarks.indices,
            signature,
            keyframes,
            backend=backend,
            maximum_rounds=maximum_refinement_rounds,
        )
        fit_seconds = refine_timing["refinement_fit_seconds"]
        interpolation_seconds = refine_timing["interpolation_seconds"]
        audit_seconds = refine_timing["audit_seconds"]
        _ = time.perf_counter() - refine_started
        approximate_rmsd = _landmark_rmsd_from_transforms(
            values,
            reference,
            landmarks.indices,
            rotations,
            translations,
        )
        profile_started = time.perf_counter()
        approximate_rmsf = sampled_rmsf(
            apply_sampled_transforms(
                values,
                rotations,
                translations,
                profile_indices,
            )
        )
        profile_seconds = time.perf_counter() - profile_started
    elif mode != "stream-scan":
        raise ValueError("mode must be stream-scan, stream-profile, or stream-align")

    total = time.perf_counter() - started
    estimated_error = (
        float(np.max(estimates[:, 4]))
        if estimates.size
        else float(keyframe_metadata["signature_coverage_P95"])
    )
    return StreamResult(
        mode=mode,
        landmark_set=landmarks,
        keyframe_indices=keyframes,
        rotations=rotations,
        translations=translations,
        approximate_rmsd=approximate_rmsd,
        approximate_rmsf=approximate_rmsf,
        profile_frame_indices=profile_indices,
        validation_frame_indices=validation_indices,
        refined_intervals=refined,
        unresolved_intervals=unresolved,
        interval_error_estimates=estimates,
        runtime_seconds=total,
        timing={
            "landmark_estimation_seconds": landmark_seconds,
            "signature_and_keyframe_seconds": signature_seconds,
            "sparse_fit_seconds": fit_seconds,
            "transform_interpolation_seconds": interpolation_seconds,
            "sampled_profile_seconds": profile_seconds,
            "sampled_audit_seconds": audit_seconds,
            "total_seconds": total,
        },
        peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        peak_vram_bytes=peak_vram,
        metadata={
            "experimental": True,
            "mode": mode,
            "landmark_count": int(landmarks.indices.size),
            "keyframe_count": int(keyframes.size),
            "frame_sampling_fraction": float(
                landmarks.representative_indices.size / values.shape[0]
            ),
            "profile_sampling_fraction": float(profile_indices.size / values.shape[0]),
            "estimated_error_A": estimated_error,
            "validation_frame_count": int(validation_indices.size),
            "refined_interval_count": int(refined.shape[0]),
            "unresolved_interval_count": int(unresolved.shape[0]),
            "approximate_RMSD_not_exact": True,
            "approximate_RMSF_not_exact": True,
            "full_aligned_trajectory_materialised": False,
            "backend": backend,
            "signature_pair_count": int(signature_pairs.shape[0]),
            **keyframe_metadata,
            "actual_keyframe_fraction": float(keyframes.size / values.shape[0]),
            "maximum_keyframe_gap": int(np.max(np.diff(keyframes))),
        },
    )
