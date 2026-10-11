"""Experimental alignment-free stable/mobile scanning from sparse distances."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .window_descriptors import (
    SparseEdgeSet,
    build_local_window_edges,
    combine_edge_sets,
    edge_squared_distances,
    validate_coordinates,
)


@dataclass(frozen=True, slots=True)
class WindowScanResult:
    """Alignment-free movement and stability estimates.

    ``movement_score`` is a distance-instability index, not RMSD. This result
    contains no rigid transform and no aligned-coordinate RMSF.
    """

    movement_score: np.ndarray
    movement_percentile: np.ndarray
    stability_weights: np.ndarray
    candidate_stable_label: np.ndarray
    support_count: np.ndarray
    support_uncertainty: np.ndarray
    edge_movement: np.ndarray
    edge_centre: np.ndarray
    edge_scale: np.ndarray
    edges: SparseEdgeSet
    statistic: str
    mapping: str
    timing: dict[str, float]
    metadata: dict[str, object]


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average-tie percentiles in ``[0, 1]``."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("values must contain only finite values")
    if scores.size == 1 or np.allclose(scores, scores[0], rtol=0.0, atol=1e-15):
        return np.zeros(scores.size, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    result = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and np.isclose(
            scores[order[stop]],
            scores[order[start]],
            rtol=1e-12,
            atol=1e-15,
        ):
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1) / (scores.size - 1)
        start = stop
    return result


def robust_relative_mad(
    squared_distances: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute robust relative MAD for every squared-distance feature."""

    values = np.asarray(squared_distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("squared_distances must have shape (frames, edges)")
    centre = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - centre[None, :]), axis=0)
    movement = scale / (np.abs(centre) + float(epsilon))
    return movement, centre, scale


def winsorised_relative_variance(
    squared_distances: np.ndarray,
    *,
    sample_count: int = 512,
    clip_sigma: float = 4.0,
    epsilon: float = 1e-12,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sampled-centre, winsorised relative standard deviation."""

    values = np.asarray(squared_distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("squared_distances must have shape (frames, edges)")
    sample_count = min(max(1, int(sample_count)), values.shape[0])
    sample_indices = np.linspace(0, values.shape[0] - 1, sample_count, dtype=int)
    sample = values[sample_indices]
    centre = np.median(sample, axis=0)
    robust_scale = 1.4826 * np.median(np.abs(sample - centre[None, :]), axis=0)
    lower = centre - float(clip_sigma) * robust_scale
    upper = centre + float(clip_sigma) * robust_scale
    total = np.zeros(values.shape[1], dtype=np.float64)
    total_squared = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, values.shape[0], int(chunk_size)):
        clipped = np.clip(values[start : start + int(chunk_size)], lower, upper)
        total += np.sum(clipped, axis=0)
        total_squared += np.einsum("te,te->e", clipped, clipped)
    mean = total / values.shape[0]
    variance = np.maximum(total_squared / values.shape[0] - mean * mean, 0.0)
    scale = np.sqrt(variance)
    movement = scale / (np.abs(centre) + float(epsilon))
    return movement, centre, scale


def aggregate_edge_movement(
    edge_movement: np.ndarray,
    edges: SparseEdgeSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate incident edge movement by a robust per-node median."""

    values = np.asarray(edge_movement, dtype=np.float64)
    if values.shape != (edges.n_edges,):
        raise ValueError("edge_movement must have shape (n_edges,)")
    incident: list[list[float]] = [[] for _ in range(edges.n_residues)]
    for source, target, value in zip(
        edges.source,
        edges.target,
        values,
        strict=True,
    ):
        incident[int(source)].append(float(value))
        incident[int(target)].append(float(value))
    movement = np.zeros(edges.n_residues, dtype=np.float64)
    support = np.zeros(edges.n_residues, dtype=np.int64)
    for index, local_values in enumerate(incident):
        support[index] = len(local_values)
        movement[index] = float(np.median(local_values)) if local_values else 0.0
    return movement, support


def map_movement_to_stability(
    movement: np.ndarray,
    *,
    mapping: str = "rank",
) -> tuple[np.ndarray, np.ndarray]:
    """Map movement to soft stability using a transparent fixed rule."""

    values = np.asarray(movement, dtype=np.float64)
    percentiles = percentile_ranks(values)
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        return np.ones(values.size, dtype=np.float64), percentiles
    if mapping == "rank":
        weights = 1.0 - percentiles
    elif mapping == "logistic":
        centre = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - centre)))
        scale = max(scale, np.finfo(np.float64).eps)
        z_score = np.clip((values - centre) / scale, -40.0, 40.0)
        weights = 1.0 / (1.0 + np.exp(z_score))
    elif mapping == "piecewise":
        lower, upper = np.quantile(values, [0.25, 0.75])
        if np.isclose(lower, upper):
            weights = 1.0 - percentiles
        else:
            weights = 1.0 - np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    else:
        raise ValueError("mapping must be 'rank', 'logistic', or 'piecewise'")
    return np.clip(weights, 0.0, 1.0), percentiles


def smooth_weights_chain_aware(
    weights: np.ndarray,
    breaks: np.ndarray,
    *,
    strength: float = 0.05,
) -> np.ndarray:
    """Apply one light sequence-smoothing pass without crossing a break."""

    values = np.asarray(weights, dtype=np.float64)
    break_values = np.asarray(breaks, dtype=bool)
    if break_values.shape != (max(0, values.size - 1),):
        raise ValueError("breaks has the wrong shape")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0.0 or values.size < 2:
        return values.copy()
    output = values.copy()
    for index in range(values.size):
        neighbours: list[float] = []
        if index > 0 and not break_values[index - 1]:
            neighbours.append(float(values[index - 1]))
        if index + 1 < values.size and not break_values[index]:
            neighbours.append(float(values[index + 1]))
        if neighbours:
            output[index] = (1.0 - strength) * values[index] + strength * float(
                np.mean(neighbours)
            )
    return np.clip(output, 0.0, 1.0)


def stable_labels_from_weights(
    weights: np.ndarray,
    *,
    core_fraction: float = 0.6,
) -> np.ndarray:
    """Select a deterministic top-fraction candidate core."""

    values = np.asarray(weights, dtype=np.float64)
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")
    count = max(1, int(np.floor(values.size * core_fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    labels = np.zeros(values.size, dtype=bool)
    labels[order[:count]] = True
    return labels


def scan_sparse_edges(
    coords: np.ndarray,
    edges: SparseEdgeSet,
    *,
    statistic: str = "relative_mad",
    mapping: str = "rank",
    core_fraction: float = 0.6,
    sequence_regularization_strength: float = 0.05,
    descriptor_chunk_size: int = 2000,
    sample_count: int = 512,
) -> WindowScanResult:
    """Scan a supplied sparse graph without rigid alignment."""

    total_start = time.perf_counter()
    values = validate_coordinates(coords)
    descriptor_start = time.perf_counter()
    squared = edge_squared_distances(
        values,
        edges,
        chunk_size=descriptor_chunk_size,
    )
    descriptor_seconds = time.perf_counter() - descriptor_start
    aggregation_start = time.perf_counter()
    if statistic == "relative_mad":
        edge_movement, edge_centre, edge_scale = robust_relative_mad(squared)
    elif statistic == "winsorised_variance":
        edge_movement, edge_centre, edge_scale = winsorised_relative_variance(
            squared,
            sample_count=sample_count,
            chunk_size=descriptor_chunk_size,
        )
    else:
        raise ValueError("statistic must be 'relative_mad' or 'winsorised_variance'")
    movement, support = aggregate_edge_movement(edge_movement, edges)
    aggregation_seconds = time.perf_counter() - aggregation_start
    mapping_start = time.perf_counter()
    weights, percentiles = map_movement_to_stability(movement, mapping=mapping)
    weights = smooth_weights_chain_aware(
        weights,
        edges.continuity_breaks,
        strength=sequence_regularization_strength,
    )
    labels = stable_labels_from_weights(weights, core_fraction=core_fraction)
    maximum_support = max(1, int(np.max(support)))
    uncertainty = 1.0 - support.astype(np.float64) / maximum_support
    mapping_seconds = time.perf_counter() - mapping_start
    total_seconds = time.perf_counter() - total_start
    temporary_bytes = squared.nbytes
    return WindowScanResult(
        movement_score=movement,
        movement_percentile=percentiles,
        stability_weights=weights,
        candidate_stable_label=labels,
        support_count=support,
        support_uncertainty=uncertainty,
        edge_movement=edge_movement,
        edge_centre=edge_centre,
        edge_scale=edge_scale,
        edges=edges,
        statistic=statistic,
        mapping=mapping,
        timing={
            "descriptor_calculation": descriptor_seconds,
            "robust_aggregation": aggregation_seconds,
            "weight_mapping_regularization": mapping_seconds,
            "total_scan_only": total_seconds,
        },
        metadata={
            "output_contract": "scan_only",
            "movement_score_semantics": "alignment-free squared-distance instability; not RMSD",
            "n_frames": int(values.shape[0]),
            "n_residues": int(values.shape[1]),
            "n_edges": edges.n_edges,
            "descriptor_values": int(values.shape[0] * edges.n_edges),
            "temporary_arrays_gt_10_mib": int(temporary_bytes > 10 * 1024**2),
            "largest_descriptor_array_bytes": int(temporary_bytes),
            "sample_count": min(int(sample_count), values.shape[0]),
            "sampling_rule": "all_frames_for_relative_mad;uniform_for_winsor_centre",
            "core_fraction": float(core_fraction),
            "sequence_regularization_strength": float(
                sequence_regularization_strength
            ),
        },
    )


def scan_window_local(
    coords: np.ndarray,
    *,
    window_size: int = 5,
    feature_set: str = "minimal",
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    statistic: str = "relative_mad",
    mapping: str = "rank",
    core_fraction: float = 0.6,
    sequence_regularization_strength: float = 0.05,
    descriptor_chunk_size: int = 2000,
    sample_count: int = 512,
) -> WindowScanResult:
    """Run the direct chain-aware local-window scan."""

    values = validate_coordinates(coords)
    edges = build_local_window_edges(
        values.shape[1],
        window_size=window_size,
        feature_set=feature_set,
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    return scan_sparse_edges(
        values,
        edges,
        statistic=statistic,
        mapping=mapping,
        core_fraction=core_fraction,
        sequence_regularization_strength=sequence_regularization_strength,
        descriptor_chunk_size=descriptor_chunk_size,
        sample_count=sample_count,
    )


def scan_window_multiscale(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    statistic: str = "relative_mad",
    mapping: str = "rank",
) -> WindowScanResult:
    """Run a W3/W5/W9 full-feature local scan."""

    values = validate_coordinates(coords)
    edge_sets = [
        build_local_window_edges(
            values.shape[1],
            window_size=size,
            feature_set="full",
            chain_ids=chain_ids,
            residue_positions=residue_positions,
        )
        for size in (3, 5, 9)
    ]
    edges = combine_edge_sets(*edge_sets)
    return scan_sparse_edges(
        values,
        edges,
        statistic=statistic,
        mapping=mapping,
    )
