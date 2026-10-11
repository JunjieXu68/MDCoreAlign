"""Experimental same-shape multi-trajectory GPU batching.

This module is intentionally outside the public default path. It batches
trajectories that share frame count, residue mapping and chain metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mdcorealign.gpu_backend import _cupy, _event_seconds, _row_operators_horn
from mdsuperpose.window_descriptors import build_local_window_edges
from mdsuperpose.window_scan import (
    aggregate_edge_movement,
    map_movement_to_stability,
    smooth_weights_chain_aware,
    stable_labels_from_weights,
)


@dataclass(slots=True)
class BatchedGPUResult:
    """Compact outputs for one same-shape GPU trajectory batch."""

    weights: np.ndarray | None
    rotations: np.ndarray | None
    translations: np.ndarray | None
    rmsd: np.ndarray | None
    weighted_rmsd: np.ndarray | None
    robust_rmsf: np.ndarray | None
    ordinary_rmsf: np.ndarray | None
    local_deformation_index: np.ndarray | None
    local_rigidity_score: np.ndarray | None
    local_stable_labels: np.ndarray | None
    timing: dict[str, float]
    peak_vram_bytes: int


def _kabsch_multi(cp, coords, reference, weights):
    """Fit BxF coordinate batches with one weight vector per trajectory."""

    weight_sum = cp.sum(weights, axis=1)
    reference_center = cp.einsum("bni,bn->bi", reference, weights) / weight_sum[:, None]
    mobile_center = cp.einsum("bfni,bn->bfi", coords, weights) / weight_sum[:, None, None]
    mobile_centered = coords - mobile_center[:, :, None, :]
    reference_centered = reference - reference_center[:, None, :]
    weighted_reference = reference_centered * (
        weights / weight_sum[:, None]
    )[:, :, None]
    covariance = cp.einsum("bfni,bnj->bfij", mobile_centered, weighted_reference)
    shape = covariance.shape
    row_operators = _row_operators_horn(cp, covariance.reshape(-1, 3, 3)).reshape(shape)
    translations = reference_center[:, None, :] - cp.einsum(
        "bfi,bfij->bfj", mobile_center, row_operators
    )
    aligned = cp.einsum("bfni,bfij->bfnj", coords, row_operators) + translations[:, :, None, :]
    return cp.swapaxes(row_operators, -1, -2), translations, aligned


def _smooth_chain_batch(cp, frame_weights, chains, strength: float = 0.05):
    if strength <= 0.0 or frame_weights.shape[-1] < 2:
        return frame_weights
    valid = np.ones(frame_weights.shape[-1] - 1, dtype=bool)
    if chains is not None:
        values = np.asarray(chains)
        valid = values[1:] == values[:-1]
    valid_gpu = cp.asarray(valid)
    neighbour_sum = cp.zeros_like(frame_weights)
    neighbour_count = cp.zeros(frame_weights.shape[-1], dtype=frame_weights.dtype)
    neighbour_sum[..., 1:] += frame_weights[..., :-1] * valid_gpu
    neighbour_sum[..., :-1] += frame_weights[..., 1:] * valid_gpu
    neighbour_count[1:] += valid_gpu
    neighbour_count[:-1] += valid_gpu
    neighbour_mean = neighbour_sum / cp.maximum(neighbour_count, 1.0)
    return cp.where(
        neighbour_count > 0,
        (1.0 - strength) * frame_weights + strength * neighbour_mean,
        frame_weights,
    )


def solve_same_shape_gpu_batch(
    coords_batch: np.ndarray,
    *,
    chain_ids: np.ndarray | None,
    representative_count: int = 256,
    update_rounds: int = 4,
    run_throughput: bool = True,
    run_w5: bool = False,
) -> BatchedGPUResult:
    """Run throughput and/or W5 for one same-shape trajectory batch."""

    total_start = time.perf_counter()
    values = np.asarray(coords_batch, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("coords_batch must have shape (batch, frames, residues, 3)")
    if not np.isfinite(values).all():
        raise ValueError("coords_batch must contain finite values")
    batch_size, n_frames, n_residues, _ = values.shape
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (n_residues,):
        raise ValueError("chain_ids must have shape (n_residues,)")
    cp = _cupy()
    cp.cuda.Device().synchronize()
    transfer_start = time.perf_counter()
    device = cp.asarray(values)
    cp.cuda.Device().synchronize()
    h2d_seconds = time.perf_counter() - transfer_start
    peak = int(cp.get_default_memory_pool().used_bytes())
    weights_host = rotations_host = translations_host = None
    rmsd_host = weighted_host = robust_rmsf_host = ordinary_rmsf_host = None
    local_index = local_rigidity = local_labels = None
    throughput_compute = w5_compute = d2h_seconds = cpu_post = 0.0

    if run_throughput:
        selected = np.unique(
            np.linspace(0, n_frames - 1, min(representative_count, n_frames), dtype=np.int64)
        )
        reps = device[:, selected]
        reference = device[:, 0]
        weights = cp.ones((batch_size, n_residues), dtype=cp.float32)

        def estimate():
            nonlocal weights
            for _ in range(int(update_rounds)):
                _, _, aligned = _kabsch_multi(cp, reps, reference, weights)
                distances = cp.linalg.norm(aligned - reference[:, None, :, :], axis=3)
                frame_weights = cp.where(
                    distances > 1.0,
                    1.0 / cp.maximum(distances, 1e-12),
                    1.0,
                )
                frame_weights = _smooth_chain_batch(cp, frame_weights, chains)
                weights = cp.mean(frame_weights, axis=1)
                weights /= cp.max(weights, axis=1)[:, None]

        _, elapsed = _event_seconds(cp, estimate)
        throughput_compute += elapsed

        def fit_all(
            device=device,
            reference=reference,
            weights=weights,
        ):
            rotations, translations, aligned = _kabsch_multi(cp, device, reference, weights)
            uniform = cp.ones_like(weights)
            _, _, ordinary = _kabsch_multi(cp, device, reference, uniform)
            delta = aligned - reference[:, None, :, :]
            squared = cp.sum(delta * delta, axis=3)
            rmsd = cp.sqrt(cp.mean(squared, axis=2))
            weighted_rmsd = cp.sqrt(
                cp.sum(squared * weights[:, None, :], axis=2)
                / cp.sum(weights, axis=1)[:, None]
            )
            robust_mean = cp.mean(aligned, axis=1)
            ordinary_mean = cp.mean(ordinary, axis=1)
            robust_rmsf = cp.sqrt(
                cp.sum(cp.mean((aligned - robust_mean[:, None]) ** 2, axis=1), axis=2)
            )
            ordinary_rmsf = cp.sqrt(
                cp.sum(cp.mean((ordinary - ordinary_mean[:, None]) ** 2, axis=1), axis=2)
            )
            return rotations, translations, rmsd, weighted_rmsd, robust_rmsf, ordinary_rmsf

        fitted, elapsed = _event_seconds(cp, fit_all)
        throughput_compute += elapsed
        peak = max(peak, int(cp.get_default_memory_pool().used_bytes()))
        transfer_start = time.perf_counter()
        (
            rotations_host,
            translations_host,
            rmsd_host,
            weighted_host,
            robust_rmsf_host,
            ordinary_rmsf_host,
        ) = [cp.asnumpy(item) for item in fitted]
        weights_host = cp.asnumpy(weights)
        cp.cuda.Device().synchronize()
        d2h_seconds += time.perf_counter() - transfer_start

    if run_w5:
        edges = build_local_window_edges(
            n_residues,
            window_size=5,
            feature_set="minimal",
            chain_ids=chains,
        )
        source = cp.asarray(edges.source)
        target = cp.asarray(edges.target)

        def w5_scan(
            device=device,
            source=source,
            target=target,
        ):
            delta = device[:, :, source, :] - device[:, :, target, :]
            distances = cp.sum(delta * delta, axis=3)
            centre = cp.median(distances, axis=1)
            scale = 1.4826 * cp.median(cp.abs(distances - centre[:, None, :]), axis=1)
            return scale / (cp.abs(centre) + 1e-12)

        edge_movement, elapsed = _event_seconds(cp, w5_scan)
        w5_compute += elapsed
        peak = max(peak, int(cp.get_default_memory_pool().used_bytes()))
        transfer_start = time.perf_counter()
        edge_host = cp.asnumpy(edge_movement)
        cp.cuda.Device().synchronize()
        d2h_seconds += time.perf_counter() - transfer_start
        post_start = time.perf_counter()
        local_index = np.empty((batch_size, n_residues), dtype=np.float64)
        local_rigidity = np.empty_like(local_index)
        local_labels = np.empty((batch_size, n_residues), dtype=bool)
        for index in range(batch_size):
            movement, _ = aggregate_edge_movement(edge_host[index], edges)
            stability, _ = map_movement_to_stability(movement, mapping="rank")
            stability = smooth_weights_chain_aware(
                stability,
                edges.continuity_breaks,
                strength=0.05,
            )
            local_index[index] = movement
            local_rigidity[index] = stability
            local_labels[index] = stable_labels_from_weights(stability, core_fraction=0.6)
        cpu_post = time.perf_counter() - post_start

    cp.cuda.Device().synchronize()
    del device
    total = time.perf_counter() - total_start
    return BatchedGPUResult(
        weights=weights_host,
        rotations=rotations_host,
        translations=translations_host,
        rmsd=rmsd_host,
        weighted_rmsd=weighted_host,
        robust_rmsf=robust_rmsf_host,
        ordinary_rmsf=ordinary_rmsf_host,
        local_deformation_index=local_index,
        local_rigidity_score=local_rigidity,
        local_stable_labels=local_labels,
        timing={
            "H2D_seconds": h2d_seconds,
            "throughput_GPU_compute_seconds": throughput_compute,
            "W5_GPU_compute_seconds": w5_compute,
            "D2H_seconds": d2h_seconds,
            "CPU_postprocess_seconds": cpu_post,
            "total_seconds": total,
        },
        peak_vram_bytes=peak,
    )
