"""Shared CPU engine for fixed-weight all-frame rigid fitting and reductions."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .fast_kabsch import apply_transforms_batch, weighted_kabsch_transform_batch


@dataclass(slots=True)
class CPUAlignmentWorkspace:
    """Reusable shape-owned metadata that never retains trajectory coordinates."""

    n_residues: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    uniform_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_residues < 3:
            raise ValueError("n_residues must be at least three")
        self.dtype = np.dtype(self.dtype)
        self.uniform_weights = np.ones(self.n_residues, dtype=self.dtype)

    def validate(self, n_residues: int, dtype: np.dtype | type) -> None:
        """Reject accidental workspace reuse across incompatible shapes."""

        if int(n_residues) != self.n_residues or np.dtype(dtype) != self.dtype:
            raise ValueError("workspace shape or dtype does not match the trajectory")


@dataclass(frozen=True, slots=True)
class AllFrameFitResult:
    """Intent-specific all-frame transforms, metrics and optional coordinates."""

    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray | None
    rmsd: np.ndarray | None
    weighted_rmsd: np.ndarray | None
    robust_rmsf: np.ndarray | None
    ordinary_rmsf: np.ndarray | None
    mean_residual: np.ndarray | None
    timing: dict[str, float]
    chunks_processed: int


@dataclass(slots=True)
class _StreamingRMSFAccumulator:
    n_residues: int
    count: int = 0
    mean: np.ndarray = field(init=False, repr=False)
    m2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros((self.n_residues, 3), dtype=np.float64)
        self.m2 = np.zeros((self.n_residues, 3), dtype=np.float64)

    def update(self, coords: np.ndarray) -> None:
        chunk_count = int(coords.shape[0])
        if chunk_count == 0:
            return
        chunk_mean = np.mean(coords, axis=0)
        centered = coords - chunk_mean[None, :, :]
        chunk_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.mean[...] = chunk_mean
            self.m2[...] = chunk_m2
            self.count = chunk_count
            return
        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean += delta * (chunk_count / total)
        self.m2 += chunk_m2 + delta * delta * (self.count * chunk_count / total)
        self.count = total

    def rmsf(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("no coordinates accumulated for RMSF")
        return np.sqrt(np.sum(self.m2 / self.count, axis=1))


def fit_all_frames_fixed_weights(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 2000,
    return_aligned: bool = False,
    compute_rmsd: bool = True,
    compute_weighted_rmsd: bool = True,
    compute_mean_residual: bool = True,
    compute_robust_rmsf: bool = True,
    compute_ordinary_rmsf: bool = True,
    workspace: CPUAlignmentWorkspace | None = None,
    validate_inputs: bool = True,
) -> AllFrameFitResult:
    """Fit each frame with fixed weights and compute only requested outputs.

    The weighted transform is always evaluated exactly in float64. Full aligned
    coordinates are retained only when ``return_aligned`` is true. Chunk-local
    coordinates are still materialised when a requested metric mathematically
    requires them.
    """

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    fixed_weights = np.asarray(weights, dtype=np.float64)
    if validate_inputs:
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("coords must have shape (n_frames, n_residues, 3)")
        if target.shape != values.shape[1:]:
            raise ValueError("reference must have shape (n_residues, 3)")
        if fixed_weights.shape != (values.shape[1],):
            raise ValueError("weights must have shape (n_residues,)")
        if values.shape[0] < 1 or values.shape[1] < 3:
            raise ValueError("at least one frame and three residues are required")
        if not (
            np.isfinite(values).all()
            and np.isfinite(target).all()
            and np.isfinite(fixed_weights).all()
        ):
            raise ValueError("coordinates and weights must be finite")
        if np.any(fixed_weights < 0.0):
            raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(fixed_weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must contain positive total mass")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    n_frames, n_residues, _ = values.shape
    if workspace is None:
        workspace = CPUAlignmentWorkspace(n_residues)
    else:
        workspace.validate(n_residues, np.float64)

    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    aligned_output = np.empty_like(values) if return_aligned else None
    rmsd = np.empty(n_frames, dtype=np.float64) if compute_rmsd else None
    weighted_rmsd = (
        np.empty(n_frames, dtype=np.float64) if compute_weighted_rmsd else None
    )
    residual_sum = (
        np.zeros(n_residues, dtype=np.float64) if compute_mean_residual else None
    )
    robust_accumulator = (
        _StreamingRMSFAccumulator(n_residues) if compute_robust_rmsf else None
    )
    ordinary_accumulator = (
        _StreamingRMSFAccumulator(n_residues) if compute_ordinary_rmsf else None
    )
    timing = {
        "all_frame_weighted_fit": 0.0,
        "all_frame_ordinary_fit": 0.0,
        "transform_application": 0.0,
        "aligned_output_materialization": 0.0,
        "RMSD": 0.0,
        "RMSF": 0.0,
    }
    needs_robust_coordinates = any(
        (
            return_aligned,
            compute_rmsd,
            compute_weighted_rmsd,
            compute_mean_residual,
            compute_robust_rmsf,
        )
    )
    chunks_processed = 0
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        chunk = values[start:stop]
        fit_start = time.perf_counter()
        transform = weighted_kabsch_transform_batch(
            chunk,
            target,
            fixed_weights,
            validate_inputs=False,
        )
        timing["all_frame_weighted_fit"] += time.perf_counter() - fit_start
        rotations[start:stop] = transform.rotations
        translations[start:stop] = transform.translations

        robust_aligned: np.ndarray | None = None
        if needs_robust_coordinates:
            apply_start = time.perf_counter()
            robust_aligned = apply_transforms_batch(
                chunk,
                transform.rotations,
                transform.translations,
            )
            timing["transform_application"] += time.perf_counter() - apply_start
            if aligned_output is not None:
                materialize_start = time.perf_counter()
                aligned_output[start:stop] = robust_aligned
                timing["aligned_output_materialization"] += (
                    time.perf_counter() - materialize_start
                )

        if any((compute_rmsd, compute_weighted_rmsd, compute_mean_residual)):
            assert robust_aligned is not None
            metric_start = time.perf_counter()
            delta = robust_aligned - target[None, :, :]
            squared = np.sum(delta * delta, axis=2)
            if rmsd is not None:
                rmsd[start:stop] = np.sqrt(np.mean(squared, axis=1))
            if weighted_rmsd is not None:
                weighted_rmsd[start:stop] = np.sqrt(
                    (squared @ fixed_weights) / weight_sum
                )
            if residual_sum is not None:
                np.sqrt(squared, out=squared)
                residual_sum += np.sum(squared, axis=0)
            timing["RMSD"] += time.perf_counter() - metric_start

        if robust_accumulator is not None:
            assert robust_aligned is not None
            rmsf_start = time.perf_counter()
            robust_accumulator.update(robust_aligned)
            timing["RMSF"] += time.perf_counter() - rmsf_start

        if ordinary_accumulator is not None:
            ordinary_start = time.perf_counter()
            ordinary_transform = weighted_kabsch_transform_batch(
                chunk,
                target,
                workspace.uniform_weights,
                validate_inputs=False,
            )
            ordinary_aligned = apply_transforms_batch(
                chunk,
                ordinary_transform.rotations,
                ordinary_transform.translations,
            )
            timing["all_frame_ordinary_fit"] += time.perf_counter() - ordinary_start
            rmsf_start = time.perf_counter()
            ordinary_accumulator.update(ordinary_aligned)
            timing["RMSF"] += time.perf_counter() - rmsf_start
        chunks_processed += 1

    timing["total"] = time.perf_counter() - total_start
    return AllFrameFitResult(
        rotations=rotations,
        translations=translations,
        aligned_coords=aligned_output,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        robust_rmsf=(robust_accumulator.rmsf() if robust_accumulator else None),
        ordinary_rmsf=(ordinary_accumulator.rmsf() if ordinary_accumulator else None),
        mean_residual=(residual_sum / n_frames if residual_sum is not None else None),
        timing=timing,
        chunks_processed=chunks_processed,
    )
