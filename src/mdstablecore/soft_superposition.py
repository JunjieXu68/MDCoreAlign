"""Soft-weighted rigid superposition primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SoftPairResult:
    """Result of soft superposition for one mobile structure."""

    aligned_coords: np.ndarray
    weights: np.ndarray
    squared_displacements: np.ndarray
    n_iterations: int
    weighted_rmsd: float
    all_atom_rmsd: float


@dataclass(slots=True)
class SoftTrajectoryResult:
    """Trajectory-level output of the soft superposition prototype."""

    aligned_coords: np.ndarray
    frame_weights: np.ndarray
    mean_weights: np.ndarray
    hard_core_mask: np.ndarray
    reference: np.ndarray
    reference_index: int | None
    iterations_per_frame: np.ndarray


def _normalized_weights(weights: np.ndarray, n_atoms: int, min_weight: float = 1e-6) -> np.ndarray:
    """Clip and normalize non-negative atom weights."""

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_atoms,):
        raise ValueError(f"weights must have shape ({n_atoms},)")
    clipped = np.clip(weights, min_weight, None)
    return clipped / clipped.sum()


def weighted_kabsch_alignment(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align one structure onto a reference with per-atom weights."""

    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must both have shape (n_atoms, 3)")

    normalized = _normalized_weights(weights, n_atoms=mobile.shape[0])
    mobile_center = np.sum(mobile * normalized[:, None], axis=0)
    reference_center = np.sum(reference * normalized[:, None], axis=0)

    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = (normalized[:, None] * mobile_centered).T @ reference_centered
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    determinant = np.linalg.det(u @ vt)
    correction = np.eye(3)
    correction[2, 2] = -1.0 if determinant < 0.0 else 1.0
    rotation = u @ correction @ vt
    aligned = mobile_centered @ rotation + reference_center
    return aligned, rotation, mobile_center, reference_center


def soft_weights_from_squared_displacements(
    squared_displacements: np.ndarray,
    temperature: float = 2.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Map per-atom squared displacements to soft core memberships."""

    squared_displacements = np.asarray(squared_displacements, dtype=float)
    if squared_displacements.ndim != 1:
        raise ValueError("squared_displacements must be one-dimensional")
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    positive = squared_displacements[squared_displacements > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale * temperature, 1e-8)
    robust = 1.0 / (1.0 + squared_displacements / scale)
    return floor + (1.0 - floor) * robust


def hard_core_from_weights(
    weights: np.ndarray,
    core_fraction: float = 0.5,
    weight_threshold: float | None = None,
) -> np.ndarray:
    """Extract a hard core set from soft atom weights."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    if weight_threshold is not None:
        return weights >= weight_threshold

    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")

    n_keep = max(1, int(round(core_fraction * weights.shape[0])))
    selected = np.argsort(weights)[-n_keep:]
    mask = np.zeros(weights.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def soft_superpose_pair(
    mobile: np.ndarray,
    reference: np.ndarray,
    init_weights: np.ndarray | None = None,
    max_iterations: int = 8,
    tolerance: float = 1e-4,
    temperature: float = 2.0,
    floor: float = 0.05,
) -> SoftPairResult:
    """Jointly optimize alignment and soft core weights for one frame."""

    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must both have shape (n_atoms, 3)")

    if init_weights is None:
        weights = np.ones(mobile.shape[0], dtype=float)
    else:
        weights = np.asarray(init_weights, dtype=float).copy()

    aligned = mobile.copy()
    squared_displacements = np.sum((mobile - reference) ** 2, axis=1)
    for iteration in range(1, max_iterations + 1):
        aligned, _, _, _ = weighted_kabsch_alignment(mobile, reference, weights)
        squared_displacements = np.sum((aligned - reference) ** 2, axis=1)
        updated = soft_weights_from_squared_displacements(
            squared_displacements,
            temperature=temperature,
            floor=floor,
        )
        if np.max(np.abs(updated - weights)) < tolerance:
            weights = updated
            break
        weights = updated

    weighted_rmsd = float(np.sqrt(np.sum(_normalized_weights(weights, mobile.shape[0]) * squared_displacements)))
    all_atom_rmsd = float(np.sqrt(np.mean(squared_displacements)))
    return SoftPairResult(
        aligned_coords=aligned,
        weights=weights,
        squared_displacements=squared_displacements,
        n_iterations=iteration,
        weighted_rmsd=weighted_rmsd,
        all_atom_rmsd=all_atom_rmsd,
    )


def soft_superpose_trajectory(
    coords: np.ndarray,
    reference: np.ndarray | None = None,
    reference_index: int | None = None,
    init_weights: np.ndarray | None = None,
    warm_start: bool = True,
    core_fraction: float = 0.5,
    weight_threshold: float | None = None,
    max_iterations: int = 8,
    tolerance: float = 1e-4,
    temperature: float = 2.0,
    floor: float = 0.05,
) -> SoftTrajectoryResult:
    """Run the soft superposition prototype on every frame of a trajectory."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")

    if reference is None:
        resolved_index = 0 if reference_index is None else int(reference_index)
        reference = coords[resolved_index].copy()
    else:
        resolved_index = reference_index
        reference = np.asarray(reference, dtype=float)
        if reference.shape != coords.shape[1:]:
            raise ValueError("reference must have shape (n_atoms, 3)")

    aligned_coords = np.zeros_like(coords)
    frame_weights = np.zeros(coords.shape[:2], dtype=float)
    iterations_per_frame = np.zeros(coords.shape[0], dtype=int)

    previous_weights = np.ones(coords.shape[1], dtype=float) if init_weights is None else np.asarray(init_weights, dtype=float)
    for frame_index, frame in enumerate(coords):
        current_init = previous_weights if warm_start else init_weights
        result = soft_superpose_pair(
            frame,
            reference,
            init_weights=current_init,
            max_iterations=max_iterations,
            tolerance=tolerance,
            temperature=temperature,
            floor=floor,
        )
        aligned_coords[frame_index] = result.aligned_coords
        frame_weights[frame_index] = result.weights
        iterations_per_frame[frame_index] = result.n_iterations
        previous_weights = result.weights

    mean_weights = frame_weights.mean(axis=0)
    hard_core_mask = hard_core_from_weights(
        mean_weights,
        core_fraction=core_fraction,
        weight_threshold=weight_threshold,
    )
    return SoftTrajectoryResult(
        aligned_coords=aligned_coords,
        frame_weights=frame_weights,
        mean_weights=mean_weights,
        hard_core_mask=hard_core_mask,
        reference=reference,
        reference_index=resolved_index,
        iterations_per_frame=iterations_per_frame,
    )
