"""Rigid alignment primitives for the scaffold package.

This module implements paired-point rigid superposition with the Kabsch
algorithm. Both unweighted and per-atom weighted variants share the same
mathematical structure:

1. compute centroids,
2. center the coordinates,
3. form a covariance matrix,
4. compute its singular value decomposition,
5. build the optimal proper rotation,
6. recover the translation and RMSD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fast_kabsch import weighted_kabsch_single
from .utils import as_float_array, ensure_atom_vector


@dataclass(slots=True)
class RigidTransform:
    """A minimal rigid transform container."""

    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class AlignmentResult:
    """Result of a paired rigid superposition solve."""

    rotation: np.ndarray
    translation: np.ndarray
    aligned_coords: np.ndarray
    rmsd: float
    weighted_rmsd: float


def _validate_paired_coords(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a pair of matched coordinate arrays."""

    mobile = as_float_array(mobile)
    reference = as_float_array(reference)
    if mobile.shape != reference.shape:
        raise ValueError("mobile and reference must have the same shape")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must have shape (n_atoms, 3)")
    if mobile.shape[0] < 3:
        raise ValueError("at least three paired points are required")
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all():
        raise ValueError("mobile and reference must contain only finite values")
    return mobile, reference


def _validate_weights(weights: np.ndarray, n_atoms: int) -> np.ndarray:
    """Validate a non-negative per-atom weight vector."""

    weights = ensure_atom_vector(weights, n_atoms, name="weights")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("at least one weight must be positive")
    return weights


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize a non-negative weight vector to sum to one."""

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("sum of weights must be positive")
    return weights / total


def _weighted_rmsd_from_squared(squared: np.ndarray, normalized_weights: np.ndarray) -> float:
    """Compute weighted RMSD from squared distances and normalized weights."""

    return float(np.sqrt(np.sum(normalized_weights * squared)))


def _rmsd(aligned_coords: np.ndarray, reference: np.ndarray) -> float:
    """Compute the unweighted RMSD between paired coordinate sets."""

    squared = np.sum((aligned_coords - reference) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def _weighted_rmsd(aligned_coords: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted RMSD between paired coordinate sets."""

    normalized = _normalized_weights(weights)
    squared = np.sum((aligned_coords - reference) ** 2, axis=1)
    return _weighted_rmsd_from_squared(squared, normalized)


def _optimal_row_operator(covariance: np.ndarray) -> np.ndarray:
    """Solve the optimal row-vector rotation operator from a covariance matrix."""

    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    determinant = np.linalg.det(u @ vt)
    correction = np.eye(3)
    correction[2, 2] = -1.0 if determinant < 0.0 else 1.0
    return u @ correction @ vt


def _build_alignment_result(
    mobile: np.ndarray,
    reference: np.ndarray,
    mobile_center: np.ndarray,
    reference_center: np.ndarray,
    row_operator: np.ndarray,
    weights: np.ndarray,
    normalized_weights: np.ndarray,
    *,
    compute_metrics: bool = True,
) -> AlignmentResult:
    """Assemble a public alignment result from an internal Kabsch solve."""

    rotation = row_operator.T
    translation = reference_center - mobile_center @ row_operator
    aligned_coords = mobile @ rotation.T + translation
    if compute_metrics:
        squared = np.sum((aligned_coords - reference) ** 2, axis=1)
        rmsd = float(np.sqrt(np.mean(squared)))
        weighted_rmsd = _weighted_rmsd_from_squared(squared, normalized_weights)
    else:
        rmsd = float("nan")
        weighted_rmsd = float("nan")
    return AlignmentResult(
        rotation=rotation,
        translation=translation,
        aligned_coords=aligned_coords,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
    )


def identity_transform() -> RigidTransform:
    """Return the identity rigid transform."""

    return RigidTransform(rotation=np.eye(3, dtype=float), translation=np.zeros(3, dtype=float))


def compose_transforms(next_transform: RigidTransform, previous_transform: RigidTransform) -> RigidTransform:
    """Compose two rigid transforms for row-vector coordinate application.

    The returned transform is equivalent to applying `previous_transform`
    first and `next_transform` second.
    """

    next_rotation = as_float_array(next_transform.rotation)
    previous_rotation = as_float_array(previous_transform.rotation)
    next_translation = as_float_array(next_transform.translation)
    previous_translation = as_float_array(previous_transform.translation)
    rotation = next_rotation @ previous_rotation
    translation = previous_translation @ next_rotation.T + next_translation
    return RigidTransform(rotation=rotation, translation=translation)


def apply_transform(coords: np.ndarray, transform: RigidTransform) -> np.ndarray:
    """Apply a rigid transform to one frame or many frames."""

    coords = as_float_array(coords)
    rotation = as_float_array(transform.rotation)
    translation = as_float_array(transform.translation)
    return coords @ rotation.T + translation


def solve_rigid_alignment(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> AlignmentResult:
    """Solve the baseline paired rigid alignment problem with Kabsch.

    The method:

    1. centers both point sets,
    2. forms the covariance matrix,
    3. computes its singular value decomposition,
    4. constructs the optimal proper rotation,
    5. recovers the translation and aligned coordinates.
    """

    mobile, reference = _validate_paired_coords(mobile, reference)
    weights = np.ones(mobile.shape[0], dtype=float)

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center

    covariance = mobile_centered.T @ reference_centered
    row_operator = _optimal_row_operator(covariance)
    normalized_weights = _normalized_weights(weights)
    return _build_alignment_result(
        mobile=mobile,
        reference=reference,
        mobile_center=mobile_center,
        reference_center=reference_center,
        row_operator=row_operator,
        weights=weights,
        normalized_weights=normalized_weights,
    )


def solve_weighted_rigid_alignment(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    validate_inputs: bool = True,
    compute_metrics: bool = True,
) -> AlignmentResult:
    """Solve the weighted paired rigid alignment problem with weighted Kabsch.

    Let `w_i >= 0` be per-atom weights. The weighted objective is:

    `min_{R, t} sum_i w_i ||R x_i + t - y_i||^2`

    with `R` constrained to be a proper rotation. The solver:

    1. computes weighted centroids,
    2. centers both point sets with those centroids,
    3. forms the weighted covariance matrix,
    4. solves the optimal proper rotation via SVD,
    5. recovers the translation,
    6. reports both ordinary RMSD and weighted RMSD.

    The weights do not need to be normalized; only their relative magnitudes
    matter for the transform.
    """

    if not validate_inputs:
        if weights is None:
            weights = np.ones(np.asarray(mobile).shape[0], dtype=float)
        fast = weighted_kabsch_single(
            mobile,
            reference,
            weights,
            validate_inputs=False,
            compute_metrics=compute_metrics,
        )
        return AlignmentResult(
            rotation=fast.rotation,
            translation=fast.translation,
            aligned_coords=fast.aligned_coords,
            rmsd=fast.rmsd,
            weighted_rmsd=fast.weighted_rmsd,
        )

    if validate_inputs:
        mobile, reference = _validate_paired_coords(mobile, reference)

    if weights is None:
        weights = np.ones(mobile.shape[0], dtype=float)
    else:
        weights = _validate_weights(weights, mobile.shape[0])

    fast = weighted_kabsch_single(
        mobile,
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=compute_metrics,
    )
    return AlignmentResult(
        rotation=fast.rotation,
        translation=fast.translation,
        aligned_coords=fast.aligned_coords,
        rmsd=fast.rmsd,
        weighted_rmsd=fast.weighted_rmsd,
    )
