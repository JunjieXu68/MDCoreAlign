"""Fast weighted Kabsch primitives.

These helpers keep the same row-vector convention used by
``mdsuperpose.alignment`` while reducing validation and allocation overhead in
the inner robust-alignment loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FastKabschResult:
    """Result from one fast weighted Kabsch solve."""

    rotation: np.ndarray
    translation: np.ndarray
    aligned_coords: np.ndarray
    rmsd: float
    weighted_rmsd: float


@dataclass(slots=True)
class FastKabschBatchResult:
    """Result from batched weighted Kabsch solves against one reference."""

    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray


@dataclass(slots=True)
class FastKabschVariableWeightBatchResult:
    """Batched Kabsch result with one atom-weight vector per frame."""

    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray


@dataclass(slots=True)
class FastKabschTransformBatchResult:
    """Rigid transforms from a batched fit without coordinate materialisation."""

    rotations: np.ndarray
    translations: np.ndarray


@dataclass(slots=True)
class FastKabschEvaluationBatchResult:
    """Intent-specific batched fit output used by internal CPU paths."""

    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray | None
    rmsd: np.ndarray | None
    weighted_rmsd: np.ndarray | None
    residuals: np.ndarray | None


def _validate_single_inputs(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate paired coordinates and non-negative weights."""

    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError("mobile and reference must have the same shape")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must have shape (n_atoms, 3)")
    if weights.shape != (mobile.shape[0],):
        raise ValueError("weights must have shape (n_atoms,)")
    if mobile.shape[0] < 3:
        raise ValueError("at least three paired points are required")
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all() or not np.isfinite(weights).all():
        raise ValueError("inputs must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must contain positive total mass")
    return mobile, reference, weights


def _validate_batch_inputs(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate batched coordinates and non-negative weights."""

    coords = np.asarray(coords, dtype=float)
    reference = np.asarray(reference, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    if reference.shape != coords.shape[1:]:
        raise ValueError("reference must have shape (n_atoms, 3)")
    if weights.shape != (coords.shape[1],):
        raise ValueError("weights must have shape (n_atoms,)")
    if coords.shape[1] < 3:
        raise ValueError("at least three paired points are required")
    if not np.isfinite(coords).all() or not np.isfinite(reference).all() or not np.isfinite(weights).all():
        raise ValueError("inputs must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must contain positive total mass")
    return coords, reference, weights


def _row_operator_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return the optimal row-vector rotation operator for one covariance."""

    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    if np.linalg.det(u @ vt) < 0.0:
        u = u.copy()
        u[:, 2] *= -1.0
    return u @ vt


def _row_operators_from_covariances(covariances: np.ndarray) -> np.ndarray:
    """Return optimal row-vector rotation operators for stacked covariances."""

    u, _, vt = np.linalg.svd(covariances, full_matrices=False)
    determinants = np.linalg.det(u @ vt)
    u_corrected = u.copy()
    u_corrected[:, :, 2] *= np.where(determinants < 0.0, -1.0, 1.0)[:, None]
    return u_corrected @ vt


def weighted_kabsch_single(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    compute_metrics: bool = True,
) -> FastKabschResult:
    """Solve one weighted rigid alignment using the Kabsch algorithm."""

    if validate_inputs:
        mobile, reference, weights = _validate_single_inputs(mobile, reference, weights)
    else:
        mobile = np.asarray(mobile, dtype=float)
        reference = np.asarray(reference, dtype=float)
        weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must contain positive total mass")

    mobile_center = (weights @ mobile) / weight_sum
    reference_center = (weights @ reference) / weight_sum
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = (mobile_centered * weights[:, None]).T @ reference_centered / weight_sum
    row_operator = _row_operator_from_covariance(covariance)
    rotation = row_operator.T
    translation = reference_center - mobile_center @ row_operator
    aligned = mobile @ row_operator + translation
    if compute_metrics:
        deltas = aligned - reference
        squared = np.einsum("ij,ij->i", deltas, deltas)
        rmsd = float(np.sqrt(np.mean(squared)))
        weighted_rmsd = float(np.sqrt(np.sum(weights * squared) / weight_sum))
    else:
        rmsd = float("nan")
        weighted_rmsd = float("nan")
    return FastKabschResult(
        rotation=rotation,
        translation=translation,
        aligned_coords=aligned,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
    )


def apply_transforms_batch(coords: np.ndarray, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Apply one row-vector rigid transform per frame."""

    coords = np.asarray(coords, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    if rotations.shape != (coords.shape[0], 3, 3):
        raise ValueError("rotations must have shape (n_frames, 3, 3)")
    if translations.shape != (coords.shape[0], 3):
        raise ValueError("translations must have shape (n_frames, 3)")
    return coords @ np.swapaxes(rotations, 1, 2) + translations[:, None, :]


def weighted_kabsch_transform_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    dtype: np.dtype | type = np.float64,
) -> FastKabschTransformBatchResult:
    """Compute fixed-weight transforms without allocating aligned coordinates."""

    if validate_inputs:
        coords, reference, weights = _validate_batch_inputs(
            coords,
            reference,
            weights,
        )
    values = np.asarray(coords, dtype=dtype)
    target = np.asarray(reference, dtype=dtype)
    fixed_weights = np.asarray(weights, dtype=dtype)
    weight_sum = float(np.sum(fixed_weights, dtype=np.float64))
    if weight_sum <= 0.0:
        raise ValueError("weights must contain positive total mass")

    normalized_weights = fixed_weights / weight_sum
    reference_center = normalized_weights @ target
    mobile_centers = (
        values.swapaxes(1, 2) @ normalized_weights
    )
    mobile_centered = values - mobile_centers[:, None, :]
    reference_centered = target - reference_center
    mobile_centered *= normalized_weights[None, :, None]
    covariances = mobile_centered.swapaxes(1, 2) @ reference_centered
    row_operators = _row_operators_from_covariances(covariances)
    rotations = np.swapaxes(row_operators, 1, 2)
    translated_centers = (
        mobile_centers[:, None, :] @ row_operators
    )[:, 0, :]
    translations = reference_center[None, :] - translated_centers
    return FastKabschTransformBatchResult(
        rotations=rotations,
        translations=translations,
    )


def weighted_kabsch_variable_weights_transform_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    dtype: np.dtype | type = np.float64,
) -> FastKabschTransformBatchResult:
    """Compute frame-weighted transforms without aligned-coordinate output."""

    values = np.asarray(coords, dtype=dtype)
    target = np.asarray(reference, dtype=dtype)
    frame_weights = np.asarray(weights, dtype=dtype)
    if validate_inputs:
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
        if target.shape != values.shape[1:]:
            raise ValueError("reference must have shape (n_atoms, 3)")
        if frame_weights.shape != values.shape[:2]:
            raise ValueError("weights must have shape (n_frames, n_atoms)")
        if values.shape[1] < 3:
            raise ValueError("at least three paired points are required")
        if (
            not np.isfinite(values).all()
            or not np.isfinite(target).all()
            or not np.isfinite(frame_weights).all()
        ):
            raise ValueError("inputs must contain only finite values")
        if np.any(frame_weights < 0.0):
            raise ValueError("weights must be non-negative")
    weight_sums = np.sum(frame_weights, axis=1)
    if np.any(weight_sums <= 0.0):
        raise ValueError("every frame must contain positive total weight")
    normalized_weights = frame_weights / weight_sums[:, None]
    mobile_centers = (
        values.swapaxes(1, 2) @ normalized_weights[:, :, None]
    )[:, :, 0]
    reference_centers = normalized_weights @ target
    mobile_centered = values - mobile_centers[:, None, :]
    reference_centered = target[None, :, :] - reference_centers[:, None, :]
    mobile_centered *= normalized_weights[:, :, None]
    covariances = mobile_centered.swapaxes(1, 2) @ reference_centered
    row_operators = _row_operators_from_covariances(covariances)
    rotations = np.swapaxes(row_operators, 1, 2)
    translations = reference_centers - (
        mobile_centers[:, None, :] @ row_operators
    )[:, 0, :]
    return FastKabschTransformBatchResult(
        rotations=rotations,
        translations=translations,
    )


def evaluate_weighted_kabsch_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    return_aligned: bool = False,
    compute_rmsd: bool = True,
    compute_weighted_rmsd: bool = True,
    compute_residuals: bool = False,
    dtype: np.dtype | type = np.float64,
) -> FastKabschEvaluationBatchResult:
    """Fit a fixed weight vector with orthogonal output controls.

    A chunk-local aligned array is created only when a coordinate or residual
    output is requested. Scalar metrics may use the same temporary array, but
    it is not retained unless ``return_aligned`` is true.
    """

    values = np.asarray(coords, dtype=dtype)
    target = np.asarray(reference, dtype=dtype)
    fixed_weights = np.asarray(weights, dtype=dtype)
    transforms = weighted_kabsch_transform_batch(
        values,
        target,
        fixed_weights,
        validate_inputs=validate_inputs,
        dtype=dtype,
    )
    needs_evaluation = (
        return_aligned
        or compute_rmsd
        or compute_weighted_rmsd
        or compute_residuals
    )
    aligned = None
    residuals = None
    rmsd = None
    weighted_rmsd = None
    if needs_evaluation:
        evaluated = apply_transforms_batch(
            values,
            transforms.rotations,
            transforms.translations,
        )
        deltas = evaluated - target[None, :, :]
        squared = np.sum(deltas * deltas, axis=2)
        if compute_rmsd:
            rmsd = np.sqrt(np.mean(squared, axis=1))
        if compute_weighted_rmsd:
            weight_sum = float(np.sum(fixed_weights, dtype=np.float64))
            weighted_rmsd = np.sqrt(
                (squared @ fixed_weights) / weight_sum
            )
        if compute_residuals:
            residuals = np.sqrt(squared)
        if return_aligned:
            aligned = evaluated
    return FastKabschEvaluationBatchResult(
        rotations=transforms.rotations,
        translations=transforms.translations,
        aligned_coords=aligned,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        residuals=residuals,
    )


def evaluate_weighted_kabsch_variable_weights_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    return_aligned: bool = False,
    compute_rmsd: bool = True,
    compute_weighted_rmsd: bool = True,
    compute_residuals: bool = False,
    dtype: np.dtype | type = np.float64,
) -> FastKabschEvaluationBatchResult:
    """Fit frame-specific weights with orthogonal output controls."""

    values = np.asarray(coords, dtype=dtype)
    target = np.asarray(reference, dtype=dtype)
    frame_weights = np.asarray(weights, dtype=dtype)
    transforms = weighted_kabsch_variable_weights_transform_batch(
        values,
        target,
        frame_weights,
        validate_inputs=validate_inputs,
        dtype=dtype,
    )
    needs_evaluation = (
        return_aligned
        or compute_rmsd
        or compute_weighted_rmsd
        or compute_residuals
    )
    aligned = None
    residuals = None
    rmsd = None
    weighted_rmsd = None
    if needs_evaluation:
        evaluated = apply_transforms_batch(
            values,
            transforms.rotations,
            transforms.translations,
        )
        deltas = evaluated - target[None, :, :]
        squared = np.sum(deltas * deltas, axis=2)
        if compute_rmsd:
            rmsd = np.sqrt(np.mean(squared, axis=1))
        if compute_weighted_rmsd:
            normalized = frame_weights / np.sum(frame_weights, axis=1)[:, None]
            weighted_rmsd = np.sqrt(
                (squared[:, None, :] @ normalized[:, :, None])[:, 0, 0]
            )
        if compute_residuals:
            residuals = np.sqrt(squared)
        if return_aligned:
            aligned = evaluated
    return FastKabschEvaluationBatchResult(
        rotations=transforms.rotations,
        translations=transforms.translations,
        aligned_coords=aligned,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        residuals=residuals,
    )


def weighted_kabsch_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    compute_metrics: bool = True,
) -> FastKabschBatchResult:
    """Solve batched weighted Kabsch alignments against one reference."""

    fit = evaluate_weighted_kabsch_batch(
        coords,
        reference,
        weights,
        validate_inputs=validate_inputs,
        return_aligned=True,
        compute_rmsd=compute_metrics,
        compute_weighted_rmsd=compute_metrics,
    )
    assert fit.aligned_coords is not None
    n_frames = np.asarray(coords).shape[0]
    rmsd = (
        fit.rmsd
        if fit.rmsd is not None
        else np.full(n_frames, np.nan, dtype=float)
    )
    weighted_rmsd = (
        fit.weighted_rmsd
        if fit.weighted_rmsd is not None
        else np.full(n_frames, np.nan, dtype=float)
    )
    return FastKabschBatchResult(
        rotations=fit.rotations,
        translations=fit.translations,
        aligned_coords=fit.aligned_coords,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
    )


def weighted_kabsch_variable_weights_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    validate_inputs: bool = True,
    compute_metrics: bool = True,
) -> FastKabschVariableWeightBatchResult:
    """Solve stacked weighted Kabsch fits with frame-specific atom weights.

    Parameters use shapes ``coords=(B, N, 3)``, ``reference=(N, 3)`` and
    ``weights=(B, N)``. The implementation performs one stacked 3x3 SVD call
    and contains no Python loop over frames.
    """

    fit = evaluate_weighted_kabsch_variable_weights_batch(
        coords,
        reference,
        weights,
        validate_inputs=validate_inputs,
        return_aligned=True,
        compute_rmsd=compute_metrics,
        compute_weighted_rmsd=compute_metrics,
    )
    assert fit.aligned_coords is not None
    n_frames = np.asarray(coords).shape[0]
    rmsd = (
        fit.rmsd
        if fit.rmsd is not None
        else np.full(n_frames, np.nan, dtype=float)
    )
    weighted_rmsd = (
        fit.weighted_rmsd
        if fit.weighted_rmsd is not None
        else np.full(n_frames, np.nan, dtype=float)
    )
    return FastKabschVariableWeightBatchResult(
        rotations=fit.rotations,
        translations=fit.translations,
        aligned_coords=fit.aligned_coords,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
    )
