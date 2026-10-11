"""Lean sequential warm-start representative-weight estimation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    from scipy.linalg import svd as _scipy_svd
except ImportError:  # SciPy is optional; NumPy remains the supported fallback.
    _scipy_svd = None

from .contacts import (
    apply_contact_consistency_regularization,
    build_contact_graph,
)
from .weights import normalize_weights, update_irls_weights


@dataclass(slots=True)
class SequentialWeightEstimationResult:
    """Compact arrays produced by sequential representative-frame IRLS."""

    weights: np.ndarray
    mean_weights: np.ndarray
    iteration_counts: np.ndarray
    converged: np.ndarray
    final_weight_change: np.ndarray
    final_transform_change: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    final_rmsd: np.ndarray
    final_weighted_rmsd: np.ndarray
    mean_residual: np.ndarray
    total_runtime_seconds: float
    timing: dict[str, float]
    call_counts: dict[str, int]
    warnings: tuple[str, ...]
    initialization: str = "sequential_warm_start"

    @property
    def average_iterations_per_frame(self) -> float:
        """Return mean IRLS iterations across representative frames."""

        return (
            float(np.mean(self.iteration_counts))
            if self.iteration_counts.size
            else 0.0
        )

    @property
    def converged_fraction(self) -> float:
        """Return the fraction of representatives satisfying a stop rule."""

        return float(np.mean(self.converged)) if self.converged.size else 0.0


@dataclass(slots=True)
class SequentialMeanWeightEstimationResult:
    """Mean-only exact estimator output with aggregate convergence metadata."""

    mean_weights: np.ndarray
    mean_residual: np.ndarray
    average_iterations_per_frame: float
    converged_fraction: float
    max_iterations_reached_count: int
    mean_final_weight_change: float
    max_final_weight_change: float
    mean_final_transform_change: float
    max_final_transform_change: float
    total_runtime_seconds: float
    timing: dict[str, float]
    call_counts: dict[str, int]
    warnings: tuple[str, ...]
    initialization: str = "sequential_warm_start"


def _chain_spans(
    chain_ids: np.ndarray | None,
    n_atoms: int,
) -> tuple[tuple[int, int], ...]:
    if chain_ids is None:
        return ((0, n_atoms),)
    values = np.asarray(chain_ids)
    if values.shape != (n_atoms,):
        raise ValueError("sequence_chain_ids must have shape (n_atoms,)")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [n_atoms]))
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    )


def _smooth_sequence(
    weights: np.ndarray,
    *,
    strength: float,
    passes: int,
    spans: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Apply the scalar Jacobi rule without crossing explicit chain spans."""

    if strength == 0.0 or passes == 0 or weights.shape[0] <= 1:
        return weights.copy()
    smoothed = weights
    updated = np.empty_like(weights)
    for _ in range(passes):
        updated[...] = smoothed
        for start, stop in spans:
            length = stop - start
            if length <= 1:
                continue
            updated[start] = (
                (1.0 - strength) * smoothed[start]
                + strength * smoothed[start + 1]
            )
            updated[stop - 1] = (
                (1.0 - strength) * smoothed[stop - 1]
                + strength * smoothed[stop - 2]
            )
            if length > 2:
                updated[start + 1 : stop - 1] = (
                    (1.0 - strength) * smoothed[start + 1 : stop - 1]
                    + 0.5
                    * strength
                    * (
                        smoothed[start : stop - 2]
                        + smoothed[start + 2 : stop]
                    )
                )
        if passes > 1:
            smoothed, updated = updated, smoothed.copy()
        else:
            smoothed = updated
    return smoothed


def _weighted_fit(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    compute_metrics: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return rotation, translation and aligned coordinates without a result object."""

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise RuntimeError("weights must contain positive total mass")
    mobile_center = (weights @ mobile) / weight_sum
    reference_center = (weights @ reference) / weight_sum
    covariance = (
        (mobile * weights[:, None]).T
        @ reference
        / weight_sum
        - np.outer(mobile_center, reference_center)
    )
    if _scipy_svd is None:
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    else:
        u, _, vt = _scipy_svd(
            covariance,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesvd",
        )
    row_operator = u @ vt
    determinant = (
        row_operator[0, 0]
        * (
            row_operator[1, 1] * row_operator[2, 2]
            - row_operator[1, 2] * row_operator[2, 1]
        )
        - row_operator[0, 1]
        * (
            row_operator[1, 0] * row_operator[2, 2]
            - row_operator[1, 2] * row_operator[2, 0]
        )
        + row_operator[0, 2]
        * (
            row_operator[1, 0] * row_operator[2, 1]
            - row_operator[1, 1] * row_operator[2, 0]
        )
    )
    if determinant < 0.0:
        u = u.copy()
        u[:, 2] *= -1.0
        row_operator = u @ vt
    rotation = row_operator.T
    translation = reference_center - mobile_center @ row_operator
    aligned = mobile @ row_operator + translation
    if not compute_metrics:
        return rotation, translation, aligned, float("nan"), float("nan")
    deltas = aligned - reference
    squared = np.einsum("ij,ij->i", deltas, deltas)
    rmsd = float(np.sqrt(np.mean(squared)))
    weighted_rmsd = float(np.sqrt(np.sum(weights * squared) / weight_sum))
    return rotation, translation, aligned, rmsd, weighted_rmsd


def estimate_representative_weights_sequential_lean(
    coords: np.ndarray,
    reference: np.ndarray,
    *,
    rule: str = "huber",
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    weight_normalize: str = "none",
    sequence_continuity_strength: float = 0.0,
    sequence_continuity_passes: int = 1,
    sequence_chain_ids: np.ndarray | None = None,
    contact_consistency_strength: float = 0.0,
    contact_consistency_passes: int = 1,
    contact_distance_cutoff: float = 8.0,
    contact_min_sequence_separation: int = 2,
    contact_max_neighbors: int | None = 8,
    objective_tolerance: float | None = None,
    objective_patience: int = 0,
    delta: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 1.0,
    record_history: bool = False,
    collect_timing: bool = False,
    _retain_details: bool = True,
    **unused_parameters: object,
) -> SequentialWeightEstimationResult | SequentialMeanWeightEstimationResult:
    """Estimate representative weights with exact sequential warm-start semantics.

    The chronological dependency is intentionally preserved. The function
    removes public per-frame result objects and history/aligned-coordinate
    retention, but it does not parallelize representatives.
    """

    total_start = time.perf_counter()
    coords = np.asarray(coords, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    if reference.shape != coords.shape[1:]:
        raise ValueError("reference must have shape (n_atoms, 3)")
    if coords.shape[0] == 0 or coords.shape[1] < 3:
        raise ValueError("at least one frame and three atoms are required")
    if not np.isfinite(coords).all() or not np.isfinite(reference).all():
        raise ValueError("coordinates must be finite")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if objective_tolerance is not None and objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be non-negative")
    if objective_patience < 0:
        raise ValueError("objective_patience must be non-negative")
    if rule == "huber" and delta <= 0.0:
        raise ValueError("delta must be positive")
    if rule == "gaussian" and sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if rule == "tukey" and cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if record_history:
        raise ValueError(
            "sequential_lean does not retain iteration history; use legacy_object"
        )

    n_frames, n_atoms, _ = coords.shape
    spans = _chain_spans(sequence_chain_ids, n_atoms)
    contact_graph = None
    if contact_consistency_strength > 0.0 and contact_consistency_passes > 0:
        contact_graph = build_contact_graph(
            reference,
            cutoff=contact_distance_cutoff,
            min_sequence_separation=contact_min_sequence_separation,
            max_neighbors=contact_max_neighbors,
        )

    weights_all = (
        np.empty((n_frames, n_atoms), dtype=float) if _retain_details else None
    )
    rotations = np.empty((n_frames, 3, 3), dtype=float) if _retain_details else None
    translations = np.empty((n_frames, 3), dtype=float) if _retain_details else None
    final_rmsd = np.empty(n_frames, dtype=float) if _retain_details else None
    final_weighted_rmsd = (
        np.empty(n_frames, dtype=float) if _retain_details else None
    )
    iteration_counts = (
        np.empty(n_frames, dtype=np.int32) if _retain_details else None
    )
    converged = np.empty(n_frames, dtype=bool) if _retain_details else None
    final_weight_change = (
        np.empty(n_frames, dtype=float) if _retain_details else None
    )
    final_transform_change = (
        np.empty(n_frames, dtype=float) if _retain_details else None
    )
    residual_sum = np.zeros(n_atoms, dtype=float)
    weight_sum = np.zeros(n_atoms, dtype=float)
    previous_weights: np.ndarray | None = None
    previous_rotation: np.ndarray | None = None
    previous_translation: np.ndarray | None = None
    timing = {
        "weighted_kabsch": 0.0,
        "residual_compute": 0.0,
        "weight_update": 0.0,
        "sequence_regularization": 0.0,
        "contact_regularization": 0.0,
        "convergence": 0.0,
        "final_refit": 0.0,
    }
    kabsch_calls = 0
    iteration_total = 0
    converged_count = 0
    weight_change_total = 0.0
    transform_change_total = 0.0
    max_weight_change = 0.0
    max_transform_change = 0.0

    for frame_index in range(n_frames):
        mobile = coords[frame_index]
        if previous_weights is None:
            weights = np.ones(n_atoms, dtype=float)
            total_rotation = np.eye(3, dtype=float)
            total_translation = np.zeros(3, dtype=float)
        else:
            weights = previous_weights
            assert previous_rotation is not None
            assert previous_translation is not None
            total_rotation = previous_rotation.copy()
            total_translation = previous_translation.copy()
        mobile_current = mobile @ total_rotation.T + total_translation
        previous_objective: float | None = None
        plateau_count = 0
        frame_converged = False
        weight_delta = float("nan")
        transform_delta = float("nan")

        for iteration in range(1, max_iterations + 1):
            fit_start = time.perf_counter() if collect_timing else 0.0
            (
                correction_rotation,
                correction_translation,
                aligned,
                _,
                _,
            ) = _weighted_fit(
                mobile_current,
                reference,
                weights,
                compute_metrics=False,
            )
            if collect_timing:
                timing["weighted_kabsch"] += time.perf_counter() - fit_start
            kabsch_calls += 1

            residual_start = time.perf_counter() if collect_timing else 0.0
            deltas = aligned - reference
            squared = np.einsum("ij,ij->i", deltas, deltas)
            distances = np.sqrt(squared)
            iteration_weighted_rmsd = float(
                np.sqrt(np.dot(weights, squared) / np.sum(weights))
            )
            if collect_timing:
                timing["residual_compute"] += time.perf_counter() - residual_start

            update_start = time.perf_counter() if collect_timing else 0.0
            if rule == "huber":
                updated_weights = np.ones_like(distances)
                outliers = distances > delta
                np.divide(
                    delta,
                    distances,
                    out=updated_weights,
                    where=outliers,
                )
            else:
                updated_weights = update_irls_weights(
                    distances,
                    rule=rule,
                    normalize="none",
                    validate=False,
                    delta=delta,
                    sigma=sigma,
                    cutoff=cutoff,
                )
            if collect_timing:
                timing["weight_update"] += time.perf_counter() - update_start
            if (
                sequence_continuity_strength > 0.0
                and sequence_continuity_passes > 0
            ):
                sequence_start = time.perf_counter() if collect_timing else 0.0
                if (
                    sequence_continuity_passes == 1
                    and spans == ((0, n_atoms),)
                ):
                    source_weights = updated_weights
                    updated_weights = np.empty_like(source_weights)
                    updated_weights[0] = (
                        (1.0 - sequence_continuity_strength)
                        * source_weights[0]
                        + sequence_continuity_strength * source_weights[1]
                    )
                    updated_weights[-1] = (
                        (1.0 - sequence_continuity_strength)
                        * source_weights[-1]
                        + sequence_continuity_strength * source_weights[-2]
                    )
                    updated_weights[1:-1] = (
                        (1.0 - sequence_continuity_strength)
                        * source_weights[1:-1]
                        + 0.5
                        * sequence_continuity_strength
                        * (source_weights[:-2] + source_weights[2:])
                    )
                else:
                    updated_weights = _smooth_sequence(
                        updated_weights,
                        strength=sequence_continuity_strength,
                        passes=sequence_continuity_passes,
                        spans=spans,
                    )
                if collect_timing:
                    timing["sequence_regularization"] += (
                        time.perf_counter() - sequence_start
                    )
            if contact_graph is not None:
                contact_start = time.perf_counter() if collect_timing else 0.0
                updated_weights = apply_contact_consistency_regularization(
                    updated_weights,
                    contact_graph,
                    strength=contact_consistency_strength,
                    passes=contact_consistency_passes,
                    validate=False,
                )
                if collect_timing:
                    timing["contact_regularization"] += (
                        time.perf_counter() - contact_start
                    )
            if weight_normalize != "none":
                updated_weights = normalize_weights(
                    updated_weights, mode=weight_normalize
                )
            if not np.any(updated_weights > 0.0):
                raise RuntimeError("weight update collapsed to all-zero weights")

            convergence_start = time.perf_counter() if collect_timing else 0.0
            weight_delta = float(
                np.max(np.abs(updated_weights - weights))
            )
            current_rotation = correction_rotation @ total_rotation
            current_translation = (
                total_translation @ correction_rotation.T
                + correction_translation
            )
            transform_delta = float(
                np.linalg.norm(current_rotation - total_rotation)
                + np.linalg.norm(current_translation - total_translation)
            )
            if objective_tolerance is not None and objective_patience > 0:
                if previous_objective is not None:
                    improvement = (
                        previous_objective - iteration_weighted_rmsd
                    )
                    if 0.0 <= improvement <= objective_tolerance:
                        plateau_count += 1
                    else:
                        plateau_count = 0
                previous_objective = iteration_weighted_rmsd
            stop_by_tolerance = (
                weight_delta <= tolerance and transform_delta <= tolerance
            )
            stop_by_objective = (
                objective_tolerance is not None
                and objective_patience > 0
                and plateau_count >= objective_patience
            )
            weights = updated_weights
            mobile_current = aligned
            total_rotation = current_rotation
            total_translation = current_translation
            if collect_timing:
                timing["convergence"] += time.perf_counter() - convergence_start
            if stop_by_tolerance or stop_by_objective:
                frame_converged = True
                break

        final_start = time.perf_counter() if collect_timing else 0.0
        (
            correction_rotation,
            correction_translation,
            final_aligned,
            rmsd_value,
            weighted_rmsd_value,
        ) = _weighted_fit(
            mobile_current,
            reference,
            weights,
            compute_metrics=_retain_details,
        )
        kabsch_calls += 1
        final_rotation = correction_rotation @ total_rotation
        final_translation = (
            total_translation @ correction_rotation.T
            + correction_translation
        )
        final_distances = np.sqrt(
            np.einsum(
                "ij,ij->i",
                final_aligned - reference,
                final_aligned - reference,
            )
        )
        if collect_timing:
            timing["final_refit"] += time.perf_counter() - final_start

        if weights_all is not None:
            weights_all[frame_index] = weights
        weight_sum += weights
        if rotations is not None:
            assert translations is not None
            assert final_rmsd is not None
            assert final_weighted_rmsd is not None
            assert iteration_counts is not None
            assert converged is not None
            assert final_weight_change is not None
            assert final_transform_change is not None
            rotations[frame_index] = final_rotation
            translations[frame_index] = final_translation
            final_rmsd[frame_index] = rmsd_value
            final_weighted_rmsd[frame_index] = weighted_rmsd_value
            iteration_counts[frame_index] = iteration
            converged[frame_index] = frame_converged
            final_weight_change[frame_index] = weight_delta
            final_transform_change[frame_index] = transform_delta
        iteration_total += int(iteration)
        converged_count += int(frame_converged)
        weight_change_total += float(weight_delta)
        transform_change_total += float(transform_delta)
        max_weight_change = max(max_weight_change, float(weight_delta))
        max_transform_change = max(max_transform_change, float(transform_delta))
        residual_sum += final_distances
        previous_weights = weights
        previous_rotation = final_rotation
        previous_translation = final_translation

    warning_values: list[str] = []
    if converged_count != n_frames:
        warning_values.append(
            f"{n_frames - converged_count} representative frames reached max_iterations"
        )
    if unused_parameters:
        warning_values.append(
            "unused robust parameters: " + ", ".join(sorted(unused_parameters))
        )
    common_call_counts = {
        "kabsch_calls": kabsch_calls,
        "svd_calls": kabsch_calls,
        "per_representative_result_objects": 0,
        "retained_aligned_coordinate_arrays": 0,
        "retained_per_representative_detail_arrays": 8 if _retain_details else 0,
    }
    total_runtime = time.perf_counter() - total_start
    if not _retain_details:
        return SequentialMeanWeightEstimationResult(
            mean_weights=weight_sum / n_frames,
            mean_residual=residual_sum / n_frames,
            average_iterations_per_frame=iteration_total / n_frames,
            converged_fraction=converged_count / n_frames,
            max_iterations_reached_count=n_frames - converged_count,
            mean_final_weight_change=weight_change_total / n_frames,
            max_final_weight_change=max_weight_change,
            mean_final_transform_change=transform_change_total / n_frames,
            max_final_transform_change=max_transform_change,
            total_runtime_seconds=total_runtime,
            timing=timing,
            call_counts=common_call_counts,
            warnings=tuple(warning_values),
        )
    assert weights_all is not None
    assert rotations is not None
    assert translations is not None
    assert final_rmsd is not None
    assert final_weighted_rmsd is not None
    assert iteration_counts is not None
    assert converged is not None
    assert final_weight_change is not None
    assert final_transform_change is not None
    return SequentialWeightEstimationResult(
        weights=weights_all,
        mean_weights=weight_sum / n_frames,
        iteration_counts=iteration_counts,
        converged=converged,
        final_weight_change=final_weight_change,
        final_transform_change=final_transform_change,
        rotations=rotations,
        translations=translations,
        final_rmsd=final_rmsd,
        final_weighted_rmsd=final_weighted_rmsd,
        mean_residual=residual_sum / n_frames,
        total_runtime_seconds=total_runtime,
        timing=timing,
        call_counts=common_call_counts,
        warnings=tuple(warning_values),
    )


def estimate_representative_weights_sequential_mean(
    coords: np.ndarray,
    reference: np.ndarray,
    **parameters: object,
) -> SequentialMeanWeightEstimationResult:
    """Run exact chronological warm-start IRLS without retaining frame details."""

    result = estimate_representative_weights_sequential_lean(
        coords,
        reference,
        _retain_details=False,
        **parameters,
    )
    assert isinstance(result, SequentialMeanWeightEstimationResult)
    return result
