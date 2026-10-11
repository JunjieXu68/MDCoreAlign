"""First iterative robust alignment prototype.

This module implements a simple alternating robust alignment loop:

1. solve a weighted rigid alignment,
2. compute residual distances,
3. update per-atom weights from those residuals,
4. optionally smooth weights along the sequence,
5. optionally smooth weights over a sparse contact graph.

It remains intentionally minimal and does not yet include trajectory
warm-start logic or heavier global regularization subproblems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .alignment import (
    RigidTransform,
    apply_transform,
    compose_transforms,
    identity_transform,
    solve_weighted_rigid_alignment,
)
from .contacts import ContactGraph, apply_contact_consistency_regularization, build_contact_graph, validate_contact_graph
from .utils import as_float_array, ensure_atom_vector
from .weights import (
    apply_sequence_continuity_regularization,
    normalize_weights,
    uniform_weights,
    update_irls_weights,
)


@dataclass(slots=True)
class RobustAlignmentHistoryEntry:
    """One iteration record for the robust alignment loop."""

    iteration: int
    weighted_rmsd: float
    weight_change: float
    transform_change: float


@dataclass(slots=True)
class RobustAlignmentResult:
    """Final output of the first iterative robust alignment prototype."""

    rotation: np.ndarray
    translation: np.ndarray
    weights: np.ndarray
    aligned_coords: np.ndarray
    residual_distances: np.ndarray
    rmsd: float
    weighted_rmsd: float
    iteration_count: int
    final_weight_change: float | None
    final_transform_change: float | None
    history: list[RobustAlignmentHistoryEntry]
    converged: bool


def _validate_paired_coords(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a pair of matched coordinate arrays for the robust loop."""

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


def residual_distances(aligned_coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute per-atom Euclidean residual distances."""

    aligned_coords = as_float_array(aligned_coords)
    reference = as_float_array(reference)
    if aligned_coords.shape != reference.shape:
        raise ValueError("aligned_coords and reference must have the same shape")
    deltas = aligned_coords - reference
    squared = np.einsum("ij,ij->i", deltas, deltas)
    return np.sqrt(squared)


def _residual_distances_fast(aligned_coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute residual distances for already validated coordinate arrays."""

    deltas = aligned_coords - reference
    squared = np.einsum("ij,ij->i", deltas, deltas)
    return np.sqrt(squared)


def _weighted_rmsd_from_distances(distances: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted RMSD from already computed residual distances."""

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("weights must contain a positive total mass")
    squared = distances * distances
    return float(np.sqrt(float(np.dot(weights, squared)) / total_weight))


def _binary_trim_weights_from_distances(distances: np.ndarray, core_fraction: float) -> np.ndarray:
    """Select a deterministic binary core mask from the lowest residual distances."""

    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.shape[0] == 0:
        raise ValueError("distances must be a non-empty one-dimensional array")
    if not np.isfinite(distances).all():
        raise ValueError("distances must contain only finite values")
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")

    k = max(1, int(np.floor(core_fraction * distances.shape[0] + 0.5)))
    order = np.lexsort((np.arange(distances.shape[0]), distances))
    weights = np.zeros(distances.shape[0], dtype=float)
    weights[order[:k]] = 1.0
    return weights


def _validate_initial_weights(weights: np.ndarray, n_atoms: int) -> np.ndarray:
    """Validate initial non-negative weights for the robust loop."""

    weights = ensure_atom_vector(weights, n_atoms, name="initial_weights")
    if not np.isfinite(weights).all():
        raise ValueError("initial_weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("initial_weights must be non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("initial_weights must contain at least one positive entry")
    return weights


def _validate_initial_transform(initial_transform: RigidTransform | None) -> RigidTransform | None:
    """Validate an optional initial rigid transform."""

    if initial_transform is None:
        return None
    rotation = as_float_array(initial_transform.rotation)
    translation = as_float_array(initial_transform.translation)
    if rotation.shape != (3, 3):
        raise ValueError("initial_transform.rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("initial_transform.translation must have shape (3,)")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("initial_transform must contain only finite values")
    return RigidTransform(rotation=rotation, translation=translation)


def transform_change(current: RigidTransform, previous: RigidTransform) -> float:
    """Measure the change between two rigid transforms."""

    rotation_difference = current.rotation - previous.rotation
    translation_difference = current.translation - previous.translation
    rotation_delta = np.sqrt(float(np.einsum("ij,ij->", rotation_difference, rotation_difference)))
    translation_delta = np.sqrt(float(np.dot(translation_difference, translation_difference)))
    return float(rotation_delta + translation_delta)


def solve_iterative_robust_alignment(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    rule: str = "huber",
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    initial_weights: np.ndarray | None = None,
    initial_transform: RigidTransform | None = None,
    weight_normalize: str = "none",
    sequence_continuity_strength: float = 0.0,
    sequence_continuity_passes: int = 1,
    contact_consistency_strength: float = 0.0,
    contact_consistency_passes: int = 1,
    contact_graph: ContactGraph | None = None,
    contact_graph_validated: bool = False,
    contact_distance_cutoff: float = 8.0,
    contact_min_sequence_separation: int = 2,
    contact_max_neighbors: int | None = 8,
    objective_tolerance: float | None = None,
    objective_patience: int = 0,
    record_history: bool = True,
    diagnostic_callback: Callable[[dict[str, Any]], None] | None = None,
    **weight_parameters: float,
) -> RobustAlignmentResult:
    """Run the first working iterative robust alignment loop.

    The loop alternates:

    1. weighted rigid alignment
    2. residual distance computation
    3. robust weight update
    4. optional light sequence smoothing
    5. optional light contact-graph smoothing

    Convergence is declared when both the transform change and the weight
    change fall below the requested tolerance.
    """

    mobile, reference = _validate_paired_coords(mobile, reference)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if objective_tolerance is not None and objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be non-negative when provided")
    if objective_patience < 0:
        raise ValueError("objective_patience must be non-negative")
    if not 0.0 <= sequence_continuity_strength <= 1.0:
        raise ValueError("sequence_continuity_strength must be in [0, 1]")
    if sequence_continuity_passes < 0:
        raise ValueError("sequence_continuity_passes must be non-negative")
    if not 0.0 <= contact_consistency_strength <= 1.0:
        raise ValueError("contact_consistency_strength must be in [0, 1]")
    if contact_consistency_passes < 0:
        raise ValueError("contact_consistency_passes must be non-negative")
    if contact_graph is None and contact_consistency_strength > 0.0:
        contact_graph = build_contact_graph(
            reference,
            cutoff=contact_distance_cutoff,
            min_sequence_separation=contact_min_sequence_separation,
            max_neighbors=contact_max_neighbors,
        )
        contact_graph_validated = True
    elif contact_graph is not None and not contact_graph_validated:
        contact_graph = validate_contact_graph(contact_graph, mobile.shape[0])
        contact_graph_validated = True

    if initial_weights is None:
        weights = uniform_weights(mobile.shape[0])
    else:
        weights = _validate_initial_weights(initial_weights, mobile.shape[0])
    total_transform = _validate_initial_transform(initial_transform) or identity_transform()
    mobile_current = apply_transform(mobile, total_transform)

    history: list[RobustAlignmentHistoryEntry] = []
    previous_transform = total_transform
    converged = False
    previous_iteration_weighted_rmsd: float | None = None
    plateau_count = 0
    iteration_count = 0
    final_weight_change: float | None = None
    final_transform_change: float | None = None

    for iteration in range(1, max_iterations + 1):
        iteration_count = iteration
        iteration_start = time.perf_counter()
        solve_start = time.perf_counter()
        solve = solve_weighted_rigid_alignment(
            mobile_current,
            reference,
            weights,
            validate_inputs=False,
            compute_metrics=False,
        )
        solve_seconds = time.perf_counter() - solve_start
        residual_start = time.perf_counter()
        distances = _residual_distances_fast(solve.aligned_coords, reference)
        iteration_weighted_rmsd = _weighted_rmsd_from_distances(distances, weights)
        residual_seconds = time.perf_counter() - residual_start
        weight_start = time.perf_counter()
        updated_weights = update_irls_weights(
            distances,
            rule=rule,
            normalize="none",
            validate=False,
            **weight_parameters,
        )
        weight_seconds = time.perf_counter() - weight_start
        sequence_seconds = 0.0
        if sequence_continuity_strength > 0.0 and sequence_continuity_passes > 0:
            sequence_start = time.perf_counter()
            updated_weights = apply_sequence_continuity_regularization(
                updated_weights,
                strength=sequence_continuity_strength,
                passes=sequence_continuity_passes,
            )
            sequence_seconds = time.perf_counter() - sequence_start
        contact_seconds = 0.0
        if contact_graph is not None and contact_consistency_strength > 0.0 and contact_consistency_passes > 0:
            contact_start = time.perf_counter()
            updated_weights = apply_contact_consistency_regularization(
                updated_weights,
                contact_graph,
                strength=contact_consistency_strength,
                passes=contact_consistency_passes,
                validate=False,
            )
            contact_seconds = time.perf_counter() - contact_start
        if weight_normalize != "none":
            updated_weights = normalize_weights(updated_weights, mode=weight_normalize)
        if not np.any(updated_weights > 0.0):
            raise RuntimeError("weight update collapsed to all-zero weights")

        weight_difference = updated_weights - weights
        weight_delta = float(np.max(np.abs(weight_difference)))
        correction = RigidTransform(rotation=solve.rotation, translation=solve.translation)
        current_transform = compose_transforms(correction, total_transform)
        transform_delta = transform_change(current_transform, previous_transform)
        if diagnostic_callback is not None:
            rotation_difference = current_transform.rotation - previous_transform.rotation
            translation_difference = current_transform.translation - previous_transform.translation
            weight_delta_l2 = np.sqrt(float(np.dot(weight_difference, weight_difference)))
            rotation_delta = np.sqrt(float(np.einsum("ij,ij->", rotation_difference, rotation_difference)))
            translation_delta = np.sqrt(float(np.dot(translation_difference, translation_difference)))
        final_weight_change = weight_delta
        final_transform_change = transform_delta
        if record_history:
            history.append(
                RobustAlignmentHistoryEntry(
                    iteration=iteration,
                    weighted_rmsd=iteration_weighted_rmsd,
                    weight_change=weight_delta,
                    transform_change=transform_delta,
                )
            )

        weights = updated_weights
        mobile_current = solve.aligned_coords
        total_transform = current_transform
        previous_transform = current_transform
        convergence_start = time.perf_counter()
        if objective_tolerance is not None and objective_patience > 0:
            if previous_iteration_weighted_rmsd is not None:
                improvement = previous_iteration_weighted_rmsd - iteration_weighted_rmsd
                if 0.0 <= improvement <= objective_tolerance:
                    plateau_count += 1
                else:
                    plateau_count = 0
            previous_iteration_weighted_rmsd = iteration_weighted_rmsd
        stop_by_tolerance = weight_delta <= tolerance and transform_delta <= tolerance
        stop_by_objective = objective_tolerance is not None and objective_patience > 0 and plateau_count >= objective_patience
        converged_this_iteration = False
        if stop_by_tolerance:
            converged = True
            converged_this_iteration = True
        if stop_by_objective:
            converged = True
            converged_this_iteration = True
        convergence_seconds = time.perf_counter() - convergence_start
        if diagnostic_callback is not None:
            diagnostic_core_count = max(1, int(np.floor(0.6 * updated_weights.shape[0] + 0.5)))
            diagnostic_callback(
                {
                    "iteration": iteration,
                    "pseudo_objective_value": iteration_weighted_rmsd,
                    "weighted_rmsd": iteration_weighted_rmsd,
                    "mean_residual": float(np.mean(distances)),
                    "median_residual": float(np.median(distances)),
                    "max_residual": float(np.max(distances)),
                    "mean_weight": float(np.mean(updated_weights)),
                    "min_weight": float(np.min(updated_weights)),
                    "max_weight": float(np.max(updated_weights)),
                    "effective_core_size": float(np.sum(updated_weights)),
                    "n_core_residues_current": diagnostic_core_count,
                    "delta_weight_l2": weight_delta_l2,
                    "delta_weight_max": weight_delta,
                    "delta_rotation_norm": rotation_delta,
                    "delta_translation_norm": translation_delta,
                    "converged": converged_this_iteration,
                    "runtime_iteration_seconds": time.perf_counter() - iteration_start,
                    "weighted_kabsch_or_refit_seconds": solve_seconds,
                    "residual_computation_seconds": residual_seconds,
                    "weight_update_seconds": weight_seconds,
                    "sequence_regularization_seconds": sequence_seconds,
                    "contact_regularization_seconds": contact_seconds,
                    "convergence_check_seconds": convergence_seconds,
                }
            )
        if converged_this_iteration:
            break

    final_solve = solve_weighted_rigid_alignment(
        mobile_current,
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=True,
    )
    final_transform = compose_transforms(
        RigidTransform(rotation=final_solve.rotation, translation=final_solve.translation),
        total_transform,
    )
    final_distances = _residual_distances_fast(final_solve.aligned_coords, reference)
    return RobustAlignmentResult(
        rotation=final_transform.rotation,
        translation=final_transform.translation,
        weights=weights,
        aligned_coords=final_solve.aligned_coords,
        residual_distances=final_distances,
        rmsd=final_solve.rmsd,
        weighted_rmsd=final_solve.weighted_rmsd,
        iteration_count=iteration_count,
        final_weight_change=final_weight_change,
        final_transform_change=final_transform_change,
        history=history,
        converged=converged,
    )


def solve_iterative_hard_trim_alignment(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    core_fraction: float,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    initial_transform: RigidTransform | None = None,
    objective_tolerance: float | None = None,
    objective_patience: int = 0,
    record_history: bool = True,
) -> RobustAlignmentResult:
    """Run a minimal hard-trim control alignment with a fixed target core fraction.

    This control baseline alternates:

    1. align on the current binary subset,
    2. compute residual distances,
    3. keep only the lowest-residual target fraction of atoms.
    """

    mobile, reference = _validate_paired_coords(mobile, reference)
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if objective_tolerance is not None and objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be non-negative when provided")
    if objective_patience < 0:
        raise ValueError("objective_patience must be non-negative")

    total_transform = _validate_initial_transform(initial_transform) or identity_transform()
    mobile_current = apply_transform(mobile, total_transform)
    seed = solve_weighted_rigid_alignment(
        mobile_current,
        reference,
        uniform_weights(mobile.shape[0]),
        validate_inputs=False,
        compute_metrics=False,
    )
    seed_distances = residual_distances(seed.aligned_coords, reference)
    weights = _binary_trim_weights_from_distances(seed_distances, core_fraction)
    total_transform = compose_transforms(
        RigidTransform(rotation=seed.rotation, translation=seed.translation),
        total_transform,
    )
    mobile_current = seed.aligned_coords

    history: list[RobustAlignmentHistoryEntry] = []
    previous_transform = total_transform
    converged = False
    previous_iteration_weighted_rmsd: float | None = None
    plateau_count = 0
    iteration_count = 0
    final_weight_change: float | None = None
    final_transform_change: float | None = None

    for iteration in range(1, max_iterations + 1):
        iteration_count = iteration
        solve = solve_weighted_rigid_alignment(
            mobile_current,
            reference,
            weights,
            validate_inputs=False,
            compute_metrics=False,
        )
        distances = residual_distances(solve.aligned_coords, reference)
        iteration_weighted_rmsd = _weighted_rmsd_from_distances(distances, weights)
        updated_weights = _binary_trim_weights_from_distances(distances, core_fraction)

        weight_delta = float(np.max(np.abs(updated_weights - weights)))
        correction = RigidTransform(rotation=solve.rotation, translation=solve.translation)
        current_transform = compose_transforms(correction, total_transform)
        transform_delta = transform_change(current_transform, previous_transform)
        final_weight_change = weight_delta
        final_transform_change = transform_delta
        if record_history:
            history.append(
                RobustAlignmentHistoryEntry(
                    iteration=iteration,
                    weighted_rmsd=iteration_weighted_rmsd,
                    weight_change=weight_delta,
                    transform_change=transform_delta,
                )
            )

        weights = updated_weights
        mobile_current = solve.aligned_coords
        total_transform = current_transform
        previous_transform = current_transform

        if objective_tolerance is not None and objective_patience > 0:
            if previous_iteration_weighted_rmsd is not None:
                improvement = previous_iteration_weighted_rmsd - iteration_weighted_rmsd
                if 0.0 <= improvement <= objective_tolerance:
                    plateau_count += 1
                else:
                    plateau_count = 0
            previous_iteration_weighted_rmsd = iteration_weighted_rmsd

        if weight_delta == 0.0 and transform_delta <= tolerance:
            converged = True
            break
        if objective_tolerance is not None and objective_patience > 0 and plateau_count >= objective_patience:
            converged = True
            break

    final_solve = solve_weighted_rigid_alignment(
        mobile_current,
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=True,
    )
    final_transform = compose_transforms(
        RigidTransform(rotation=final_solve.rotation, translation=final_solve.translation),
        total_transform,
    )
    final_distances = residual_distances(final_solve.aligned_coords, reference)
    return RobustAlignmentResult(
        rotation=final_transform.rotation,
        translation=final_transform.translation,
        weights=weights,
        aligned_coords=final_solve.aligned_coords,
        residual_distances=final_distances,
        rmsd=final_solve.rmsd,
        weighted_rmsd=final_solve.weighted_rmsd,
        iteration_count=iteration_count,
        final_weight_change=final_weight_change,
        final_transform_change=final_transform_change,
        history=history,
        converged=converged,
    )
