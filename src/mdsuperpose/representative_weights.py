"""Batched weights-only IRLS for long-mode representative frames."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .fast_kabsch import weighted_kabsch_variable_weights_batch


@dataclass(slots=True)
class RepresentativeWeightEstimationResult:
    """Compact result needed by long-mode representative weight estimation."""

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
    initialization: str = "uniform_independent"

    @property
    def average_iterations_per_frame(self) -> float:
        """Return mean IRLS iterations over representative frames."""

        return float(np.mean(self.iteration_counts)) if self.iteration_counts.size else 0.0

    @property
    def converged_fraction(self) -> float:
        """Return the fraction of representatives satisfying a stop rule."""

        return float(np.mean(self.converged)) if self.converged.size else 0.0


def _chain_slices(chain_ids: np.ndarray | None, n_atoms: int) -> tuple[slice, ...]:
    """Return contiguous sequence spans without crossing chain boundaries."""

    if chain_ids is None:
        return (slice(0, n_atoms),)
    values = np.asarray(chain_ids)
    if values.shape != (n_atoms,):
        raise ValueError("sequence_chain_ids must have shape (n_atoms,)")
    if n_atoms == 0:
        return ()
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [n_atoms]))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))


def apply_sequence_continuity_regularization_batch(
    weights: np.ndarray,
    *,
    strength: float = 0.0,
    passes: int = 1,
    chain_ids: np.ndarray | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Apply the scalar sequence-neighbor rule to a stack of weight vectors.

    Python iteration is limited to smoothing passes and contiguous chains; all
    representative frames are updated together. Chain boundaries are never
    smoothed across.
    """

    weights = np.asarray(weights, dtype=float)
    if validate:
        if weights.ndim != 2:
            raise ValueError("weights must have shape (n_frames, n_atoms)")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if passes < 0:
            raise ValueError("passes must be non-negative")
    if weights.shape[1] <= 1 or strength == 0.0 or passes == 0:
        return weights.copy()

    spans = _chain_slices(chain_ids, weights.shape[1])
    smoothed = weights.copy()
    updated = weights.copy()
    for _ in range(passes):
        updated[...] = smoothed
        for span in spans:
            start = int(span.start or 0)
            stop = int(span.stop or weights.shape[1])
            length = stop - start
            if length <= 1:
                continue
            updated[:, start] = (
                (1.0 - strength) * smoothed[:, start]
                + strength * smoothed[:, start + 1]
            )
            updated[:, stop - 1] = (
                (1.0 - strength) * smoothed[:, stop - 1]
                + strength * smoothed[:, stop - 2]
            )
            if length > 2:
                updated[:, start + 1 : stop - 1] = (
                    (1.0 - strength) * smoothed[:, start + 1 : stop - 1]
                    + 0.5
                    * strength
                    * (
                        smoothed[:, start : stop - 2]
                        + smoothed[:, start + 2 : stop]
                    )
                )
        smoothed, updated = updated, smoothed
    return smoothed


def _update_weights_batch(
    distances: np.ndarray,
    *,
    rule: str,
    delta: float,
    sigma: float,
    cutoff: float,
) -> np.ndarray:
    """Vectorize the existing Gaussian, Huber and Tukey weight rules."""

    if rule == "huber":
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        updated = np.ones_like(distances)
        mask = distances > delta
        updated[mask] = delta / distances[mask]
        return updated
    if rule == "gaussian":
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        return np.exp(-0.5 * (distances / sigma) ** 2)
    if rule == "tukey":
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        scaled = distances / cutoff
        updated = np.zeros_like(distances)
        mask = scaled < 1.0
        updated[mask] = (1.0 - scaled[mask] ** 2) ** 2
        return updated
    raise ValueError("rule must be one of: 'gaussian', 'huber', 'tukey'")


def _normalize_weights_batch(weights: np.ndarray, mode: str) -> np.ndarray:
    """Normalize each representative weight row when requested."""

    if mode == "none":
        return weights
    if mode == "sum":
        scales = np.sum(weights, axis=1)
    elif mode == "max":
        scales = np.max(weights, axis=1)
    else:
        raise ValueError("weight_normalize must be one of: 'none', 'sum', 'max'")
    if np.any(scales <= 0.0):
        raise RuntimeError("weight normalization encountered an all-zero frame")
    return weights / scales[:, None]


def _compose_batch(
    correction_rotations: np.ndarray,
    correction_translations: np.ndarray,
    previous_rotations: np.ndarray,
    previous_translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose row-vector transforms for a complete representative batch."""

    rotations = correction_rotations @ previous_rotations
    translations = (
        np.einsum(
            "bi,bji->bj",
            previous_translations,
            correction_rotations,
        )
        + correction_translations
    )
    return rotations, translations


def _estimate_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    *,
    initial_weights: np.ndarray | None,
    initial_rotations: np.ndarray | None,
    initial_translations: np.ndarray | None,
    rule: str,
    max_iterations: int,
    tolerance: float,
    weight_normalize: str,
    sequence_continuity_strength: float,
    sequence_continuity_passes: int,
    sequence_chain_ids: np.ndarray | None,
    objective_tolerance: float | None,
    objective_patience: int,
    delta: float,
    sigma: float,
    cutoff: float,
) -> RepresentativeWeightEstimationResult:
    """Estimate one representative chunk without frame-level Python loops."""

    start_total = time.perf_counter()
    n_frames, n_atoms, _ = coords.shape
    weights = (
        np.ones((n_frames, n_atoms), dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float).copy()
    )
    total_rotations = (
        np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
        if initial_rotations is None
        else np.asarray(initial_rotations, dtype=float).copy()
    )
    total_translations = (
        np.zeros((n_frames, 3), dtype=float)
        if initial_translations is None
        else np.asarray(initial_translations, dtype=float).copy()
    )
    mobile_current = (
        np.einsum("bni,bji->bnj", coords, total_rotations)
        + total_translations[:, None, :]
    )
    previous_objective = np.full(n_frames, np.nan, dtype=float)
    plateau_counts = np.zeros(n_frames, dtype=np.int32)
    iteration_counts = np.zeros(n_frames, dtype=np.int32)
    final_weight_change = np.full(n_frames, np.nan, dtype=float)
    final_transform_change = np.full(n_frames, np.nan, dtype=float)
    converged = np.zeros(n_frames, dtype=bool)
    active = np.ones(n_frames, dtype=bool)
    timing = {
        "batched_kabsch": 0.0,
        "residual_compute": 0.0,
        "weight_update": 0.0,
        "sequence_regularization": 0.0,
        "convergence": 0.0,
        "final_refit": 0.0,
    }
    stacked_svd_calls = 0

    for _ in range(max_iterations):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        kabsch_start = time.perf_counter()
        fit = weighted_kabsch_variable_weights_batch(
            mobile_current[active_indices],
            reference,
            weights[active_indices],
            validate_inputs=False,
            compute_metrics=False,
        )
        timing["batched_kabsch"] += time.perf_counter() - kabsch_start
        stacked_svd_calls += 1

        residual_start = time.perf_counter()
        deltas = fit.aligned_coords - reference[None, :, :]
        squared = np.einsum("bni,bni->bn", deltas, deltas)
        distances = np.sqrt(squared)
        active_weights = weights[active_indices]
        weighted_rmsd = np.sqrt(
            np.einsum("bn,bn->b", active_weights, squared)
            / np.sum(active_weights, axis=1)
        )
        timing["residual_compute"] += time.perf_counter() - residual_start

        weight_start = time.perf_counter()
        updated_weights = _update_weights_batch(
            distances,
            rule=rule,
            delta=delta,
            sigma=sigma,
            cutoff=cutoff,
        )
        timing["weight_update"] += time.perf_counter() - weight_start
        if sequence_continuity_strength > 0.0 and sequence_continuity_passes > 0:
            sequence_start = time.perf_counter()
            updated_weights = apply_sequence_continuity_regularization_batch(
                updated_weights,
                strength=sequence_continuity_strength,
                passes=sequence_continuity_passes,
                chain_ids=sequence_chain_ids,
                validate=False,
            )
            timing["sequence_regularization"] += time.perf_counter() - sequence_start
        updated_weights = _normalize_weights_batch(updated_weights, weight_normalize)
        if np.any(np.sum(updated_weights, axis=1) <= 0.0):
            raise RuntimeError("weight update collapsed to all-zero weights")

        convergence_start = time.perf_counter()
        weight_delta = np.max(np.abs(updated_weights - active_weights), axis=1)
        current_rotations, current_translations = _compose_batch(
            fit.rotations,
            fit.translations,
            total_rotations[active_indices],
            total_translations[active_indices],
        )
        transform_delta = np.linalg.norm(
            current_rotations - total_rotations[active_indices],
            axis=(1, 2),
        ) + np.linalg.norm(
            current_translations - total_translations[active_indices],
            axis=1,
        )
        active_previous_objective = previous_objective[active_indices]
        improvements = active_previous_objective - weighted_rmsd
        plateau = (
            np.isfinite(active_previous_objective)
            & (improvements >= 0.0)
            & (
                improvements
                <= (objective_tolerance if objective_tolerance is not None else -1.0)
            )
        )
        active_plateaus = np.where(
            plateau,
            plateau_counts[active_indices] + 1,
            0,
        )
        stopped = (weight_delta <= tolerance) & (transform_delta <= tolerance)
        if objective_tolerance is not None and objective_patience > 0:
            stopped |= active_plateaus >= objective_patience

        weights[active_indices] = updated_weights
        mobile_current[active_indices] = fit.aligned_coords
        total_rotations[active_indices] = current_rotations
        total_translations[active_indices] = current_translations
        previous_objective[active_indices] = weighted_rmsd
        plateau_counts[active_indices] = active_plateaus
        iteration_counts[active_indices] += 1
        final_weight_change[active_indices] = weight_delta
        final_transform_change[active_indices] = transform_delta
        converged[active_indices[stopped]] = True
        active[active_indices[stopped]] = False
        timing["convergence"] += time.perf_counter() - convergence_start

    final_start = time.perf_counter()
    final_fit = weighted_kabsch_variable_weights_batch(
        mobile_current,
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=True,
    )
    stacked_svd_calls += 1
    final_rotations, final_translations = _compose_batch(
        final_fit.rotations,
        final_fit.translations,
        total_rotations,
        total_translations,
    )
    final_deltas = final_fit.aligned_coords - reference[None, :, :]
    mean_residual = np.mean(
        np.sqrt(np.einsum("bni,bni->bn", final_deltas, final_deltas)),
        axis=0,
    )
    timing["final_refit"] += time.perf_counter() - final_start
    total_runtime = time.perf_counter() - start_total
    warnings = ()
    if not np.all(converged):
        warnings = (
            f"{int(np.sum(~converged))} representative frames reached max_iterations",
        )
    return RepresentativeWeightEstimationResult(
        weights=weights,
        mean_weights=np.mean(weights, axis=0),
        iteration_counts=iteration_counts,
        converged=converged,
        final_weight_change=final_weight_change,
        final_transform_change=final_transform_change,
        rotations=final_rotations,
        translations=final_translations,
        final_rmsd=final_fit.rmsd,
        final_weighted_rmsd=final_fit.weighted_rmsd,
        mean_residual=mean_residual,
        total_runtime_seconds=total_runtime,
        timing=timing,
        call_counts={
            "stacked_svd_calls": stacked_svd_calls,
            "per_frame_kabsch_calls": 0,
            "per_frame_result_objects": 0,
        },
        warnings=warnings,
    )


def estimate_representative_weights(
    coords_subset: np.ndarray,
    reference: np.ndarray,
    *,
    batch_size: int = 2000,
    return_diagnostics: bool = False,
    sequence_chain_ids: np.ndarray | None = None,
    warm_start_refinement_passes: int = 0,
    rule: str = "huber",
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    weight_normalize: str = "none",
    sequence_continuity_strength: float = 0.0,
    sequence_continuity_passes: int = 1,
    contact_consistency_strength: float = 0.0,
    contact_consistency_passes: int = 1,
    objective_tolerance: float | None = None,
    objective_patience: int = 0,
    delta: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 1.0,
    **unused_parameters: Any,
) -> RepresentativeWeightEstimationResult:
    """Estimate representative-frame weights with stacked float64 IRLS.

    Representative frames are initialized independently with uniform weights.
    This removes the legacy sequential warm-start dependency and permits true
    frame batching. The legacy estimator remains the public default until its
    aggregate long-mode outputs pass the Phase 3 quality decision.
    """

    del return_diagnostics
    ignored = {"record_history", "diagnostic_callback"}
    unknown = sorted(set(unused_parameters) - ignored)
    if unknown:
        raise TypeError(f"unsupported batched representative parameters: {unknown}")
    coords = np.asarray(coords_subset, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords_subset must have shape (n_frames, n_atoms, 3)")
    if reference.shape != coords.shape[1:]:
        raise ValueError("reference must have shape (n_atoms, 3)")
    if coords.shape[0] == 0 or coords.shape[1] < 3:
        raise ValueError("at least one frame and three atoms are required")
    if not np.isfinite(coords).all() or not np.isfinite(reference).all():
        raise ValueError("coordinates must contain only finite values")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if objective_tolerance is not None and objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be non-negative")
    if objective_patience < 0:
        raise ValueError("objective_patience must be non-negative")
    if not 0.0 <= sequence_continuity_strength <= 1.0:
        raise ValueError("sequence_continuity_strength must be in [0, 1]")
    if sequence_continuity_passes < 0:
        raise ValueError("sequence_continuity_passes must be non-negative")
    if warm_start_refinement_passes < 0:
        raise ValueError("warm_start_refinement_passes must be non-negative")
    if contact_consistency_strength > 0.0 and contact_consistency_passes > 0:
        raise NotImplementedError(
            "batched representative estimation does not change contact regularization; "
            "use representative_weight_estimator='legacy'"
        )
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _chain_slices(sequence_chain_ids, coords.shape[1])

    start = time.perf_counter()

    def run_once(
        initial_weights: np.ndarray | None,
        initial_rotations: np.ndarray | None,
        initial_translations: np.ndarray | None,
    ) -> RepresentativeWeightEstimationResult:
        chunks: list[RepresentativeWeightEstimationResult] = []
        for chunk_start in range(0, coords.shape[0], batch_size):
            chunk_stop = min(chunk_start + batch_size, coords.shape[0])
            chunks.append(
                _estimate_batch(
                    coords[chunk_start:chunk_stop],
                    reference,
                    initial_weights=(
                        None
                        if initial_weights is None
                        else initial_weights[chunk_start:chunk_stop]
                    ),
                    initial_rotations=(
                        None
                        if initial_rotations is None
                        else initial_rotations[chunk_start:chunk_stop]
                    ),
                    initial_translations=(
                        None
                        if initial_translations is None
                        else initial_translations[chunk_start:chunk_stop]
                    ),
                    rule=rule,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    weight_normalize=weight_normalize,
                    sequence_continuity_strength=sequence_continuity_strength,
                    sequence_continuity_passes=sequence_continuity_passes,
                    sequence_chain_ids=sequence_chain_ids,
                    objective_tolerance=objective_tolerance,
                    objective_patience=objective_patience,
                    delta=delta,
                    sigma=sigma,
                    cutoff=cutoff,
                )
            )
        weights = np.concatenate([chunk.weights for chunk in chunks], axis=0)
        timing = {
            key: float(sum(chunk.timing[key] for chunk in chunks))
            for key in chunks[0].timing
        }
        warnings = tuple(
            warning for chunk in chunks for warning in chunk.warnings
        )
        return RepresentativeWeightEstimationResult(
            weights=weights,
            mean_weights=np.mean(weights, axis=0),
            iteration_counts=np.concatenate(
                [chunk.iteration_counts for chunk in chunks]
            ),
            converged=np.concatenate([chunk.converged for chunk in chunks]),
            final_weight_change=np.concatenate(
                [chunk.final_weight_change for chunk in chunks]
            ),
            final_transform_change=np.concatenate(
                [chunk.final_transform_change for chunk in chunks]
            ),
            rotations=np.concatenate(
                [chunk.rotations for chunk in chunks],
                axis=0,
            ),
            translations=np.concatenate(
                [chunk.translations for chunk in chunks],
                axis=0,
            ),
            final_rmsd=np.concatenate([chunk.final_rmsd for chunk in chunks]),
            final_weighted_rmsd=np.concatenate(
                [chunk.final_weighted_rmsd for chunk in chunks]
            ),
            mean_residual=(
                sum(
                    chunk.mean_residual * chunk.weights.shape[0]
                    for chunk in chunks
                )
                / coords.shape[0]
            ),
            total_runtime_seconds=sum(
                chunk.total_runtime_seconds for chunk in chunks
            ),
            timing=timing,
            call_counts={
                "stacked_svd_calls": sum(
                    chunk.call_counts["stacked_svd_calls"] for chunk in chunks
                ),
                "per_frame_kabsch_calls": 0,
                "per_frame_result_objects": 0,
                "representative_chunks": len(chunks),
            },
            warnings=warnings,
        )

    result = run_once(None, None, None)
    aggregate_timing = dict(result.timing)
    aggregate_call_counts = dict(result.call_counts)
    for _ in range(warm_start_refinement_passes):
        initial_weights = np.ones_like(result.weights)
        initial_rotations = np.broadcast_to(
            np.eye(3),
            result.rotations.shape,
        ).copy()
        initial_translations = np.zeros_like(result.translations)
        initial_weights[1:] = result.weights[:-1]
        initial_rotations[1:] = result.rotations[:-1]
        initial_translations[1:] = result.translations[:-1]
        result = run_once(
            initial_weights,
            initial_rotations,
            initial_translations,
        )
        for key, value in result.timing.items():
            aggregate_timing[key] += value
        for key, value in result.call_counts.items():
            aggregate_call_counts[key] = aggregate_call_counts.get(key, 0) + value

    result.total_runtime_seconds = time.perf_counter() - start
    result.timing = aggregate_timing
    result.call_counts = aggregate_call_counts
    result.call_counts["warm_start_refinement_passes"] = (
        warm_start_refinement_passes
    )
    result.initialization = (
        "uniform_independent"
        if warm_start_refinement_passes == 0
        else f"uniform_then_{warm_start_refinement_passes}_shifted_refinement"
    )
    return result
