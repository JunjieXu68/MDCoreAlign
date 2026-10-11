"""Internal approximate representative-weight estimators for CPU studies.

These candidates alter only stable-weight estimation. They never replace the
chronological float64 reference estimator and are intentionally not exposed as
separate public CLI modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .fast_kabsch import apply_transforms_batch, weighted_kabsch_transform_batch
from .representative_weights import (
    _estimate_batch,
    apply_sequence_continuity_regularization_batch,
    estimate_representative_weights,
)
from .sequential_weights import _weighted_fit
from .window_refine import bounded_global_weight_refinement


@dataclass(slots=True)
class ParallelWeightResult:
    """Compact output shared by internal parallel-weight candidates."""

    candidate: str
    mean_weights: np.ndarray
    total_runtime_seconds: float
    timing: dict[str, float]
    refinement_fraction: float
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _validate_inputs(
    representatives: np.ndarray,
    reference: np.ndarray,
    chain_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    coords = np.asarray(representatives, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("representatives must have shape (frames, residues, 3)")
    if target.shape != coords.shape[1:]:
        raise ValueError("reference must have shape (residues, 3)")
    if coords.shape[0] < 1 or coords.shape[1] < 3:
        raise ValueError("at least one frame and three residues are required")
    if not np.isfinite(coords).all() or not np.isfinite(target).all():
        raise ValueError("coordinates must be finite")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (coords.shape[1],):
        raise ValueError("chain_ids must have shape (residues,)")
    return coords, target, chains


def _normalise_max(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    maximum = float(np.max(values))
    if maximum <= 0.0:
        raise RuntimeError("weight update collapsed to all-zero weights")
    return np.clip(values / maximum, 0.0, 1.0)


def estimate_independent_batch(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    batch_size: int = 2000,
    max_iterations: int = 20,
) -> ParallelWeightResult:
    """Candidate A: independently initialise and batch all representatives."""

    coords, target, chains = _validate_inputs(representatives, reference, chain_ids)
    start = time.perf_counter()
    estimate = estimate_representative_weights(
        coords,
        target,
        batch_size=batch_size,
        rule="huber",
        delta=1.0,
        max_iterations=max_iterations,
        tolerance=1e-6,
        objective_tolerance=1e-5,
        objective_patience=2,
        sequence_continuity_strength=0.05,
        sequence_chain_ids=chains,
    )
    elapsed = time.perf_counter() - start
    return ParallelWeightResult(
        candidate="independent_batch",
        mean_weights=_normalise_max(estimate.mean_weights),
        total_runtime_seconds=elapsed,
        timing={"independent_batch": elapsed},
        refinement_fraction=1.0,
        metadata={
            "average_iterations": estimate.average_iterations_per_frame,
            "converged_fraction": estimate.converged_fraction,
            "representative_count": int(coords.shape[0]),
        },
        warnings=estimate.warnings,
    )


def estimate_shared_weight(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    update_rounds: int = 4,
    precision: str = "float64",
) -> ParallelWeightResult:
    """Candidate B: update one shared weight vector for all representatives."""

    coords, target, chains = _validate_inputs(representatives, reference, chain_ids)
    start = time.perf_counter()
    weights, changes, converged_counts, stage_timing = bounded_global_weight_refinement(
        coords,
        target,
        np.ones(coords.shape[1], dtype=np.float64),
        update_rounds=update_rounds,
        sequence_chain_ids=chains,
        precision=precision,
    )
    elapsed = time.perf_counter() - start
    return ParallelWeightResult(
        candidate="shared_weight",
        mean_weights=_normalise_max(weights),
        total_runtime_seconds=elapsed,
        timing={"shared_global_rounds": elapsed, **stage_timing},
        refinement_fraction=0.0,
        metadata={
            "update_rounds": int(update_rounds),
            "precision": precision,
            "final_weight_change": float(changes[-1]),
            "final_converged_representatives": int(converged_counts[-1]),
            "representative_count": int(coords.shape[0]),
        },
    )


def _bounded_chronological_correction(
    coords: np.ndarray,
    reference: np.ndarray,
    initial_weights: np.ndarray,
    *,
    correction_iterations: int,
    chain_ids: np.ndarray | None,
    refine_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """Apply a bounded sequential correction without result-object retention."""

    n_frames, n_residues, _ = coords.shape
    if correction_iterations not in {1, 2, 4}:
        raise ValueError("correction_iterations must be one of: 1, 2, 4")
    mask = (
        np.ones(n_frames, dtype=bool)
        if refine_mask is None
        else np.asarray(refine_mask, dtype=bool)
    )
    if mask.shape != (n_frames,):
        raise ValueError("refine_mask must have shape (frames,)")
    accumulated = np.zeros(n_residues, dtype=np.float64)
    previous_weights = np.asarray(initial_weights, dtype=np.float64).copy()
    previous_rotation = np.eye(3, dtype=np.float64)
    previous_translation = np.zeros(3, dtype=np.float64)
    previous_was_refined = False
    refined = 0
    start = time.perf_counter()
    for frame_index in range(n_frames):
        if not mask[frame_index]:
            accumulated += initial_weights
            previous_was_refined = False
            continue
        if not previous_was_refined:
            previous_weights = np.asarray(initial_weights, dtype=np.float64).copy()
            previous_rotation = np.eye(3, dtype=np.float64)
            previous_translation = np.zeros(3, dtype=np.float64)
        weights = previous_weights
        total_rotation = previous_rotation.copy()
        total_translation = previous_translation.copy()
        mobile_current = (
            coords[frame_index] @ total_rotation.T + total_translation
        )
        for _ in range(correction_iterations):
            correction_rotation, correction_translation, aligned, _, _ = _weighted_fit(
                mobile_current,
                reference,
                weights,
                compute_metrics=False,
            )
            distances = np.linalg.norm(aligned - reference, axis=1)
            updated = np.ones_like(distances)
            outliers = distances > 1.0
            np.divide(1.0, distances, out=updated, where=outliers)
            updated = apply_sequence_continuity_regularization_batch(
                updated[None, :],
                strength=0.05,
                passes=1,
                chain_ids=chain_ids,
                validate=False,
            )[0]
            weights = updated
            total_rotation = correction_rotation @ total_rotation
            total_translation = (
                total_translation @ correction_rotation.T
                + correction_translation
            )
            mobile_current = aligned
        correction_rotation, correction_translation, _, _, _ = _weighted_fit(
            mobile_current,
            reference,
            weights,
            compute_metrics=False,
        )
        previous_rotation = correction_rotation @ total_rotation
        previous_translation = (
            total_translation @ correction_rotation.T + correction_translation
        )
        previous_weights = weights
        previous_was_refined = True
        accumulated += weights
        refined += 1
    return _normalise_max(accumulated / n_frames), time.perf_counter() - start, refined


def estimate_batch_then_sequential(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    correction_iterations: int = 1,
) -> ParallelWeightResult:
    """Candidate C: shared batch initialisation then one chronological pass."""

    coords, target, chains = _validate_inputs(representatives, reference, chain_ids)
    total_start = time.perf_counter()
    initial = estimate_shared_weight(coords, target, chain_ids=chains)
    corrected, correction_seconds, refined = _bounded_chronological_correction(
        coords,
        target,
        initial.mean_weights,
        correction_iterations=correction_iterations,
        chain_ids=chains,
    )
    total = time.perf_counter() - total_start
    return ParallelWeightResult(
        candidate="batch_plus_sequential",
        mean_weights=corrected,
        total_runtime_seconds=total,
        timing={
            "batched_initialisation": initial.total_runtime_seconds,
            "chronological_correction": correction_seconds,
        },
        refinement_fraction=refined / coords.shape[0],
        metadata={
            "correction_iterations": int(correction_iterations),
            "representative_count": int(coords.shape[0]),
        },
    )


def estimate_blockwise_warm_start(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    block_size: int = 16,
    max_iterations: int = 20,
) -> ParallelWeightResult:
    """Candidate D: preserve chronology between independently batched blocks."""

    coords, target, chains = _validate_inputs(representatives, reference, chain_ids)
    if block_size not in {8, 16, 32, 64}:
        raise ValueError("block_size must be one of: 8, 16, 32, 64")
    total_start = time.perf_counter()
    accumulated = np.zeros(coords.shape[1], dtype=np.float64)
    previous = np.ones(coords.shape[1], dtype=np.float64)
    timing: dict[str, float] = {}
    warnings: list[str] = []
    n_blocks = 0
    for start in range(0, coords.shape[0], block_size):
        stop = min(start + block_size, coords.shape[0])
        block_count = stop - start
        estimate = _estimate_batch(
            coords[start:stop],
            target,
            initial_weights=np.broadcast_to(previous, (block_count, previous.size)),
            initial_rotations=None,
            initial_translations=None,
            rule="huber",
            max_iterations=max_iterations,
            tolerance=1e-6,
            weight_normalize="none",
            sequence_continuity_strength=0.05,
            sequence_continuity_passes=1,
            sequence_chain_ids=chains,
            objective_tolerance=1e-5,
            objective_patience=2,
            delta=1.0,
            sigma=1.0,
            cutoff=1.0,
        )
        previous = estimate.mean_weights
        accumulated += previous * block_count
        warnings.extend(estimate.warnings)
        for key, value in estimate.timing.items():
            timing[key] = timing.get(key, 0.0) + float(value)
        n_blocks += 1
    elapsed = time.perf_counter() - total_start
    timing["blockwise_total"] = elapsed
    return ParallelWeightResult(
        candidate="blockwise_warm_start",
        mean_weights=_normalise_max(accumulated / coords.shape[0]),
        total_runtime_seconds=elapsed,
        timing=timing,
        refinement_fraction=1.0,
        metadata={
            "block_size": int(block_size),
            "block_count": int(n_blocks),
            "representative_count": int(coords.shape[0]),
        },
        warnings=tuple(warnings),
    )


def _representative_risk_mask(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    quantile: float,
    interval_padding: int,
) -> tuple[np.ndarray, dict[str, float]]:
    transform = weighted_kabsch_transform_batch(
        coords,
        reference,
        weights,
        validate_inputs=False,
    )
    aligned = apply_transforms_batch(coords, transform.rotations, transform.translations)
    residuals = np.linalg.norm(aligned - reference[None, :, :], axis=2)
    objectives = np.sqrt(np.mean(residuals * residuals, axis=1))
    tail_ratio = np.quantile(residuals, 0.95, axis=1) / np.maximum(
        np.median(residuals, axis=1), 1e-12
    )
    objective_threshold = float(np.quantile(objectives, quantile))
    tail_threshold = float(np.quantile(tail_ratio, quantile))
    mask = (objectives >= objective_threshold) | (tail_ratio >= tail_threshold)
    if interval_padding > 0 and np.any(mask):
        expanded = mask.copy()
        for offset in range(1, interval_padding + 1):
            expanded[offset:] |= mask[:-offset]
            expanded[:-offset] |= mask[offset:]
        mask = expanded
    return mask, {
        "objective_threshold": objective_threshold,
        "residual_tail_threshold": tail_threshold,
    }


def estimate_risk_triggered_refinement(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    risk_quantile: float = 0.90,
    interval_padding: int = 1,
    correction_iterations: int = 2,
) -> ParallelWeightResult:
    """Candidate E: refine only residual-tail/fit-objective risk intervals."""

    coords, target, chains = _validate_inputs(representatives, reference, chain_ids)
    if not 0.5 <= risk_quantile < 1.0:
        raise ValueError("risk_quantile must be in [0.5, 1.0)")
    if interval_padding < 0:
        raise ValueError("interval_padding must be non-negative")
    total_start = time.perf_counter()
    initial = estimate_shared_weight(coords, target, chain_ids=chains)
    risk_start = time.perf_counter()
    mask, risk_metadata = _representative_risk_mask(
        coords,
        target,
        initial.mean_weights,
        quantile=risk_quantile,
        interval_padding=interval_padding,
    )
    risk_seconds = time.perf_counter() - risk_start
    corrected, correction_seconds, refined = _bounded_chronological_correction(
        coords,
        target,
        initial.mean_weights,
        correction_iterations=correction_iterations,
        chain_ids=chains,
        refine_mask=mask,
    )
    elapsed = time.perf_counter() - total_start
    return ParallelWeightResult(
        candidate="risk_triggered_refinement",
        mean_weights=corrected,
        total_runtime_seconds=elapsed,
        timing={
            "batched_initialisation": initial.total_runtime_seconds,
            "risk_scoring": risk_seconds,
            "chronological_refinement": correction_seconds,
        },
        refinement_fraction=refined / coords.shape[0],
        metadata={
            "risk_quantile": float(risk_quantile),
            "interval_padding": int(interval_padding),
            "correction_iterations": int(correction_iterations),
            "flagged_representatives": int(np.count_nonzero(mask)),
            "representative_count": int(coords.shape[0]),
            **risk_metadata,
        },
    )
