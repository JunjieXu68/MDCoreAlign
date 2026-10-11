"""Bounded global refinement with optional window-derived soft initial weights."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .all_frame_engine import AllFrameFitResult, fit_all_frames_fixed_weights
from .fast_kabsch import weighted_kabsch_transform_batch
from .representative_weights import apply_sequence_continuity_regularization_batch
from .stability_graph import scan_window_graph
from .window_scan import (
    WindowScanResult,
    scan_window_local,
    scan_window_multiscale,
)


@dataclass(frozen=True, slots=True)
class InitialWeightResult:
    """One soft initializer and its construction cost."""

    weights: np.ndarray
    initializer: str
    scan_seconds: float
    metadata: dict[str, object]


ExactAllFrameFitResult = AllFrameFitResult


@dataclass(frozen=True, slots=True)
class WindowPreconditionedRefinementResult:
    """Bounded-refinement output followed by an explicit exact global fit."""

    initializer: str
    initial_weights: np.ndarray
    final_weights: np.ndarray
    representative_indices: np.ndarray
    update_rounds: int
    round_weight_changes: np.ndarray
    converged_representative_counts: np.ndarray
    active_set_size: int
    effective_core_size: float
    fit: ExactAllFrameFitResult
    timing: dict[str, float]
    metadata: dict[str, object]


def temporally_stratified_indices(
    n_frames: int,
    representative_count: int,
) -> np.ndarray:
    """Select deterministic, evenly stratified frame indices."""

    n_frames = int(n_frames)
    representative_count = int(representative_count)
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if representative_count < 1:
        raise ValueError("representative_count must be positive")
    count = min(n_frames, representative_count)
    if count == 1:
        return np.asarray([0], dtype=np.int64)
    return np.unique(np.linspace(0, n_frames - 1, count, dtype=np.int64))


def _normalize_prior(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("initializer weights must be a residue vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("initializer weights must be finite and non-negative")
    maximum = float(np.max(values))
    if maximum <= 0.0:
        raise ValueError("initializer weights must contain positive mass")
    return np.clip(values / maximum, 0.0, 1.0)


def blend_soft_prior(weights: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a normalized soft prior with uniform weights."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    prior = _normalize_prior(weights)
    return (1.0 - alpha) + alpha * prior


def conservative_exclusion_prior(
    movement_score: np.ndarray,
    *,
    top_mobility_fraction: float,
    minimum_retained_weight: float = 0.25,
) -> np.ndarray:
    """Softly downweight a fixed top-mobility fraction without zero weights."""

    movement = np.asarray(movement_score, dtype=np.float64)
    fraction = float(top_mobility_fraction)
    minimum = float(minimum_retained_weight)
    if movement.ndim != 1 or movement.size < 3:
        raise ValueError("movement_score must be a residue vector")
    if not 0.0 < fraction < 1.0:
        raise ValueError("top_mobility_fraction must be in (0, 1)")
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_retained_weight must be in (0, 1]")
    count = max(1, int(np.ceil(fraction * movement.size)))
    order = np.lexsort((np.arange(movement.size), -movement))
    weights = np.ones(movement.size, dtype=np.float64)
    weights[order[:count]] = minimum
    return weights


def compute_frozen_window_signals(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    initializers: tuple[str, ...] | None = None,
) -> dict[str, tuple[WindowScanResult, float]]:
    """Compute requested frozen Phase 6 window signals exactly once."""

    values = np.asarray(coords, dtype=np.float64)
    outputs: dict[str, tuple[WindowScanResult, float]] = {}
    specifications = (
        (
            "w5_relative_mad_soft",
            lambda: scan_window_local(
                values,
                feature_set="minimal",
                chain_ids=chain_ids,
                statistic="relative_mad",
            ),
        ),
        (
            "w5_winsorised_variance_soft",
            lambda: scan_window_local(
                values,
                feature_set="minimal",
                chain_ids=chain_ids,
                statistic="winsorised_variance",
            ),
        ),
        (
            "window_local_multiscale_soft",
            lambda: scan_window_multiscale(
                values,
                chain_ids=chain_ids,
                statistic="winsorised_variance",
            ),
        ),
        (
            "window_graph_soft",
            lambda: scan_window_graph(
                values,
                chain_ids=chain_ids,
                variant="local_contact",
                statistic="winsorised_variance",
            ).scan,
        ),
    )
    requested = (
        {name for name, _ in specifications}
        if initializers is None
        else set(initializers)
    )
    unknown = requested.difference(name for name, _ in specifications)
    if unknown:
        raise ValueError(
            f"unknown frozen window signal(s): {', '.join(sorted(unknown))}"
        )
    for name, function in specifications:
        if name not in requested:
            continue
        start = time.perf_counter()
        result = function()
        outputs[name] = (result, time.perf_counter() - start)
    return outputs


def build_initial_weights(
    n_residues: int,
    initializer: str,
    *,
    window_signals: dict[str, tuple[WindowScanResult, float]] | None = None,
    alpha: float | None = None,
    top_mobility_fraction: float | None = None,
    minimum_retained_weight: float = 0.25,
) -> InitialWeightResult:
    """Construct one fixed-grid initializer from cached window signals."""

    if initializer == "uniform":
        return InitialWeightResult(
            weights=np.ones(int(n_residues), dtype=np.float64),
            initializer=initializer,
            scan_seconds=0.0,
            metadata={"alpha": None, "output_contract": "global_refinement"},
        )
    if window_signals is None:
        raise ValueError("window_signals are required for a window initializer")
    if initializer == "w5_conservative_exclusion":
        if top_mobility_fraction is None:
            raise ValueError("top_mobility_fraction is required")
        signal, scan_seconds = window_signals["w5_winsorised_variance_soft"]
        weights = conservative_exclusion_prior(
            signal.movement_score,
            top_mobility_fraction=top_mobility_fraction,
            minimum_retained_weight=minimum_retained_weight,
        )
        return InitialWeightResult(
            weights=weights,
            initializer=initializer,
            scan_seconds=scan_seconds,
            metadata={
                "alpha": None,
                "top_mobility_fraction": float(top_mobility_fraction),
                "minimum_retained_weight": float(minimum_retained_weight),
                "window_signal": "w5_winsorised_variance_soft",
                "soft_prior_only": True,
            },
        )
    if initializer not in window_signals:
        raise ValueError(f"unknown initializer: {initializer}")
    if alpha is None:
        raise ValueError("alpha is required for a soft window initializer")
    signal, scan_seconds = window_signals[initializer]
    return InitialWeightResult(
        weights=blend_soft_prior(signal.stability_weights, alpha),
        initializer=initializer,
        scan_seconds=scan_seconds,
        metadata={
            "alpha": float(alpha),
            "top_mobility_fraction": None,
            "window_signal": initializer,
            "soft_prior_only": True,
        },
    )


def validate_weighting_parameters(delta: float, sequence_continuity_strength: float) -> None:
    """Reject non-finite scales/strengths before fitting or device allocation."""

    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    if not np.isfinite(sequence_continuity_strength) or not 0.0 <= sequence_continuity_strength <= 1.0:
        raise ValueError("sequence_continuity_strength must be finite and in [0, 1]")


def bounded_global_weight_refinement(
    representatives: np.ndarray,
    reference: np.ndarray,
    initial_weights: np.ndarray,
    *,
    update_rounds: int,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    sequence_chain_ids: np.ndarray | None = None,
    convergence_tolerance: float = 1e-6,
    trace: dict[str, np.ndarray] | None = None,
    precision: str = "float64",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Apply an exact fixed number of shared-weight global robust updates."""

    if precision not in {"float64", "float32", "mixed"}:
        raise ValueError("precision must be 'float64', 'float32', or 'mixed'")
    compute_dtype = np.float64 if precision == "float64" else np.float32
    coords = np.asarray(representatives, dtype=compute_dtype)
    target = np.asarray(reference, dtype=compute_dtype)
    weights = _normalize_prior(initial_weights).astype(compute_dtype, copy=False)
    update_rounds = int(update_rounds)
    if coords.ndim != 3 or coords.shape[1:] != target.shape:
        raise ValueError("representatives and reference shapes are incompatible")
    if update_rounds < 1:
        raise ValueError("update_rounds must be at least 1")
    validate_weighting_parameters(delta, sequence_continuity_strength)
    changes = np.empty(update_rounds, dtype=np.float64)
    converged_counts = np.empty(update_rounds, dtype=np.int64)
    fit_seconds = 0.0
    residual_seconds = 0.0
    update_seconds = 0.0
    regularization_seconds = 0.0
    if trace is not None:
        weight_history = np.empty(
            (update_rounds + 1, weights.size),
            dtype=np.float64,
        )
        weight_history[0] = weights
        objective_history = np.empty(update_rounds, dtype=np.float64)
        l1_changes = np.empty(update_rounds, dtype=np.float64)
        mean_l1_changes = np.empty(update_rounds, dtype=np.float64)
        core_jaccards = np.empty(update_rounds, dtype=np.float64)
        core_count = max(
            1,
            int(np.floor(weights.size * 0.6 + 0.5)),
        )
    for round_index in range(update_rounds):
        fit_start = time.perf_counter()
        fit = weighted_kabsch_transform_batch(
            coords,
            target,
            weights,
            validate_inputs=False,
            dtype=compute_dtype,
        )
        fit_seconds += time.perf_counter() - fit_start
        residual_start = time.perf_counter()
        aligned = (
            coords @ np.swapaxes(fit.rotations, 1, 2)
            + fit.translations[:, None, :]
        )
        deltas = aligned - target[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        residual_seconds += time.perf_counter() - residual_start
        if trace is not None:
            squared = distances * distances
            objective_history[round_index] = float(
                np.sqrt(
                    np.mean(
                        np.einsum("bn,n->b", squared, weights)
                        / np.sum(weights)
                    )
                )
            )
        update_start = time.perf_counter()
        frame_weights = np.ones_like(distances)
        outliers = distances > delta
        np.divide(
            delta,
            distances,
            out=frame_weights,
            where=outliers,
        )
        update_seconds += time.perf_counter() - update_start
        regularization_start = time.perf_counter()
        frame_weights = apply_sequence_continuity_regularization_batch(
            frame_weights,
            strength=sequence_continuity_strength,
            passes=1,
            chain_ids=sequence_chain_ids,
            validate=False,
        )
        regularization_seconds += time.perf_counter() - regularization_start
        new_weights = np.mean(frame_weights, axis=0, dtype=compute_dtype)
        maximum = float(np.max(new_weights))
        if maximum <= 0.0:
            raise RuntimeError("weight update collapsed to all-zero weights")
        new_weights = np.clip(new_weights / maximum, 0.0, 1.0).astype(
            compute_dtype,
            copy=False,
        )
        per_frame_change = np.max(
            np.abs(frame_weights - weights[None, :]),
            axis=1,
        )
        converged_counts[round_index] = int(
            np.count_nonzero(per_frame_change <= convergence_tolerance)
        )
        changes[round_index] = float(np.max(np.abs(new_weights - weights)))
        if trace is not None:
            absolute_change = np.abs(new_weights - weights)
            l1_changes[round_index] = float(np.sum(absolute_change))
            mean_l1_changes[round_index] = float(np.mean(absolute_change))
            old_order = np.lexsort(
                (np.arange(weights.size), -weights)
            )
            new_order = np.lexsort(
                (np.arange(weights.size), -new_weights)
            )
            old_core = np.zeros(weights.size, dtype=bool)
            new_core = np.zeros(weights.size, dtype=bool)
            old_core[old_order[:core_count]] = True
            new_core[new_order[:core_count]] = True
            union = np.count_nonzero(old_core | new_core)
            core_jaccards[round_index] = (
                1.0
                if union == 0
                else float(
                    np.count_nonzero(old_core & new_core) / union
                )
            )
            weight_history[round_index + 1] = new_weights
        weights = new_weights
    if trace is not None:
        trace.update(
            {
                "weight_history": weight_history,
                "weighted_objective_history": objective_history,
                "round_weight_l1_changes": l1_changes,
                "round_weight_mean_l1_changes": mean_l1_changes,
                "round_weight_linf_changes": changes.copy(),
                "round_core_jaccards": core_jaccards,
            }
        )
    return (
        np.asarray(weights, dtype=np.float64),
        changes,
        converged_counts,
        {
            "representative_weighted_fit": fit_seconds,
            "representative_residual": residual_seconds,
            "representative_weight_update": update_seconds,
            "representative_sequence_regularization": regularization_seconds,
        },
    )


def exact_all_frame_fit(
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
) -> ExactAllFrameFitResult:
    """Compatibility wrapper around the shared fixed-weight CPU engine."""

    fixed_weights = _normalize_prior(weights)
    return fit_all_frames_fixed_weights(
        coords,
        reference,
        fixed_weights,
        chunk_size=chunk_size,
        return_aligned=return_aligned,
        compute_rmsd=compute_rmsd,
        compute_weighted_rmsd=compute_weighted_rmsd,
        compute_mean_residual=compute_mean_residual,
        compute_robust_rmsf=compute_robust_rmsf,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
    )


def solve_window_preconditioned_refinement(
    coords: np.ndarray,
    initial: InitialWeightResult,
    *,
    representative_count: int,
    update_rounds: int,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    chain_ids: np.ndarray | None = None,
    chunk_size: int = 2000,
    refinement_trace: dict[str, np.ndarray] | None = None,
    representative_indices: np.ndarray | None = None,
    refinement_precision: str = "float64",
    return_aligned: bool = False,
    compute_ordinary_rmsf: bool = True,
) -> WindowPreconditionedRefinementResult:
    """Run bounded global refinement and one explicit exact all-frame fit."""

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    selection_start = time.perf_counter()
    if representative_indices is None:
        selected = temporally_stratified_indices(
            values.shape[0],
            representative_count,
        )
    else:
        selected = np.asarray(representative_indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size < 1:
            raise ValueError("representative_indices must be a non-empty vector")
        if np.any(selected < 0) or np.any(selected >= values.shape[0]):
            raise ValueError("representative_indices are outside frame bounds")
        if np.any(selected[1:] <= selected[:-1]):
            raise ValueError(
                "representative_indices must be strictly increasing"
            )
    selection_seconds = time.perf_counter() - selection_start
    refinement_start = time.perf_counter()
    final_weights, changes, converged_counts, refinement_timing = (
        bounded_global_weight_refinement(
            values[selected],
            values[0],
            initial.weights,
            update_rounds=update_rounds,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
            sequence_chain_ids=chain_ids,
            trace=refinement_trace,
            precision=refinement_precision,
        )
    )
    refinement_seconds = time.perf_counter() - refinement_start
    exact_start = time.perf_counter()
    exact = exact_all_frame_fit(
        values,
        values[0],
        final_weights,
        chunk_size=chunk_size,
        return_aligned=return_aligned,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
    )
    exact_seconds = time.perf_counter() - exact_start
    effective_core = float(
        np.sum(final_weights) / max(float(np.max(final_weights)), 1e-12)
    )
    total_algorithm = time.perf_counter() - total_start
    return WindowPreconditionedRefinementResult(
        initializer=initial.initializer,
        initial_weights=initial.weights.copy(),
        final_weights=final_weights,
        representative_indices=selected,
        update_rounds=int(update_rounds),
        round_weight_changes=changes,
        converged_representative_counts=converged_counts,
        active_set_size=int(np.count_nonzero(final_weights >= 0.5)),
        effective_core_size=effective_core,
        fit=exact,
        timing={
            "scan": float(initial.scan_seconds),
            "representative_selection": selection_seconds,
            "limited_global_refinement": refinement_seconds,
            **refinement_timing,
            **exact.timing,
            "exact_all_frame_total": exact_seconds,
            "algorithm_excluding_scan": total_algorithm,
            "algorithm_including_scan": total_algorithm + initial.scan_seconds,
        },
        metadata={
            **initial.metadata,
            "output_contract": "exact_global_fit_after_bounded_refinement",
            "representative_count_requested": int(representative_count),
            "representative_count_actual": int(selected.size),
            "update_rounds_fixed": int(update_rounds),
            "early_stopping": False,
            "movement_index_called_rmsd": False,
            "float_dtype": refinement_precision,
            "all_frame_float_dtype": "float64",
            "ordinary_rmsf_computed": compute_ordinary_rmsf,
        },
    )
