"""Optional diagnostics helpers for robust trajectory alignment."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .trajectory import Trajectory
from .trajectory_mode import TrajectoryAlignmentResult, solve_trajectory_robust_alignment


TRACE_FIELDS = [
    "system",
    "run_id",
    "frame_index",
    "iteration",
    "objective_value",
    "pseudo_objective_value",
    "weighted_RMSD",
    "unified_posthoc_RMSD",
    "mean_residual",
    "median_residual",
    "max_residual",
    "mean_weight",
    "min_weight",
    "max_weight",
    "effective_core_size",
    "delta_weight_L2",
    "delta_weight_max",
    "delta_transform_rotation_norm",
    "delta_transform_translation_norm",
    "converged",
    "runtime_iteration_seconds",
]

RUNTIME_PROFILE_FIELDS = [
    "input_parsing_seconds",
    "initial_alignment_seconds",
    "residual_computation_seconds",
    "weight_update_seconds",
    "sequence_regularization_seconds",
    "contact_regularization_seconds",
    "weighted_kabsch_or_refit_seconds",
    "convergence_check_seconds",
    "export_seconds",
    "rmsf_computation_seconds",
    "flexible_segment_detection_seconds",
    "total_algorithm_seconds",
    "total_with_export_seconds",
]


@dataclass(frozen=True)
class DiagnosticTrajectoryResult:
    """Trajectory result plus optional diagnostic tables."""

    result: TrajectoryAlignmentResult
    trace_rows: list[dict[str, object]]
    runtime_profile: dict[str, object]


def solve_trajectory_with_optional_diagnostics(
    trajectory: Trajectory | np.ndarray,
    *,
    diagnostics: bool = False,
    system: str = "unknown",
    run_id: str = "run0",
    **kwargs: object,
) -> DiagnosticTrajectoryResult:
    """Run robust trajectory alignment and optionally derive diagnostics.

    When `diagnostics=False`, this is a thin wrapper around the existing
    trajectory solver. When `diagnostics=True`, the final solver output is not
    altered; trace rows are derived after the alignment has completed.
    """

    start = time.perf_counter()
    result = solve_trajectory_robust_alignment(trajectory, **kwargs)
    total = time.perf_counter() - start
    if not diagnostics:
        return DiagnosticTrajectoryResult(result=result, trace_rows=[], runtime_profile={})
    trace_rows = trace_rows_from_result(result, system=system, run_id=run_id)
    profile = empty_runtime_profile()
    profile["total_algorithm_seconds"] = result.total_runtime_seconds
    profile["total_with_export_seconds"] = total
    profile["export_seconds"] = 0.0
    return DiagnosticTrajectoryResult(result=result, trace_rows=trace_rows, runtime_profile=profile)


def trace_rows_from_result(
    result: TrajectoryAlignmentResult,
    *,
    system: str,
    run_id: str,
) -> list[dict[str, object]]:
    """Derive per-iteration trace rows from a trajectory result."""

    rows: list[dict[str, object]] = []
    for frame in result.frame_results:
        weights = frame.result.weights
        residuals = frame.result.residual_distances
        effective_core_size = float(np.sum(weights))
        history = frame.result.history or []
        for entry in history:
            rows.append(
                {
                    "system": system,
                    "run_id": run_id,
                    "frame_index": frame.frame_index,
                    "iteration": entry.iteration,
                    "objective_value": "",
                    "pseudo_objective_value": entry.weighted_rmsd,
                    "weighted_RMSD": entry.weighted_rmsd,
                    "unified_posthoc_RMSD": frame.result.rmsd if entry.iteration == frame.iteration_count else "",
                    "mean_residual": float(np.mean(residuals)),
                    "median_residual": float(np.median(residuals)),
                    "max_residual": float(np.max(residuals)),
                    "mean_weight": float(np.mean(weights)),
                    "min_weight": float(np.min(weights)),
                    "max_weight": float(np.max(weights)),
                    "effective_core_size": effective_core_size,
                    "delta_weight_L2": entry.weight_change,
                    "delta_weight_max": frame.final_weight_change if entry.iteration == frame.iteration_count else "",
                    "delta_transform_rotation_norm": "",
                    "delta_transform_translation_norm": frame.final_transform_change if entry.iteration == frame.iteration_count else "",
                    "converged": frame.converged if entry.iteration == frame.iteration_count else "",
                    "runtime_iteration_seconds": frame.runtime_seconds / max(frame.iteration_count, 1),
                }
            )
    return rows


def empty_runtime_profile() -> dict[str, object]:
    """Return a runtime-profile skeleton with all required fields."""

    return {field: 0.0 for field in RUNTIME_PROFILE_FIELDS}


def final_solution_signature(result: TrajectoryAlignmentResult) -> dict[str, object]:
    """Return deterministic values useful for diagnostics equality tests."""

    weights = np.asarray([frame.result.weights for frame in result.frame_results], dtype=float)
    rmsds = np.asarray([frame.result.rmsd for frame in result.frame_results], dtype=float)
    return {
        "mean_rmsd": float(np.mean(rmsds)),
        "weights": weights,
    }
