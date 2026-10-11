"""Trajectory-mode robust alignment with optional warm start."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from .alignment import RigidTransform
from .contacts import build_contact_graph
from .robust import RobustAlignmentResult, solve_iterative_robust_alignment
from .trajectory import Trajectory
from .utils import ensure_xyz_frames


@dataclass(slots=True)
class TrajectoryFrameResult:
    """Per-frame output for trajectory-mode robust alignment."""

    frame_index: int
    reference_frame_index: int
    runtime_seconds: float
    iteration_count: int
    converged: bool
    used_warm_start: bool
    initial_transform_available: bool
    initial_weights_available: bool
    final_weight_change: float | None
    final_transform_change: float | None
    result: RobustAlignmentResult


@dataclass(slots=True)
class TrajectoryAlignmentResult:
    """Trajectory-level output for robust alignment across many frames."""

    reference_mode: str
    warm_start: bool
    frame_results: list[TrajectoryFrameResult]
    total_runtime_seconds: float
    timing: dict[str, float] | None = None

    @property
    def average_iterations_per_frame(self) -> float:
        """Return the mean number of solver iterations per frame."""

        if not self.frame_results:
            return 0.0
        return float(np.mean([frame.iteration_count for frame in self.frame_results]))

    @property
    def average_runtime_per_frame(self) -> float:
        """Return the mean wall-clock time per frame."""

        if not self.frame_results:
            return 0.0
        return float(np.mean([frame.runtime_seconds for frame in self.frame_results]))

    @property
    def mean_rmsd(self) -> float:
        """Return the mean final RMSD across frames."""

        if not self.frame_results:
            return 0.0
        return float(np.mean([frame.result.rmsd for frame in self.frame_results]))

    @property
    def mean_weighted_rmsd(self) -> float:
        """Return the mean final weighted RMSD across frames."""

        if not self.frame_results:
            return 0.0
        return float(np.mean([frame.result.weighted_rmsd for frame in self.frame_results]))

    @property
    def converged_fraction(self) -> float:
        """Return the fraction of frames that reached convergence."""

        if not self.frame_results:
            return 0.0
        return float(np.mean([frame.converged for frame in self.frame_results]))


def _ensure_trajectory(trajectory: Trajectory | np.ndarray) -> Trajectory:
    """Normalize a trajectory input into the Trajectory container."""

    if isinstance(trajectory, Trajectory):
        return trajectory
    return Trajectory(coords=ensure_xyz_frames(trajectory))


def _resolve_reference_frame_index(frame_index: int, reference_mode: str) -> int:
    """Resolve the reference-frame index used for one frame."""

    if reference_mode == "first_frame":
        return 0
    if reference_mode == "previous_frame":
        return 0 if frame_index == 0 else frame_index - 1
    raise ValueError("reference_mode must be 'first_frame' or 'previous_frame'")


def solve_trajectory_robust_alignment(
    trajectory: Trajectory | np.ndarray,
    *,
    reference_mode: str = "first_frame",
    warm_start: bool = False,
    collect_timing: bool = False,
    timing_output_path: str | Path | None = None,
    **robust_kwargs: object,
) -> TrajectoryAlignmentResult:
    """Run the robust solver over a trajectory with optional warm start.

    Supported reference modes:

    - `first_frame`: every frame is aligned against frame 0
    - `previous_frame`: frame `t` is aligned against frame `t - 1`

    Warm start reuses the previous frame's weights and rigid transform in both
    modes. When the reference stays fixed to frame 0, that transform is often
    directly compatible and therefore more beneficial. In `previous_frame`
    mode the transform is only an approximate guess, so the gain may be much
    smaller.
    """

    prepare_start = time.perf_counter()
    trajectory = _ensure_trajectory(trajectory)
    prepare_seconds = time.perf_counter() - prepare_start
    if trajectory.n_frames == 0:
        raise ValueError("trajectory must contain at least one frame")

    frame_results: list[TrajectoryFrameResult] = []
    previous_result: RobustAlignmentResult | None = None
    contact_graph_cache: dict[int, object] = {}
    contact_strength = float(robust_kwargs.get("contact_consistency_strength", 0.0))
    user_diagnostic_callback = robust_kwargs.pop("diagnostic_callback", None)
    timing = None
    if collect_timing:
        timing = {
            "total_runtime_sec": 0.0,
            "prepare_input_sec": prepare_seconds,
            "initialize_alignment_sec": 0.0,
            "weighted_fit_sec": 0.0,
            "residual_update_sec": 0.0,
            "weight_update_sec": 0.0,
            "regularization_sec": 0.0,
            "convergence_check_sec": 0.0,
            "rmsf_sec": 0.0,
            "flexible_segments_sec": 0.0,
            "output_build_sec": 0.0,
            "output_write_sec": 0.0,
            "unified_metric_sec": 0.0,
        }

    total_start = time.perf_counter()
    for frame_index, mobile in enumerate(trajectory.coords):
        reference_frame_index = _resolve_reference_frame_index(frame_index, reference_mode)
        reference = trajectory.coords[reference_frame_index]

        initial_weights = None
        initial_transform = None
        if warm_start and previous_result is not None:
            initial_weights = previous_result.weights
            initial_transform = RigidTransform(
                rotation=previous_result.rotation.copy(),
                translation=previous_result.translation.copy(),
            )

        frame_kwargs = dict(robust_kwargs)
        if collect_timing or user_diagnostic_callback is not None:
            def timing_callback(payload: dict[str, object]) -> None:
                if timing is not None:
                    timing["weighted_fit_sec"] += float(payload.get("weighted_kabsch_or_refit_seconds", 0.0))
                    timing["residual_update_sec"] += float(payload.get("residual_computation_seconds", 0.0))
                    timing["weight_update_sec"] += float(payload.get("weight_update_seconds", 0.0))
                    timing["regularization_sec"] += float(payload.get("sequence_regularization_seconds", 0.0))
                    timing["regularization_sec"] += float(payload.get("contact_regularization_seconds", 0.0))
                    timing["convergence_check_sec"] += float(payload.get("convergence_check_seconds", 0.0))
                if user_diagnostic_callback is not None:
                    user_diagnostic_callback(payload)

            frame_kwargs["diagnostic_callback"] = timing_callback
        if contact_strength > 0.0 and "contact_graph" not in frame_kwargs:
            cached_graph = contact_graph_cache.get(reference_frame_index)
            if cached_graph is None:
                cached_graph = build_contact_graph(
                    reference,
                    cutoff=float(frame_kwargs.get("contact_distance_cutoff", 8.0)),
                    min_sequence_separation=int(frame_kwargs.get("contact_min_sequence_separation", 2)),
                    max_neighbors=frame_kwargs.get("contact_max_neighbors", 8),
                )
                contact_graph_cache[reference_frame_index] = cached_graph
            frame_kwargs["contact_graph"] = cached_graph

        frame_start = time.perf_counter()
        result = solve_iterative_robust_alignment(
            mobile,
            reference,
            initial_weights=initial_weights,
            initial_transform=initial_transform,
            contact_graph_validated="contact_graph" in frame_kwargs,
            **frame_kwargs,
        )
        frame_runtime = time.perf_counter() - frame_start

        frame_results.append(
            TrajectoryFrameResult(
                frame_index=frame_index,
                reference_frame_index=reference_frame_index,
                runtime_seconds=frame_runtime,
                iteration_count=result.iteration_count,
                converged=result.converged,
                used_warm_start=bool(warm_start and previous_result is not None),
                initial_transform_available=initial_transform is not None,
                initial_weights_available=initial_weights is not None,
                final_weight_change=result.final_weight_change,
                final_transform_change=result.final_transform_change,
                result=result,
            )
        )
        previous_result = result

    total_runtime = time.perf_counter() - total_start
    if timing is not None:
        timing["total_runtime_sec"] = total_runtime + prepare_seconds
        if timing_output_path is not None:
            path = Path(timing_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    return TrajectoryAlignmentResult(
        reference_mode=reference_mode,
        warm_start=warm_start,
        frame_results=frame_results,
        total_runtime_seconds=total_runtime,
        timing=timing,
    )


def summarize_trajectory_solution_difference(
    first: TrajectoryAlignmentResult,
    second: TrajectoryAlignmentResult,
) -> dict[str, float]:
    """Summarize how different two trajectory-mode solutions are."""

    if len(first.frame_results) != len(second.frame_results):
        raise ValueError("trajectory solutions must contain the same number of frames")

    aligned_differences: list[float] = []
    weight_differences: list[float] = []
    for first_frame, second_frame in zip(first.frame_results, second.frame_results, strict=True):
        aligned_delta = first_frame.result.aligned_coords - second_frame.result.aligned_coords
        aligned_differences.append(float(np.sqrt(np.mean(np.sum(aligned_delta**2, axis=1)))))
        weight_differences.append(float(np.max(np.abs(first_frame.result.weights - second_frame.result.weights))))
    return {
        "mean_aligned_coordinate_difference": float(np.mean(aligned_differences)) if aligned_differences else 0.0,
        "max_aligned_coordinate_difference": float(np.max(aligned_differences)) if aligned_differences else 0.0,
        "mean_weight_linf_difference": float(np.mean(weight_differences)) if weight_differences else 0.0,
        "max_weight_linf_difference": float(np.max(weight_differences)) if weight_differences else 0.0,
    }
