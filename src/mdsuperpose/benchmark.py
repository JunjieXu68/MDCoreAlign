"""Benchmark helpers for ordinary, robust, and external alignment methods."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .alignment import solve_rigid_alignment
from .contacts import build_contact_graph, contact_graph_disagreement
from .core_selection import select_core_by_target_fraction
from .mdlovofit import run_mdlovofit_on_coordinates
from .metrics import mean_rmsd
from .robust import solve_iterative_hard_trim_alignment
from .theseus import run_theseus_on_coordinates
from .trajectory import Trajectory
from .trajectory_mode import solve_trajectory_robust_alignment
from .trajectory_mode import summarize_trajectory_solution_difference
from .utils import ensure_xyz_frames


@dataclass(slots=True)
class BenchmarkResult:
    """Standardized result record used by benchmark scripts."""

    method_name: str
    runtime_seconds: float
    algorithm_runtime_seconds: float | None
    overhead_seconds: float | None
    rmsd: float | None
    weighted_rmsd: float | None
    iteration_count: float | None
    core_size: int | None
    core_fraction: float | None
    parsing_success: bool | None
    transform_if_available: dict[str, Any] | None
    extra_metadata: dict[str, Any]


_ROW_FIELDS = [
    "benchmark_mode",
    "dataset_name",
    "method_name",
    "repeat_index",
    "preprocessing_steps",
    "runtime_seconds",
    "algorithm_runtime_seconds",
    "overhead_seconds",
    "rmsd",
    "unified_posthoc_rmsd",
    "mdlovofit_native_rmsd",
    "mdlovofit_native_all_atom_rmsd",
    "mdlovofit_native_least_mobile_rmsd",
    "mdlovofit_native_most_mobile_rmsd",
    "theseus_native_classical_pairwise_rmsd",
    "theseus_native_maximum_likelihood_rmsd",
    "theseus_native_least_squares_sigma",
    "theseus_native_maximum_likelihood_sigma",
    "theseus_output_core_atom_count",
    "theseus_output_core_atom_fraction",
    "mdlovofit_reported_fraction",
    "mdlovofit_reported_total_atoms",
    "mdlovofit_reported_selection_atoms",
    "mdlovofit_reported_total_atoms_found",
    "mdlovofit_frame_rmsd_row_count",
    "mdlovofit_output_subset_atom_count",
    "mdlovofit_output_subset_atom_fraction",
    "weighted_rmsd",
    "iteration_count",
    "core_size",
    "core_fraction",
    "parsing_success",
]

_NUMERIC_SUMMARY_FIELDS = [
    "runtime_seconds",
    "algorithm_runtime_seconds",
    "overhead_seconds",
    "rmsd",
    "unified_posthoc_rmsd",
    "mdlovofit_native_rmsd",
    "mdlovofit_native_all_atom_rmsd",
    "mdlovofit_native_least_mobile_rmsd",
    "mdlovofit_native_most_mobile_rmsd",
    "theseus_native_classical_pairwise_rmsd",
    "theseus_native_maximum_likelihood_rmsd",
    "theseus_native_least_squares_sigma",
    "theseus_native_maximum_likelihood_sigma",
    "theseus_output_core_atom_count",
    "theseus_output_core_atom_fraction",
    "mdlovofit_reported_fraction",
    "mdlovofit_reported_total_atoms",
    "mdlovofit_reported_selection_atoms",
    "mdlovofit_reported_total_atoms_found",
    "mdlovofit_frame_rmsd_row_count",
    "mdlovofit_output_subset_atom_count",
    "mdlovofit_output_subset_atom_fraction",
    "weighted_rmsd",
    "iteration_count",
    "core_size",
    "core_fraction",
]


def _ensure_trajectory(trajectory: Trajectory | np.ndarray) -> Trajectory:
    """Normalize trajectory-like input."""

    return trajectory if isinstance(trajectory, Trajectory) else Trajectory(coords=trajectory)


def _mean_frame_rmsd(aligned_coords: np.ndarray, reference: np.ndarray) -> float:
    """Compute the mean per-frame RMSD to a fixed reference."""

    return mean_rmsd(aligned_coords, reference)


def _as_jsonable(value: Any) -> Any:
    """Convert NumPy and Path objects into JSON-friendly values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _mean_or_none(values: list[float]) -> float | None:
    """Return the mean of non-missing values."""

    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(finite))


def _std_or_none(values: list[float]) -> float | None:
    """Return population standard deviation of non-missing values."""

    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.std(finite, ddof=0))


def _coerce_float(value: Any) -> float | None:
    """Best-effort float conversion for summary tables."""

    if value in ("", None):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return converted


def _stack_aligned_frames(results: list[Any]) -> np.ndarray:
    """Stack per-frame alignment results into one trajectory array."""

    return np.asarray([result.aligned_coords for result in results], dtype=float)


def _core_transition_count(core_mask: np.ndarray) -> int:
    """Count sequence-adjacent changes in a binary core mask."""

    core_mask = np.asarray(core_mask, dtype=bool)
    if core_mask.shape[0] < 2:
        return 0
    return int(np.count_nonzero(core_mask[1:] != core_mask[:-1]))


def benchmark_ordinary_alignment(trajectory: Trajectory | np.ndarray) -> BenchmarkResult:
    """Benchmark ordinary Kabsch alignment of every frame to frame 0."""

    trajectory = _ensure_trajectory(trajectory)
    reference = trajectory.coords[0]
    start = time.perf_counter()
    aligned = []
    for frame in trajectory.coords:
        aligned.append(solve_rigid_alignment(frame, reference).aligned_coords)
    aligned_coords = np.asarray(aligned, dtype=float)
    runtime = time.perf_counter() - start
    return BenchmarkResult(
        method_name="ordinary_alignment",
        runtime_seconds=runtime,
        algorithm_runtime_seconds=runtime,
        overhead_seconds=0.0,
        rmsd=_mean_frame_rmsd(aligned_coords, reference),
        weighted_rmsd=None,
        iteration_count=None,
        core_size=None,
        core_fraction=None,
        parsing_success=None,
        transform_if_available=None,
        extra_metadata={"n_frames": trajectory.n_frames, "n_atoms": trajectory.n_atoms},
    )


def benchmark_soft_weight_robust_alignment(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    warm_start: bool = False,
    frame_order: str = "input",
    reference_mode: str = "first_frame",
    **robust_kwargs: object,
) -> BenchmarkResult:
    """Benchmark the soft-weight robust trajectory alignment pipeline."""

    trajectory = _ensure_trajectory(trajectory)
    method_name = str(robust_kwargs.pop("method_name", "soft_weight_robust_alignment"))
    ordering_start = time.perf_counter()
    frame_permutation = _frame_permutation(trajectory.coords, mode=frame_order)
    ordering_runtime = time.perf_counter() - ordering_start

    solver_start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode=reference_mode,
        warm_start=warm_start,
        record_history=bool(robust_kwargs.pop("record_history", False)),
        **robust_kwargs,
    )
    solver_runtime = time.perf_counter() - solver_start
    runtime = ordering_runtime + solver_runtime

    mean_weights = np.mean(
        np.asarray([frame.result.weights for frame in result.frame_results], dtype=float),
        axis=0,
    )
    core = select_core_by_target_fraction(mean_weights, core_fraction)
    metadata: dict[str, Any] = {
        "n_frames": trajectory.n_frames,
        "n_atoms": trajectory.n_atoms,
        "warm_start": warm_start,
        "reference_mode": reference_mode,
        "frame_order": frame_order,
        "frame_permutation": frame_permutation.tolist(),
        "ordering_runtime_seconds": ordering_runtime,
        "solver_runtime_seconds": solver_runtime,
        "per_frame_iterations": [frame.iteration_count for frame in result.frame_results],
        "core_transition_count": _core_transition_count(core.core_mask),
        "selected_core_indices": core.core_indices.tolist(),
    }
    for key in (
        "sequence_continuity_strength",
        "sequence_continuity_passes",
        "contact_consistency_strength",
        "contact_consistency_passes",
    ):
        if key in robust_kwargs:
            metadata[key] = robust_kwargs[key]

    contact_strength = float(robust_kwargs.get("contact_consistency_strength", 0.0))
    if contact_strength > 0.0:
        graph = build_contact_graph(
            trajectory.coords[0],
            cutoff=float(robust_kwargs.get("contact_distance_cutoff", 8.0)),
            min_sequence_separation=int(robust_kwargs.get("contact_min_sequence_separation", 2)),
            max_neighbors=robust_kwargs.get("contact_max_neighbors", 8),
        )
        metadata["contact_graph_edge_count"] = graph.edge_count
        metadata["contact_graph_mean_degree"] = graph.mean_degree
        metadata["contact_graph_disagreement"] = contact_graph_disagreement(mean_weights, graph)

    return BenchmarkResult(
        method_name=method_name,
        runtime_seconds=runtime,
        algorithm_runtime_seconds=runtime,
        overhead_seconds=0.0,
        rmsd=result.mean_rmsd,
        weighted_rmsd=result.mean_weighted_rmsd,
        iteration_count=result.average_iterations_per_frame,
        core_size=core.core_size,
        core_fraction=core.core_fraction,
        parsing_success=None,
        transform_if_available=None,
        extra_metadata=metadata,
    )


def benchmark_trajectory_robust_alignment(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    reference_mode: str = "first_frame",
    warm_start: bool = False,
    **robust_kwargs: object,
) -> BenchmarkResult:
    """Benchmark trajectory-mode robust alignment with optional warm start."""

    return benchmark_soft_weight_robust_alignment(
        trajectory,
        core_fraction=core_fraction,
        reference_mode=reference_mode,
        warm_start=warm_start,
        **robust_kwargs,
    )


def benchmark_hard_trim_control_alignment(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    **kwargs: object,
) -> BenchmarkResult:
    """Benchmark a hard-trim control over trajectory frames."""

    trajectory = _ensure_trajectory(trajectory)
    reference = trajectory.coords[0]
    start = time.perf_counter()
    frame_results = [
        solve_iterative_hard_trim_alignment(
            frame,
            reference,
            core_fraction=core_fraction,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
        )
        for frame in trajectory.coords
    ]
    runtime = time.perf_counter() - start
    aligned = _stack_aligned_frames(frame_results)
    weights = np.mean(np.asarray([result.weights for result in frame_results], dtype=float), axis=0)
    core = select_core_by_target_fraction(weights, core_fraction)
    return BenchmarkResult(
        method_name="hard_trim_control_alignment",
        runtime_seconds=runtime,
        algorithm_runtime_seconds=runtime,
        overhead_seconds=0.0,
        rmsd=_mean_frame_rmsd(aligned, reference),
        weighted_rmsd=float(np.mean([result.weighted_rmsd for result in frame_results])),
        iteration_count=float(np.mean([result.iteration_count for result in frame_results])),
        core_size=core.core_size,
        core_fraction=core.core_fraction,
        parsing_success=None,
        transform_if_available=None,
        extra_metadata={
            "warm_start": False,
            "selected_core_indices": core.core_indices.tolist(),
            "core_transition_count": _core_transition_count(core.core_mask),
        },
    )


def benchmark_external_mdlovofit(
    trajectory: Trajectory | np.ndarray,
    *,
    fraction: float = 0.6,
    workdir: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> BenchmarkResult:
    """Benchmark the real external MDLovoFit executable."""

    trajectory = _ensure_trajectory(trajectory)
    run = run_mdlovofit_on_coordinates(
        trajectory.coords,
        fraction=fraction,
        workdir=workdir,
        executable=executable,
        config_path=config_path,
        timeout_seconds=timeout_seconds,
    )
    parsed = run.parsed_output
    rmsd = _mean_frame_rmsd(run.aligned_coords, trajectory.coords[0]) if run.aligned_coords is not None else None
    return BenchmarkResult(
        method_name="external_mdlovofit",
        runtime_seconds=run.total_runtime_seconds,
        algorithm_runtime_seconds=None,
        overhead_seconds=run.input_write_seconds + run.output_read_seconds,
        rmsd=rmsd,
        weighted_rmsd=None,
        iteration_count=None,
        core_size=run.output_subset_atom_count,
        core_fraction=run.output_subset_atom_fraction,
        parsing_success=run.return_code == 0,
        transform_if_available=None,
        extra_metadata={
            "requested_fraction": fraction,
            "stdout": run.stdout,
            "stderr": run.stderr,
            "command": run.command,
            "output_path": str(run.output_path),
            "input_write_seconds": run.input_write_seconds,
            "subprocess_runtime_seconds": run.subprocess_runtime_seconds,
            "output_read_seconds": run.output_read_seconds,
            "mdlovofit_native_all_atom_rmsd": parsed.average_rmsd_all_atoms,
            "mdlovofit_native_least_mobile_rmsd": parsed.average_rmsd_least_mobile,
            "mdlovofit_native_most_mobile_rmsd": parsed.average_rmsd_most_mobile,
            "mdlovofit_reported_fraction": parsed.fraction_of_best_aligned_atoms,
            "mdlovofit_reported_total_atoms": parsed.total_number_of_atoms,
            "mdlovofit_reported_selection_atoms": parsed.number_of_atoms_of_selection,
            "mdlovofit_reported_total_atoms_found": parsed.total_number_of_atoms_found,
            "mdlovofit_frame_rmsd_row_count": parsed.frame_rmsd_row_count,
            "mdlovofit_output_subset_atom_count": run.output_subset_atom_count,
            "mdlovofit_output_subset_atom_fraction": run.output_subset_atom_fraction,
        },
    )


def benchmark_external_theseus(
    trajectory: Trajectory | np.ndarray,
    *,
    workdir: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> BenchmarkResult:
    """Benchmark the real external THESEUS executable."""

    trajectory = _ensure_trajectory(trajectory)
    run = run_theseus_on_coordinates(
        trajectory.coords,
        workdir=workdir,
        executable=executable,
        config_path=config_path,
        timeout_seconds=timeout_seconds,
    )
    parsed = run.parsed_output
    rmsd = None
    consensus_orientation_rmsd = None
    if run.aligned_coords is not None:
        # THESEUS writes the ensemble in its own consensus coordinate frame.
        # Apply one common rigid transform to orient that consensus frame to
        # the benchmark reference; do not independently refit each frame.
        consensus = np.mean(run.aligned_coords, axis=0)
        orientation = solve_rigid_alignment(
            consensus, trajectory.coords[0]
        )
        oriented = (
            run.aligned_coords @ orientation.rotation.T
            + orientation.translation
        )
        rmsd = _mean_frame_rmsd(oriented, trajectory.coords[0])
        consensus_orientation_rmsd = orientation.rmsd
    return BenchmarkResult(
        method_name="external_theseus",
        runtime_seconds=run.total_runtime_seconds,
        algorithm_runtime_seconds=None,
        overhead_seconds=run.input_write_seconds + run.output_read_seconds,
        rmsd=rmsd,
        weighted_rmsd=None,
        iteration_count=None,
        core_size=run.output_core_atom_count,
        core_fraction=run.output_core_atom_fraction,
        parsing_success=run.return_code == 0,
        transform_if_available=None,
        extra_metadata={
            "stdout": run.stdout,
            "stderr": run.stderr,
            "command": run.command,
            "aligned_pdb_path": str(run.aligned_pdb_path),
            "input_write_seconds": run.input_write_seconds,
            "subprocess_runtime_seconds": run.subprocess_runtime_seconds,
            "output_read_seconds": run.output_read_seconds,
            "reported_classical_pairwise_rmsd": parsed.classical_pairwise_rmsd,
            "reported_maximum_likelihood_rmsd": parsed.maximum_likelihood_rmsd,
            "reported_least_squares_sigma": parsed.least_squares_sigma,
            "reported_maximum_likelihood_sigma": parsed.maximum_likelihood_sigma,
            "output_core_atom_count": run.output_core_atom_count,
            "output_core_atom_fraction": run.output_core_atom_fraction,
            "unified_rmsd_orientation": (
                "one_common_consensus_to_reference_rigid_transform"
            ),
            "consensus_orientation_rmsd": consensus_orientation_rmsd,
        },
    )


def compare_methods_on_trajectory(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    include_external_mdlovofit: bool = False,
    mdlovofit_workdir: str | Path | None = None,
    mdlovofit_executable: str | Path | None = None,
    include_external_theseus: bool = False,
    theseus_workdir: str | Path | None = None,
    theseus_executable: str | Path | None = None,
    **robust_kwargs: object,
) -> list[BenchmarkResult]:
    """Compare ordinary, soft robust, and optional external baselines."""

    trajectory = _ensure_trajectory(trajectory)
    results = [
        benchmark_ordinary_alignment(trajectory),
        benchmark_soft_weight_robust_alignment(trajectory, core_fraction=core_fraction, **robust_kwargs),
    ]
    if include_external_mdlovofit:
        if mdlovofit_workdir is None:
            raise ValueError("mdlovofit_workdir is required when include_external_mdlovofit=True")
        results.append(
            benchmark_external_mdlovofit(
                trajectory,
                fraction=core_fraction,
                workdir=mdlovofit_workdir,
                executable=mdlovofit_executable,
            )
        )
    if include_external_theseus:
        if theseus_workdir is None:
            raise ValueError("theseus_workdir is required when include_external_theseus=True")
        results.append(
            benchmark_external_theseus(
                trajectory,
                workdir=theseus_workdir,
                executable=theseus_executable,
            )
        )
    return results


def compare_methods_with_soft_configurations(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    soft_configurations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    include_ordinary: bool = True,
    include_external_mdlovofit: bool = False,
    mdlovofit_workdir: str | Path | None = None,
    mdlovofit_executable: str | Path | None = None,
    include_external_theseus: bool = False,
    theseus_workdir: str | Path | None = None,
    theseus_executable: str | Path | None = None,
    **shared_robust_kwargs: object,
) -> list[BenchmarkResult]:
    """Compare ordinary alignment, one or more soft configurations, and externals."""

    trajectory = _ensure_trajectory(trajectory)
    results: list[BenchmarkResult] = []
    if include_ordinary:
        results.append(benchmark_ordinary_alignment(trajectory))

    if soft_configurations is None:
        soft_configurations = ({},)
    for config in soft_configurations:
        kwargs = dict(shared_robust_kwargs)
        kwargs.update(config)
        method_label = kwargs.pop("method_name", None)
        local_core_fraction = float(kwargs.pop("core_fraction", core_fraction))
        result = benchmark_soft_weight_robust_alignment(
            trajectory,
            core_fraction=local_core_fraction,
            **kwargs,
        )
        if method_label is not None:
            result.method_name = str(method_label)
        results.append(result)

    if include_external_mdlovofit:
        if mdlovofit_workdir is None:
            raise ValueError("mdlovofit_workdir is required when include_external_mdlovofit=True")
        results.append(
            benchmark_external_mdlovofit(
                trajectory,
                fraction=core_fraction,
                workdir=mdlovofit_workdir,
                executable=mdlovofit_executable,
            )
        )
    if include_external_theseus:
        if theseus_workdir is None:
            raise ValueError("theseus_workdir is required when include_external_theseus=True")
        results.append(
            benchmark_external_theseus(
                trajectory,
                workdir=theseus_workdir,
                executable=theseus_executable,
            )
        )
    return results


def compare_trajectory_warm_start(
    trajectory: Trajectory | np.ndarray,
    *,
    core_fraction: float = 0.6,
    reference_mode: str = "first_frame",
    **robust_kwargs: object,
) -> tuple[list[BenchmarkResult], dict[str, float]]:
    """Compare cold-start and warm-start trajectory-mode robust alignment."""

    trajectory = _ensure_trajectory(trajectory)
    cold_solution = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode=reference_mode,
        warm_start=False,
        **robust_kwargs,
    )
    warm_solution = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode=reference_mode,
        warm_start=True,
        **robust_kwargs,
    )
    difference = summarize_trajectory_solution_difference(cold_solution, warm_solution)

    results = []
    for label, solution in (("trajectory_cold_start", cold_solution), ("trajectory_warm_start", warm_solution)):
        weights = np.mean(np.asarray([frame.result.weights for frame in solution.frame_results], dtype=float), axis=0)
        core = select_core_by_target_fraction(weights, core_fraction)
        results.append(
            BenchmarkResult(
                method_name=label,
                runtime_seconds=solution.total_runtime_seconds,
                algorithm_runtime_seconds=solution.total_runtime_seconds,
                overhead_seconds=0.0,
                rmsd=solution.mean_rmsd,
                weighted_rmsd=solution.mean_weighted_rmsd,
                iteration_count=solution.average_iterations_per_frame,
                core_size=core.core_size,
                core_fraction=core.core_fraction,
                parsing_success=None,
                transform_if_available=None,
                extra_metadata={
                    "reference_mode": reference_mode,
                    "warm_start": solution.warm_start,
                    "per_frame_iterations": [frame.iteration_count for frame in solution.frame_results],
                    "core_transition_count": _core_transition_count(core.core_mask),
                },
            )
        )
    return results, difference


def benchmark_result_to_record(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize one benchmark result without benchmark-run context."""

    record = asdict(result)
    return _as_jsonable(record)


def benchmark_result_to_row(
    result: BenchmarkResult,
    *,
    benchmark_mode: str,
    dataset_name: str,
    repeat_index: int | None = None,
    preprocessing_steps: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Flatten one result into a CSV row."""

    metadata = result.extra_metadata or {}
    row = {field: None for field in _ROW_FIELDS}
    row.update(
        {
            "benchmark_mode": benchmark_mode,
            "dataset_name": dataset_name,
            "method_name": result.method_name,
            "repeat_index": repeat_index,
            "preprocessing_steps": "; ".join(preprocessing_steps or ()),
            "runtime_seconds": result.runtime_seconds,
            "algorithm_runtime_seconds": result.algorithm_runtime_seconds,
            "overhead_seconds": result.overhead_seconds,
            "rmsd": result.rmsd,
            "unified_posthoc_rmsd": result.rmsd,
            "weighted_rmsd": result.weighted_rmsd,
            "iteration_count": result.iteration_count,
            "core_size": result.core_size,
            "core_fraction": result.core_fraction,
            "parsing_success": result.parsing_success,
        }
    )

    row["mdlovofit_native_rmsd"] = metadata.get("mdlovofit_native_all_atom_rmsd")
    row["mdlovofit_native_all_atom_rmsd"] = metadata.get("mdlovofit_native_all_atom_rmsd")
    row["mdlovofit_native_least_mobile_rmsd"] = metadata.get("mdlovofit_native_least_mobile_rmsd")
    row["mdlovofit_native_most_mobile_rmsd"] = metadata.get("mdlovofit_native_most_mobile_rmsd")
    row["mdlovofit_reported_fraction"] = metadata.get("mdlovofit_reported_fraction")
    row["mdlovofit_reported_total_atoms"] = metadata.get("mdlovofit_reported_total_atoms")
    row["mdlovofit_reported_selection_atoms"] = metadata.get("mdlovofit_reported_selection_atoms")
    row["mdlovofit_reported_total_atoms_found"] = metadata.get("mdlovofit_reported_total_atoms_found")
    row["mdlovofit_frame_rmsd_row_count"] = metadata.get("mdlovofit_frame_rmsd_row_count")
    row["mdlovofit_output_subset_atom_count"] = metadata.get("mdlovofit_output_subset_atom_count")
    row["mdlovofit_output_subset_atom_fraction"] = metadata.get("mdlovofit_output_subset_atom_fraction")
    row["theseus_native_classical_pairwise_rmsd"] = metadata.get("reported_classical_pairwise_rmsd")
    row["theseus_native_maximum_likelihood_rmsd"] = metadata.get("reported_maximum_likelihood_rmsd")
    row["theseus_native_least_squares_sigma"] = metadata.get("reported_least_squares_sigma")
    row["theseus_native_maximum_likelihood_sigma"] = metadata.get("reported_maximum_likelihood_sigma")
    row["theseus_output_core_atom_count"] = metadata.get("output_core_atom_count")
    row["theseus_output_core_atom_fraction"] = metadata.get("output_core_atom_fraction")
    return _as_jsonable(row)


def benchmark_results_to_rows(
    results: list[BenchmarkResult] | tuple[BenchmarkResult, ...],
    *,
    benchmark_mode: str,
    dataset_name: str,
    repeat_index: int | None = None,
    preprocessing_steps: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Flatten multiple benchmark results into CSV rows."""

    return [
        benchmark_result_to_row(
            result,
            benchmark_mode=benchmark_mode,
            dataset_name=dataset_name,
            repeat_index=repeat_index,
            preprocessing_steps=preprocessing_steps,
        )
        for result in results
    ]


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize benchmark rows by mode, dataset, and method."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("benchmark_mode")), str(row.get("dataset_name")), str(row.get("method_name")))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (mode, dataset, method), group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "benchmark_mode": mode,
            "dataset_name": dataset,
            "method_name": method,
            "n_repeats": len(group_rows),
        }
        for field in _NUMERIC_SUMMARY_FIELDS:
            values = [_coerce_float(row.get(field)) for row in group_rows]
            summary[f"{field}_mean"] = _mean_or_none(values)
            summary[f"{field}_std"] = _std_or_none(values)

        parsing_values = [row.get("parsing_success") for row in group_rows if row.get("parsing_success") is not None]
        if parsing_values:
            summary["parsing_success_rate"] = float(np.mean([bool(value) for value in parsing_values]))
        else:
            summary["parsing_success_rate"] = None
        summary_rows.append(summary)
    return summary_rows


def write_benchmark_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write benchmark rows to CSV, preserving a stable field order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields: list[str] = []
    for candidate in _ROW_FIELDS + list(rows[0]):
        if candidate not in fields and any(candidate in row for row in rows):
            fields.append(candidate)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def write_benchmark_json(path: str | Path, payload: Any) -> None:
    """Write a benchmark payload as formatted JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True))


def _frame_permutation(coords: np.ndarray, *, mode: str) -> np.ndarray:
    """Return a deterministic frame permutation used for optional diagnostics."""

    coords = ensure_xyz_frames(coords)
    if mode == "input":
        return np.arange(coords.shape[0], dtype=int)
    reference = coords[0]
    centered_reference = reference - np.mean(reference, axis=0)
    scores: list[float] = []
    for frame in coords:
        if mode == "reference_rmsd":
            diff = frame - reference
        elif mode == "reference_centered_rmsd":
            diff = (frame - np.mean(frame, axis=0)) - centered_reference
        else:
            raise ValueError("frame_order must be 'input', 'reference_rmsd', or 'reference_centered_rmsd'")
        scores.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
    return np.asarray(np.argsort(scores, kind="mergesort"), dtype=int)


def _has_regularization(kwargs: dict[str, object]) -> bool:
    """Return whether sequence or contact regularization is enabled."""

    return (
        float(kwargs.get("sequence_continuity_strength", 0.0)) > 0.0
        or float(kwargs.get("contact_consistency_strength", 0.0)) > 0.0
    )
