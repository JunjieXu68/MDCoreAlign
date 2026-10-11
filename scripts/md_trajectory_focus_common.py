"""Shared helpers for MD-trajectory-focused validation analyses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (ROOT, ROOT / "src", SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import sha256_file, write_json  # noqa: E402
from submission_critical_common import (  # noqa: E402
    DEFAULT_CONTACT_LAMBDA,
    DEFAULT_CORE_THRESHOLD,
    DEFAULT_DELTA,
    DEFAULT_FLEXIBLE_PERCENTILE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REFERENCE_MODE,
    DEFAULT_RULE,
    DEFAULT_SEQUENCE_LAMBDA,
    DEFAULT_TOLERANCE,
    DISPLAY_NAMES,
    FORMAL_SYSTEMS,
    LoadedSystem,
    count_segments,
    load_formal_system,
    mean_frame_rmsd,
    robust_kwargs,
    write_table,
)
from mdsuperpose import (  # noqa: E402
    Trajectory,
    TrajectoryAlignmentResult,
    TrajectoryFrameResult,
    align_trajectory_ordinary,
    build_contact_graph,
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    residual_distances,
    robust_aligned_coords_from_result,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
    solve_trajectory_robust_alignment,
    spearman_correlation,
)


MD_FOCUS_ROOT = ROOT / "results" / "md_trajectory_focus"
REFERENCE_STRATEGIES = [
    "first_frame",
    "middle_frame",
    "last_frame",
    "ordinary_mean_structure",
    "robust_mean_structure",
]


@dataclass(frozen=True)
class MDSolveSummary:
    """Full result bundle for one MD trajectory validation rerun."""

    metrics: dict[str, object]
    trajectory_result: TrajectoryAlignmentResult
    aligned_coords: np.ndarray
    weights: np.ndarray
    mobility: np.ndarray
    core_mask: np.ndarray
    flexible_mask: np.ndarray
    flexible_segment_id: np.ndarray
    reference_coords: np.ndarray


def timestamp() -> str:
    """Return an ISO timestamp."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def md_block_definitions(n_frames: int) -> list[dict[str, object]]:
    """Return standard temporal block definitions using half-open indices."""

    if n_frames < 2:
        raise ValueError("at least two frames are required for block analysis")
    q1 = n_frames // 4
    q2 = n_frames // 2
    q3 = (3 * n_frames) // 4
    return [
        {"block_id": "block_0_25", "frame_start": 0, "frame_end": q1},
        {"block_id": "block_25_50", "frame_start": q1, "frame_end": q2},
        {"block_id": "block_50_75", "frame_start": q2, "frame_end": q3},
        {"block_id": "block_75_100", "frame_start": q3, "frame_end": n_frames},
        {"block_id": "first_half", "frame_start": 0, "frame_end": q2},
        {"block_id": "second_half", "frame_start": q2, "frame_end": n_frames},
        {"block_id": "full", "frame_start": 0, "frame_end": n_frames},
    ]


def stride_values() -> list[int]:
    """Return the formal stride sensitivity grid."""

    return [1, 2, 5, 10, 20]


def prefix_definitions(n_frames: int) -> list[dict[str, object]]:
    """Return standard trajectory prefix definitions."""

    if n_frames < 2:
        raise ValueError("at least two frames are required for prefix analysis")
    fractions = [
        ("first_10_percent", 0.10),
        ("first_25_percent", 0.25),
        ("first_50_percent", 0.50),
        ("first_75_percent", 0.75),
        ("full_100_percent", 1.00),
    ]
    definitions = []
    for label, fraction in fractions:
        end = max(1, int(np.ceil(n_frames * fraction)))
        definitions.append({"prefix_id": label, "prefix_fraction": fraction, "frame_end": min(end, n_frames)})
    return definitions


def subset_loaded_system(system: LoadedSystem, coords: np.ndarray, *, note_suffix: str) -> LoadedSystem:
    """Return a LoadedSystem using a coordinate subset and identical metadata."""

    metadata = dict(system.metadata)
    metadata["n_frames"] = int(coords.shape[0])
    metadata["subset_note"] = note_suffix
    return LoadedSystem(
        key=system.key,
        label=system.label,
        trajectory=Trajectory(coords=np.asarray(coords, dtype=float)),
        residue_labels=system.residue_labels,
        metadata_rows=system.metadata_rows,
        metadata=metadata,
    )


def chain_values(system: LoadedSystem) -> tuple[str, ...]:
    """Return chain labels from formal residue metadata."""

    return tuple(str(row["chain"]) for row in system.metadata_rows)


def default_soft_kwargs() -> dict[str, object]:
    """Return default soft sequence+contact kwargs for MD trajectory analyses."""

    return robust_kwargs(DEFAULT_SEQUENCE_LAMBDA, DEFAULT_CONTACT_LAMBDA)


def solve_default_soft(
    system: LoadedSystem,
    *,
    warm_start: bool = False,
    reference_strategy: str = "first_frame",
) -> MDSolveSummary:
    """Run default soft sequence+contact alignment and summarize it."""

    start = time.perf_counter()
    reference = reference_for_strategy(system, reference_strategy)
    if reference_strategy == "first_frame":
        result = solve_trajectory_robust_alignment(
            system.trajectory,
            reference_mode=DEFAULT_REFERENCE_MODE,
            warm_start=warm_start,
            **default_soft_kwargs(),
        )
    else:
        result = solve_against_fixed_reference(system.trajectory, reference, warm_start=warm_start)
    runtime = time.perf_counter() - start
    return summarize_solution(system, result, reference_coords=reference, runtime=runtime)


def solve_against_fixed_reference(
    trajectory: Trajectory,
    reference_coords: np.ndarray,
    *,
    warm_start: bool = False,
) -> TrajectoryAlignmentResult:
    """Run robust alignment against an explicit fixed reference structure."""

    kwargs = default_soft_kwargs()
    contact_graph = None
    if float(kwargs["contact_consistency_strength"]) > 0.0:
        contact_graph = build_contact_graph(
            reference_coords,
            cutoff=float(kwargs["contact_distance_cutoff"]),
            min_sequence_separation=int(kwargs["contact_min_sequence_separation"]),
            max_neighbors=kwargs["contact_max_neighbors"],
        )
    frames: list[TrajectoryFrameResult] = []
    previous = None
    total_start = time.perf_counter()
    for frame_index, mobile in enumerate(trajectory.coords):
        frame_kwargs = dict(kwargs)
        if contact_graph is not None:
            frame_kwargs["contact_graph"] = contact_graph
            frame_kwargs["contact_graph_validated"] = True
        if warm_start and previous is not None:
            frame_kwargs["initial_weights"] = previous.weights
            from mdsuperpose import RigidTransform

            frame_kwargs["initial_transform"] = RigidTransform(
                rotation=previous.rotation.copy(),
                translation=previous.translation.copy(),
            )
        frame_start = time.perf_counter()
        result = solve_iterative_robust_alignment(mobile, reference_coords, **frame_kwargs)
        frame_runtime = time.perf_counter() - frame_start
        frames.append(
            TrajectoryFrameResult(
                frame_index=frame_index,
                reference_frame_index=-1,
                runtime_seconds=frame_runtime,
                iteration_count=result.iteration_count,
                converged=result.converged,
                used_warm_start=bool(warm_start and previous is not None),
                initial_transform_available=bool(warm_start and previous is not None),
                initial_weights_available=bool(warm_start and previous is not None),
                final_weight_change=result.final_weight_change,
                final_transform_change=result.final_transform_change,
                result=result,
            )
        )
        previous = result
    return TrajectoryAlignmentResult(
        reference_mode="explicit_fixed_reference",
        warm_start=warm_start,
        frame_results=frames,
        total_runtime_seconds=time.perf_counter() - total_start,
    )


def reference_for_strategy(system: LoadedSystem, reference_strategy: str) -> np.ndarray:
    """Build a reference structure for one reference strategy."""

    coords = system.trajectory.coords
    if reference_strategy == "first_frame":
        return coords[0].copy()
    if reference_strategy == "middle_frame":
        return coords[coords.shape[0] // 2].copy()
    if reference_strategy == "last_frame":
        return coords[-1].copy()
    if reference_strategy == "ordinary_mean_structure":
        return np.mean(align_trajectory_ordinary(system.trajectory), axis=0)
    if reference_strategy == "robust_mean_structure":
        seed = solve_trajectory_robust_alignment(
            system.trajectory,
            reference_mode=DEFAULT_REFERENCE_MODE,
            warm_start=False,
            **default_soft_kwargs(),
        )
        return np.mean(robust_aligned_coords_from_result(seed), axis=0)
    raise ValueError(f"unknown reference strategy: {reference_strategy}")


def summarize_solution(
    system: LoadedSystem,
    result: TrajectoryAlignmentResult,
    *,
    reference_coords: np.ndarray,
    runtime: float,
) -> MDSolveSummary:
    """Summarize one trajectory solve into metrics and comparable vectors."""

    aligned = robust_aligned_coords_from_result(result)
    weights = mean_soft_weights_from_result(result)
    mobility = np.clip(1.0 - weights, 0.0, 1.0)
    core = select_core_by_target_fraction(weights, DEFAULT_CORE_THRESHOLD)
    core_mask = core.core_mask
    chains = chain_values(system)
    rmsf = compute_rmsf(aligned)
    flex_ids = detect_flexible_segments(
        mobility,
        percentile_cutoff=DEFAULT_FLEXIBLE_PERCENTILE / 100.0,
        min_segment_length=3,
        merge_gap=1,
        chain=chains,
    )
    distances = np.asarray([residual_distances(frame, reference_coords) for frame in aligned], dtype=float)
    mean_core = float(np.mean(rmsf[core_mask])) if np.any(core_mask) else ""
    mean_noncore = float(np.mean(rmsf[~core_mask])) if np.any(~core_mask) else ""
    metrics = {
        "unified_posthoc_RMSD": mean_frame_rmsd(aligned, reference_coords),
        "weighted_RMSD": result.mean_weighted_rmsd,
        "runtime_algorithm_seconds": runtime,
        "runtime_total_seconds": runtime,
        "effective_core_size": float(np.sum(weights)),
        "n_core_residues": int(np.sum(core_mask)),
        "n_core_segments": count_segments(core_mask, chains),
        "mean_core_RMSF": mean_core,
        "mean_noncore_RMSF": mean_noncore,
        "noncore_core_RMSF_ratio": (mean_noncore / mean_core) if mean_core not in ("", 0.0) and mean_noncore != "" else "",
        "spearman_robust_RMSF_vs_mobility": spearman_correlation(rmsf, mobility),
        "converged": all(frame.converged for frame in result.frame_results),
        "n_iterations": result.average_iterations_per_frame,
        "mean_residual": float(np.mean(distances)),
        "median_residual": float(np.median(distances)),
        "max_residual": float(np.max(distances)),
    }
    return MDSolveSummary(
        metrics=metrics,
        trajectory_result=result,
        aligned_coords=aligned,
        weights=weights,
        mobility=mobility,
        core_mask=core_mask,
        flexible_mask=flex_ids > 0,
        flexible_segment_id=flex_ids,
        reference_coords=reference_coords,
    )


def compare_to_reference(solution: MDSolveSummary, reference: MDSolveSummary, *, suffix: str) -> dict[str, object]:
    """Compare core, mobility, and flexible masks to a reference solution."""

    return {
        f"jaccard_core_vs_{suffix}": jaccard(solution.core_mask, reference.core_mask),
        f"spearman_mobility_vs_{suffix}": spearman_correlation(solution.mobility, reference.mobility),
        f"flexible_segment_overlap_vs_{suffix}": jaccard(solution.flexible_mask, reference.flexible_mask),
    }


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Return Jaccard overlap for two boolean masks."""

    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("masks must have the same shape")
    union = np.logical_or(first, second)
    if not np.any(union):
        return 1.0
    return float(np.sum(np.logical_and(first, second)) / np.sum(union))


def common_manifest(
    *,
    script: str,
    output_dir: Path,
    systems: list[str],
    output_files: list[Path],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a standard manifest for MD trajectory focus analyses."""

    payload: dict[str, Any] = {
        "script": script,
        "generated_at": timestamp(),
        "systems": systems,
        "default_method": "soft_sequence_plus_contact",
        "default_parameters": {
            "rule": DEFAULT_RULE,
            "delta": DEFAULT_DELTA,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "tolerance": DEFAULT_TOLERANCE,
            "lambda_sequence": DEFAULT_SEQUENCE_LAMBDA,
            "lambda_contact": DEFAULT_CONTACT_LAMBDA,
            "core_fraction": DEFAULT_CORE_THRESHOLD,
            "flexible_percentile": DEFAULT_FLEXIBLE_PERCENTILE,
        },
        "output_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in output_files
            if path.exists()
        },
    }
    if extra:
        payload.update(extra)
    write_json(output_dir / "manifest.json", payload)


def formatted_range(values: list[float]) -> str:
    """Return compact min-max text."""

    if not values:
        return "unavailable"
    return f"{min(values):.4g} to {max(values):.4g}"


__all__ = [
    "DISPLAY_NAMES",
    "FORMAL_SYSTEMS",
    "MD_FOCUS_ROOT",
    "REFERENCE_STRATEGIES",
    "LoadedSystem",
    "MDSolveSummary",
    "Trajectory",
    "align_trajectory_ordinary",
    "chain_values",
    "common_manifest",
    "compare_to_reference",
    "default_soft_kwargs",
    "formatted_range",
    "jaccard",
    "load_formal_system",
    "md_block_definitions",
    "mean_frame_rmsd",
    "prefix_definitions",
    "reference_for_strategy",
    "residual_distances",
    "robust_aligned_coords_from_result",
    "select_core_by_target_fraction",
    "solve_default_soft",
    "solve_trajectory_robust_alignment",
    "stride_values",
    "subset_loaded_system",
    "timestamp",
    "write_json",
    "write_table",
]
