"""Shared helpers for submission-critical rerun scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
for path in (ROOT, SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.formal_flexibility_profiles import DATASETS, load_dataset_with_metadata  # noqa: E402
from mdsuperpose import (  # noqa: E402
    Trajectory,
    align_trajectory_ordinary,
    build_flexibility_profile,
    compute_rmsf,
    mean_soft_weights_from_result,
    residual_distances,
    robust_aligned_coords_from_result,
    select_core_by_target_fraction,
    solve_iterative_hard_trim_alignment,
    solve_trajectory_robust_alignment,
    spearman_correlation,
)


FORMAL_SYSTEMS = ["ifabp", "adk_equilibrium", "nhaa_equilibrium", "yiip_equilibrium_short"]
DISPLAY_NAMES = {
    "ifabp": "I-FABP",
    "adk_equilibrium": "ADK",
    "nhaa_equilibrium": "NhaA",
    "yiip_equilibrium_short": "YiiP short",
}
DEFAULT_CORE_THRESHOLD = 0.6
DEFAULT_FLEXIBLE_PERCENTILE = 80
DEFAULT_SEQUENCE_LAMBDA = 0.6
DEFAULT_CONTACT_LAMBDA = 0.5
DEFAULT_RULE = "huber"
DEFAULT_DELTA = 0.25
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TOLERANCE = 1e-6
DEFAULT_REFERENCE_MODE = "first_frame"
DEFAULT_WARM_START = False


@dataclass(frozen=True)
class LoadedSystem:
    """Coordinates and metadata for one formal system."""

    key: str
    label: str
    trajectory: Trajectory
    residue_labels: tuple[str, ...]
    metadata_rows: list[dict[str, object]]
    metadata: dict[str, object]


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str], *, delimiter: str = "\t") -> None:
    """Write a table with a stable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_formal_inputs(system_key: str) -> None:
    """Fail loudly if formal input files for one system are missing."""

    config = DATASETS[system_key]
    missing = []
    for field in ("topology", "trajectory"):
        path = Path(config[field])
        if not path.exists():
            missing.append(f"{field}={path}")
    if missing:
        raise FileNotFoundError(f"Missing formal input files for {system_key}: {', '.join(missing)}")


def load_formal_system(system_key: str) -> LoadedSystem:
    """Load one formal system with official preprocessing."""

    require_formal_inputs(system_key)
    trajectory, residue_labels, metadata_rows, metadata = load_dataset_with_metadata(system_key)
    return LoadedSystem(
        key=system_key,
        label=DISPLAY_NAMES[system_key],
        trajectory=trajectory,
        residue_labels=residue_labels,
        metadata_rows=metadata_rows,
        metadata=metadata,
    )


def robust_kwargs(lambda_sequence: float, lambda_contact: float) -> dict[str, object]:
    """Build fixed robust alignment parameters."""

    return {
        "rule": DEFAULT_RULE,
        "delta": DEFAULT_DELTA,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "tolerance": DEFAULT_TOLERANCE,
        "sequence_continuity_strength": lambda_sequence,
        "sequence_continuity_passes": 1,
        "contact_consistency_strength": lambda_contact,
        "contact_consistency_passes": 1,
        "contact_distance_cutoff": 8.0,
        "contact_min_sequence_separation": 2,
        "contact_max_neighbors": 8,
    }


def run_ordinary(system: LoadedSystem) -> tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run ordinary alignment and return metrics plus aligned frames."""

    start = time.perf_counter()
    aligned = align_trajectory_ordinary(system.trajectory)
    runtime = time.perf_counter() - start
    reference = system.trajectory.coords[0]
    distances = frame_residuals(aligned, reference)
    metrics = {
        "runtime_algorithm_seconds": runtime,
        "runtime_total_seconds": runtime,
        "unified_posthoc_RMSD": mean_frame_rmsd(aligned, reference),
        "weighted_RMSD": "",
        "mean_residual": float(np.mean(distances)),
        "median_residual": float(np.median(distances)),
        "effective_core_size": "",
        "n_core_residues": "",
        "n_mobile_residues": "",
        "n_core_segments": "",
        "core_fragmentation_score": "",
        "mean_core_RMSF": "",
        "mean_noncore_RMSF": "",
        "noncore_core_RMSF_ratio": "",
        "spearman_robust_RMSF_vs_mobility": "",
        "n_flexible_segments": "",
        "converged": "not_applicable",
        "n_iterations": "",
    }
    return metrics, aligned, None, None


def run_hard_trim(system: LoadedSystem, *, core_fraction: float = DEFAULT_CORE_THRESHOLD) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Run hard-trim control on every frame."""

    reference = system.trajectory.coords[0]
    start = time.perf_counter()
    results = [
        solve_iterative_hard_trim_alignment(
            frame,
            reference,
            core_fraction=core_fraction,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            tolerance=DEFAULT_TOLERANCE,
        )
        for frame in system.trajectory.coords
    ]
    runtime = time.perf_counter() - start
    aligned = np.asarray([result.aligned_coords for result in results], dtype=float)
    weights_by_frame = np.asarray([result.weights for result in results], dtype=float)
    mean_weights = np.mean(weights_by_frame, axis=0)
    metrics = summarize_alignment(
        system,
        aligned,
        mean_weights,
        weighted_rmsd=float(np.mean([result.weighted_rmsd for result in results])),
        iteration_count=float(np.mean([result.iteration_count for result in results])),
        converged=all(result.converged for result in results),
        runtime=runtime,
    )
    return metrics, aligned, mean_weights, weights_by_frame


def run_soft(
    system: LoadedSystem,
    *,
    lambda_sequence: float,
    lambda_contact: float,
    diagnostic_callback_factory=None,
) -> tuple[dict[str, object], Any, np.ndarray]:
    """Run soft robust alignment with optional regularization."""

    kwargs = robust_kwargs(lambda_sequence, lambda_contact)
    callback_kwargs = dict(kwargs)
    frame_callbacks = []
    if diagnostic_callback_factory is not None:
        # The current trajectory-mode API applies the same callback to all frames,
        # so diagnostics reruns use a custom runner instead.
        callback_kwargs["diagnostic_callback"] = diagnostic_callback_factory(frame_callbacks)
    start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        system.trajectory,
        reference_mode=DEFAULT_REFERENCE_MODE,
        warm_start=DEFAULT_WARM_START,
        **callback_kwargs,
    )
    runtime = time.perf_counter() - start
    aligned = robust_aligned_coords_from_result(result)
    mean_weights = mean_soft_weights_from_result(result)
    profile = build_flexibility_profile(
        system.trajectory,
        residue_labels=system.residue_labels,
        trajectory_result=result,
        reference_mode=DEFAULT_REFERENCE_MODE,
        warm_start=DEFAULT_WARM_START,
        flexible_percentile_cutoff=DEFAULT_FLEXIBLE_PERCENTILE / 100.0,
        min_segment_length=3,
        merge_gap=1,
        mobility_score_mode="soft_weight",
    )
    metrics = summarize_alignment(
        system,
        aligned,
        mean_weights,
        weighted_rmsd=result.mean_weighted_rmsd,
        iteration_count=result.average_iterations_per_frame,
        converged=all(frame.converged for frame in result.frame_results),
        runtime=runtime,
        profile=profile,
    )
    return metrics, result, mean_weights


def summarize_alignment(
    system: LoadedSystem,
    aligned: np.ndarray,
    weights: np.ndarray,
    *,
    weighted_rmsd: float | str,
    iteration_count: float | str,
    converged: bool | str,
    runtime: float,
    profile=None,
) -> dict[str, object]:
    """Summarize final alignment, core, and RMSF metrics."""

    reference = system.trajectory.coords[0]
    distances = frame_residuals(aligned, reference)
    core_mask = select_core_by_target_fraction(weights, DEFAULT_CORE_THRESHOLD).core_mask
    n_core = int(np.sum(core_mask))
    n_mobile = int(core_mask.shape[0] - n_core)
    chains = tuple(str(row["chain"]) for row in system.metadata_rows)
    n_core_segments = count_segments(core_mask, chains)
    if profile is None:
        rmsf = compute_rmsf(aligned)
        mobility = np.clip(1.0 - np.asarray(weights, dtype=float), 0.0, 1.0)
        flexible_ids = np.zeros(weights.shape[0], dtype=int)
    else:
        rmsf = profile.robust_aligned_rmsf
        mobility = profile.mobility_score
        flexible_ids = profile.flexible_segment_id
    mean_core = float(np.mean(rmsf[core_mask])) if np.any(core_mask) else ""
    mean_noncore = float(np.mean(rmsf[~core_mask])) if np.any(~core_mask) else ""
    return {
        "runtime_algorithm_seconds": runtime,
        "runtime_total_seconds": runtime,
        "unified_posthoc_RMSD": mean_frame_rmsd(aligned, reference),
        "weighted_RMSD": weighted_rmsd,
        "mean_residual": float(np.mean(distances)),
        "median_residual": float(np.median(distances)),
        "effective_core_size": float(np.sum(weights)),
        "n_core_residues": n_core,
        "n_mobile_residues": n_mobile,
        "n_core_segments": n_core_segments,
        "core_fragmentation_score": n_core_segments,
        "mean_core_RMSF": mean_core,
        "mean_noncore_RMSF": mean_noncore,
        "noncore_core_RMSF_ratio": (mean_noncore / mean_core) if mean_core not in ("", 0.0) and mean_noncore != "" else "",
        "spearman_robust_RMSF_vs_mobility": spearman_correlation(rmsf, mobility),
        "n_flexible_segments": len({int(value) for value in flexible_ids if int(value) > 0}),
        "converged": converged,
        "n_iterations": iteration_count,
    }


def frame_residuals(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return flattened per-frame residual distances."""

    return np.asarray([residual_distances(frame, reference) for frame in aligned], dtype=float).reshape(-1)


def mean_frame_rmsd(aligned: np.ndarray, reference: np.ndarray) -> float:
    """Return mean frame RMSD against one reference structure."""

    squared = np.sum((aligned - reference[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.sqrt(np.mean(squared, axis=1))))


def count_segments(mask: np.ndarray, chains: tuple[str, ...] | list[str]) -> int:
    """Count contiguous True segments without merging across chains."""

    count = 0
    previous = False
    previous_chain = None
    for value, chain in zip(mask.astype(bool), chains, strict=True):
        if value and (not previous or chain != previous_chain):
            count += 1
        previous = bool(value)
        previous_chain = chain
    return count


def base_output_row(system: LoadedSystem, method: str, lambda_sequence: float, lambda_contact: float) -> dict[str, object]:
    """Return common metadata columns for one result row."""

    return {
        "system": system.label,
        "method": method,
        "input_coordinate_file": system.metadata["trajectory_path"],
        "topology_file": system.metadata["topology_path"],
        "selection": system.metadata["selection"],
        "reference_strategy": DEFAULT_REFERENCE_MODE,
        "n_residues": system.metadata["n_residues"],
        "n_frames_or_models": system.metadata["n_frames"],
        "lambda_sequence": lambda_sequence,
        "lambda_contact": lambda_contact,
        "core_threshold": DEFAULT_CORE_THRESHOLD,
        "flexible_percentile": DEFAULT_FLEXIBLE_PERCENTILE,
    }
