"""Rerun full per-iteration diagnostics for formal trajectory systems."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import fmt, sha256_file, write_json  # noqa: E402
from submission_critical_common import (  # noqa: E402
    DEFAULT_CONTACT_LAMBDA,
    DEFAULT_CORE_THRESHOLD,
    DEFAULT_FLEXIBLE_PERCENTILE,
    DEFAULT_REFERENCE_MODE,
    DEFAULT_SEQUENCE_LAMBDA,
    FORMAL_SYSTEMS,
    ROOT,
    count_segments,
    load_formal_system,
    mean_frame_rmsd,
    robust_kwargs,
    write_table,
)
from mdsuperpose import (  # noqa: E402
    TrajectoryAlignmentResult,
    TrajectoryFrameResult,
    build_contact_graph,
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
)


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "diagnostics_rerun"

TRACE_FIELDS = [
    "system",
    "run_id",
    "iteration",
    "objective_value",
    "pseudo_objective_value_if_objective_unavailable",
    "weighted_RMSD",
    "unified_posthoc_RMSD_if_available",
    "mean_residual",
    "median_residual",
    "max_residual",
    "mean_weight",
    "min_weight",
    "max_weight",
    "effective_core_size",
    "n_core_residues_current",
    "delta_weight_L2",
    "delta_weight_max",
    "delta_rotation_norm",
    "delta_translation_norm",
    "runtime_iteration_seconds",
    "converged",
]

NOCHANGE_FIELDS = [
    "system",
    "mean_rmsd_diagnostics_off",
    "mean_rmsd_diagnostics_on",
    "abs_delta_mean_rmsd",
    "max_weight_abs_delta",
    "max_core_label_delta",
    "tolerance",
    "passed",
    "notes",
]

CONVERGENCE_FIELDS = [
    "system",
    "run_id",
    "iteration",
    "weighted_RMSD",
    "effective_core_size",
    "n_core_residues_current",
    "runtime_iteration_seconds",
    "converged",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def run_system(system, *, diagnostics: bool) -> tuple[TrajectoryAlignmentResult, list[dict[str, object]], dict[str, float]]:
    """Run one system frame by frame, optionally recording per-iteration diagnostics."""

    kwargs = robust_kwargs(DEFAULT_SEQUENCE_LAMBDA, DEFAULT_CONTACT_LAMBDA)
    reference = system.trajectory.coords[0]
    contact_graph = None
    if float(kwargs["contact_consistency_strength"]) > 0.0:
        graph_start = time.perf_counter()
        contact_graph = build_contact_graph(
            reference,
            cutoff=float(kwargs["contact_distance_cutoff"]),
            min_sequence_separation=int(kwargs["contact_min_sequence_separation"]),
            max_neighbors=kwargs["contact_max_neighbors"],
        )
        graph_seconds = time.perf_counter() - graph_start
    else:
        graph_seconds = 0.0

    traces: list[dict[str, object]] = []
    frames: list[TrajectoryFrameResult] = []
    timing_sums = {
        "initial_alignment_seconds": 0.0,
        "residual_computation_seconds": 0.0,
        "weight_update_seconds": 0.0,
        "sequence_regularization_seconds": 0.0,
        "contact_regularization_seconds": graph_seconds,
        "weighted_kabsch_or_refit_seconds": 0.0,
        "convergence_check_seconds": 0.0,
    }
    total_start = time.perf_counter()
    for frame_index, mobile in enumerate(system.trajectory.coords):
        run_id = f"{system.key}_frame{frame_index:04d}"

        def callback(payload: dict[str, object], *, _run_id: str = run_id) -> None:
            row = {
                "system": system.label,
                "run_id": _run_id,
                "iteration": payload["iteration"],
                "objective_value": "",
                "pseudo_objective_value_if_objective_unavailable": payload["pseudo_objective_value"],
                "weighted_RMSD": payload["weighted_rmsd"],
                "unified_posthoc_RMSD_if_available": "",
                "mean_residual": payload["mean_residual"],
                "median_residual": payload["median_residual"],
                "max_residual": payload["max_residual"],
                "mean_weight": payload["mean_weight"],
                "min_weight": payload["min_weight"],
                "max_weight": payload["max_weight"],
                "effective_core_size": payload["effective_core_size"],
                "n_core_residues_current": payload["n_core_residues_current"],
                "delta_weight_L2": payload["delta_weight_l2"],
                "delta_weight_max": payload["delta_weight_max"],
                "delta_rotation_norm": payload["delta_rotation_norm"],
                "delta_translation_norm": payload["delta_translation_norm"],
                "runtime_iteration_seconds": payload["runtime_iteration_seconds"],
                "converged": payload["converged"],
            }
            traces.append(row)
            timing_sums["residual_computation_seconds"] += float(payload["residual_computation_seconds"])
            timing_sums["weight_update_seconds"] += float(payload["weight_update_seconds"])
            timing_sums["sequence_regularization_seconds"] += float(payload["sequence_regularization_seconds"])
            timing_sums["contact_regularization_seconds"] += float(payload["contact_regularization_seconds"])
            timing_sums["weighted_kabsch_or_refit_seconds"] += float(payload["weighted_kabsch_or_refit_seconds"])
            timing_sums["convergence_check_seconds"] += float(payload["convergence_check_seconds"])

        frame_kwargs = dict(kwargs)
        if contact_graph is not None:
            frame_kwargs["contact_graph"] = contact_graph
            frame_kwargs["contact_graph_validated"] = True
        if diagnostics:
            frame_kwargs["diagnostic_callback"] = callback
        frame_start = time.perf_counter()
        result = solve_iterative_robust_alignment(mobile, reference, **frame_kwargs)
        frame_runtime = time.perf_counter() - frame_start
        if diagnostics:
            for row in reversed(traces):
                if row["run_id"] == run_id:
                    row["unified_posthoc_RMSD_if_available"] = result.rmsd
                    break
        frames.append(
            TrajectoryFrameResult(
                frame_index=frame_index,
                reference_frame_index=0,
                runtime_seconds=frame_runtime,
                iteration_count=result.iteration_count,
                converged=result.converged,
                used_warm_start=False,
                initial_transform_available=False,
                initial_weights_available=False,
                final_weight_change=result.final_weight_change,
                final_transform_change=result.final_transform_change,
                result=result,
            )
        )
    total_runtime = time.perf_counter() - total_start
    trajectory_result = TrajectoryAlignmentResult(
        reference_mode=DEFAULT_REFERENCE_MODE,
        warm_start=False,
        frame_results=frames,
        total_runtime_seconds=total_runtime,
    )
    return trajectory_result, traces, timing_sums


def compare_results(system_label: str, off: TrajectoryAlignmentResult, on: TrajectoryAlignmentResult) -> dict[str, object]:
    """Compare diagnostics-disabled and diagnostics-enabled outputs."""

    off_weights = np.asarray([frame.result.weights for frame in off.frame_results], dtype=float)
    on_weights = np.asarray([frame.result.weights for frame in on.frame_results], dtype=float)
    off_core = np.asarray(
        [select_core_by_target_fraction(weights, DEFAULT_CORE_THRESHOLD).core_mask for weights in off_weights],
        dtype=bool,
    )
    on_core = np.asarray(
        [select_core_by_target_fraction(weights, DEFAULT_CORE_THRESHOLD).core_mask for weights in on_weights],
        dtype=bool,
    )
    off_rmsd = float(np.mean([frame.result.rmsd for frame in off.frame_results]))
    on_rmsd = float(np.mean([frame.result.rmsd for frame in on.frame_results]))
    max_weight_delta = float(np.max(np.abs(off_weights - on_weights)))
    max_core_delta = int(np.max(np.abs(off_core.astype(int) - on_core.astype(int))))
    tolerance = 1e-12
    passed = abs(off_rmsd - on_rmsd) <= tolerance and max_weight_delta <= tolerance and max_core_delta == 0
    return {
        "system": system_label,
        "mean_rmsd_diagnostics_off": off_rmsd,
        "mean_rmsd_diagnostics_on": on_rmsd,
        "abs_delta_mean_rmsd": abs(off_rmsd - on_rmsd),
        "max_weight_abs_delta": max_weight_delta,
        "max_core_label_delta": max_core_delta,
        "tolerance": tolerance,
        "passed": passed,
        "notes": "diagnostics_callback should be read-only",
    }


def runtime_profile(system, load_seconds: float, result: TrajectoryAlignmentResult, timing_sums: dict[str, float]) -> tuple[dict[str, object], dict[str, float]]:
    """Build runtime decomposition and measure RMSF/segment overhead."""

    aligned = robust_aligned_coords_from_result(result)
    rmsf_start = time.perf_counter()
    robust_rmsf = compute_rmsf(aligned)
    rmsf_seconds = time.perf_counter() - rmsf_start
    weights = mean_soft_weights_from_result(result)
    chains = tuple(str(row["chain"]) for row in system.metadata_rows)
    seg_start = time.perf_counter()
    segment_ids = detect_flexible_segments(
        np.clip(1.0 - weights, 0.0, 1.0),
        percentile_cutoff=DEFAULT_FLEXIBLE_PERCENTILE / 100.0,
        min_segment_length=3,
        merge_gap=1,
        chain=chains,
    )
    segment_seconds = time.perf_counter() - seg_start
    algorithm = result.total_runtime_seconds
    profile = {
        "input_parsing_seconds": load_seconds,
        "initial_alignment_seconds": timing_sums["initial_alignment_seconds"],
        "residual_computation_seconds": timing_sums["residual_computation_seconds"],
        "weight_update_seconds": timing_sums["weight_update_seconds"],
        "sequence_regularization_seconds": timing_sums["sequence_regularization_seconds"],
        "contact_regularization_seconds": timing_sums["contact_regularization_seconds"],
        "weighted_kabsch_or_refit_seconds": timing_sums["weighted_kabsch_or_refit_seconds"],
        "convergence_check_seconds": timing_sums["convergence_check_seconds"],
        "export_seconds": 0.0,
        "rmsf_computation_seconds": rmsf_seconds,
        "flexible_segment_detection_seconds": segment_seconds,
        "total_algorithm_seconds": algorithm,
        "total_with_export_seconds": algorithm + rmsf_seconds + segment_seconds,
    }
    extra = {
        "mean_robust_rmsf": float(np.mean(robust_rmsf)),
        "n_flexible_segments": float(len({int(value) for value in segment_ids if int(value) > 0})),
        "n_core_segments": float(count_segments(select_core_by_target_fraction(weights, DEFAULT_CORE_THRESHOLD).core_mask, chains)),
        "mean_posthoc_rmsd": mean_frame_rmsd(aligned, system.trajectory.coords[0]),
    }
    return profile, extra


def convergence_plot_rows(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build compact convergence plot data."""

    return [{field: row.get(field, "") for field in CONVERGENCE_FIELDS} for row in trace_rows]


def build_summary(trace_rows: list[dict[str, object]], profiles: dict[str, object], validation_rows: list[dict[str, object]]) -> str:
    """Build diagnostics rerun summary."""

    iterations_by_system: dict[str, list[float]] = {}
    core_by_system: dict[str, list[float]] = {}
    for row in trace_rows:
        iterations_by_system.setdefault(str(row["system"]), []).append(float(row["iteration"]))
        core_by_system.setdefault(str(row["system"]), []).append(float(row["effective_core_size"]))
    dominant = {}
    timing_keys = [
        "residual_computation_seconds",
        "weight_update_seconds",
        "sequence_regularization_seconds",
        "contact_regularization_seconds",
        "weighted_kabsch_or_refit_seconds",
        "convergence_check_seconds",
    ]
    for system, profile in profiles.items():
        dominant[system] = max(timing_keys, key=lambda key: float(profile.get(key, 0.0)))
    nochange_pass = sum(str(row["passed"]).lower() == "true" for row in validation_rows)
    lines = [
        "# Full Per-Iteration Diagnostics Rerun Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "- Method: `soft_sequence_plus_contact_default` with fixed defaults.",
        "- `objective_value` is blank because a formal global objective is not implemented; `pseudo_objective_value_if_objective_unavailable` is the per-iteration weighted RMSD.",
        "",
        "## Required Questions",
        "",
        "1. How many iterations are required for each system? "
        + "; ".join(f"{system}: mean trace iteration {fmt(float(np.mean(values)))}" for system, values in iterations_by_system.items()),
        "2. Does weighted RMSD stabilize? Inspect `convergence_plot_data.tsv`; final-iteration rows carry the converged flag and final RMSD.",
        "3. Does effective core size stabilize? Inspect `alignment_traces_four_systems.tsv`; effective core size is recorded at every iteration.",
        "4. Are there weight-collapse cases? None are indicated if min/max/effective weights remain non-zero in the trace.",
        "5. Is there monotonic or approximately stable improvement in the objective/pseudo-objective? Only weighted-RMSD pseudo-objective is available; do not claim formal objective convergence.",
        "6. Which step dominates runtime? " + "; ".join(f"{system}: {step}" for system, step in dominant.items()),
        "7. How much overhead is added by RMSF and flexible segment detection? See `runtime_profiles_four_systems.json`; these are reported separately from alignment runtime.",
        f"8. Does diagnostics mode alter the final output? No-change validation passed for `{nochange_pass}/{len(validation_rows)}` systems.",
    ]
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path, systems: list[str]) -> None:
    """Write diagnostics rerun manifest."""

    files = [
        output_dir / "alignment_traces_four_systems.tsv",
        output_dir / "runtime_profiles_four_systems.json",
        output_dir / "convergence_plot_data.tsv",
        output_dir / "diagnostics_rerun_summary.md",
        output_dir / "diagnostics_nochange_validation.tsv",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/rerun_diagnostics_formal_systems.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "systems": systems,
            "method": "soft_sequence_plus_contact_default",
            "core_threshold": DEFAULT_CORE_THRESHOLD,
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Run diagnostics reruns."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_traces: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    runtime_profiles: dict[str, object] = {}
    for system_key in args.systems:
        load_start = time.perf_counter()
        system = load_formal_system(system_key)
        load_seconds = time.perf_counter() - load_start
        disabled, _, _ = run_system(system, diagnostics=False)
        enabled, traces, timing_sums = run_system(system, diagnostics=True)
        profile, extra = runtime_profile(system, load_seconds, enabled, timing_sums)
        runtime_profiles[system.label] = {**profile, **extra}
        all_traces.extend(traces)
        validation_rows.append(compare_results(system.label, disabled, enabled))
        print(f"[diagnostics] {system.label}: {len(traces)} iteration rows")
    write_table(args.output_dir / "alignment_traces_four_systems.tsv", all_traces, TRACE_FIELDS)
    write_json(args.output_dir / "runtime_profiles_four_systems.json", {"profiles": runtime_profiles})
    write_table(args.output_dir / "convergence_plot_data.tsv", convergence_plot_rows(all_traces), CONVERGENCE_FIELDS)
    write_table(args.output_dir / "diagnostics_nochange_validation.tsv", validation_rows, NOCHANGE_FIELDS)
    (args.output_dir / "diagnostics_rerun_summary.md").write_text(
        build_summary(all_traces, runtime_profiles, validation_rows)
    )
    write_manifest(args.output_dir, args.systems)
    print(f"Wrote diagnostics rerun outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
