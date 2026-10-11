#!/usr/bin/env python
"""Profile MDCoreAlign runtime hotspots on mdCATH and formal MD systems.

The script is intentionally observational: it reuses the current solver path
and records stage timings without changing convergence settings or outputs.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import dataclass
from pathlib import Path
import pstats
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose.core_selection import select_core_by_target_fraction  # noqa: E402
from mdsuperpose.flexibility import (  # noqa: E402
    align_trajectory_ordinary,
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
)
from mdsuperpose.metrics import mean_rmsd  # noqa: E402
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment  # noqa: E402
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory  # noqa: E402
from scripts.mdcath_benchmark.mdcath_common import (  # noqa: E402
    HF_REPO_ID,
    RESULT_ROOT,
    download_hf_file,
    read_tsv,
    stable_spearman,
    write_tsv,
)
from submission_critical_common import (  # noqa: E402
    DEFAULT_CONTACT_LAMBDA,
    DEFAULT_CORE_THRESHOLD,
    DEFAULT_FLEXIBLE_PERCENTILE,
    DEFAULT_REFERENCE_MODE,
    DEFAULT_SEQUENCE_LAMBDA,
    FORMAL_SYSTEMS,
    load_formal_system,
    robust_kwargs,
)


PROFILE_FIELDS = [
    "trajectory_id",
    "benchmark_group",
    "n_frames",
    "n_residues",
    "stage",
    "runtime_sec",
    "percent_total",
    "call_count",
    "notes",
]


@dataclass(frozen=True)
class ProfileCase:
    """A loaded trajectory plus metadata required by the profiler."""

    trajectory_id: str
    benchmark_group: str
    coords: np.ndarray
    residue_metadata: list[dict[str, Any]]
    robust_kwargs: dict[str, object]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=RESULT_ROOT / "subsets/mdcath_cath_balanced_5_per_class.tsv",
    )
    parser.add_argument("--cache-dir", type=Path, default=RESULT_ROOT / "cache")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--mdcath-max-trajectories", type=int, default=None)
    parser.add_argument("--formal-systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    parser.add_argument("--skip-mdcath", action="store_true")
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("results/performance_optimization/profile/mdcorealign_hotspot_profile.tsv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/mdcorealign_runtime_hotspot_report.md"),
    )
    return parser.parse_args()


def resolve_h5(row: dict[str, str], args: argparse.Namespace) -> Path:
    """Resolve a cached mdCATH HDF5 file, downloading only when requested."""

    h5_path = row.get("h5_path", "")
    local_candidates = [Path(h5_path), args.cache_dir / h5_path]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    completed_summary = (
        RESULT_ROOT
        / "cath_balanced_subset_fullframes"
        / "aggregate"
        / "cath_balanced_5_per_class_fullframes_summary.tsv"
    )
    if completed_summary.exists():
        for completed in read_tsv(completed_summary):
            if completed.get("trajectory_id") != row.get("trajectory_id"):
                continue
            for note_part in str(completed.get("notes", "")).split(";"):
                note_part = note_part.strip()
                if not note_part.startswith("h5_file="):
                    continue
                candidate = Path(note_part.split("=", 1)[1])
                if candidate.exists():
                    return candidate
    if not args.allow_download:
        raise FileNotFoundError(f"missing cached HDF5 for {row.get('trajectory_id')}: {h5_path}")
    return download_hf_file(h5_path, cache_dir=args.cache_dir, repo_id=args.repo_id)


def load_mdcath_cases(args: argparse.Namespace) -> list[ProfileCase]:
    """Load mdCATH subset trajectories for profiling."""

    rows = read_tsv(args.subset_manifest)
    if args.mdcath_max_trajectories is not None:
        rows = rows[: args.mdcath_max_trajectories]
    cases: list[ProfileCase] = []
    for row in rows:
        h5_file = resolve_h5(row, args)
        coords, _labels, residue_metadata, _metadata = extract_ca_trajectory(
            h5_file,
            coordinate_path="auto",
            max_frames=None,
            stride=1,
        )
        cases.append(
            ProfileCase(
                trajectory_id=row["trajectory_id"],
                benchmark_group="mdcath_cath_balanced_fullframes",
                coords=coords,
                residue_metadata=residue_metadata,
                robust_kwargs={
                    "rule": "huber",
                    "max_iterations": 20,
                    "tolerance": 1e-6,
                    "objective_tolerance": 1e-5,
                    "objective_patience": 2,
                    "sequence_continuity_strength": 0.05,
                    "contact_consistency_strength": 0.0,
                },
            )
        )
    return cases


def load_formal_cases(args: argparse.Namespace) -> list[ProfileCase]:
    """Load the four formal MD systems using the official loader."""

    cases: list[ProfileCase] = []
    for system_key in args.formal_systems:
        system = load_formal_system(system_key)
        cases.append(
            ProfileCase(
                trajectory_id=system.label,
                benchmark_group="formal_four_md_systems",
                coords=system.trajectory.coords,
                residue_metadata=system.metadata_rows,
                robust_kwargs=robust_kwargs(DEFAULT_SEQUENCE_LAMBDA, DEFAULT_CONTACT_LAMBDA),
            )
        )
    return cases


def add_stage(
    rows: list[dict[str, object]],
    *,
    case: ProfileCase,
    stage: str,
    runtime: float,
    total: float,
    call_count: int | str = 1,
    notes: str = "",
) -> None:
    """Append one stage timing row."""

    rows.append(
        {
            "trajectory_id": case.trajectory_id,
            "benchmark_group": case.benchmark_group,
            "n_frames": int(case.coords.shape[0]),
            "n_residues": int(case.coords.shape[1]),
            "stage": stage,
            "runtime_sec": float(runtime),
            "percent_total": float(100.0 * runtime / total) if total > 0.0 else 0.0,
            "call_count": call_count,
            "notes": notes,
        }
    )


def top_cprofile_rows(profile: cProfile.Profile, limit: int = 12) -> list[tuple[str, int, float]]:
    """Return top cumulative-time functions from a cProfile run."""

    stats = pstats.Stats(profile)
    items = []
    for (filename, line_no, func_name), stat in stats.stats.items():
        call_count = int(stat[1])
        cumulative = float(stat[3])
        short_file = str(Path(filename).name)
        items.append((f"{short_file}:{line_no}:{func_name}", call_count, cumulative))
    items.sort(key=lambda item: item[2], reverse=True)
    return items[:limit]


def profile_case(case: ProfileCase) -> list[dict[str, object]]:
    """Profile one trajectory case and return flat timing rows."""

    rows: list[dict[str, object]] = []
    stage_times = {
        "ordinary_initialization_fit": 0.0,
        "unified_metric_ordinary": 0.0,
        "weighted_fit": 0.0,
        "residual_update": 0.0,
        "weight_update": 0.0,
        "sequence_regularization": 0.0,
        "contact_regularization": 0.0,
        "convergence_check": 0.0,
        "iteration_total": 0.0,
    }
    stage_counts = {key: 0 for key in stage_times}

    total_start = time.perf_counter()

    ordinary_start = time.perf_counter()
    ordinary_aligned = align_trajectory_ordinary(case.coords, reference_frame_index=0)
    stage_times["ordinary_initialization_fit"] = time.perf_counter() - ordinary_start
    stage_counts["ordinary_initialization_fit"] = int(case.coords.shape[0])

    metric_start = time.perf_counter()
    ordinary_rmsd = mean_rmsd(ordinary_aligned, case.coords[0])
    stage_times["unified_metric_ordinary"] = time.perf_counter() - metric_start
    stage_counts["unified_metric_ordinary"] = 1

    trace_count = 0

    def callback(payload: dict[str, object]) -> None:
        nonlocal trace_count
        trace_count += 1
        stage_times["weighted_fit"] += float(payload["weighted_kabsch_or_refit_seconds"])
        stage_times["residual_update"] += float(payload["residual_computation_seconds"])
        stage_times["weight_update"] += float(payload["weight_update_seconds"])
        stage_times["sequence_regularization"] += float(payload["sequence_regularization_seconds"])
        stage_times["contact_regularization"] += float(payload["contact_regularization_seconds"])
        stage_times["convergence_check"] += float(payload["convergence_check_seconds"])
        stage_times["iteration_total"] += float(payload["runtime_iteration_seconds"])
        for key in (
            "weighted_fit",
            "residual_update",
            "weight_update",
            "sequence_regularization",
            "contact_regularization",
            "convergence_check",
            "iteration_total",
        ):
            stage_counts[key] += 1

    robust_kwargs = dict(case.robust_kwargs)
    robust_kwargs["diagnostic_callback"] = callback
    profiler = cProfile.Profile()
    robust_start = time.perf_counter()
    profiler.enable()
    result = solve_trajectory_robust_alignment(
        case.coords,
        reference_mode=DEFAULT_REFERENCE_MODE,
        warm_start=True if case.benchmark_group.startswith("mdcath") else False,
        **robust_kwargs,
    )
    profiler.disable()
    robust_runtime = time.perf_counter() - robust_start

    stack_start = time.perf_counter()
    robust_aligned = robust_aligned_coords_from_result(result)
    stack_runtime = time.perf_counter() - stack_start

    rmsf_start = time.perf_counter()
    robust_rmsf = compute_rmsf(robust_aligned)
    ordinary_rmsf = compute_rmsf(ordinary_aligned)
    rmsf_runtime = time.perf_counter() - rmsf_start

    profile_start = time.perf_counter()
    stability = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
    mobility = np.clip(1.0 - stability, 0.0, 1.0)
    core = select_core_by_target_fraction(stability, DEFAULT_CORE_THRESHOLD)
    segment_ids = detect_flexible_segments(
        mobility,
        percentile_cutoff=DEFAULT_FLEXIBLE_PERCENTILE / 100.0,
        min_segment_length=3,
        merge_gap=1,
        chain=[str(item.get("chain", "_")) for item in case.residue_metadata],
    )
    corr, _ = stable_spearman(robust_rmsf, mobility)
    profile_rows = [
        {
            "index": index,
            "chain": case.residue_metadata[index].get("chain", "_") if index < len(case.residue_metadata) else "_",
            "robust_RMSF": float(robust_rmsf[index]),
            "ordinary_RMSF": float(ordinary_rmsf[index]),
            "stability": float(stability[index]),
            "mobility": float(mobility[index]),
            "core": int(core.core_mask[index]),
            "segment_id": int(segment_ids[index]),
        }
        for index in range(case.coords.shape[1])
    ]
    profile_runtime = time.perf_counter() - profile_start

    robust_metric_start = time.perf_counter()
    robust_rmsd = mean_rmsd(robust_aligned, case.coords[0])
    robust_metric_runtime = time.perf_counter() - robust_metric_start

    total_runtime = time.perf_counter() - total_start

    add_stage(rows, case=case, stage="full_profiled_workflow_total", runtime=total_runtime, total=total_runtime, notes="ordinary+robust+RMSF/profile; no external tools")
    add_stage(rows, case=case, stage="ordinary_initialization_fit", runtime=stage_times["ordinary_initialization_fit"], total=total_runtime, call_count=stage_counts["ordinary_initialization_fit"], notes=f"ordinary_mean_RMSD={ordinary_rmsd:.6g}")
    add_stage(rows, case=case, stage="mdcorealign_robust_solver_total", runtime=robust_runtime, total=total_runtime, call_count=int(case.coords.shape[0]), notes=f"mean_iterations={result.average_iterations_per_frame:.6g}; converged_fraction={result.converged_fraction:.6g}")
    for stage in (
        "weighted_fit",
        "residual_update",
        "weight_update",
        "sequence_regularization",
        "contact_regularization",
        "convergence_check",
    ):
        add_stage(rows, case=case, stage=stage, runtime=stage_times[stage], total=total_runtime, call_count=stage_counts[stage])
    unattributed = max(0.0, robust_runtime - stage_times["iteration_total"])
    add_stage(rows, case=case, stage="robust_unattributed_final_refit_and_overhead", runtime=unattributed, total=total_runtime, call_count=int(case.coords.shape[0]), notes="includes final refits, Python dispatch, trajectory container overhead, and callback overhead")
    add_stage(rows, case=case, stage="stack_robust_aligned_coords", runtime=stack_runtime, total=total_runtime)
    add_stage(rows, case=case, stage="rmsf_calculation", runtime=rmsf_runtime, total=total_runtime, call_count=2)
    add_stage(rows, case=case, stage="residue_profile_table_construction", runtime=profile_runtime, total=total_runtime, call_count=len(profile_rows), notes=f"RMSF_mobility_spearman={corr:.6g}")
    add_stage(rows, case=case, stage="unified_metric_robust", runtime=robust_metric_runtime, total=total_runtime, notes=f"robust_mean_RMSD={robust_rmsd:.6g}")

    for func_name, call_count, cumulative in top_cprofile_rows(profiler):
        add_stage(
            rows,
            case=case,
            stage=f"cprofile:{func_name}",
            runtime=cumulative,
            total=robust_runtime,
            call_count=call_count,
            notes="top cumulative-time function inside robust solver",
        )
    return rows


def summarize_report(rows: list[dict[str, object]], output_tsv: Path, report_path: Path) -> None:
    """Write a concise human-readable hotspot report."""

    aggregate: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row["stage"])
        if stage.startswith("cprofile:"):
            continue
        aggregate.setdefault(stage, {"runtime": 0.0, "percent": 0.0})
        aggregate[stage]["runtime"] += float(row["runtime_sec"])
        aggregate[stage]["percent"] += float(row["percent_total"])
        counts[stage] = counts.get(stage, 0) + 1
    ranked = sorted(aggregate.items(), key=lambda item: item[1]["runtime"], reverse=True)
    case_count = len({str(row["trajectory_id"]) for row in rows})
    lines = [
        "# MDCoreAlign Runtime Hotspot Report",
        "",
        f"- Profile table: `{output_tsv}`",
        f"- Profiled cases: {case_count}",
        "- Scope: existing Python implementation, current solver settings, no external MDLovoFit/THESEUS calls.",
        "",
        "## Aggregate Stage Ranking",
        "",
        "| Stage | Total runtime (s) | Mean percent per case |",
        "|---|---:|---:|",
    ]
    for stage, payload in ranked:
        mean_percent = payload["percent"] / max(1, counts[stage])
        lines.append(f"| `{stage}` | {payload['runtime']:.6f} | {mean_percent:.2f}% |")
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- `mdcorealign_robust_solver_total` is the primary target if it dominates the full profiled workflow.",
            "- `weighted_fit`, `residual_update`, and `weight_update` reflect per-iteration inner-loop cost from the diagnostic callback.",
            "- `robust_unattributed_final_refit_and_overhead` includes final weighted refits, Python dispatch, object construction, and diagnostic callback overhead; if large, the optimization target is loop overhead rather than pure SVD.",
            "- RMSF/profile stages are outside the mdCATH runner's MDCoreAlign algorithm timer but are still reported for full-workflow awareness.",
            "",
            "## Low-Risk Optimization Candidates",
            "",
            "1. Disable per-iteration history recording in benchmark runners when history is not consumed, while preserving final weights, transforms, RMSD, RMSF, and core labels.",
            "2. Add fast weighted Kabsch primitives and regression-test them against the current Kabsch implementation.",
            "3. Add a named fast convergence mode only if output-equivalence checks pass; otherwise keep it opt-in and do not use it for default claims.",
            "",
            "## Higher-Risk Changes",
            "",
            "- Changing default convergence thresholds may alter weights and flexible segments.",
            "- Disabling contact regularization in formal systems changes the intended biological regularization and must remain a labeled ablation, not a silent default.",
            "- Float32 acceleration should not be used in formal benchmarks unless strict regression tolerances pass.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the hotspot profiler."""

    args = parse_args()
    cases: list[ProfileCase] = []
    load_rows: list[dict[str, object]] = []
    if not args.skip_mdcath:
        start = time.perf_counter()
        cases.extend(load_mdcath_cases(args))
        load_rows.append({"benchmark_group": "mdcath_cath_balanced_fullframes", "runtime_sec": time.perf_counter() - start})
    if not args.skip_formal:
        start = time.perf_counter()
        cases.extend(load_formal_cases(args))
        load_rows.append({"benchmark_group": "formal_four_md_systems", "runtime_sec": time.perf_counter() - start})
    all_rows: list[dict[str, object]] = []
    for case in cases:
        case_rows = profile_case(case)
        all_rows.extend(case_rows)
    for row in load_rows:
        pseudo = ProfileCase(
            trajectory_id=f"{row['benchmark_group']}:loader",
            benchmark_group=str(row["benchmark_group"]),
            coords=np.zeros((1, 1, 3), dtype=float),
            residue_metadata=[{}],
            robust_kwargs={},
        )
        add_stage(
            all_rows,
            case=pseudo,
            stage="input_loading_total",
            runtime=float(row["runtime_sec"]),
            total=float(row["runtime_sec"]) if float(row["runtime_sec"]) > 0 else 1.0,
            notes="aggregate loader/extraction time for this benchmark group",
        )
    write_tsv(args.output_tsv, all_rows, PROFILE_FIELDS)
    summarize_report(all_rows, args.output_tsv, args.report)
    print(args.output_tsv)
    print(args.report)


if __name__ == "__main__":
    main()
