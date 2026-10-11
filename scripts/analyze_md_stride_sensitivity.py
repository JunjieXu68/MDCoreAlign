"""Analyze frame-stride sensitivity for formal MD trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from md_trajectory_focus_common import (  # noqa: E402
    FORMAL_SYSTEMS,
    MD_FOCUS_ROOT,
    common_manifest,
    compare_to_reference,
    formatted_range,
    load_formal_system,
    solve_default_soft,
    stride_values,
    subset_loaded_system,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "stride_sensitivity"

FIELDS = [
    "system",
    "stride",
    "n_frames",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "runtime_algorithm_seconds",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "jaccard_core_vs_stride1",
    "spearman_mobility_vs_stride1",
    "flexible_segment_overlap_vs_stride1",
    "converged",
    "n_iterations",
    "notes",
]

METRIC_FIELDS = [
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "runtime_algorithm_seconds",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "converged",
    "n_iterations",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def stride_subset_indices(n_frames: int, stride: int) -> np.ndarray:
    """Return frame indices for one analysis stride."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    return np.arange(0, n_frames, stride, dtype=int)


def run_analysis(systems: list[str]) -> list[dict[str, object]]:
    """Run stride sensitivity analysis."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        reference_solution = None
        for stride in stride_values():
            indices = stride_subset_indices(system.trajectory.n_frames, stride)
            stride_system = subset_loaded_system(
                system,
                system.trajectory.coords[indices],
                note_suffix=f"analysis_stride_{stride}",
            )
            solution = solve_default_soft(stride_system)
            if stride == 1:
                reference_solution = solution
            assert reference_solution is not None
            comparison = compare_to_reference(solution, reference_solution, suffix="stride1")
            rows.append(
                {
                    "system": system.label,
                    "stride": stride,
                    "n_frames": int(indices.shape[0]),
                    **{field: solution.metrics.get(field, "") for field in METRIC_FIELDS},
                    **comparison,
                    "notes": "formal_md_stride_rerun_over_formal_loaded_frames",
                }
            )
            print(f"[stride] {system.label} stride={stride}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    non_ref = [row for row in rows if str(row["stride"]) != "1"]
    core = [float(row["jaccard_core_vs_stride1"]) for row in non_ref]
    mobility = [float(row["spearman_mobility_vs_stride1"]) for row in non_ref if row["spearman_mobility_vs_stride1"] != ""]
    runtime_ratio = []
    by_system_stride1 = {row["system"]: row for row in rows if str(row["stride"]) == "1"}
    for row in non_ref:
        base = by_system_stride1[row["system"]]
        runtime_ratio.append(float(row["runtime_algorithm_seconds"]) / float(base["runtime_algorithm_seconds"]))
    by_system: dict[str, list[float]] = {}
    for row in non_ref:
        by_system.setdefault(str(row["system"]), []).append(float(row["jaccard_core_vs_stride1"]))
    sensitive = min(by_system, key=lambda system: float(np.mean(by_system[system]))) if by_system else "unavailable"
    safe = "stride 5 or 10" if core and min(core) >= 0.5 else "system-dependent; inspect table before downsampling"
    lines = [
        "# MD Frame Stride Sensitivity Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Does reducing frame density change stable-core assignments? Core Jaccard vs stride1 ranges from `{formatted_range(core)}`.",
        f"2. Does stride mainly affect runtime or scientific conclusions? Runtime ratios vs stride1 range from `{formatted_range(runtime_ratio)}`; mobility Spearman ranges from `{formatted_range(mobility)}`.",
        f"3. What stride appears safe for practical use? `{safe}`.",
        f"4. Are any systems unusually stride-sensitive? `{sensitive}` has the lowest mean core Jaccard across non-reference strides.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "stride_sensitivity.tsv", rows, FIELDS)
    write_table(args.output_dir / "stride_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "stride_sensitivity_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_stride_sensitivity.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "stride_sensitivity.tsv",
            args.output_dir / "stride_sensitivity_summary.md",
            args.output_dir / "stride_plot_data.tsv",
        ],
        extra={"stride_values": stride_values()},
    )
    print(f"Wrote stride sensitivity outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
