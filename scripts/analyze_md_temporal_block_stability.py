"""Analyze temporal block stability for formal MD trajectories."""

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
    md_block_definitions,
    solve_default_soft,
    subset_loaded_system,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "temporal_block_stability"

FIELDS = [
    "system",
    "block_id",
    "frame_start",
    "frame_end",
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
    "jaccard_core_vs_full",
    "spearman_mobility_vs_full",
    "flexible_segment_overlap_vs_full",
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


def run_analysis(systems: list[str]) -> list[dict[str, object]]:
    """Run temporal block stability analysis."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        definitions = md_block_definitions(system.trajectory.n_frames)
        full_def = next(item for item in definitions if item["block_id"] == "full")
        full_system = subset_loaded_system(
            system,
            system.trajectory.coords[int(full_def["frame_start"]): int(full_def["frame_end"])],
            note_suffix="temporal_full_reference",
        )
        full_solution = solve_default_soft(full_system)
        for definition in definitions:
            start = int(definition["frame_start"])
            end = int(definition["frame_end"])
            block_system = subset_loaded_system(
                system,
                system.trajectory.coords[start:end],
                note_suffix=str(definition["block_id"]),
            )
            solution = full_solution if definition["block_id"] == "full" else solve_default_soft(block_system)
            comparison = compare_to_reference(solution, full_solution, suffix="full")
            row = {
                "system": system.label,
                "block_id": definition["block_id"],
                "frame_start": start,
                "frame_end": end,
                "n_frames": end - start,
                **{field: solution.metrics.get(field, "") for field in METRIC_FIELDS},
                **comparison,
                "notes": "formal_md_temporal_block_rerun",
            }
            rows.append(row)
            print(f"[temporal] {system.label} {definition['block_id']}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    block_rows = [row for row in rows if row["block_id"] != "full"]
    core_values = [float(row["jaccard_core_vs_full"]) for row in block_rows]
    mobility_values = [float(row["spearman_mobility_vs_full"]) for row in block_rows if row["spearman_mobility_vs_full"] != ""]
    flex_values = [float(row["flexible_segment_overlap_vs_full"]) for row in block_rows]
    by_system: dict[str, list[float]] = {}
    for row in block_rows:
        by_system.setdefault(str(row["system"]), []).append(float(row["jaccard_core_vs_full"]))
    sensitive = min(by_system, key=lambda system: float(np.mean(by_system[system]))) if by_system else "unavailable"
    failures = [row for row in rows if str(row["converged"]).lower() == "false" and row["block_id"] != "full"]
    lines = [
        "# MD Temporal Block Stability Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Are stable cores consistent across trajectory blocks? Core Jaccard vs full ranges from `{formatted_range(core_values)}`.",
        f"2. Are mobility profiles consistent across blocks? Mobility Spearman vs full ranges from `{formatted_range(mobility_values)}`.",
        f"3. Which system is most time-window sensitive? `{sensitive}` has the lowest mean block-vs-full core Jaccard.",
        f"4. Does any block show a failure or weight-collapse case? `{len(failures)}` non-full block rows have `converged=False`; inspect table before interpreting as failure because max-iteration stopping is common.",
        f"5. Are flexible segments stable across time windows? Flexible-segment mask overlap vs full ranges from `{formatted_range(flex_values)}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "temporal_block_stability.tsv", rows, FIELDS)
    write_table(args.output_dir / "temporal_block_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "temporal_block_stability_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_temporal_block_stability.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "temporal_block_stability.tsv",
            args.output_dir / "temporal_block_stability_summary.md",
            args.output_dir / "temporal_block_plot_data.tsv",
        ],
        extra={"block_definitions": md_block_definitions(100)},
    )
    print(f"Wrote temporal block stability outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
