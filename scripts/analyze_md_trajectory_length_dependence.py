"""Analyze trajectory-prefix length dependence for formal MD trajectories."""

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
    prefix_definitions,
    solve_default_soft,
    subset_loaded_system,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "trajectory_length_dependence"

FIELDS = [
    "system",
    "prefix_fraction",
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
    """Run trajectory length dependence analysis."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        definitions = prefix_definitions(system.trajectory.n_frames)
        full_definition = definitions[-1]
        full_system = subset_loaded_system(
            system,
            system.trajectory.coords[: int(full_definition["frame_end"])],
            note_suffix="prefix_full_reference",
        )
        full_solution = solve_default_soft(full_system)
        for definition in definitions:
            end = int(definition["frame_end"])
            prefix_system = subset_loaded_system(
                system,
                system.trajectory.coords[:end],
                note_suffix=str(definition["prefix_id"]),
            )
            solution = full_solution if definition["prefix_fraction"] == 1.0 else solve_default_soft(prefix_system)
            comparison = compare_to_reference(solution, full_solution, suffix="full")
            rows.append(
                {
                    "system": system.label,
                    "prefix_fraction": definition["prefix_fraction"],
                    "n_frames": end,
                    **{field: solution.metrics.get(field, "") for field in METRIC_FIELDS},
                    **comparison,
                    "notes": f"formal_md_prefix_rerun:{definition['prefix_id']}",
                }
            )
            print(f"[length] {system.label} {definition['prefix_id']}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    non_full = [row for row in rows if float(row["prefix_fraction"]) < 1.0]
    core = [float(row["jaccard_core_vs_full"]) for row in non_full]
    mobility = [float(row["spearman_mobility_vs_full"]) for row in non_full if row["spearman_mobility_vs_full"] != ""]
    ratio_values = [float(row["noncore_core_RMSF_ratio"]) for row in non_full if row["noncore_core_RMSF_ratio"] != ""]
    by_system: dict[str, list[tuple[float, float]]] = {}
    for row in non_full:
        by_system.setdefault(str(row["system"]), []).append((float(row["prefix_fraction"]), float(row["jaccard_core_vs_full"])))
    thresholds = {}
    for system, values in by_system.items():
        passing = [fraction for fraction, jaccard in values if jaccard >= 0.8]
        thresholds[system] = min(passing) if passing else None
    longest = [system for system, threshold in thresholds.items() if threshold is None or threshold >= 0.75]
    lines = [
        "# MD Trajectory Length Dependence Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        "1. How much trajectory is needed before the stable core resembles the full-trajectory result? "
        + "; ".join(f"{system}: {threshold if threshold is not None else 'not reached'}" for system, threshold in thresholds.items()),
        f"2. Which systems require longer sampling? `{', '.join(longest) if longest else 'none by core Jaccard >= 0.8 criterion'}`.",
        f"3. Is RMSF/core separation stable before full-length sampling? Noncore/core RMSF ratio across prefixes ranges from `{formatted_range(ratio_values)}`.",
        f"4. Are early trajectory prefixes misleading for any system? Core Jaccard vs full ranges from `{formatted_range(core)}` and mobility Spearman from `{formatted_range(mobility)}`; low 10% values should be treated cautiously.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "trajectory_length_dependence.tsv", rows, FIELDS)
    write_table(args.output_dir / "trajectory_length_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "trajectory_length_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_trajectory_length_dependence.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "trajectory_length_dependence.tsv",
            args.output_dir / "trajectory_length_summary.md",
            args.output_dir / "trajectory_length_plot_data.tsv",
        ],
        extra={"prefix_definitions": prefix_definitions(100)},
    )
    print(f"Wrote trajectory length outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
