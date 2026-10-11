"""Analyze reference-choice sensitivity for formal MD trajectories."""

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
    REFERENCE_STRATEGIES,
    common_manifest,
    compare_to_reference,
    formatted_range,
    load_formal_system,
    solve_default_soft,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "reference_sensitivity"

FIELDS = [
    "system",
    "reference_strategy",
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
    "jaccard_core_vs_default_reference",
    "spearman_mobility_vs_default_reference",
    "flexible_segment_overlap_vs_default_reference",
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
    parser.add_argument("--strategies", nargs="+", choices=REFERENCE_STRATEGIES, default=REFERENCE_STRATEGIES)
    return parser.parse_args()


def run_analysis(systems: list[str], strategies: list[str]) -> list[dict[str, object]]:
    """Run reference sensitivity analysis."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        default_solution = solve_default_soft(system, reference_strategy="first_frame")
        for strategy in strategies:
            solution = default_solution if strategy == "first_frame" else solve_default_soft(system, reference_strategy=strategy)
            comparison = compare_to_reference(solution, default_solution, suffix="default_reference")
            rows.append(
                {
                    "system": system.label,
                    "reference_strategy": strategy,
                    **{field: solution.metrics.get(field, "") for field in METRIC_FIELDS},
                    **comparison,
                    "notes": "formal_md_reference_strategy_rerun",
                }
            )
            print(f"[reference] {system.label} {strategy}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    non_default = [row for row in rows if row["reference_strategy"] != "first_frame"]
    core = [float(row["jaccard_core_vs_default_reference"]) for row in non_default]
    mobility = [float(row["spearman_mobility_vs_default_reference"]) for row in non_default if row["spearman_mobility_vs_default_reference"] != ""]
    by_system: dict[str, list[float]] = {}
    for row in non_default:
        by_system.setdefault(str(row["system"]), []).append(float(row["jaccard_core_vs_default_reference"]))
    sensitive = min(by_system, key=lambda system: float(np.mean(by_system[system]))) if by_system else "unavailable"
    lines = [
        "# MD Reference Sensitivity Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Are stable-core assignments robust to reference-frame choice? Core Jaccard vs first-frame default ranges from `{formatted_range(core)}`.",
        f"2. Does reference choice mainly affect RMSD or mobility interpretation? Mobility Spearman vs default ranges from `{formatted_range(mobility)}`; inspect RMSD columns for geometry shifts.",
        "3. Which reference strategy should be recommended as default? `first_frame` remains the most reproducible and least computationally nested default for formal MD trajectory mode.",
        f"4. Are any systems reference-sensitive? `{sensitive}` has the lowest mean core Jaccard across non-default reference strategies.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems, args.strategies)
    write_table(args.output_dir / "reference_sensitivity.tsv", rows, FIELDS)
    write_table(args.output_dir / "reference_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "reference_sensitivity_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_reference_sensitivity.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "reference_sensitivity.tsv",
            args.output_dir / "reference_sensitivity_summary.md",
            args.output_dir / "reference_plot_data.tsv",
        ],
        extra={"reference_strategies": args.strategies},
    )
    print(f"Wrote reference sensitivity outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
