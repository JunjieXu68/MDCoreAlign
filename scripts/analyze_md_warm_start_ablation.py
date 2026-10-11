"""Analyze warm-start behavior in MD trajectory mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from md_trajectory_focus_common import (  # noqa: E402
    FORMAL_SYSTEMS,
    MD_FOCUS_ROOT,
    common_manifest,
    compare_to_reference,
    load_formal_system,
    solve_default_soft,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "warm_start_ablation"

FIELDS = [
    "system",
    "warm_start",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "runtime_algorithm_seconds",
    "runtime_total_seconds",
    "n_iterations",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "spearman_mobility_vs_warm_start_default",
    "jaccard_core_vs_warm_start_default",
    "converged",
    "notes",
]

METRIC_FIELDS = [
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "runtime_algorithm_seconds",
    "runtime_total_seconds",
    "n_iterations",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "converged",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def run_analysis(systems: list[str]) -> list[dict[str, object]]:
    """Run warm-start ablation."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        warm = solve_default_soft(system, warm_start=True)
        cold = solve_default_soft(system, warm_start=False)
        for label, solution in (("True", warm), ("False", cold)):
            comparison = compare_to_reference(solution, warm, suffix="warm_start_default")
            rows.append(
                {
                    "system": system.label,
                    "warm_start": label,
                    **{field: solution.metrics.get(field, "") for field in METRIC_FIELDS},
                    "spearman_mobility_vs_warm_start_default": comparison["spearman_mobility_vs_warm_start_default"],
                    "jaccard_core_vs_warm_start_default": comparison["jaccard_core_vs_warm_start_default"],
                    "notes": "formal_md_warm_start_ablation",
                }
            )
        print(f"[warm-start] {system.label}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    by_system = {}
    for row in rows:
        by_system.setdefault(row["system"], {})[row["warm_start"]] = row
    faster = 0
    iteration_reduced = 0
    changed = []
    for system, pair in by_system.items():
        warm = pair.get("True")
        cold = pair.get("False")
        if not warm or not cold:
            continue
        faster += float(warm["runtime_algorithm_seconds"]) < float(cold["runtime_algorithm_seconds"])
        iteration_reduced += float(warm["n_iterations"]) < float(cold["n_iterations"])
        if float(cold["jaccard_core_vs_warm_start_default"]) < 0.99:
            changed.append(system)
    lines = [
        "# MD Warm-Start Ablation Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Does warm start reduce runtime or iterations? Warm start was faster in `{faster}/{len(by_system)}` systems and reduced average iterations in `{iteration_reduced}/{len(by_system)}` systems.",
        f"2. Does warm start change final alignment or core assignment? Systems with core Jaccard < 0.99 vs warm-start result: `{', '.join(changed) if changed else 'none'}`.",
        "3. Is warm start mainly an efficiency feature or a scientific feature? Treat it as an efficiency feature unless the comparison table shows material core/mobility changes.",
        "4. Should warm start remain default for MD trajectory mode? Keep it optional unless repeated benchmarks show consistent speedup without changing scientific outputs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "warm_start_ablation.tsv", rows, FIELDS)
    write_table(args.output_dir / "warm_start_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "warm_start_ablation_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_warm_start_ablation.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "warm_start_ablation.tsv",
            args.output_dir / "warm_start_ablation_summary.md",
            args.output_dir / "warm_start_plot_data.tsv",
        ],
        extra={"warm_start_values": [True, False], "comparison_default": True},
    )
    print(f"Wrote warm-start ablation outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
