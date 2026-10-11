#!/usr/bin/env python
"""Rerun optimized MDCoreAlign on the four formal trajectory systems."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.external_baseline_theseus_formal import DATASETS, load_dataset  # noqa: E402
from mdsuperpose import benchmark_soft_weight_robust_alignment  # noqa: E402


FIELDS = [
    "system",
    "n_frames",
    "n_residues",
    "old_mdcorealign_runtime_sec",
    "optimized_mdcorealign_runtime_sec",
    "mdlovofit_runtime_sec",
    "speedup_vs_old_mdcorealign",
    "speedup_vs_mdlovofit",
    "old_mdcorealign_unified_RMSD",
    "optimized_mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "RMSD_diff_old_vs_optimized",
    "core_jaccard_old_vs_optimized",
    "mobility_spearman_old_vs_optimized",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument(
        "--old-summary",
        type=Path,
        default=Path("results/higher_tier_external_baseline_theseus_formal/combined_theseus_formal_summary.csv"),
    )
    parser.add_argument(
        "--old-runs-root",
        type=Path,
        default=Path("results/higher_tier_external_baseline_theseus_formal"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/performance_optimization/post_optimization_timing/formal_four_systems_optimized.tsv"),
    )
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=Path("results/performance_optimization/post_optimization_timing/formal_four_systems_optimized_runs.json"),
    )
    return parser.parse_args()


def fnum(value: object) -> float | None:
    """Convert a table value to float when possible."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def read_old_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read old formal benchmark summary rows keyed by dataset and method."""

    with path.open(newline="") as handle:
        return {
            (row["dataset_name"], row["method_name"]): row
            for row in csv.DictReader(handle)
        }


def old_core_indices(old_runs_root: Path, dataset: str) -> set[int] | None:
    """Load old selected core indices from previous formal run records."""

    path = old_runs_root / dataset / f"{dataset}_theseus_formal_runs.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    runs = payload.get("runs", [])
    for run in runs:
        for result in run.get("results", []):
            if result.get("method_name") != "soft_weight_no_regularization":
                continue
            indices = result.get("extra_metadata", {}).get("selected_core_indices")
            if indices is not None:
                return {int(index) for index in indices}
    return None


def jaccard(first: set[int] | None, second: set[int] | None) -> float | str:
    """Return Jaccard overlap for two optional core-index sets."""

    if first is None or second is None:
        return "NA"
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""

    return float(statistics.mean(values))


def main() -> None:
    """Run optimized formal trajectory timings."""

    args = parse_args()
    old = read_old_summary(args.old_summary)
    rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for dataset in args.datasets:
        trajectory, preprocessing_steps = load_dataset(dataset)
        old_soft = old[(dataset, "soft_weight_no_regularization")]
        old_mdlovo = old[(dataset, "external_mdlovofit")]
        optimized_results = []
        optimized_core_sets = []
        for repeat_index in range(args.repeats):
            result = benchmark_soft_weight_robust_alignment(
                trajectory,
                method_name="soft_weight_no_regularization",
                core_fraction=args.fraction,
                rule="huber",
                delta=0.25,
                max_iterations=20,
                tolerance=1e-6,
                record_history=False,
            )
            optimized_results.append(result)
            indices = result.extra_metadata.get("selected_core_indices")
            optimized_core_sets.append({int(index) for index in indices} if indices is not None else None)
            run_records.append(
                {
                    "dataset": dataset,
                    "repeat_index": repeat_index,
                    "runtime_seconds": result.runtime_seconds,
                    "unified_posthoc_rmsd": result.rmsd,
                    "weighted_rmsd": result.weighted_rmsd,
                    "iteration_count": result.iteration_count,
                    "core_fraction": result.core_fraction,
                    "selected_core_indices": sorted(optimized_core_sets[-1]) if optimized_core_sets[-1] is not None else None,
                    "preprocessing_steps": preprocessing_steps,
                }
            )

        optimized_runtime = mean([float(result.runtime_seconds) for result in optimized_results])
        optimized_rmsd = mean([float(result.rmsd) for result in optimized_results])
        old_runtime = fnum(old_soft["runtime_seconds_mean"])
        old_rmsd = fnum(old_soft["unified_posthoc_rmsd_mean"])
        mdlovo_runtime = fnum(old_mdlovo["runtime_seconds_mean"])
        mdlovo_rmsd = fnum(old_mdlovo["unified_posthoc_rmsd_mean"])
        old_core = old_core_indices(args.old_runs_root, dataset)
        optimized_core = optimized_core_sets[0] if optimized_core_sets else None
        rows.append(
            {
                "system": dataset,
                "n_frames": trajectory.n_frames,
                "n_residues": trajectory.n_atoms,
                "old_mdcorealign_runtime_sec": old_runtime,
                "optimized_mdcorealign_runtime_sec": optimized_runtime,
                "mdlovofit_runtime_sec": mdlovo_runtime,
                "speedup_vs_old_mdcorealign": old_runtime / optimized_runtime if old_runtime else "NA",
                "speedup_vs_mdlovofit": mdlovo_runtime / optimized_runtime if mdlovo_runtime else "NA",
                "old_mdcorealign_unified_RMSD": old_rmsd,
                "optimized_mdcorealign_unified_RMSD": optimized_rmsd,
                "mdlovofit_unified_RMSD": mdlovo_rmsd,
                "RMSD_diff_old_vs_optimized": abs(optimized_rmsd - old_rmsd) if old_rmsd is not None else "NA",
                "core_jaccard_old_vs_optimized": jaccard(old_core, optimized_core),
                "mobility_spearman_old_vs_optimized": "NA_old_mobility_not_stored",
                "notes": "optimized rerun uses the same no-regularization formal soft settings; external baseline values are copied from the frozen formal summary",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{value:.10g}"
                        if isinstance(value, float) and np.isfinite(value)
                        else value
                    )
                    for field, value in row.items()
                }
            )
    args.runs_json.write_text(json.dumps(run_records, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
