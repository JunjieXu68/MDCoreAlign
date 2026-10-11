#!/usr/bin/env python
"""Aggregate mdCATH per-trajectory summaries into manuscript-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    PER_TRAJECTORY_COLUMNS,
    RESULT_ROOT,
    ensure_result_dirs,
    read_tsv,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-trajectory-root", type=Path, default=RESULT_ROOT / "per_trajectory")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv")
    parser.add_argument("--mode", choices=("all", "domain_canonical", "full_trajectory", "cath_balanced_subset"), default="all")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "aggregate")
    return parser.parse_args()


def load_summaries(root: Path) -> list[dict[str, object]]:
    """Load all per-trajectory summary JSON files."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception:
            rows.append({"trajectory_id": path.parent.name, "status": "failed", "failure_reason": "summary_json_unreadable"})
    return rows


def as_float(value: object) -> float | None:
    """Convert a value to float or None."""

    try:
        if value in (None, "NA", ""):
            return None
        return float(value)
    except Exception:
        return None


def summarize_group(rows: list[dict[str, object]], group_name: str, group_value: str) -> dict[str, object]:
    """Summarize a group of per-trajectory rows."""

    success = [row for row in rows if row.get("status") == "success"]
    values = lambda key: [val for row in success if (val := as_float(row.get(key))) is not None]
    attempted = len(rows)
    output: dict[str, object] = {
        group_name: group_value,
        "n_attempted": attempted,
        "n_success": len(success),
        "n_failed": attempted - len(success),
        "success_rate": len(success) / attempted if attempted else 0.0,
    }
    metric_map = {
        "median_runtime_sec": "runtime_seconds",
        "mean_runtime_sec": "runtime_seconds",
        "p95_runtime_sec": "runtime_seconds",
        "median_peak_memory_mb": "peak_memory_mb",
        "median_n_iterations": "iteration_count_mean",
        "convergence_rate": "converged_fraction",
        "median_unified_RMSD": "unified_RMSD",
        "median_weighted_RMSD": "weighted_RMSD",
        "median_stable_core_fraction": "stable_core_fraction",
        "median_mobile_fraction": "mobile_fraction",
        "median_n_core_segments": "n_core_segments",
        "median_core_fragmentation_score": "core_fragmentation_score",
        "median_robust_RMSF_mean": "robust_RMSF_mean",
        "median_robust_to_ordinary_RMSF_ratio": "robust_to_ordinary_RMSF_ratio",
        "median_noncore_core_RMSF_ratio": "noncore_core_RMSF_ratio",
        "median_RMSF_mobility_spearman": "RMSF_mobility_spearman",
        "median_delta_unified_RMSD_vs_fullset": "delta_unified_RMSD_vs_fullset",
    }
    for output_key, source_key in metric_map.items():
        vals = values(source_key)
        if not vals:
            output[output_key] = "NA"
        elif output_key.startswith("mean_"):
            output[output_key] = sum(vals) / len(vals)
        elif output_key.startswith("p95_"):
            vals = sorted(vals)
            output[output_key] = vals[min(len(vals) - 1, int(0.95 * (len(vals) - 1)))]
        elif output_key == "convergence_rate":
            output[output_key] = sum(vals) / len(vals)
        else:
            output[output_key] = median(vals)
    spearman_vals = values("RMSF_mobility_spearman")
    ratio_vals = values("noncore_core_RMSF_ratio")
    output["percent_positive_RMSF_mobility_spearman"] = (
        sum(1 for val in spearman_vals if val > 0) / len(spearman_vals) if spearman_vals else "NA"
    )
    output["percent_noncore_RMSF_greater_than_core"] = (
        sum(1 for val in ratio_vals if val > 1) / len(ratio_vals) if ratio_vals else "NA"
    )
    return output


def grouped(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    """Group rows by a key."""

    out: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        value = str(row.get(key, "NA"))
        out.setdefault(value, []).append(row)
    return out


def length_bin(row: dict[str, object]) -> str:
    """Return a simple residue-count bin."""

    n_res = as_float(row.get("n_residues"))
    if n_res is None:
        return "unknown"
    if n_res < 75:
        return "<75"
    if n_res < 150:
        return "75-149"
    if n_res < 250:
        return "150-249"
    return ">=250"


def main() -> None:
    """Aggregate outputs."""

    args = parse_args()
    if args.mode == "cath_balanced_subset":
        if args.input_dir is None:
            args.input_dir = RESULT_ROOT / "cath_balanced_subset_fullframes"
        args.per_trajectory_root = args.input_dir / "per_trajectory"
        if args.output_dir == RESULT_ROOT / "aggregate":
            args.output_dir = args.input_dir / "aggregate"
    ensure_result_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summaries(args.per_trajectory_root)
    if args.mode == "cath_balanced_subset":
        from scripts.mdcath_benchmark.run_mdcath_subset_benchmark import SUMMARY_COLUMNS, write_class_summary

        summary_path = args.output_dir / "cath_balanced_5_per_class_fullframes_summary.tsv"
        write_tsv(summary_path, rows, SUMMARY_COLUMNS)
        write_class_summary(args.output_dir / "cath_balanced_5_per_class_class_summary.tsv", rows)
        write_json(
            args.output_dir / "cath_balanced_5_per_class_aggregation_manifest.json",
            {
                "mode": args.mode,
                "input_dir": str(args.input_dir),
                "per_trajectory_root": str(args.per_trajectory_root),
                "n_loaded_summaries": len(rows),
                "summary": str(summary_path),
                "note": "CATH-balanced subset aggregation; not a full mdCATH benchmark.",
            },
        )
        print(summary_path)
        return
    manifest = read_tsv(args.manifest)
    write_tsv(args.output_dir / "mdcath_full_trajectory_summary.tsv", rows, PER_TRAJECTORY_COLUMNS)

    canonical_ids = {row["trajectory_id"] for row in manifest if row.get("include_in_domain_canonical_benchmark") == "1"}
    canonical_rows = [row for row in rows if row.get("trajectory_id") in canonical_ids]
    write_tsv(args.output_dir / "mdcath_domain_canonical_summary.tsv", canonical_rows, PER_TRAJECTORY_COLUMNS)
    write_tsv(
        args.output_dir / "mdcath_domain_canonical_failure_summary.tsv",
        [
            {
                "failure_reason": reason,
                "n_failed": len(group_rows),
                "example_trajectory_id": group_rows[0].get("trajectory_id", "NA") if group_rows else "NA",
            }
            for reason, group_rows in sorted(
                grouped([row for row in canonical_rows if row.get("status") != "success"], "failure_reason").items()
            )
        ],
        ("failure_reason", "n_failed", "example_trajectory_id"),
    )

    temperature_rows = [
        summarize_group(group_rows, "temperature_K", temperature)
        for temperature, group_rows in sorted(grouped(rows, "temperature_K").items())
    ]
    write_tsv(args.output_dir / "mdcath_by_temperature_summary.tsv", temperature_rows)

    cath_rows = [
        summarize_group(group_rows, "cath_class", cath_class)
        for cath_class, group_rows in sorted(grouped(rows, "cath_class").items())
    ]
    write_tsv(args.output_dir / "mdcath_by_cath_class_summary.tsv", cath_rows)

    length_groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        length_groups.setdefault(length_bin(row), []).append(row)
    length_rows = [
        summarize_group(group_rows, "length_bin", bin_name)
        for bin_name, group_rows in sorted(length_groups.items())
    ]
    write_tsv(args.output_dir / "mdcath_by_length_bin_summary.tsv", length_rows)

    failure_groups = grouped([row for row in rows if row.get("status") != "success"], "failure_reason")
    failure_rows = [
        {
            "failure_reason": reason,
            "n_failed": len(group_rows),
            "example_trajectory_id": group_rows[0].get("trajectory_id", "NA") if group_rows else "NA",
        }
        for reason, group_rows in sorted(failure_groups.items())
    ]
    write_tsv(args.output_dir / "mdcath_failure_summary.tsv", failure_rows)
    write_tsv(RESULT_ROOT / "failures/mdcath_failed_trajectories.tsv", [row for row in rows if row.get("status") != "success"])

    summary = summarize_group(rows, "scope", "all_loaded_per_trajectory_outputs")
    write_tsv(RESULT_ROOT / "manuscript_tables/mdcath_main_table_summary_metrics.tsv", [summary])
    write_tsv(RESULT_ROOT / "manuscript_tables/mdcath_failure_summary.tsv", failure_rows)
    temperature_trends = [
        {
            "trend_scope": "not_computed",
            "n_domains_with_all_temperatures": 0,
            "stable_core_fraction_vs_temperature_spearman_median": "NA",
            "robust_RMSF_mean_vs_temperature_spearman_median": "NA",
            "mobile_fraction_vs_temperature_spearman_median": "NA",
            "n_flexible_segments_vs_temperature_spearman_median": "NA",
            "notes": "Temperature trend analysis requires successful trajectories across temperatures for the same domains.",
        }
    ]
    write_tsv(RESULT_ROOT / "manuscript_tables/mdcath_temperature_trends.tsv", temperature_trends)
    external_summary = RESULT_ROOT / "external_baselines_subset/external_subset_summary.tsv"
    external_rows = read_tsv(external_summary)
    write_tsv(RESULT_ROOT / "manuscript_tables/mdcath_external_subset_comparison.tsv", external_rows)
    write_json(
        RESULT_ROOT / "aggregate/mdcath_aggregation_manifest.json",
        {
            "n_loaded_summaries": len(rows),
            "n_manifest_rows": len(manifest),
            "aggregate_tables": [
                "aggregate/mdcath_full_trajectory_summary.tsv",
                "aggregate/mdcath_domain_canonical_summary.tsv",
                "aggregate/mdcath_by_temperature_summary.tsv",
                "aggregate/mdcath_by_cath_class_summary.tsv",
                "aggregate/mdcath_by_length_bin_summary.tsv",
                "aggregate/mdcath_failure_summary.tsv",
            ],
        },
    )
    print(args.output_dir / "mdcath_full_trajectory_summary.tsv")


if __name__ == "__main__":
    main()
