#!/usr/bin/env python
"""Aggregate all-domain canonical mdCATH benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import read_tsv, write_tsv


PER_DOMAIN_COLUMNS = [
    "domain_id",
    "trajectory_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "cath_code",
    "selected_temperature_K",
    "selected_replica_label_one_based",
    "n_frames",
    "n_residues",
    "mdcorealign_status",
    "fullset_status",
    "mdlovofit_status",
    "theseus_status",
    "mdcorealign_runtime_sec",
    "fullset_runtime_sec",
    "mdlovofit_runtime_sec",
    "theseus_runtime_sec",
    "mdcorealign_unified_RMSD",
    "fullset_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_faster_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "runtime_delta_mdcorealign_minus_mdlovofit",
    "runtime_ratio_mdcorealign_over_mdlovofit",
    "RMSD_delta_mdcorealign_minus_mdlovofit",
    "stable_core_fraction",
    "mobile_fraction",
    "n_core_segments",
    "n_flexible_segments",
    "robust_RMSF_mean",
    "noncore_core_RMSF_ratio",
    "RMSF_mobility_spearman",
    "failure_reason",
    "notes",
]

GROUP_COLUMNS = [
    "group_level",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "n_domains",
    "n_success_both_mdcorealign_mdlovofit",
    "median_mdcorealign_runtime_sec",
    "median_mdlovofit_runtime_sec",
    "median_speedup_vs_mdlovofit",
    "percent_mdcorealign_faster",
    "median_mdcorealign_unified_RMSD",
    "median_mdlovofit_unified_RMSD",
    "percent_mdcorealign_lower_RMSD",
    "median_stable_core_fraction",
    "median_RMSF_mobility_spearman",
    "median_noncore_core_RMSF_ratio",
    "failure_rate",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/mdcath_all_domain_canonical_manifest.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/run_fullframes"),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate"),
    )
    return parser.parse_args()


def fnum(value: object) -> float | None:
    """Parse finite float values."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted


def median(values: list[float]) -> float | str:
    """Return median or NA."""

    return statistics.median(values) if values else "NA"


def pct(numerator: int, denominator: int) -> float | str:
    """Return percentage or NA."""

    return 100.0 * numerator / denominator if denominator else "NA"


def load_summary(path: Path) -> dict[str, Any] | None:
    """Load one per-trajectory summary."""

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def comparison_row(manifest_row: dict[str, str], summary: dict[str, Any] | None) -> dict[str, object]:
    """Merge manifest row and summary row into one per-domain result row."""

    row = dict(summary or {})
    mdca_rt = fnum(row.get("mdcorealign_runtime_sec"))
    mdlovo_rt = fnum(row.get("mdlovofit_runtime_sec"))
    mdca_rmsd = fnum(row.get("mdcorealign_unified_RMSD"))
    mdlovo_rmsd = fnum(row.get("mdlovofit_unified_RMSD"))
    return {
        "domain_id": manifest_row.get("domain_id", row.get("domain_id", "NA")),
        "trajectory_id": manifest_row.get("trajectory_id", row.get("trajectory_id", "NA")),
        "cath_class": manifest_row.get("cath_class", row.get("cath_class", "NA")),
        "cath_architecture": manifest_row.get("cath_architecture", row.get("cath_architecture", "NA")),
        "cath_topology": manifest_row.get("cath_topology", row.get("cath_topology", "NA")),
        "cath_homologous_superfamily": manifest_row.get("cath_homologous_superfamily", row.get("cath_homologous_superfamily", "NA")),
        "cath_code": manifest_row.get("cath_code", row.get("cath_code", "NA")),
        "selected_temperature_K": manifest_row.get("selected_temperature_K", row.get("selected_temperature_K", "NA")),
        "selected_replica_label_one_based": manifest_row.get("selected_replica_label_one_based", row.get("selected_replica_label_one_based", "NA")),
        "n_frames": row.get("n_frames", "NA"),
        "n_residues": row.get("n_residues", "NA"),
        "mdcorealign_status": row.get("mdcorealign_status", "missing"),
        "fullset_status": row.get("fullset_status", "missing"),
        "mdlovofit_status": row.get("mdlovofit_status", "missing"),
        "theseus_status": row.get("theseus_status", "missing"),
        "mdcorealign_runtime_sec": row.get("mdcorealign_runtime_sec", "NA"),
        "fullset_runtime_sec": row.get("fullset_runtime_sec", "NA"),
        "mdlovofit_runtime_sec": row.get("mdlovofit_runtime_sec", "NA"),
        "theseus_runtime_sec": row.get("theseus_runtime_sec", "NA"),
        "mdcorealign_unified_RMSD": row.get("mdcorealign_unified_RMSD", "NA"),
        "fullset_unified_RMSD": row.get("fullset_unified_RMSD", "NA"),
        "mdlovofit_unified_RMSD": row.get("mdlovofit_unified_RMSD", "NA"),
        "theseus_unified_RMSD": row.get("theseus_unified_RMSD", "NA"),
        "mdcorealign_faster_than_mdlovofit": int(mdca_rt is not None and mdlovo_rt is not None and mdca_rt < mdlovo_rt),
        "mdcorealign_lower_RMSD_than_mdlovofit": int(mdca_rmsd is not None and mdlovo_rmsd is not None and mdca_rmsd < mdlovo_rmsd),
        "runtime_delta_mdcorealign_minus_mdlovofit": mdca_rt - mdlovo_rt if mdca_rt is not None and mdlovo_rt is not None else "NA",
        "runtime_ratio_mdcorealign_over_mdlovofit": mdca_rt / mdlovo_rt if mdca_rt is not None and mdlovo_rt not in (None, 0.0) else "NA",
        "RMSD_delta_mdcorealign_minus_mdlovofit": mdca_rmsd - mdlovo_rmsd if mdca_rmsd is not None and mdlovo_rmsd is not None else "NA",
        "stable_core_fraction": row.get("stable_core_fraction", "NA"),
        "mobile_fraction": row.get("mobile_fraction", "NA"),
        "n_core_segments": row.get("n_core_segments", "NA"),
        "n_flexible_segments": row.get("n_flexible_segments", "NA"),
        "robust_RMSF_mean": row.get("robust_RMSF_mean", "NA"),
        "noncore_core_RMSF_ratio": row.get("noncore_core_RMSF_ratio", "NA"),
        "RMSF_mobility_spearman": row.get("RMSF_mobility_spearman", "NA"),
        "failure_reason": row.get("failure_reason", "missing_summary" if summary is None else "NA"),
        "notes": row.get("notes", "NA"),
    }


def group_summary(rows: list[dict[str, object]], group_keys: tuple[str, ...], level: str) -> list[dict[str, object]]:
    """Summarize rows by one CATH hierarchy level."""

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(key, "NA")) for key in group_keys), []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        both = [
            row for row in group
            if row.get("mdcorealign_status") == "success" and row.get("mdlovofit_status") == "success"
        ]
        mdca_rt = [fnum(row.get("mdcorealign_runtime_sec")) for row in both]
        mdlovo_rt = [fnum(row.get("mdlovofit_runtime_sec")) for row in both]
        mdca_rmsd = [fnum(row.get("mdcorealign_unified_RMSD")) for row in both]
        mdlovo_rmsd = [fnum(row.get("mdlovofit_unified_RMSD")) for row in both]
        mdca_rt = [value for value in mdca_rt if value is not None]
        mdlovo_rt = [value for value in mdlovo_rt if value is not None]
        mdca_rmsd = [value for value in mdca_rmsd if value is not None]
        mdlovo_rmsd = [value for value in mdlovo_rmsd if value is not None]
        median_speedup = median([float(row["mdlovofit_runtime_sec"]) / float(row["mdcorealign_runtime_sec"]) for row in both if fnum(row.get("mdcorealign_runtime_sec")) and fnum(row.get("mdlovofit_runtime_sec"))])
        values = dict(zip(group_keys, key, strict=False))
        output.append(
            {
                "group_level": level,
                "cath_class": values.get("cath_class", "all"),
                "cath_architecture": values.get("cath_architecture", "all"),
                "cath_topology": values.get("cath_topology", "all"),
                "cath_homologous_superfamily": values.get("cath_homologous_superfamily", "all"),
                "n_domains": len(group),
                "n_success_both_mdcorealign_mdlovofit": len(both),
                "median_mdcorealign_runtime_sec": median(mdca_rt),
                "median_mdlovofit_runtime_sec": median(mdlovo_rt),
                "median_speedup_vs_mdlovofit": median_speedup,
                "percent_mdcorealign_faster": pct(sum(int(row["mdcorealign_faster_than_mdlovofit"]) for row in both), len(both)),
                "median_mdcorealign_unified_RMSD": median(mdca_rmsd),
                "median_mdlovofit_unified_RMSD": median(mdlovo_rmsd),
                "percent_mdcorealign_lower_RMSD": pct(sum(int(row["mdcorealign_lower_RMSD_than_mdlovofit"]) for row in both), len(both)),
                "median_stable_core_fraction": median([value for row in both for value in [fnum(row.get("stable_core_fraction"))] if value is not None]),
                "median_RMSF_mobility_spearman": median([value for row in both for value in [fnum(row.get("RMSF_mobility_spearman"))] if value is not None]),
                "median_noncore_core_RMSF_ratio": median([value for row in both for value in [fnum(row.get("noncore_core_RMSF_ratio"))] if value is not None]),
                "failure_rate": 1.0 - (len(both) / len(group) if group else 0.0),
            }
        )
    return output


def main() -> None:
    """Aggregate current all-domain results."""

    args = parse_args()
    args.aggregate_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = [row for row in read_tsv(args.manifest) if row.get("include_in_all_domain_benchmark", "1") == "1"]
    summaries = {
        path.parent.name: load_summary(path)
        for path in (args.output_dir / "per_trajectory").glob("*/summary.json")
    }
    rows = [comparison_row(row, summaries.get(row["trajectory_id"])) for row in manifest_rows]
    write_tsv(args.aggregate_dir / "all_domain_canonical_per_domain_summary.tsv", rows, PER_DOMAIN_COLUMNS)

    both = [row for row in rows if row["mdcorealign_status"] == "success" and row["mdlovofit_status"] == "success"]
    overall = group_summary(rows, tuple(), "overall")[0]
    overall_rows = [{"metric": key, "value": value} for key, value in {
        "n_target_domains": len(rows),
        "n_attempted": sum(row["failure_reason"] != "missing_summary" for row in rows),
        "n_success_both_mdcorealign_mdlovofit": len(both),
        "n_mdcorealign_success": sum(row["mdcorealign_status"] == "success" for row in rows),
        "n_mdlovofit_success": sum(row["mdlovofit_status"] == "success" for row in rows),
        "n_fullset_success": sum(row["fullset_status"] == "success" for row in rows),
        "n_theseus_success": sum(row["theseus_status"] == "success" for row in rows),
        "n_failed": sum(row["failure_reason"] not in ("NA", "missing_summary") for row in rows),
        "success_rate": len(both) / len(rows) if rows else 0.0,
        "mdcorealign_median_runtime_sec": overall["median_mdcorealign_runtime_sec"],
        "mdlovofit_median_runtime_sec": overall["median_mdlovofit_runtime_sec"],
        "median_speedup_vs_mdlovofit": overall["median_speedup_vs_mdlovofit"],
        "n_mdcorealign_faster_than_mdlovofit": sum(int(row["mdcorealign_faster_than_mdlovofit"]) for row in both),
        "percent_mdcorealign_faster_than_mdlovofit": overall["percent_mdcorealign_faster"],
        "n_mdcorealign_lower_RMSD_than_mdlovofit": sum(int(row["mdcorealign_lower_RMSD_than_mdlovofit"]) for row in both),
        "percent_mdcorealign_lower_RMSD_than_mdlovofit": overall["percent_mdcorealign_lower_RMSD"],
        "mdcorealign_median_unified_RMSD": overall["median_mdcorealign_unified_RMSD"],
        "mdlovofit_median_unified_RMSD": overall["median_mdlovofit_unified_RMSD"],
        "median_RMSD_delta_A": median([fnum(row["RMSD_delta_mdcorealign_minus_mdlovofit"]) for row in both if fnum(row["RMSD_delta_mdcorealign_minus_mdlovofit"]) is not None]),
        "median_stable_core_fraction": overall["median_stable_core_fraction"],
        "median_RMSF_mobility_spearman": overall["median_RMSF_mobility_spearman"],
        "median_noncore_core_RMSF_ratio": overall["median_noncore_core_RMSF_ratio"],
        "median_n_flexible_segments": median([value for row in both for value in [fnum(row.get("n_flexible_segments"))] if value is not None]),
    }.items()]
    write_tsv(args.aggregate_dir / "all_domain_canonical_overall_summary.tsv", overall_rows, ["metric", "value"])
    write_tsv(args.aggregate_dir / "all_domain_canonical_by_class.tsv", group_summary(rows, ("cath_class",), "class"), GROUP_COLUMNS)
    write_tsv(args.aggregate_dir / "all_domain_canonical_by_architecture.tsv", group_summary(rows, ("cath_class", "cath_architecture"), "architecture"), GROUP_COLUMNS)
    write_tsv(args.aggregate_dir / "all_domain_canonical_by_topology.tsv", group_summary(rows, ("cath_class", "cath_architecture", "cath_topology"), "topology"), GROUP_COLUMNS)
    write_tsv(args.aggregate_dir / "all_domain_canonical_by_homologous_superfamily.tsv", group_summary(rows, ("cath_class", "cath_architecture", "cath_topology", "cath_homologous_superfamily"), "homologous_superfamily"), GROUP_COLUMNS)
    print(args.aggregate_dir / "all_domain_canonical_overall_summary.tsv")


if __name__ == "__main__":
    main()
