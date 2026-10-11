#!/usr/bin/env python
"""Validate a CATH-balanced mdCATH subset manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=Path,
        default=RESULT_ROOT / "subsets/mdcath_cath_balanced_5_per_class.tsv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULT_ROOT / "subsets/mdcath_cath_balanced_5_per_class_summary.tsv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/mdcath_cath_balanced_subset_report.md"),
    )
    return parser.parse_args()


def compact(counter: Counter[str]) -> str:
    """Format a counter for compact report cells."""

    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str)) or "NA"


def unique_count(rows: list[dict[str, str]], key: str) -> int:
    """Count unique non-missing values."""

    return len({row.get(key, "NA") for row in rows if row.get(key, "NA") not in {"", "NA"}})


def duplicate_count(values: list[str]) -> int:
    """Count duplicate values."""

    counter = Counter(values)
    return sum(count - 1 for count in counter.values() if count > 1)


def main() -> None:
    """Validate subset and write summary/report."""

    args = parse_args()
    rows = read_tsv(args.subset)
    required = (
        "domain_id",
        "trajectory_id",
        "h5_path",
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "selected_temperature_K",
        "selected_replica_label_one_based",
    )
    missing_rows = [
        row["domain_id"]
        for row in rows
        if any(row.get(key, "NA") in {"", "NA"} for key in required)
    ]
    class_counts = Counter(row["cath_class"] for row in rows)
    temp_counts = Counter(row["selected_temperature_K"] for row in rows)
    rep_counts = Counter(row["selected_replica_label_one_based"] for row in rows)
    class_to_arch: dict[str, set[str]] = defaultdict(set)
    class_to_top: dict[str, set[str]] = defaultdict(set)
    class_to_hsf: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        class_to_arch[row["cath_class"]].add(row["cath_architecture"])
        class_to_top[row["cath_class"]].add(row["cath_topology"])
        class_to_hsf[row["cath_class"]].add(row["cath_homologous_superfamily"])
    imbalance = len(set(class_counts.values())) > 1
    fallback_rows = [row["domain_id"] for row in rows if row.get("fallback_used") not in {"0", "False", "false", ""}]
    summary = [
        {
            "n_selected_total": len(rows),
            "n_selected_per_class": compact(class_counts),
            "n_unique_architectures": unique_count(rows, "cath_architecture"),
            "n_unique_topologies": unique_count(rows, "cath_topology"),
            "n_unique_homologous_superfamilies": unique_count(rows, "cath_homologous_superfamily"),
            "selected_temperature_distribution": compact(temp_counts),
            "selected_replicate_distribution": compact(rep_counts),
            "duplicate_domain_id_count": duplicate_count([row["domain_id"] for row in rows]),
            "duplicate_trajectory_id_count": duplicate_count([row["trajectory_id"] for row in rows]),
            "rows_with_missing_required_fields": len(missing_rows),
            "missing_or_fallback_cases": ";".join(missing_rows + fallback_rows) or "NA",
            "class_balance_status": "imbalanced_explained_by_available_domains" if imbalance else "balanced",
            "validation_status": "pass"
            if not missing_rows
            and not fallback_rows
            and duplicate_count([row["domain_id"] for row in rows]) == 0
            and duplicate_count([row["trajectory_id"] for row in rows]) == 0
            else "warning",
        }
    ]
    write_tsv(args.summary, summary)
    selected_table = "\n".join(
        f"| {row['domain_id']} | {row['trajectory_id']} | {row['cath_class']} | "
        f"{row['cath_architecture']} | {row['cath_topology']} | {row['cath_homologous_superfamily']} |"
        for row in rows
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# mdCATH CATH-Balanced Subset Report",
                "",
                f"- subset file: `{args.subset}`",
                f"- selected trajectories: {len(rows)}",
                f"- selected per class: {compact(class_counts)}",
                f"- unique architectures: {unique_count(rows, 'cath_architecture')}",
                f"- unique topologies: {unique_count(rows, 'cath_topology')}",
                f"- unique homologous superfamilies: {unique_count(rows, 'cath_homologous_superfamily')}",
                f"- temperature distribution: {compact(temp_counts)}",
                f"- replicate distribution: {compact(rep_counts)}",
                f"- validation status: {summary[0]['validation_status']}",
                "",
                "This is a CATH-balanced subset manifest, not a completed full mdCATH benchmark.",
                "",
                "| domain | trajectory | class | architecture | topology | homologous superfamily |",
                "|---|---|---|---|---|---|",
                selected_table,
                "",
            ]
        )
    )
    print(args.summary)


if __name__ == "__main__":
    main()
