#!/usr/bin/env python
"""Build an all-domain one-canonical-trajectory mdCATH manifest."""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import read_tsv, write_tsv


BENCHMARK_NAME = "mdcath_all_domain_canonical_one_replicate"

OUTPUT_COLUMNS = [
    "benchmark_name",
    "subset_name",
    "domain_id",
    "trajectory_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "cath_code",
    "h5_path",
    "h5_size_bytes",
    "selected_temperature_K",
    "selected_replica_index_zero_based",
    "selected_replica_label_one_based",
    "expected_n_frames_if_available",
    "domain_length_if_available",
    "fallback_used",
    "fallback_reason",
    "has_complete_cath_hierarchy",
    "include_in_all_domain_benchmark",
    "exclusion_reason",
    "appeared_in_previous_subset",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchy-manifest",
        type=Path,
        default=Path("results/mdcath_benchmark/manifest/mdcath_domain_hierarchy_manifest.tsv"),
    )
    parser.add_argument(
        "--previous-subsets",
        nargs="*",
        type=Path,
        default=[
            Path("results/mdcath_benchmark/subsets/mdcath_cath_balanced_5_per_class.tsv"),
            Path("results/mdcath_benchmark/subsets/mdcath_cath_balanced_10_per_class.tsv"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/mdcath_all_domain_canonical_manifest.tsv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/mdcath_all_domain_canonical_manifest_summary.tsv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/mdcath_all_domain_canonical_manifest_report.md"),
    )
    return parser.parse_args()


def previous_subset_domains(paths: list[Path]) -> set[str]:
    """Collect domains that appeared in earlier preliminary subsets."""

    domains: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_tsv(path):
            domain_id = row.get("domain_id")
            if domain_id and domain_id != "NA":
                domains.add(domain_id)
    return domains


def complete_hierarchy(row: dict[str, str]) -> bool:
    """Return whether all four CATH levels are present."""

    return all(
        row.get(key) not in ("", "NA", None)
        for key in ("cath_class", "cath_architecture", "cath_topology", "cath_homologous_superfamily", "cath_code")
    )


def canonical_row(row: dict[str, str], previous_domains: set[str]) -> dict[str, object]:
    """Convert one hierarchy row into one canonical benchmark row."""

    has_h5 = row.get("has_h5") == "1" and row.get("h5_path") not in ("", "NA", None)
    include = bool(has_h5)
    domain_id = row["domain_id"]
    selected_temperature = row.get("selected_temperature_K_default") or "320"
    selected_replica_zero = row.get("selected_replica_index_zero_based_default") or "0"
    selected_replica_one = row.get("selected_replica_label_one_based_default") or "1"
    trajectory_id = row.get("default_trajectory_id") or f"{domain_id}__T{selected_temperature}__R{selected_replica_one}"
    return {
        "benchmark_name": BENCHMARK_NAME,
        "subset_name": BENCHMARK_NAME,
        "domain_id": domain_id,
        "trajectory_id": trajectory_id,
        "cath_class": row.get("cath_class", "NA") or "NA",
        "cath_architecture": row.get("cath_architecture", "NA") or "NA",
        "cath_topology": row.get("cath_topology", "NA") or "NA",
        "cath_homologous_superfamily": row.get("cath_homologous_superfamily", "NA") or "NA",
        "cath_code": row.get("cath_code", "NA") or "NA",
        "h5_path": row.get("h5_path", "NA") or "NA",
        "h5_size_bytes": row.get("h5_size_bytes", "NA") or "NA",
        "selected_temperature_K": selected_temperature,
        "selected_replica_index_zero_based": selected_replica_zero,
        "selected_replica_label_one_based": selected_replica_one,
        "expected_n_frames_if_available": row.get("expected_n_frames", "NA") or "NA",
        "domain_length_if_available": row.get("domain_length", "NA") or "NA",
        "fallback_used": "0",
        "fallback_reason": "NA",
        "has_complete_cath_hierarchy": "1" if complete_hierarchy(row) else "0",
        "include_in_all_domain_benchmark": "1" if include else "0",
        "exclusion_reason": "NA" if include else "missing_h5",
        "appeared_in_previous_subset": "yes" if domain_id in previous_domains else "no",
        "notes": "canonical policy: prefer 320 K replica R1; hierarchy from CATH crosswalk, not domain_id",
    }


def sort_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    """Sort rows by CATH hierarchy then domain."""

    return (
        str(row.get("cath_class", "NA")),
        str(row.get("cath_architecture", "NA")),
        str(row.get("cath_topology", "NA")),
        str(row.get("cath_homologous_superfamily", "NA")),
        str(row.get("domain_id", "NA")),
    )


def count_distinct(rows: list[dict[str, object]], keys: tuple[str, ...]) -> int:
    """Count distinct non-missing key tuples."""

    values = {
        tuple(str(row.get(key, "NA")) for key in keys)
        for row in rows
        if all(str(row.get(key, "NA")) not in ("", "NA") for key in keys)
    }
    return len(values)


def main() -> None:
    """Build manifest, summary and report."""

    args = parse_args()
    previous_domains = previous_subset_domains(args.previous_subsets)
    hierarchy_rows = read_tsv(args.hierarchy_manifest)
    rows = [canonical_row(row, previous_domains) for row in hierarchy_rows]
    rows.sort(key=sort_key)
    write_tsv(args.output, rows, OUTPUT_COLUMNS)

    selected = [row for row in rows if row["include_in_all_domain_benchmark"] == "1"]
    missing_hierarchy = [row for row in rows if row["has_complete_cath_hierarchy"] != "1"]
    per_class = collections.Counter(str(row["cath_class"]) for row in selected)
    per_arch = collections.Counter((str(row["cath_class"]), str(row["cath_architecture"])) for row in selected)
    per_top = collections.Counter((str(row["cath_class"]), str(row["cath_architecture"]), str(row["cath_topology"])) for row in selected)
    per_hom = collections.Counter(
        (
            str(row["cath_class"]),
            str(row["cath_architecture"]),
            str(row["cath_topology"]),
            str(row["cath_homologous_superfamily"]),
        )
        for row in selected
    )
    summary_rows = [
        {"metric": "n_total_domains", "value": len(rows)},
        {"metric": "n_selected_canonical_trajectories", "value": len(selected)},
        {"metric": "n_per_class", "value": ";".join(f"{key}:{per_class[key]}" for key in sorted(per_class))},
        {"metric": "n_per_architecture", "value": len(per_arch)},
        {"metric": "n_per_topology", "value": len(per_top)},
        {"metric": "n_per_homologous_superfamily", "value": len(per_hom)},
        {"metric": "n_fallback_temperature", "value": 0},
        {"metric": "n_fallback_replica", "value": 0},
        {"metric": "n_missing_h5", "value": sum(row["include_in_all_domain_benchmark"] != "1" for row in rows)},
        {"metric": "n_missing_cath_hierarchy", "value": len(missing_hierarchy)},
        {"metric": "n_appeared_in_previous_subset", "value": sum(row["appeared_in_previous_subset"] == "yes" for row in rows)},
    ]
    write_tsv(args.summary_output, summary_rows, ["metric", "value"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# mdCATH All-Domain Canonical Manifest Report\n\n"
        f"- Manifest: `{args.output}`\n"
        f"- Summary: `{args.summary_output}`\n"
        f"- Total domains: {len(rows)}\n"
        f"- Selected canonical trajectories: {len(selected)}\n"
        f"- Missing HDF5 rows: {sum(row['include_in_all_domain_benchmark'] != '1' for row in rows)}\n"
        f"- Missing complete CATH hierarchy: {len(missing_hierarchy)}\n"
        f"- CATH classes: {', '.join(sorted(per_class))}\n"
        f"- Domains per class: {'; '.join(f'{key}:{per_class[key]}' for key in sorted(per_class))}\n"
        f"- Distinct architectures: {len(per_arch)}\n"
        f"- Distinct topologies: {len(per_top)}\n"
        f"- Distinct homologous-superfamily bins: {len(per_hom)}\n"
        "- Canonical selection: 320 K, replica R1 where available; no frame cap; no stride.\n"
        "- Current hierarchy manifest exposes one HDF5 per domain, so no fallback temperature or replica was required.\n"
    )
    print(args.output)
    print(args.summary_output)
    print(args.report)


if __name__ == "__main__":
    main()
