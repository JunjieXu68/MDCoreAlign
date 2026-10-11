#!/usr/bin/env python
"""Join the mdCATH Hugging Face file manifest with CATH hierarchy metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, trajectory_id, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-file-manifest",
        type=Path,
        default=RESULT_ROOT / "manifest/hf_mdcath_file_manifest.tsv",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=RESULT_ROOT / "manifest/cath_domain_crosswalk.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "manifest/mdcath_domain_hierarchy_manifest.tsv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULT_ROOT / "manifest/mdcath_domain_hierarchy_summary.tsv",
    )
    return parser.parse_args()


def compact(counter: Counter[str]) -> str:
    """Format a counter for compact TSV cells."""

    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str)) or "NA"


def unique_per_class(rows: list[dict[str, str]], key: str) -> str:
    """Count unique values per class for a field."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if str(row["include_available"]) not in {"1", "1.0"}:
            continue
        grouped[row["cath_class"]].add(row[key])
    return ";".join(f"{class_id}:{len(values)}" for class_id, values in sorted(grouped.items())) or "NA"


def main() -> None:
    """Join manifests and write a domain-level hierarchy table."""

    args = parse_args()
    hf_rows = read_tsv(args.hf_file_manifest)
    crosswalk_rows = read_tsv(args.crosswalk)
    crosswalk = {row["domain_id"]: row for row in crosswalk_rows}
    h5_by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hf_rows:
        domain_id = row.get("inferred_domain_id", "NA")
        if row.get("is_h5_or_hdf5") != "1" or domain_id in {"", "NA"}:
            continue
        h5_by_domain[domain_id].append(row)
    all_domains = sorted(set(h5_by_domain) | set(crosswalk))
    rows: list[dict[str, str]] = []
    for domain_id in all_domains:
        h5_rows = h5_by_domain.get(domain_id, [])
        h5_path = h5_rows[0]["path"] if h5_rows else "NA"
        h5_size = h5_rows[0].get("size_bytes", "NA") if h5_rows else "NA"
        cw = crosswalk.get(domain_id, {})
        complete_cath = all(
            cw.get(key, "NA") not in {"", "NA"}
            for key in ("cath_class", "cath_architecture", "cath_topology", "cath_homologous_superfamily")
        )
        has_h5 = bool(h5_rows)
        rows.append(
            {
                "domain_id": domain_id,
                "h5_path": h5_path,
                "h5_size_bytes": h5_size,
                "cath_class": cw.get("cath_class", "NA"),
                "cath_architecture": cw.get("cath_architecture", "NA"),
                "cath_topology": cw.get("cath_topology", "NA"),
                "cath_homologous_superfamily": cw.get("cath_homologous_superfamily", "NA"),
                "cath_code": cw.get("cath_code", "NA"),
                "n_h5_files_for_domain": len(h5_rows),
                "has_h5": int(has_h5),
                "metadata_source": cw.get("source", "NA"),
                "selected_temperature_K_default": 320,
                "selected_replica_index_zero_based_default": 0,
                "selected_replica_label_one_based_default": 1,
                "default_trajectory_id": trajectory_id(domain_id, 320, 1),
                "include_available": int(has_h5 and complete_cath),
                "notes": "ready_for_balanced_subset" if has_h5 and complete_cath else "missing_h5_or_cath_crosswalk",
            }
        )
    rows.sort(
        key=lambda row: (
            row["cath_class"],
            row["cath_architecture"],
            row["cath_topology"],
            row["cath_homologous_superfamily"],
            row["domain_id"],
        )
    )
    columns = [
        "domain_id",
        "h5_path",
        "h5_size_bytes",
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "cath_code",
        "n_h5_files_for_domain",
        "has_h5",
        "metadata_source",
        "selected_temperature_K_default",
        "selected_replica_index_zero_based_default",
        "selected_replica_label_one_based_default",
        "default_trajectory_id",
        "include_available",
        "notes",
    ]
    write_tsv(args.output, rows, columns)
    available = [row for row in rows if row["include_available"] == 1]
    class_counter = Counter(row["cath_class"] for row in available)
    summary = [
        {
            "total_domains_in_hf_manifest": len(h5_by_domain),
            "domains_with_cath_crosswalk": sum(1 for row in rows if row["cath_code"] != "NA"),
            "domains_missing_cath_crosswalk": sum(1 for row in rows if row["cath_code"] == "NA"),
            "domains_with_h5": sum(1 for row in rows if row["has_h5"] == 1),
            "domains_without_h5": sum(1 for row in rows if row["has_h5"] != 1),
            "classes_present": ",".join(sorted(class_counter, key=str)) or "NA",
            "domains_per_class": compact(class_counter),
            "architectures_per_class": unique_per_class(rows, "cath_architecture"),
            "topologies_per_class": unique_per_class(rows, "cath_topology"),
            "homologous_superfamilies_per_class": unique_per_class(rows, "cath_homologous_superfamily"),
        }
    ]
    write_tsv(args.summary, summary)
    print(args.output)


if __name__ == "__main__":
    main()
