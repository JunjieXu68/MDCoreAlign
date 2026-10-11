#!/usr/bin/env python
"""Build a CATH hierarchy crosswalk for mdCATH domains.

The CATH hierarchy is read from official CATH domain-list data. Domain IDs are
matched to mdCATH domain IDs; hierarchy levels are never inferred from the
domain identifier string itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import sys
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


DEFAULT_CATH_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains-file",
        type=Path,
        default=RESULT_ROOT / "metadata_files/mdCATH_domains.txt",
        help="mdCATH domain list downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--hf-file-manifest",
        type=Path,
        default=RESULT_ROOT / "manifest/hf_mdcath_file_manifest.tsv",
        help="Fallback source of mdCATH domain IDs if --domains-file is unavailable.",
    )
    parser.add_argument(
        "--cath-domain-list",
        type=Path,
        default=RESULT_ROOT / "metadata_files/cath-domain-list.txt",
        help="Local CATH domain-list file; downloaded if absent.",
    )
    parser.add_argument("--cath-url", default=DEFAULT_CATH_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "manifest/cath_domain_crosswalk.tsv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULT_ROOT / "manifest/cath_crosswalk_summary.tsv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/mdcath_cath_hierarchy_summary.md"),
    )
    return parser.parse_args()


def read_mdcath_domains(domains_file: Path, hf_file_manifest: Path) -> list[str]:
    """Read mdCATH domain IDs from a small domain list or from the HF manifest."""

    if domains_file.exists():
        domains = [line.strip() for line in domains_file.read_text().splitlines() if line.strip()]
        return sorted(set(domains))
    rows = read_tsv(hf_file_manifest)
    domains = {
        row.get("inferred_domain_id", "")
        for row in rows
        if row.get("is_h5_or_hdf5") == "1" and row.get("inferred_domain_id") not in {"", "NA", None}
    }
    return sorted(domains)


def ensure_cath_domain_list(path: Path, url: str) -> str:
    """Download official CATH domain-list data if needed."""

    if path.exists() and path.stat().st_size > 0:
        return "local_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return url


def parse_cath_domain_list(path: Path) -> dict[str, dict[str, str]]:
    """Parse a CATH domain-list file."""

    records: dict[str, dict[str, str]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        domain_id = parts[0]
        cath_class, architecture, topology, homology = parts[1:5]
        records[domain_id] = {
            "domain_id": domain_id,
            "cath_class": cath_class,
            "cath_architecture": architecture,
            "cath_topology": topology,
            "cath_homologous_superfamily": homology,
            "cath_code": ".".join([cath_class, architecture, topology, homology]),
            "s35_if_available": parts[5] if len(parts) > 5 else "NA",
            "s60_if_available": parts[6] if len(parts) > 6 else "NA",
            "s95_if_available": parts[7] if len(parts) > 7 else "NA",
            "s100_if_available": parts[8] if len(parts) > 8 else "NA",
        }
    return records


def compact_counter(counter: Counter[str]) -> str:
    """Format a counter as a stable semicolon-separated string."""

    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str)) or "NA"


def per_class_unique(rows: list[dict[str, str]], key: str) -> str:
    """Return counts of unique values per CATH class."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["parse_status"] != "matched":
            continue
        grouped[row["cath_class"]].add(row[key])
    return ";".join(f"{class_id}:{len(values)}" for class_id, values in sorted(grouped.items())) or "NA"


def main() -> None:
    """Build the crosswalk and write summary reports."""

    args = parse_args()
    md_domains = read_mdcath_domains(args.domains_file, args.hf_file_manifest)
    source = ensure_cath_domain_list(args.cath_domain_list, args.cath_url)
    cath_records = parse_cath_domain_list(args.cath_domain_list)
    source_version = f"latest-release_or_local;retrieved={date.today().isoformat()}"
    rows: list[dict[str, str]] = []
    for domain_id in md_domains:
        record = cath_records.get(domain_id)
        if record is None:
            rows.append(
                {
                    "domain_id": domain_id,
                    "cath_class": "NA",
                    "cath_architecture": "NA",
                    "cath_topology": "NA",
                    "cath_homologous_superfamily": "NA",
                    "cath_code": "NA",
                    "s35_if_available": "NA",
                    "s60_if_available": "NA",
                    "s95_if_available": "NA",
                    "s100_if_available": "NA",
                    "source": source,
                    "source_version_or_date": source_version,
                    "parse_status": "missing_in_cath_domain_list",
                    "notes": "No CATH hierarchy was inferred from domain_id.",
                }
            )
            continue
        rows.append(
            {
                **record,
                "source": source,
                "source_version_or_date": source_version,
                "parse_status": "matched",
                "notes": "CATH hierarchy parsed from official CATH domain-list.",
            }
        )
    columns = [
        "domain_id",
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "cath_code",
        "s35_if_available",
        "s60_if_available",
        "s95_if_available",
        "s100_if_available",
        "source",
        "source_version_or_date",
        "parse_status",
        "notes",
    ]
    write_tsv(args.output, rows, columns)
    matched = [row for row in rows if row["parse_status"] == "matched"]
    class_counter = Counter(row["cath_class"] for row in matched)
    summary = [
        {
            "total_mdcath_domains": len(rows),
            "domains_with_cath_crosswalk": len(matched),
            "domains_missing_cath_crosswalk": len(rows) - len(matched),
            "classes_present": ",".join(sorted(class_counter, key=str)) or "NA",
            "domains_per_class": compact_counter(class_counter),
            "architectures_per_class": per_class_unique(rows, "cath_architecture"),
            "topologies_per_class": per_class_unique(rows, "cath_topology"),
            "homologous_superfamilies_per_class": per_class_unique(rows, "cath_homologous_superfamily"),
            "source": source,
            "source_version_or_date": source_version,
        }
    ]
    write_tsv(args.summary, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# mdCATH CATH Hierarchy Summary",
                "",
                "CATH hierarchy annotations were joined from official CATH domain-list data.",
                "No hierarchy level was inferred from mdCATH domain IDs.",
                "",
                f"- mdCATH domains inspected: {len(rows)}",
                f"- domains with CATH crosswalk: {len(matched)}",
                f"- domains missing CATH crosswalk: {len(rows) - len(matched)}",
                f"- observed CATH classes: {', '.join(sorted(class_counter, key=str)) or 'NA'}",
                f"- domains per class: {compact_counter(class_counter)}",
                f"- source: {source}",
                "",
                "The joined hierarchy is suitable for selecting CATH-balanced mdCATH subsets.",
            ]
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
