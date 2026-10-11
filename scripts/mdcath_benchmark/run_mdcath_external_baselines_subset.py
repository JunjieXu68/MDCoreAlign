#!/usr/bin/env python
"""Prepare or run the mdCATH external-baseline subset.

This script intentionally treats MDLovoFit/THESEUS as optional external tools.
It records clear availability status rather than failing the internal
MDCoreAlign benchmark when external binaries are absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_json, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, default=RESULT_ROOT / "external_baselines_subset/mdcath_external_subset_manifest.tsv")
    parser.add_argument("--manifest", type=Path, default=RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--include-mdlovofit", action="store_true")
    parser.add_argument("--include-theseus", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create subset manifest and external-baseline status table."""

    args = parse_args()
    rows = read_tsv(args.manifest)
    candidates = [row for row in rows if row.get("include_in_external_baseline_subset") == "1"]
    if not candidates:
        candidates = rows
    subset = candidates[: args.subset_size]
    write_tsv(args.subset_manifest, subset)
    mdlovofit_path = shutil.which("MDLovoFit") or shutil.which("mdlovofit")
    theseus_path = shutil.which("theseus")
    status_rows = []
    for row in subset:
        status_rows.append(
            {
                "trajectory_id": row.get("trajectory_id"),
                "domain_id": row.get("domain_id"),
                "temperature_K": row.get("temperature_K"),
                "replica": row.get("replica"),
                "mdcorealign_RMSD": "not_run_by_this_script",
                "mdlovofit_RMSD": "NA",
                "theseus_RMSD": "NA",
                "fullset_runtime_sec": "NA",
                "mdcorealign_runtime_sec": "NA",
                "mdlovofit_runtime_sec": "NA" if args.include_mdlovofit else "not_requested",
                "theseus_runtime_sec": "NA" if args.include_theseus else "not_requested",
                "mdcorealign_status": "see_internal_summary",
                "mdlovofit_status": "available" if mdlovofit_path and args.include_mdlovofit else "not_available_or_not_requested",
                "theseus_status": "available" if theseus_path and args.include_theseus else "not_available_or_not_requested",
                "notes": "External baseline execution is intentionally subset-only and should be run after CA extraction policy is finalized.",
            }
        )
    output = RESULT_ROOT / "external_baselines_subset/external_subset_summary.tsv"
    write_tsv(output, status_rows)
    write_json(
        RESULT_ROOT / "external_baselines_subset/external_subset_manifest.json",
        {
            "subset_manifest": str(args.subset_manifest),
            "subset_size_requested": args.subset_size,
            "subset_size_written": len(subset),
            "include_mdlovofit": args.include_mdlovofit,
            "include_theseus": args.include_theseus,
            "mdlovofit_path": mdlovofit_path,
            "theseus_path": theseus_path,
            "status": "subset_manifest_prepared",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
