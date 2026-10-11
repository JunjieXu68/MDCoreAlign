#!/usr/bin/env python
"""Audit current mdCATH production benchmark completion status."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv")
    parser.add_argument("--summary", type=Path, default=RESULT_ROOT / "aggregate/mdcath_full_trajectory_summary.tsv")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "aggregate/current_production_status.tsv")
    parser.add_argument("--doc", type=Path, default=Path("docs/mdcath_current_status_audit.md"))
    return parser.parse_args()


def main() -> None:
    """Write production status audit."""

    args = parse_args()
    manifest = read_tsv(args.manifest)
    summaries = read_tsv(args.summary)
    manifest_ids = {row["trajectory_id"] for row in manifest}
    canonical_ids = {
        row["trajectory_id"]
        for row in manifest
        if row.get("include_in_domain_canonical_benchmark") == "1"
    }
    attempted_ids = {row.get("trajectory_id", "") for row in summaries if row.get("trajectory_id")}
    success_ids = {row.get("trajectory_id", "") for row in summaries if row.get("status") == "success"}
    failed_rows = [row for row in summaries if row.get("status") == "failed"]
    download_failed = sum(1 for row in failed_rows if row.get("failure_reason") == "download_failed")
    mdcorealign_failed = sum(1 for row in failed_rows if str(row.get("failure_reason", "")).startswith("mdcorealign"))
    canonical_success_count = len(success_ids & canonical_ids)
    full_success_count = len(success_ids & manifest_ids)
    skipped = len(manifest_ids - attempted_ids)
    domain_complete = canonical_success_count >= 5000
    full_complete = full_success_count >= 100000
    row = {
        "manifest_total_domains": len({row["domain_id"] for row in manifest}),
        "manifest_total_trajectories": len(manifest),
        "attempted_trajectories": len(attempted_ids),
        "successful_trajectories": len(success_ids),
        "failed_trajectories": len(failed_rows),
        "skipped_trajectories": skipped,
        "download_failed": download_failed,
        "mdcorealign_failed": mdcorealign_failed,
        "domain_canonical_success_count": canonical_success_count,
        "full_trajectory_success_count": full_success_count,
        "whether_domain_canonical_complete": "yes" if domain_complete else "no",
        "whether_full_trajectory_complete": "yes" if full_complete else "no",
    }
    write_tsv(args.output, [row])
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(
        "# mdCATH Current Production Status Audit\n\n"
        "Pipeline and feasibility validation have completed, but the mdCATH production benchmark has not yet completed. "
        "The current scientific result is limited to one real mdCATH trajectory and should not be used as database-scale evidence.\n\n"
        "## Status Table\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in row.items())
        + "\n\n"
        "## Interpretation\n\n"
        "- Manifest coverage is complete at the file-list level.\n"
        "- Domain-canonical production is not complete until at least 5,000 canonical trajectories succeed or failures are fully justified.\n"
        "- Full-trajectory production is not complete until large-scale execution covers the 134,950 trajectory target.\n"
        "- no-download smoke failures validate failure handling only; they are not scientific benchmark evidence.\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()

