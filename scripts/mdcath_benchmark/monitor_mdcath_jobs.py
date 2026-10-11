#!/usr/bin/env python
"""Monitor mdCATH production progress from per-trajectory summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("domain_canonical", "full_trajectory"), default="domain_canonical")
    parser.add_argument("--manifest", type=Path, default=RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv")
    parser.add_argument("--per-trajectory-root", type=Path, default=RESULT_ROOT / "per_trajectory")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "aggregate/mdcath_monitor_status.tsv")
    return parser.parse_args()


def load_jsons(root: Path) -> list[dict[str, object]]:
    """Load per-trajectory summaries."""

    rows = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception:
            rows.append({"trajectory_id": path.parent.name, "status": "failed", "failure_reason": "summary_json_unreadable"})
    return rows


def as_float(value: object) -> float | None:
    """Convert value to float."""

    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    """Write monitor status."""

    args = parse_args()
    manifest = read_tsv(args.manifest)
    if args.mode == "domain_canonical":
        target_ids = {
            row["trajectory_id"]
            for row in manifest
            if row.get("include_in_domain_canonical_benchmark") == "1"
        }
    else:
        target_ids = {row["trajectory_id"] for row in manifest}
    rows = [row for row in load_jsons(args.per_trajectory_root) if row.get("trajectory_id") in target_ids]
    success = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") == "failed"]
    runtimes = [val for row in success if (val := as_float(row.get("runtime_seconds"))) is not None]
    reasons: dict[str, int] = {}
    for row in failed:
        reason = str(row.get("failure_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    completed = len(rows)
    remaining = max(0, len(target_ids) - completed)
    median_runtime = median(runtimes) if runtimes else None
    estimated_remaining = remaining * median_runtime if median_runtime is not None else None
    status = {
        "mode": args.mode,
        "total_targets": len(target_ids),
        "completed": completed,
        "running": "unknown_without_scheduler_query",
        "failed": len(failed),
        "download_failed": reasons.get("download_failed", 0),
        "success_rate": len(success) / completed if completed else 0.0,
        "median_runtime_sec_so_far": median_runtime if median_runtime is not None else "NA",
        "estimated_remaining_sec": estimated_remaining if estimated_remaining is not None else "NA",
        "top_failure_reasons": sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:10],
    }
    write_tsv(args.output, [status])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

