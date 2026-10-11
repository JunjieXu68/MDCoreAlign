#!/usr/bin/env python
"""Monitor the all-domain canonical mdCATH benchmark progress."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import read_tsv, write_tsv


STATUS_FIELDS = [
    "timestamp",
    "total_target_domains",
    "attempted",
    "completed",
    "mdcorealign_success",
    "mdcorealign_failed",
    "mdlovofit_success",
    "mdlovofit_failed",
    "fullset_success",
    "theseus_success",
    "theseus_failed",
    "download_failed",
    "skipped",
    "success_rate",
    "completed_last_10_min",
    "completed_last_60_min",
    "throughput_trajectories_per_hour",
    "estimated_time_remaining_hours",
    "cache_size",
    "active_runner_pid",
    "top_failure_reasons",
    "latest_log_lines",
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
        "--status-tsv",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/progress/current_status.tsv"),
    )
    parser.add_argument(
        "--progress-md",
        type=Path,
        default=Path("docs/mdcath_all_domain_progress.md"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/run_fullframes/all_domain_canonical.log"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("results/mdcath_benchmark/cache"))
    return parser.parse_args()


def cache_size(path: Path) -> str:
    """Return human-readable cache size."""

    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "NA"


def active_pid(output_dir: Path) -> str:
    """Return a likely active runner PID."""

    pid_file = output_dir / "runner.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        if pid:
            try:
                output = subprocess.check_output(["ps", "-p", pid, "-o", "pid=,stat=,etime=,cmd="], text=True)
            except Exception:
                output = ""
            if output.strip():
                return output.strip()
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_mdcath_subset_benchmark.py"], text=True)
    except Exception:
        return "NA"
    lines = [line for line in output.splitlines() if "all_domain_canonical" in line and "pgrep" not in line]
    return "; ".join(lines) if lines else "NA"


def tail(path: Path, n_lines: int = 50) -> str:
    """Return final log lines."""

    if not path.exists():
        return "NA"
    try:
        return subprocess.check_output(["tail", "-n", str(n_lines), str(path)], text=True, errors="replace")
    except Exception:
        return "NA"


def read_summary(path: Path) -> dict[str, object] | None:
    """Read one per-trajectory JSON summary."""

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    """Write current progress status."""

    args = parse_args()
    now = time.time()
    manifest_rows = [row for row in read_tsv(args.manifest) if row.get("include_in_all_domain_benchmark", "1") == "1"]
    summary_paths = list((args.output_dir / "per_trajectory").glob("*/summary.json"))
    summaries = [summary for path in summary_paths for summary in [read_summary(path)] if summary is not None]
    completed = len(summaries)
    attempted = len(summary_paths)

    def count_status(field: str, status: str) -> int:
        return sum(str(row.get(field)) == status for row in summaries)

    failures: dict[str, int] = {}
    for row in summaries:
        reason = str(row.get("failure_reason", "NA"))
        if reason and reason != "NA":
            failures[reason] = failures.get(reason, 0) + 1
    completed_last_10 = sum(now - path.stat().st_mtime <= 600 for path in summary_paths)
    completed_last_60 = sum(now - path.stat().st_mtime <= 3600 for path in summary_paths)
    throughput = completed_last_60
    remaining = max(0, len(manifest_rows) - completed)
    eta = remaining / throughput if throughput > 0 else "NA"
    both_success = sum(
        row.get("mdcorealign_status") == "success" and row.get("mdlovofit_status") == "success"
        for row in summaries
    )
    success_rate = both_success / len(manifest_rows) if manifest_rows else 0.0
    latest_log = tail(args.log_path)
    status_row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_target_domains": len(manifest_rows),
        "attempted": attempted,
        "completed": completed,
        "mdcorealign_success": count_status("mdcorealign_status", "success"),
        "mdcorealign_failed": count_status("mdcorealign_status", "failed"),
        "mdlovofit_success": count_status("mdlovofit_status", "success"),
        "mdlovofit_failed": count_status("mdlovofit_status", "failed"),
        "fullset_success": count_status("fullset_status", "success"),
        "theseus_success": count_status("theseus_status", "success"),
        "theseus_failed": count_status("theseus_status", "failed"),
        "download_failed": sum(str(row.get("failure_reason")) == "download_failed" for row in summaries),
        "skipped": sum(str(row.get("mdcorealign_status")) == "not_requested" for row in summaries),
        "success_rate": success_rate,
        "completed_last_10_min": completed_last_10,
        "completed_last_60_min": completed_last_60,
        "throughput_trajectories_per_hour": throughput,
        "estimated_time_remaining_hours": eta,
        "cache_size": cache_size(args.cache_dir),
        "active_runner_pid": active_pid(args.output_dir),
        "top_failure_reasons": ";".join(f"{key}:{value}" for key, value in sorted(failures.items(), key=lambda item: item[1], reverse=True)[:10]) or "NA",
        "latest_log_lines": latest_log.replace("\n", "\\n"),
    }
    write_tsv(args.status_tsv, [status_row], STATUS_FIELDS)
    args.progress_md.parent.mkdir(parents=True, exist_ok=True)
    args.progress_md.write_text(
        "# mdCATH All-Domain Canonical Progress\n\n"
        f"- Timestamp: {status_row['timestamp']}\n"
        f"- Total target domains: {status_row['total_target_domains']}\n"
        f"- Completed summaries: {status_row['completed']}\n"
        f"- MDCoreAlign success: {status_row['mdcorealign_success']}\n"
        f"- MDLovoFit success: {status_row['mdlovofit_success']}\n"
        f"- THESEUS success: {status_row['theseus_success']}\n"
        f"- Download failed: {status_row['download_failed']}\n"
        f"- Last 10 min completed: {status_row['completed_last_10_min']}\n"
        f"- Last 60 min completed: {status_row['completed_last_60_min']}\n"
        f"- Throughput: {status_row['throughput_trajectories_per_hour']} trajectories/hour\n"
        f"- ETA: {status_row['estimated_time_remaining_hours']} hours\n"
        f"- Cache size: {status_row['cache_size']}\n"
        f"- Active runner PID: `{status_row['active_runner_pid']}`\n"
        f"- Top failure reasons: {status_row['top_failure_reasons']}\n\n"
        "## Latest Log Lines\n\n"
        "```text\n"
        f"{latest_log}\n"
        "```\n"
    )
    print(args.status_tsv)
    print(args.progress_md)


if __name__ == "__main__":
    main()
