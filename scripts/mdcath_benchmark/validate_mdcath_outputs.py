#!/usr/bin/env python
"""Validate mdCATH benchmark outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_json, write_tsv


REQUIRED_SUCCESS_NUMERIC = (
    "n_frames",
    "n_residues",
    "runtime_seconds",
    "algorithm_runtime_seconds",
    "weighted_RMSD",
    "unified_RMSD",
    "fullset_unified_RMSD",
    "stable_core_fraction",
    "RMSF_mobility_spearman",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULT_ROOT / "aggregate/mdcath_full_trajectory_summary.tsv")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv")
    parser.add_argument("--subset-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "aggregate/mdcath_validation_report.tsv")
    parser.add_argument("--mode", choices=("all", "domain_canonical", "full_trajectory", "cath_balanced_subset"), default="all")
    parser.add_argument("--min-success", type=int, default=0)
    parser.add_argument("--require-completion", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float | None:
    """Convert a value to float."""

    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except Exception:
        return None


def validate_row(row: dict[str, str], seen: set[str]) -> list[dict[str, object]]:
    """Validate one summary row."""

    issues: list[dict[str, object]] = []
    tid = row.get("trajectory_id", "NA")
    if tid in seen:
        issues.append(issue(tid, "duplicate_trajectory_id", "trajectory appears more than once"))
    seen.add(tid)
    status = row.get("status")
    if status == "success":
        for key in REQUIRED_SUCCESS_NUMERIC:
            value = as_float(row.get(key))
            if value is None or not math.isfinite(value):
                issues.append(issue(tid, "nonfinite_required_numeric", key))
        stable_core_fraction = as_float(row.get("stable_core_fraction"))
        if stable_core_fraction is not None and not 0.0 <= stable_core_fraction <= 1.0:
            issues.append(issue(tid, "stable_core_fraction_out_of_bounds", stable_core_fraction))
        runtime = as_float(row.get("runtime_seconds"))
        if runtime is not None and runtime < 0:
            issues.append(issue(tid, "negative_runtime", runtime))
        summary_json = row.get("summary_json")
        if not summary_json or summary_json == "NA" or not Path(summary_json).exists():
            issues.append(issue(tid, "missing_summary_json", summary_json))
    elif status == "failed":
        if not row.get("failure_reason") or row.get("failure_reason") == "NA":
            issues.append(issue(tid, "failed_without_reason", "failure_reason missing"))
    else:
        issues.append(issue(tid, "unknown_status", status))
    return issues


def issue(trajectory_id: str, category: str, detail: object) -> dict[str, object]:
    """Build one validation issue row."""

    return {"trajectory_id": trajectory_id, "issue_category": category, "detail": detail}


def validate_subset_row(row: dict[str, str], seen: set[str]) -> list[dict[str, object]]:
    """Validate one CATH-balanced subset summary row."""

    issues: list[dict[str, object]] = []
    tid = row.get("trajectory_id", "NA")
    if tid in seen:
        issues.append(issue(tid, "duplicate_trajectory_id", "trajectory appears more than once"))
    seen.add(tid)
    for key in ("mdcorealign_status", "fullset_status", "mdlovofit_status", "theseus_status"):
        if row.get(key) != "success":
            issues.append(issue(tid, f"{key}_not_success", row.get(key, "missing")))
    for key in (
        "n_frames",
        "n_residues",
        "mdcorealign_runtime_sec",
        "fullset_runtime_sec",
        "mdlovofit_runtime_sec",
        "theseus_runtime_sec",
        "mdcorealign_unified_RMSD",
        "fullset_unified_RMSD",
        "mdlovofit_unified_RMSD",
        "theseus_unified_RMSD",
        "stable_core_fraction",
        "RMSF_mobility_spearman",
    ):
        value = as_float(row.get(key))
        if value is None or not math.isfinite(value):
            issues.append(issue(tid, "nonfinite_required_numeric", key))
    return issues


def main() -> None:
    """Validate outputs."""

    args = parse_args()
    if args.mode == "cath_balanced_subset":
        if args.input_dir is None:
            args.input_dir = RESULT_ROOT / "cath_balanced_subset_fullframes"
        if args.summary == RESULT_ROOT / "aggregate/mdcath_full_trajectory_summary.tsv":
            args.summary = args.input_dir / "aggregate/cath_balanced_5_per_class_fullframes_summary.tsv"
        if args.output == RESULT_ROOT / "aggregate/mdcath_validation_report.tsv":
            args.output = args.input_dir / "aggregate/cath_balanced_5_per_class_validation_report.tsv"
    rows = read_tsv(args.summary)
    manifest_path = args.subset_manifest if args.mode == "cath_balanced_subset" and args.subset_manifest else args.manifest
    manifest = read_tsv(manifest_path)
    target_ids = target_trajectory_ids(manifest, args.mode)
    if target_ids:
        rows = [row for row in rows if row.get("trajectory_id") in target_ids]
    seen: set[str] = set()
    issues: list[dict[str, object]] = []
    if args.mode == "cath_balanced_subset":
        for row in rows:
            issues.extend(validate_subset_row(row, seen))
        success_ids = {row.get("trajectory_id") for row in rows if row.get("mdcorealign_status") == "success"}
        failed_ids = {row.get("trajectory_id") for row in rows if row.get("mdcorealign_status") == "failed"}
    else:
        for row in rows:
            issues.extend(validate_row(row, seen))
        success_ids = {row.get("trajectory_id") for row in rows if row.get("status") == "success"}
        failed_ids = {row.get("trajectory_id") for row in rows if row.get("status") == "failed"}
    overlap = success_ids & failed_ids
    for tid in sorted(overlap):
        issues.append(issue(str(tid), "both_success_and_failed", "trajectory has conflicting statuses"))
    success_count = (
        sum(1 for row in rows if row.get("mdcorealign_status") == "success")
        if args.mode == "cath_balanced_subset"
        else sum(1 for row in rows if row.get("status") == "success")
    )
    attempted_ids = {row.get("trajectory_id") for row in rows}
    skipped_count = len(target_ids - attempted_ids) if target_ids else 0
    if args.min_success and success_count < args.min_success:
        issues.append(issue(args.mode, "min_success_not_met", f"{success_count} < {args.min_success}"))
    if args.require_completion and skipped_count > 0:
        issues.append(issue(args.mode, "completion_not_met", f"{skipped_count} target trajectories have no summary"))
    write_tsv(args.output, issues, ("trajectory_id", "issue_category", "detail"))
    payload = {
        "summary": str(args.summary),
        "manifest": str(manifest_path),
        "n_summary_rows": len(rows),
        "n_manifest_rows": len(manifest),
        "mode": args.mode,
        "target_count": len(target_ids) if target_ids else len(manifest),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "min_success": args.min_success,
        "require_completion": args.require_completion,
        "n_validation_issues": len(issues),
        "status": "pass" if not issues else "issues_found",
    }
    write_json(args.output.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2))
    if issues:
        raise SystemExit(1)


def target_trajectory_ids(manifest: list[dict[str, str]], mode: str) -> set[str]:
    """Return target IDs for a validation mode."""

    if mode == "domain_canonical":
        return {
            row["trajectory_id"]
            for row in manifest
            if row.get("include_in_domain_canonical_benchmark") == "1"
        }
    if mode == "full_trajectory":
        return {row["trajectory_id"] for row in manifest}
    if mode == "cath_balanced_subset":
        return {row["trajectory_id"] for row in manifest}
    return set()


if __name__ == "__main__":
    main()
