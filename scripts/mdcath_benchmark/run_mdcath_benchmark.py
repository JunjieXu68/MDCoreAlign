#!/usr/bin/env python
"""Run mdCATH benchmark modes with resume-safe per-trajectory outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    HF_REPO_ID,
    MANIFEST_PATH,
    PER_TRAJECTORY_COLUMNS,
    RESULT_ROOT,
    append_tsv,
    cache_size_gb,
    download_hf_file,
    ensure_result_dirs,
    purge_path,
    read_tsv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke_test", "domain_canonical", "full_trajectory", "external_subset"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--cache-dir", type=Path, default=RESULT_ROOT / "cache")
    parser.add_argument("--max-cache-gb", type=float, default=None)
    parser.add_argument("--purge-after-success", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow on-demand mdCATH HDF5 download.")
    parser.add_argument("--no-download", action="store_true", help="Deprecated alias for default no-download behavior.")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=0, help="Start index within selected rows.")
    parser.add_argument("--row-stop", type=int, default=None, help="Stop index within selected rows.")
    parser.add_argument("--job-index", type=int, default=None, help="Shard index for deterministic job-array execution.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Total number of deterministic job-array shards.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--timeout-sec", type=float, dest="timeout_seconds", help="Alias for --timeout-seconds.")
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT / "per_trajectory")
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], mode: str, max_trajectories: int | None) -> list[dict[str, str]]:
    """Select manifest rows for a benchmark mode."""

    if mode == "domain_canonical":
        selected = [row for row in rows if row.get("include_in_domain_canonical_benchmark") == "1"]
    elif mode == "external_subset":
        selected = [row for row in rows if row.get("include_in_external_baseline_subset") == "1"]
    elif mode == "smoke_test":
        selected = []
        seen_keys: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (row.get("domain_id", ""), row.get("temperature_K", ""), row.get("replica", ""))
            if key in seen_keys:
                continue
            if row.get("temperature_K") not in {"320", "348", "379"}:
                continue
            if row.get("replica") not in {"1", "2"}:
                continue
            selected.append(row)
            seen_keys.add(key)
            if len(selected) >= 10:
                break
    else:
        selected = rows
    if max_trajectories is not None:
        selected = selected[:max_trajectories]
    return selected


def row_summary_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return the per-trajectory summary path."""

    return output_root / row["trajectory_id"] / "summary.json"


def h5_path_for_row(row: dict[str, str], cache_dir: Path, repo_id: str, allow_download: bool) -> tuple[Path | None, str | None]:
    """Resolve or download the HDF5 file for one manifest row."""

    for key in ("local_cache_path", "h5_file"):
        value = row.get(key, "")
        if value and value != "NA":
            path = Path(value)
            if path.exists():
                return path, None
    if not allow_download:
        return None, "download_failed: local HDF5 unavailable and --allow-download was not set"
    h5_file = row.get("h5_file", "")
    if not h5_file or h5_file == "NA":
        h5_file = f"data/mdcath_dataset_{row['domain_id']}.h5"
    try:
        return download_hf_file(h5_file, cache_dir=cache_dir, repo_id=repo_id), None
    except Exception as exc:
        return None, f"download_failed: {exc}"


def run_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    """Run or skip one manifest row."""

    summary_path = row_summary_path(args.output_root, row)
    if args.resume and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            if reusable_existing_summary(existing, args):
                return existing
            summary_path.unlink(missing_ok=True)
        except Exception:
            pass
    h5_file, h5_error = h5_path_for_row(row, args.cache_dir, args.repo_id, args.allow_download)
    if h5_error:
        summary = failure_summary(row, h5_error, summary_path)
        write_json(summary_path, summary)
        return summary

    command = [
        sys.executable,
        "scripts/mdcath_benchmark/run_mdcath_single_trajectory.py",
        "--trajectory-id",
        row["trajectory_id"],
        "--domain-id",
        row["domain_id"],
        "--temperature-K",
        row["temperature_K"],
        "--replica",
        row["replica"],
        "--h5-file",
        str(h5_file),
        "--coordinate-path",
        row.get("coordinate_path") or "auto",
        "--output-dir",
        str(args.output_root),
        "--stride",
        str(args.stride),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])

    start = time.perf_counter()
    log_dir = RESULT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{row['trajectory_id']}.log"
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr
        )
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = failure_summary(row, f"mdcorealign_exception: returncode={completed.returncode}", summary_path)
            write_json(summary_path, summary)
        if completed.returncode != 0 and summary.get("status") == "success":
            summary["status"] = "failed"
            summary["failure_reason"] = f"mdcorealign_exception_returncode_{completed.returncode}"
            write_json(summary_path, summary)
        if args.purge_after_success and summary.get("status") == "success" and h5_file is not None:
            purge_downloaded_domain_file(h5_file, args.cache_dir)
        return summary
    except subprocess.TimeoutExpired:
        summary = failure_summary(row, "mdcorealign_timeout", summary_path)
        summary["runtime_seconds"] = time.perf_counter() - start
        write_json(summary_path, summary)
        log_path.write_text("TIMEOUT\nCOMMAND:\n" + " ".join(command))
        return summary


def failure_summary(row: dict[str, str], reason: str, summary_path: Path) -> dict[str, object]:
    """Build a failure summary compatible with aggregate tables."""

    return {
        "trajectory_id": row.get("trajectory_id", "NA"),
        "domain_id": row.get("domain_id", "NA"),
        "temperature_K": row.get("temperature_K", "NA"),
        "replica": row.get("replica", "NA"),
        "status": "failed",
        "failure_reason": reason.split(":", 1)[0],
        "max_frames": "NA",
        "stride": "NA",
        "n_frames": "NA",
        "n_residues": "NA",
        "n_atoms": "NA",
        "runtime_seconds": 0.0,
        "algorithm_runtime_seconds": 0.0,
        "io_runtime_seconds": 0.0,
        "peak_memory_mb": "NA",
        "iteration_count_mean": "NA",
        "converged_fraction": "NA",
        "weighted_RMSD": "NA",
        "unified_RMSD": "NA",
        "fullset_unified_RMSD": "NA",
        "delta_unified_RMSD_vs_fullset": "NA",
        "robust_RMSF_mean": "NA",
        "robust_RMSF_median": "NA",
        "ordinary_RMSF_mean": "NA",
        "ordinary_RMSF_median": "NA",
        "robust_to_ordinary_RMSF_ratio": "NA",
        "stable_core_fraction": "NA",
        "mobile_fraction": "NA",
        "n_core_segments": "NA",
        "core_fragmentation_score": "NA",
        "noncore_core_RMSF_ratio": "NA",
        "RMSF_mobility_spearman": "NA",
        "RMSF_mobility_spearman_pvalue": "NA",
        "min_weight": "NA",
        "max_weight": "NA",
        "mean_weight": "NA",
        "median_weight": "NA",
        "summary_json": str(summary_path),
        "profile_tsv": "NA",
        "notes": reason,
    }


def reusable_existing_summary(summary: dict[str, object], args: argparse.Namespace) -> bool:
    """Return whether a prior summary is valid for the requested run settings."""

    if args.allow_download and summary.get("failure_reason") == "download_failed":
        return False
    if summary.get("status") == "success":
        previous_max_frames = summary.get("max_frames", "missing")
        if previous_max_frames == "missing":
            previous_max_frames = previous_max_frames_from_metadata(summary)
        if args.max_frames is None and previous_max_frames not in (None, "NA", "", "None"):
            return False
        if args.max_frames is not None and str(previous_max_frames) not in {str(args.max_frames), "NA"}:
            return False
    return True


def previous_max_frames_from_metadata(summary: dict[str, object]) -> object:
    """Read legacy max-frame metadata from the per-trajectory input metadata."""

    try:
        summary_path = Path(str(summary.get("summary_json", "")))
        metadata_path = summary_path.parent / "input_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            return metadata.get("max_frames")
    except Exception:
        return "missing"
    return "missing"


def purge_downloaded_domain_file(h5_file: Path, cache_dir: Path) -> None:
    """Best-effort purge for one Hugging Face-cached domain file."""

    try:
        h5_file = Path(h5_file)
        cache_dir = Path(cache_dir).resolve()
        resolved = h5_file.resolve()
        if str(h5_file.resolve()).startswith(str(cache_dir)) or str(h5_file).startswith(str(cache_dir)):
            if h5_file.is_symlink() or h5_file.exists():
                purge_path(h5_file)
            if str(resolved).startswith(str(cache_dir)) and resolved.exists():
                purge_path(resolved)
    except Exception:
        return


def main() -> None:
    """Run a benchmark mode."""

    args = parse_args()
    if args.no_download and args.allow_download:
        raise SystemExit("--no-download and --allow-download cannot be used together")
    if args.no_download:
        args.allow_download = False
    if (args.job_index is None) ^ (args.n_jobs is None):
        raise SystemExit("--job-index and --n-jobs must be provided together")
    if args.n_jobs is not None and args.n_jobs <= 0:
        raise SystemExit("--n-jobs must be positive")
    if args.job_index is not None and not 0 <= args.job_index < args.n_jobs:
        raise SystemExit("--job-index must be in [0, n_jobs)")
    ensure_result_dirs()
    rows = read_tsv(args.manifest)
    if not rows:
        raise SystemExit(f"Manifest is empty or missing: {args.manifest}")
    selected = select_rows(rows, args.mode, args.max_trajectories)
    selected = selected[args.row_start : args.row_stop]
    if args.job_index is not None and args.n_jobs is not None:
        selected = selected[args.job_index :: args.n_jobs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_summary = RESULT_ROOT / "aggregate" / f"mdcath_{args.mode}_raw_summary.tsv"
    if raw_summary.exists() and not args.resume:
        raw_summary.unlink()
    summaries: list[dict[str, object]] = []
    if args.n_workers <= 1:
        for row in selected:
            summary = run_row(row, args)
            summaries.append(summary)
            append_tsv(raw_summary, summary, PER_TRAJECTORY_COLUMNS)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(run_row, row, args) for row in selected]
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                append_tsv(raw_summary, summary, PER_TRAJECTORY_COLUMNS)
    write_json(
        RESULT_ROOT / "logs" / f"mdcath_{args.mode}_run_manifest.json",
        {
            "mode": args.mode,
            "manifest": str(args.manifest),
            "n_selected": len(selected),
            "n_success": sum(1 for row in summaries if row.get("status") == "success"),
            "n_failed": sum(1 for row in summaries if row.get("status") != "success"),
            "raw_summary": str(raw_summary),
            "cache_dir": str(args.cache_dir),
            "cache_size_gb": cache_size_gb(args.cache_dir),
            "max_cache_gb": args.max_cache_gb,
            "allow_download": args.allow_download,
            "job_index": args.job_index,
            "n_jobs": args.n_jobs,
        },
    )
    print(raw_summary)


if __name__ == "__main__":
    main()
