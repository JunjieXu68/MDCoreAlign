#!/usr/bin/env python
"""Run a CATH-balanced mdCATH subset benchmark.

This runner is intentionally subset-only. It does not attempt to process the
full mdCATH repository, and all external baselines are optional.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.benchmark import benchmark_external_mdlovofit, benchmark_external_theseus
from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.flexibility import (
    align_trajectory_ordinary,
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
)
from mdsuperpose.metrics import mean_rmsd
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment

from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory
from scripts.mdcath_benchmark.mdcath_common import (
    HF_REPO_ID,
    RESULT_ROOT,
    cache_size_gb,
    download_hf_file,
    json_ready,
    read_tsv,
    stable_spearman,
    write_json,
    write_traceback,
    write_tsv,
)


SUMMARY_COLUMNS = [
    "subset_name",
    "domain_id",
    "trajectory_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "cath_code",
    "n_residues",
    "n_frames",
    "selected_temperature_K",
    "selected_replica_label_one_based",
    "mdcorealign_status",
    "fullset_status",
    "mdlovofit_status",
    "theseus_status",
    "mdcorealign_runtime_sec",
    "fullset_runtime_sec",
    "mdlovofit_runtime_sec",
    "theseus_runtime_sec",
    "mdcorealign_unified_RMSD",
    "fullset_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "stable_core_fraction",
    "mobile_fraction",
    "n_core_segments",
    "n_flexible_segments",
    "robust_RMSF_mean",
    "noncore_core_RMSF_ratio",
    "RMSF_mobility_spearman",
    "mdlovofit_native_all_atom_rmsd",
    "mdlovofit_native_least_mobile_rmsd",
    "mdlovofit_output_subset_atom_fraction",
    "theseus_native_maximum_likelihood_rmsd",
    "failure_reason",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=RESULT_ROOT / "subsets/mdcath_cath_balanced_5_per_class.tsv",
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--cache-dir", type=Path, default=RESULT_ROOT / "cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--include-fullset", action="store_true")
    parser.add_argument("--include-mdcorealign", action="store_true")
    parser.add_argument("--include-mdlovofit", action="store_true")
    parser.add_argument("--include-theseus", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--core-fraction", type=float, default=0.6)
    parser.add_argument("--mobile-percentile", type=float, default=0.8)
    parser.add_argument("--external-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--mdlovofit-exe", type=Path, default=None)
    parser.add_argument("--theseus-exe", type=Path, default=None)
    parser.add_argument("--no-frame-cap", action="store_true", help="Explicitly disable any frame cap.")
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_root",
        type=Path,
        default=RESULT_ROOT / "cath_balanced_subset",
    )
    return parser.parse_args()


def contiguous_segment_count(mask: np.ndarray) -> int:
    """Count contiguous true segments in a one-dimensional mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.sum(starts))


def resolve_h5(row: dict[str, str], args: argparse.Namespace) -> tuple[Path | None, str | None]:
    """Resolve one selected domain HDF5 path, downloading only if allowed."""

    h5_path = row.get("h5_path", "")
    if h5_path and h5_path != "NA":
        local = Path(h5_path)
        if local.exists():
            return local, None
    if not args.allow_download:
        return None, "download_failed: HDF5 unavailable locally and --allow-download was not set"
    try:
        return download_hf_file(h5_path, cache_dir=args.cache_dir, repo_id=args.repo_id), None
    except Exception as exc:
        return None, f"download_failed: {exc}"


def empty_row(row: dict[str, str], reason: str) -> dict[str, Any]:
    """Build a failure row with the required schema."""

    return {
        "subset_name": row.get("subset_name", "NA"),
        "domain_id": row.get("domain_id", "NA"),
        "trajectory_id": row.get("trajectory_id", "NA"),
        "cath_class": row.get("cath_class", "NA"),
        "cath_architecture": row.get("cath_architecture", "NA"),
        "cath_topology": row.get("cath_topology", "NA"),
        "cath_homologous_superfamily": row.get("cath_homologous_superfamily", "NA"),
        "cath_code": row.get("cath_code", "NA"),
        "n_residues": "NA",
        "n_frames": "NA",
        "selected_temperature_K": row.get("selected_temperature_K", "NA"),
        "selected_replica_label_one_based": row.get("selected_replica_label_one_based", "NA"),
        "mdcorealign_status": "failed",
        "fullset_status": "failed",
        "mdlovofit_status": "not_run",
        "theseus_status": "not_run",
        "mdcorealign_runtime_sec": "NA",
        "fullset_runtime_sec": "NA",
        "mdlovofit_runtime_sec": "NA",
        "theseus_runtime_sec": "NA",
        "mdcorealign_unified_RMSD": "NA",
        "fullset_unified_RMSD": "NA",
        "mdlovofit_unified_RMSD": "NA",
        "theseus_unified_RMSD": "NA",
        "stable_core_fraction": "NA",
        "mobile_fraction": "NA",
        "n_core_segments": "NA",
        "n_flexible_segments": "NA",
        "robust_RMSF_mean": "NA",
        "noncore_core_RMSF_ratio": "NA",
        "RMSF_mobility_spearman": "NA",
        "mdlovofit_native_all_atom_rmsd": "NA",
        "mdlovofit_native_least_mobile_rmsd": "NA",
        "mdlovofit_output_subset_atom_fraction": "NA",
        "theseus_native_maximum_likelihood_rmsd": "NA",
        "failure_reason": reason.split(":", 1)[0],
        "notes": reason,
    }


def write_profile(
    path: Path,
    residue_metadata: list[dict[str, str]],
    robust_rmsf: np.ndarray,
    ordinary_rmsf: np.ndarray,
    stability: np.ndarray,
    mobility: np.ndarray,
    core_mask: np.ndarray,
    flexible_segments: np.ndarray,
) -> None:
    """Write a compact per-residue subset profile."""

    rows = []
    for index, meta in enumerate(residue_metadata):
        rows.append(
            {
                "atom_index": index,
                "chain": meta.get("chain", "_"),
                "residue_id": meta.get("residue_id", str(index + 1)),
                "resname": meta.get("resname", "UNK"),
                "robust_RMSF": float(robust_rmsf[index]),
                "ordinary_RMSF": float(ordinary_rmsf[index]),
                "soft_stability_weight": float(stability[index]),
                "mobility_score": float(mobility[index]),
                "core_label": int(bool(core_mask[index])),
                "flexible_segment_id": int(flexible_segments[index]),
            }
        )
    write_tsv(path, rows)


def run_external_mdlovofit(coords: np.ndarray, outdir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run external MDLovoFit if requested and available."""

    if not args.include_mdlovofit:
        return {"status": "not_requested"}
    executable = args.mdlovofit_exe
    if executable is not None and not executable.exists():
        return {"status": "not_available", "error": f"explicit executable missing: {executable}"}
    if executable is None and shutil.which("mdlovofit") is None:
        return {"status": "not_available"}
    try:
        result = benchmark_external_mdlovofit(
            coords,
            fraction=args.core_fraction,
            workdir=outdir,
            executable=executable,
            timeout_seconds=args.external_timeout_seconds,
        )
        return {
            "status": "success" if result.parsing_success else "failed",
            "runtime": result.runtime_seconds,
            "rmsd": result.rmsd,
            "core_fraction": result.core_fraction,
            "native_all_atom_rmsd": result.extra_metadata.get("mdlovofit_native_all_atom_rmsd"),
            "native_least_mobile_rmsd": result.extra_metadata.get("mdlovofit_native_least_mobile_rmsd"),
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_external_theseus(coords: np.ndarray, outdir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run external THESEUS if requested and available."""

    if not args.include_theseus:
        return {"status": "not_requested"}
    executable = args.theseus_exe
    if executable is not None and not executable.exists():
        return {"status": "not_available", "error": f"explicit executable missing: {executable}"}
    if executable is None and shutil.which("theseus") is None:
        return {"status": "not_available"}
    try:
        result = benchmark_external_theseus(
            coords,
            workdir=outdir,
            executable=executable,
            timeout_seconds=args.external_timeout_seconds,
        )
        return {
            "status": "success" if result.parsing_success else "failed",
            "runtime": result.runtime_seconds,
            "rmsd": result.rmsd,
            "native_maximum_likelihood_rmsd": result.extra_metadata.get("reported_maximum_likelihood_rmsd"),
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    """Run one selected trajectory and return a flat summary row."""

    trajectory_id = row["trajectory_id"]
    outdir = args.output_root / "per_trajectory" / trajectory_id
    summary_path = outdir / "summary.json"
    if args.resume and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            if reusable_existing_summary(existing, args):
                return existing
        except Exception:
            pass
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        h5_file, h5_error = resolve_h5(row, args)
        if h5_error or h5_file is None:
            summary = empty_row(row, h5_error or "download_failed")
            write_json(summary_path, summary)
            return summary
        coords, _labels, residue_metadata, metadata = extract_ca_trajectory(
            h5_file,
            coordinate_path=row.get("coordinate_path", "auto"),
            max_frames=args.max_frames,
            stride=args.stride,
        )
        reference = coords[0]

        ordinary_runtime = None
        fullset_rmsd = None
        ordinary_rmsf = None
        if args.include_fullset:
            start = time.perf_counter()
            ordinary_aligned = align_trajectory_ordinary(coords, reference_frame_index=0)
            ordinary_runtime = time.perf_counter() - start
            fullset_rmsd = mean_rmsd(ordinary_aligned, reference)
            ordinary_rmsf = compute_rmsf(ordinary_aligned)
        else:
            ordinary_rmsf = np.zeros(coords.shape[1], dtype=float)

        robust_runtime = None
        robust_rmsd = None
        robust_rmsf = np.zeros(coords.shape[1], dtype=float)
        stability = np.ones(coords.shape[1], dtype=float)
        mobility = np.zeros(coords.shape[1], dtype=float)
        core = select_core_by_target_fraction(stability, args.core_fraction)
        flexible_segments = np.zeros(coords.shape[1], dtype=int)
        noncore_core_ratio = None
        corr = None
        if args.include_mdcorealign:
            start = time.perf_counter()
            trajectory_result = solve_trajectory_robust_alignment(
                coords,
                reference_mode="first_frame",
                warm_start=True,
                rule="huber",
                max_iterations=20,
                tolerance=1e-6,
                objective_tolerance=1e-5,
                objective_patience=2,
                sequence_continuity_strength=0.05,
                contact_consistency_strength=0.0,
                record_history=False,
            )
            robust_runtime = time.perf_counter() - start
            robust_aligned = robust_aligned_coords_from_result(trajectory_result)
            robust_rmsd = mean_rmsd(robust_aligned, reference)
            robust_rmsf = compute_rmsf(robust_aligned)
            stability = np.clip(mean_soft_weights_from_result(trajectory_result), 0.0, 1.0)
            mobility = np.clip(1.0 - stability, 0.0, 1.0)
            core = select_core_by_target_fraction(stability, args.core_fraction)
            flexible_segments = detect_flexible_segments(
                mobility,
                percentile_cutoff=args.mobile_percentile,
                min_segment_length=3,
                merge_gap=1,
                chain=[item.get("chain", "_") for item in residue_metadata],
            )
            core_rmsf = robust_rmsf[core.core_mask]
            noncore_rmsf = robust_rmsf[~core.core_mask]
            if core_rmsf.size and noncore_rmsf.size and float(np.mean(core_rmsf)) != 0.0:
                noncore_core_ratio = float(np.mean(noncore_rmsf) / np.mean(core_rmsf))
            corr, _ = stable_spearman(robust_rmsf, mobility)

        profile_path = outdir / "residue_profile.tsv"
        write_profile(profile_path, residue_metadata, robust_rmsf, ordinary_rmsf, stability, mobility, core.core_mask, flexible_segments)
        mdlo = run_external_mdlovofit(coords, outdir / "external_mdlovofit", args)
        the = run_external_theseus(coords, outdir / "external_theseus", args)
        summary = {
            "subset_name": row["subset_name"],
            "domain_id": row["domain_id"],
            "trajectory_id": trajectory_id,
            "cath_class": row["cath_class"],
            "cath_architecture": row["cath_architecture"],
            "cath_topology": row["cath_topology"],
            "cath_homologous_superfamily": row["cath_homologous_superfamily"],
            "cath_code": row.get("cath_code", "NA"),
            "n_residues": int(coords.shape[1]),
            "n_frames": int(coords.shape[0]),
            "selected_temperature_K": row["selected_temperature_K"],
            "selected_replica_label_one_based": row["selected_replica_label_one_based"],
            "mdcorealign_status": "success" if args.include_mdcorealign else "not_requested",
            "fullset_status": "success" if args.include_fullset else "not_requested",
            "mdlovofit_status": mdlo.get("status", "not_run"),
            "theseus_status": the.get("status", "not_run"),
            "mdcorealign_runtime_sec": robust_runtime,
            "fullset_runtime_sec": ordinary_runtime,
            "mdlovofit_runtime_sec": mdlo.get("runtime"),
            "theseus_runtime_sec": the.get("runtime"),
            "mdcorealign_unified_RMSD": robust_rmsd,
            "fullset_unified_RMSD": fullset_rmsd,
            "mdlovofit_unified_RMSD": mdlo.get("rmsd"),
            "theseus_unified_RMSD": the.get("rmsd"),
            "stable_core_fraction": core.core_fraction,
            "mobile_fraction": 1.0 - core.core_fraction,
            "n_core_segments": contiguous_segment_count(core.core_mask),
            "n_flexible_segments": len({int(value) for value in flexible_segments if int(value) > 0}),
            "robust_RMSF_mean": float(np.mean(robust_rmsf)),
            "noncore_core_RMSF_ratio": noncore_core_ratio,
            "RMSF_mobility_spearman": corr,
            "mdlovofit_native_all_atom_rmsd": mdlo.get("native_all_atom_rmsd"),
            "mdlovofit_native_least_mobile_rmsd": mdlo.get("native_least_mobile_rmsd"),
            "mdlovofit_output_subset_atom_fraction": mdlo.get("core_fraction"),
            "theseus_native_maximum_likelihood_rmsd": the.get("native_maximum_likelihood_rmsd"),
            "failure_reason": "NA",
            "notes": (
                f"h5_file={h5_file}; max_frames={args.max_frames}; stride={args.stride}; "
                f"profile={profile_path}; mdlovofit_error={mdlo.get('error', 'NA')}; "
                f"theseus_error={the.get('error', 'NA')}"
            ),
            "input_metadata": metadata,
        }
        write_json(summary_path, summary)
        return summary
    except Exception as exc:
        write_traceback(outdir / "traceback.txt", exc)
        summary = empty_row(row, f"{type(exc).__name__}: {exc}")
        write_json(summary_path, summary)
        return summary


def executable_requested_and_available(path: Path | None, command_name: str) -> bool:
    """Return whether an optional external executable was requested and can be used."""

    if path is not None:
        return path.exists()
    return shutil.which(command_name) is not None


def reusable_existing_summary(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    """Return whether an existing per-trajectory summary satisfies this run request."""

    if args.max_frames is None and summary.get("n_frames") in {"50", 50}:
        return False
    if args.max_frames is not None and str(summary.get("n_frames")) != str(args.max_frames):
        return False
    if args.include_fullset and summary.get("fullset_status") != "success":
        return False
    if args.include_mdcorealign and summary.get("mdcorealign_status") != "success":
        return False
    if args.include_mdlovofit and executable_requested_and_available(args.mdlovofit_exe, "mdlovofit"):
        if summary.get("mdlovofit_status") != "success":
            return False
    if args.include_theseus and executable_requested_and_available(args.theseus_exe, "theseus"):
        if summary.get("theseus_status") != "success":
            return False
    return True


def main() -> None:
    """Run the subset benchmark."""

    args = parse_args()
    if args.no_frame_cap:
        args.max_frames = None
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(args.subset_manifest)
    if not rows:
        raise SystemExit(f"Subset manifest is empty or missing: {args.subset_manifest}")
    if args.max_trajectories is not None:
        rows = rows[: args.max_trajectories]
    summaries: list[dict[str, Any]] = []
    if args.n_workers <= 1:
        for row in rows:
            summaries.append(run_row(row, args))
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(run_row, row, args) for row in rows]
            for future in as_completed(futures):
                summaries.append(future.result())
    summaries.sort(key=lambda row: (str(row.get("cath_class", "")), str(row.get("domain_id", ""))))
    output_root_text = str(args.output_root)
    if "all_domain_canonical" in output_root_text:
        aggregate_name = "all_domain_canonical_raw_summary.tsv"
    elif "fullframes" in args.output_root.name:
        aggregate_name = "cath_balanced_5_per_class_fullframes_summary.tsv"
    else:
        aggregate_name = "cath_balanced_5_per_class_summary.tsv"
    aggregate = args.output_root / "aggregate" / aggregate_name
    write_tsv(aggregate, summaries, SUMMARY_COLUMNS)
    write_class_summary(args.output_root / "aggregate/cath_balanced_5_per_class_class_summary.tsv", summaries)
    write_json(
        args.output_root / "aggregate/cath_balanced_5_per_class_manifest.json",
        {
            "subset_manifest": str(args.subset_manifest),
            "aggregate": str(aggregate),
            "n_selected": len(rows),
            "n_success_mdcorealign": sum(1 for row in summaries if row.get("mdcorealign_status") == "success"),
            "n_success_mdlovofit": sum(1 for row in summaries if row.get("mdlovofit_status") == "success"),
            "n_success_theseus": sum(1 for row in summaries if row.get("theseus_status") == "success"),
            "allow_download": args.allow_download,
            "max_frames": args.max_frames,
            "stride": args.stride,
            "cache_size_gb": cache_size_gb(args.cache_dir),
            "note": "CATH-balanced subset benchmark; not a full mdCATH benchmark.",
        },
    )
    print(aggregate)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Collect numeric values from summary rows."""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except Exception:
            pass
    return values


def median(values: list[float]) -> float | str:
    """Return the median or NA for an empty list."""

    if not values:
        return "NA"
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def write_class_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write class-level medians for the CATH-balanced subset."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(str(row.get("cath_class", "NA")), []).append(row)
    rows = []
    for class_id in sorted(grouped, key=str):
        group = grouped[class_id]
        rows.append(
            {
                "cath_class": class_id,
                "n_selected": len(group),
                "n_success_mdcorealign": sum(row.get("mdcorealign_status") == "success" for row in group),
                "n_success_fullset": sum(row.get("fullset_status") == "success" for row in group),
                "n_success_mdlovofit": sum(row.get("mdlovofit_status") == "success" for row in group),
                "n_success_theseus": sum(row.get("theseus_status") == "success" for row in group),
                "median_mdcorealign_runtime_sec": median(numeric_values(group, "mdcorealign_runtime_sec")),
                "median_fullset_runtime_sec": median(numeric_values(group, "fullset_runtime_sec")),
                "median_mdlovofit_runtime_sec": median(numeric_values(group, "mdlovofit_runtime_sec")),
                "median_theseus_runtime_sec": median(numeric_values(group, "theseus_runtime_sec")),
                "median_mdcorealign_unified_RMSD": median(numeric_values(group, "mdcorealign_unified_RMSD")),
                "median_fullset_unified_RMSD": median(numeric_values(group, "fullset_unified_RMSD")),
                "median_mdlovofit_unified_RMSD": median(numeric_values(group, "mdlovofit_unified_RMSD")),
                "median_theseus_unified_RMSD": median(numeric_values(group, "theseus_unified_RMSD")),
                "median_stable_core_fraction": median(numeric_values(group, "stable_core_fraction")),
                "median_noncore_core_RMSF_ratio": median(numeric_values(group, "noncore_core_RMSF_ratio")),
                "median_RMSF_mobility_spearman": median(numeric_values(group, "RMSF_mobility_spearman")),
            }
        )
    write_tsv(path, rows)


if __name__ == "__main__":
    main()
