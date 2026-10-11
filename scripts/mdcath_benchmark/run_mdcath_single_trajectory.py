#!/usr/bin/env python
"""Run MDCoreAlign and ordinary full-set baseline on one mdCATH trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    PER_TRAJECTORY_COLUMNS,
    append_tsv,
    memory_mb,
    now_utc,
    parse_domain_id_from_h5_path,
    stable_spearman,
    write_json,
    write_traceback,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--temperature-K", type=int, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--h5-file", type=Path, help="Input mdCATH HDF5 file.")
    parser.add_argument("--ca-npz", type=Path, help="Pre-extracted CA trajectory NPZ.")
    parser.add_argument("--coordinate-path", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-table", type=Path, help="Optional table to append one summary row.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--core-fraction", type=float, default=0.6)
    parser.add_argument("--mobile-percentile", type=float, default=0.8)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--objective-tolerance", type=float, default=1e-5)
    parser.add_argument("--objective-patience", type=int, default=2)
    parser.add_argument("--rule", default="huber", choices=("gaussian", "huber", "tukey"))
    parser.add_argument("--sequence-continuity-strength", type=float, default=0.05)
    parser.add_argument("--contact-consistency-strength", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    return parser.parse_args()


def load_ca_input(args: argparse.Namespace) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, object], float]:
    """Load or extract CA coordinates."""

    start = time.perf_counter()
    if args.ca_npz:
        payload = np.load(args.ca_npz, allow_pickle=True)
        coords = np.asarray(payload["coords"], dtype=float)
        labels = [str(item) for item in payload.get("labels", np.asarray([], dtype=object)).tolist()]
        if not labels:
            labels = [f"_:{index + 1}:UNK" for index in range(coords.shape[1])]
        residue_metadata = [_label_to_metadata(label, index) for index, label in enumerate(labels)]
        metadata = {
            "input_source": str(args.ca_npz),
            "domain_id": args.domain_id,
            "n_frames": int(coords.shape[0]),
            "n_ca_atoms": int(coords.shape[1]),
            "labels": labels,
            "residue_metadata": residue_metadata,
        }
    elif args.h5_file:
        coords, labels, residue_metadata, metadata = extract_ca_trajectory(
            args.h5_file,
            coordinate_path=args.coordinate_path,
            max_frames=args.max_frames,
            stride=args.stride,
        )
    else:
        raise ValueError("Either --h5-file or --ca-npz is required")
    return coords, labels, residue_metadata, metadata, time.perf_counter() - start


def _label_to_metadata(label: str, index: int) -> dict[str, str]:
    """Convert a chain:residue:resname label to metadata."""

    parts = label.split(":")
    if len(parts) >= 3:
        return {"chain": parts[0], "residue_id": parts[1], "resname": parts[2]}
    return {"chain": "_", "residue_id": str(index + 1), "resname": "UNK"}


def contiguous_segment_count(mask: np.ndarray) -> int:
    """Count contiguous true segments in a one-dimensional mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.sum(starts))


def run_single(args: argparse.Namespace) -> dict[str, object]:
    """Run one mdCATH trajectory and return a summary row."""

    total_start = time.perf_counter()
    output_dir = args.output_dir / args.trajectory_id
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_tsv = output_dir / "residue_profile.tsv"
    summary_json = output_dir / "summary.json"
    input_metadata_json = output_dir / "input_metadata.json"
    traceback_path = output_dir / "traceback.txt"

    try:
        coords, labels, residue_metadata, input_metadata, io_seconds = load_ca_input(args)
        write_json(input_metadata_json, input_metadata)
        reference = coords[0]

        algorithm_start = time.perf_counter()
        ordinary_aligned = align_trajectory_ordinary(coords, reference_frame_index=0)
        fullset_unified_rmsd = mean_rmsd(ordinary_aligned, reference)
        ordinary_rmsf = compute_rmsf(ordinary_aligned)

        trajectory_result = solve_trajectory_robust_alignment(
            coords,
            reference_mode="first_frame",
            warm_start=True,
            rule=args.rule,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            objective_tolerance=args.objective_tolerance,
            objective_patience=args.objective_patience,
            sequence_continuity_strength=args.sequence_continuity_strength,
            contact_consistency_strength=args.contact_consistency_strength,
        )
        robust_aligned = robust_aligned_coords_from_result(trajectory_result)
        unified_rmsd = mean_rmsd(robust_aligned, reference)
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
        algorithm_seconds = time.perf_counter() - algorithm_start

        core_rmsf = robust_rmsf[core.core_mask]
        noncore_rmsf = robust_rmsf[~core.core_mask]
        corr, pvalue = stable_spearman(robust_rmsf, mobility)
        noncore_core_ratio = (
            float(np.mean(noncore_rmsf) / np.mean(core_rmsf))
            if core_rmsf.size and noncore_rmsf.size and float(np.mean(core_rmsf)) != 0.0
            else None
        )
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
                    "core_label": int(bool(core.core_mask[index])),
                    "flexible_segment_id": int(flexible_segments[index]),
                }
            )
        write_tsv(profile_tsv, rows)
        runtime_seconds = time.perf_counter() - total_start
        summary = {
            "trajectory_id": args.trajectory_id,
            "domain_id": args.domain_id or parse_domain_id_from_h5_path(args.h5_file or ""),
            "temperature_K": args.temperature_K,
            "replica": args.replica,
            "status": "success",
            "failure_reason": "NA",
            "generated_at": now_utc(),
            "max_frames": args.max_frames,
            "stride": args.stride,
            "n_frames": int(coords.shape[0]),
            "n_residues": int(coords.shape[1]),
            "n_atoms": int(coords.shape[1]),
            "runtime_seconds": runtime_seconds,
            "algorithm_runtime_seconds": algorithm_seconds,
            "io_runtime_seconds": io_seconds,
            "peak_memory_mb": memory_mb(),
            "iteration_count_mean": trajectory_result.average_iterations_per_frame,
            "converged_fraction": trajectory_result.converged_fraction,
            "weighted_RMSD": trajectory_result.mean_weighted_rmsd,
            "unified_RMSD": unified_rmsd,
            "fullset_unified_RMSD": fullset_unified_rmsd,
            "delta_unified_RMSD_vs_fullset": unified_rmsd - fullset_unified_rmsd,
            "robust_RMSF_mean": float(np.mean(robust_rmsf)),
            "robust_RMSF_median": float(np.median(robust_rmsf)),
            "ordinary_RMSF_mean": float(np.mean(ordinary_rmsf)),
            "ordinary_RMSF_median": float(np.median(ordinary_rmsf)),
            "robust_to_ordinary_RMSF_ratio": float(np.mean(robust_rmsf) / np.mean(ordinary_rmsf))
            if float(np.mean(ordinary_rmsf)) != 0.0
            else None,
            "stable_core_fraction": core.core_fraction,
            "mobile_fraction": 1.0 - core.core_fraction,
            "n_core_segments": contiguous_segment_count(core.core_mask),
            "core_fragmentation_score": contiguous_segment_count(core.core_mask) / max(core.core_size, 1),
            "noncore_core_RMSF_ratio": noncore_core_ratio,
            "RMSF_mobility_spearman": corr,
            "RMSF_mobility_spearman_pvalue": pvalue,
            "min_weight": float(np.min(stability)),
            "max_weight": float(np.max(stability)),
            "mean_weight": float(np.mean(stability)),
            "median_weight": float(np.median(stability)),
            "summary_json": str(summary_json),
            "profile_tsv": str(profile_tsv),
            "notes": "MDCoreAlign and ordinary full-set baseline completed.",
        }
        write_json(summary_json, summary)
        return summary
    except Exception as exc:
        runtime_seconds = time.perf_counter() - total_start
        write_traceback(traceback_path, exc)
        failure = {
            "trajectory_id": args.trajectory_id,
            "domain_id": args.domain_id,
            "temperature_K": args.temperature_K,
            "replica": args.replica,
            "status": "failed",
            "failure_reason": str(exc).split(":", 1)[0],
            "generated_at": now_utc(),
            "max_frames": args.max_frames,
            "stride": args.stride,
            "n_frames": "NA",
            "n_residues": "NA",
            "n_atoms": "NA",
            "runtime_seconds": runtime_seconds,
            "algorithm_runtime_seconds": 0.0,
            "io_runtime_seconds": runtime_seconds,
            "peak_memory_mb": memory_mb(),
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
            "summary_json": str(summary_json),
            "profile_tsv": str(profile_tsv),
            "notes": f"traceback={traceback_path}; error={exc}",
        }
        write_json(summary_json, failure)
        return failure


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = run_single(args)
    if args.summary_table:
        append_tsv(args.summary_table, summary, PER_TRAJECTORY_COLUMNS)
    print(summary["summary_json"])
    if summary["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
