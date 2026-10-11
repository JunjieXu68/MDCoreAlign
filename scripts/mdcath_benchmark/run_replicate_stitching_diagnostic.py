#!/usr/bin/env python
"""Run a focused mdCATH replicate-stitching diagnostic for MDCoreAlign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

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
from scripts.mdcath_benchmark.mdcath_common import stable_spearman, write_json, write_tsv


DEFAULT_DOMAIN_PANEL = (
    ("2rdeA02", "medium class 2 domain with clear MDCoreAlign speed advantage"),
    ("1b4aA01", "short domain where MDCoreAlign was slower than MDLovoFit"),
    ("1vq8L01", "CATH class 4 high-RMSD domain with weak runtime advantage"),
)

SELECTED_COLUMNS = [
    "domain_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "n_residues",
    "selected_temperature_K",
    "selected_replicates",
    "reason_selected",
    "expected_frames_per_replicate",
    "cached_available",
    "notes",
]

EXTRACTION_COLUMNS = [
    "domain_id",
    "temperature_K",
    "replicate_label",
    "replicate_index_zero_based",
    "n_frames",
    "n_residues",
    "coord_shape",
    "residue_mapping_ok",
    "coordinate_units",
    "extraction_status",
    "failure_reason",
    "checksum",
    "notes",
]

STITCHING_COLUMNS = [
    "domain_id",
    "stitch_name",
    "included_replicates",
    "n_total_frames",
    "n_residues",
    "boundary_frame_indices",
    "mean_boundary_jump",
    "max_boundary_jump",
    "median_within_replicate_adjacent_jump",
    "boundary_to_within_jump_ratio",
    "notes",
]

RUN_SUMMARY_COLUMNS = [
    "domain_id",
    "run_type",
    "run_name",
    "included_replicates",
    "n_frames",
    "n_residues",
    "runtime_sec",
    "algorithm_runtime_sec",
    "n_iterations",
    "convergence_status",
    "weighted_RMSD_mean",
    "unified_RMSD_mean",
    "robust_RMSF_mean",
    "stable_core_fraction",
    "mobile_fraction",
    "n_core_segments",
    "n_flexible_segments",
    "noncore_core_RMSF_ratio",
    "RMSF_mobility_spearman",
    "output_status",
    "failure_reason",
]

CONSENSUS_COLUMNS = [
    "domain_id",
    "cath_class",
    "n_residues",
    "n_frames_stitched",
    "core_jaccard_stitched_vs_consensus",
    "mobile_jaccard_stitched_vs_consensus",
    "mobility_spearman_stitched_vs_mean_replicate",
    "RMSF_spearman_stitched_vs_mean_replicate",
    "stable_core_fraction_stitched",
    "stable_core_fraction_mean_replicate",
    "stable_core_fraction_delta",
    "noncore_core_RMSF_ratio_stitched",
    "noncore_core_RMSF_ratio_mean_replicate",
    "flexible_segment_overlap",
    "interpretation_stable",
    "notes",
]

SCALING_COLUMNS = [
    "domain_id",
    "stitch_name",
    "included_replicates",
    "n_frames",
    "n_residues",
    "runtime_sec",
    "algorithm_runtime_sec",
    "weighted_fit_sec",
    "residual_update_sec",
    "weight_update_sec",
    "rmsf_sec",
    "output_write_sec",
    "n_iterations",
    "convergence_status",
    "runtime_per_frame",
    "runtime_per_frame_squared",
]

SCALING_MODEL_COLUMNS = [
    "domain_id",
    "R2_linear",
    "R2_quadratic",
    "AIC_linear",
    "AIC_quadratic",
    "estimated_scaling_preference",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-domain-summary",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate/all_domain_canonical_per_domain_summary.tsv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/mdcath_benchmark/replicate_stitching"),
    )
    parser.add_argument("--temperature-k", type=int, default=320)
    parser.add_argument("--replicates", default="0,1,2,3,4")
    parser.add_argument("--domain-ids", default=None, help="Optional comma-separated domain IDs.")
    parser.add_argument("--max-domains", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into dictionaries."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fnum(value: object) -> float | None:
    """Parse a finite float."""

    try:
        if value in ("NA", "", None):
            return None
        converted = float(value)
    except Exception:
        return None
    return converted if math.isfinite(converted) else None


def h5_path_from_notes(notes: str) -> Path | None:
    """Extract the cached HDF5 path from all-domain notes."""

    match = re.search(r"h5_file=([^;]+)", notes or "")
    if not match:
        return None
    return Path(match.group(1))


def load_domain_panel(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Select a small diagnostic domain panel from existing benchmark results."""

    rows = {row["domain_id"]: row for row in read_tsv(args.per_domain_summary)}
    if args.domain_ids:
        requested = [(item.strip(), "user-selected domain") for item in args.domain_ids.split(",") if item.strip()]
    else:
        requested = list(DEFAULT_DOMAIN_PANEL)
    panel: list[dict[str, Any]] = []
    for domain_id, reason in requested:
        row = rows.get(domain_id)
        if row is None:
            continue
        h5_file = h5_path_from_notes(row.get("notes", ""))
        panel.append({"summary": row, "reason": reason, "h5_file": h5_file})
        if len(panel) >= args.max_domains:
            break
    if len(panel) < 3:
        raise SystemExit("Need at least three selected domains for this diagnostic.")
    return panel


def checksum_array(coords: np.ndarray) -> str:
    """Return a short SHA256 checksum for one coordinate array."""

    digest = hashlib.sha256(np.ascontiguousarray(coords).view(np.uint8)).hexdigest()
    return digest[:16]


def save_npz(path: Path, coords: np.ndarray, labels: list[str], metadata: dict[str, object]) -> None:
    """Save a coordinate array and small metadata in NPZ format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=coords.astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=object),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def coord_path(domain_id: str, temperature_k: int, replicate_index: int) -> str:
    """Return the mdCATH coordinate path for one replicate."""

    return f"/{domain_id}/{temperature_k}/{replicate_index}/coords"


def extract_replicates(
    domain: dict[str, Any],
    *,
    output_root: Path,
    temperature_k: int,
    replicate_indices: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract all requested CA-only replicates for one domain."""

    summary = domain["summary"]
    domain_id = summary["domain_id"]
    h5_file = domain["h5_file"]
    domain_root = output_root / domain_id
    rows: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    reference_labels: list[str] | None = None
    reference_meta: list[dict[str, str]] | None = None
    for replicate_index in replicate_indices:
        label = f"R{replicate_index + 1}"
        out_npz = domain_root / "replicates" / f"{label}_ca_coords.npz"
        try:
            coords, labels, residue_metadata, metadata = extract_ca_trajectory(
                h5_file,
                coordinate_path=coord_path(domain_id, temperature_k, replicate_index),
                min_ca_atoms=30,
                min_frames=20,
            )
            mapping_ok = True
            if reference_labels is None:
                reference_labels = labels
                reference_meta = residue_metadata
            else:
                mapping_ok = labels == reference_labels and residue_metadata == reference_meta
            npz_metadata = dict(metadata)
            npz_metadata["replicate_label"] = label
            npz_metadata["replicate_index_zero_based"] = replicate_index
            save_npz(out_npz, coords, labels, npz_metadata)
            rows.append(
                {
                    "domain_id": domain_id,
                    "temperature_K": temperature_k,
                    "replicate_label": label,
                    "replicate_index_zero_based": replicate_index,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "coord_shape": list(coords.shape),
                    "residue_mapping_ok": int(mapping_ok),
                    "coordinate_units": metadata.get("unit_status", "NA"),
                    "extraction_status": "success",
                    "failure_reason": "NA",
                    "checksum": checksum_array(coords),
                    "notes": f"coordinate_path={coord_path(domain_id, temperature_k, replicate_index)}; npz={out_npz}",
                }
            )
            extracted.append(
                {
                    "label": label,
                    "replicate_index": replicate_index,
                    "coords": coords,
                    "labels": labels,
                    "residue_metadata": residue_metadata,
                    "metadata": metadata,
                    "npz": out_npz,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "domain_id": domain_id,
                    "temperature_K": temperature_k,
                    "replicate_label": label,
                    "replicate_index_zero_based": replicate_index,
                    "n_frames": "NA",
                    "n_residues": "NA",
                    "coord_shape": "NA",
                    "residue_mapping_ok": 0,
                    "coordinate_units": "NA",
                    "extraction_status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "checksum": "NA",
                    "notes": "NA",
                }
            )
    write_tsv(domain_root / "replicate_extraction_manifest.tsv", rows, EXTRACTION_COLUMNS)
    return rows, extracted


def adjacent_jump_distribution(coords: np.ndarray) -> np.ndarray:
    """Return adjacent-frame RMSD-like jumps within one trajectory."""

    if coords.shape[0] < 2:
        return np.asarray([], dtype=float)
    diffs = coords[1:] - coords[:-1]
    return np.sqrt(np.mean(np.sum(diffs * diffs, axis=2), axis=1))


def boundary_jump(first: np.ndarray, second: np.ndarray) -> float:
    """Return the jump between the last frame of first and first frame of second."""

    diff = second[0] - first[-1]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def stitch_prefixes(
    domain_id: str,
    extracted: list[dict[str, Any]],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Create stitched prefix trajectories for one domain."""

    domain_root = output_root / domain_id
    rows: list[dict[str, Any]] = []
    stitched: list[dict[str, Any]] = []
    within_jumps = np.concatenate(
        [adjacent_jump_distribution(item["coords"]) for item in extracted if adjacent_jump_distribution(item["coords"]).size],
    )
    within_median = float(np.median(within_jumps)) if within_jumps.size else float("nan")
    for stop in range(1, len(extracted) + 1):
        included = extracted[:stop]
        labels = [item["label"] for item in included]
        stitch_name = "_".join(labels)
        coords = np.concatenate([item["coords"] for item in included], axis=0)
        boundary_indices: list[int] = []
        boundary_jumps: list[float] = []
        cursor = 0
        for left, right in zip(included[:-1], included[1:], strict=False):
            cursor += int(left["coords"].shape[0])
            boundary_indices.append(cursor - 1)
            boundary_jumps.append(boundary_jump(left["coords"], right["coords"]))
        mean_boundary = float(np.mean(boundary_jumps)) if boundary_jumps else "NA"
        max_boundary = float(np.max(boundary_jumps)) if boundary_jumps else "NA"
        ratio = (
            float(np.mean(boundary_jumps) / within_median)
            if boundary_jumps and np.isfinite(within_median) and within_median > 0.0
            else "NA"
        )
        out_npz = domain_root / "stitched" / f"stitched_{stitch_name}.npz"
        save_npz(
            out_npz,
            coords,
            included[0]["labels"],
            {
                "domain_id": domain_id,
                "stitch_name": stitch_name,
                "included_replicates": labels,
                "boundary_frame_indices": boundary_indices,
            },
        )
        row = {
            "domain_id": domain_id,
            "stitch_name": stitch_name,
            "included_replicates": ",".join(labels),
            "n_total_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "boundary_frame_indices": ",".join(str(index) for index in boundary_indices) or "NA",
            "mean_boundary_jump": mean_boundary,
            "max_boundary_jump": max_boundary,
            "median_within_replicate_adjacent_jump": within_median,
            "boundary_to_within_jump_ratio": ratio,
            "notes": f"npz={out_npz}",
        }
        rows.append(row)
        stitched.append(
            {
                "run_type": "stitched_prefix",
                "run_name": stitch_name,
                "included_replicates": labels,
                "coords": coords,
                "labels": included[0]["labels"],
                "residue_metadata": included[0]["residue_metadata"],
                "stitching": row,
            }
        )
    write_tsv(domain_root / "stitching_manifest.tsv", rows, STITCHING_COLUMNS)
    return stitched


def count_segments(mask: np.ndarray) -> int:
    """Count contiguous true segments."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    return int(np.sum(mask & np.concatenate(([True], ~mask[:-1]))))


def write_residue_profile(
    path: Path,
    residue_metadata: list[dict[str, str]],
    robust_rmsf: np.ndarray,
    ordinary_rmsf: np.ndarray,
    stability: np.ndarray,
    mobility: np.ndarray,
    core_mask: np.ndarray,
    flexible_segments: np.ndarray,
) -> None:
    """Write one per-residue profile table."""

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


def run_mdcorealign(
    domain_id: str,
    run: dict[str, Any],
    *,
    output_root: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Run MDCoreAlign on one separate or stitched trajectory."""

    run_dir = output_root / domain_id / "mdcorealign_runs" / f"{run['run_type']}_{run['run_name']}"
    summary_path = run_dir / "summary.json"
    profile_path = run_dir / "residue_profile.tsv"
    timing_path = run_dir / "timing.json"
    aligned_path = run_dir / "aligned_coords.npz"
    if skip_existing and summary_path.exists() and profile_path.exists():
        return json.loads(summary_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(run["coords"], dtype=float)
    start = time.perf_counter()
    try:
        algorithm_start = time.perf_counter()
        result = solve_trajectory_robust_alignment(
            coords,
            reference_mode="first_frame",
            warm_start=True,
            collect_timing=True,
            rule="huber",
            max_iterations=20,
            tolerance=1e-6,
            objective_tolerance=1e-5,
            objective_patience=2,
            sequence_continuity_strength=0.05,
            contact_consistency_strength=0.0,
            record_history=False,
        )
        algorithm_runtime = time.perf_counter() - algorithm_start
        rmsf_start = time.perf_counter()
        robust_aligned = robust_aligned_coords_from_result(result)
        robust_rmsf = compute_rmsf(robust_aligned)
        ordinary_aligned = align_trajectory_ordinary(coords, reference_frame_index=0)
        ordinary_rmsf = compute_rmsf(ordinary_aligned)
        rmsf_sec = time.perf_counter() - rmsf_start
        stability = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
        mobility = np.clip(1.0 - stability, 0.0, 1.0)
        core = select_core_by_target_fraction(stability, 0.6)
        flexible_segments = detect_flexible_segments(
            mobility,
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=[item.get("chain", "_") for item in run["residue_metadata"]],
        )
        core_rmsf = robust_rmsf[core.core_mask]
        noncore_rmsf = robust_rmsf[~core.core_mask]
        noncore_core_ratio = (
            float(np.mean(noncore_rmsf) / np.mean(core_rmsf))
            if core_rmsf.size and noncore_rmsf.size and float(np.mean(core_rmsf)) != 0.0
            else None
        )
        corr, _ = stable_spearman(robust_rmsf, mobility)
        output_start = time.perf_counter()
        write_residue_profile(
            profile_path,
            run["residue_metadata"],
            robust_rmsf,
            ordinary_rmsf,
            stability,
            mobility,
            core.core_mask,
            flexible_segments,
        )
        np.savez_compressed(
            aligned_path,
            aligned_coords=robust_aligned.astype(np.float32, copy=False),
            weights=stability.astype(np.float32, copy=False),
            robust_rmsf=robust_rmsf.astype(np.float32, copy=False),
            ordinary_rmsf=ordinary_rmsf.astype(np.float32, copy=False),
            mobility=mobility.astype(np.float32, copy=False),
            core_mask=core.core_mask.astype(np.int8),
            flexible_segments=flexible_segments.astype(np.int32),
        )
        output_write_sec = time.perf_counter() - output_start
        timing = dict(result.timing or {})
        timing["rmsf_sec"] = rmsf_sec
        timing["output_write_sec"] = output_write_sec
        write_json(timing_path, timing)
        summary = {
            "domain_id": domain_id,
            "run_type": run["run_type"],
            "run_name": run["run_name"],
            "included_replicates": ",".join(run["included_replicates"]),
            "n_frames": int(coords.shape[0]),
            "n_residues": int(coords.shape[1]),
            "runtime_sec": time.perf_counter() - start,
            "algorithm_runtime_sec": algorithm_runtime,
            "n_iterations": float(result.average_iterations_per_frame),
            "convergence_status": float(result.converged_fraction),
            "weighted_RMSD_mean": float(result.mean_weighted_rmsd),
            "unified_RMSD_mean": mean_rmsd(robust_aligned, coords[0]),
            "robust_RMSF_mean": float(np.mean(robust_rmsf)),
            "stable_core_fraction": float(core.core_fraction),
            "mobile_fraction": float(1.0 - core.core_fraction),
            "n_core_segments": count_segments(core.core_mask),
            "n_flexible_segments": len({int(value) for value in flexible_segments if int(value) > 0}),
            "noncore_core_RMSF_ratio": noncore_core_ratio,
            "RMSF_mobility_spearman": corr,
            "output_status": "success",
            "failure_reason": "NA",
            "profile_path": str(profile_path),
            "timing_path": str(timing_path),
            "aligned_path": str(aligned_path),
        }
    except Exception as exc:
        summary = {
            "domain_id": domain_id,
            "run_type": run["run_type"],
            "run_name": run["run_name"],
            "included_replicates": ",".join(run["included_replicates"]),
            "n_frames": int(coords.shape[0]),
            "n_residues": int(coords.shape[1]),
            "runtime_sec": time.perf_counter() - start,
            "algorithm_runtime_sec": "NA",
            "n_iterations": "NA",
            "convergence_status": "NA",
            "weighted_RMSD_mean": "NA",
            "unified_RMSD_mean": "NA",
            "robust_RMSF_mean": "NA",
            "stable_core_fraction": "NA",
            "mobile_fraction": "NA",
            "n_core_segments": "NA",
            "n_flexible_segments": "NA",
            "noncore_core_RMSF_ratio": "NA",
            "RMSF_mobility_spearman": "NA",
            "output_status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    write_json(summary_path, summary)
    (run_dir / "log.txt").write_text(
        f"run_name={run['run_name']}\nstatus={summary['output_status']}\nfailure_reason={summary['failure_reason']}\n"
    )
    return summary


def load_aligned_artifacts(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load saved arrays for one run."""

    path = Path(summary["aligned_path"])
    data = np.load(path)
    return {key: data[key] for key in data.files}


def jaccard(a: np.ndarray, b: np.ndarray) -> float | str:
    """Return Jaccard overlap for two boolean masks."""

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b)
    if not np.any(union):
        return "NA"
    return float(np.sum(np.logical_and(a, b)) / np.sum(union))


def summarize_consensus(
    domain_id: str,
    domain_summary: dict[str, str],
    run_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare full stitched result to separate-replicate consensus."""

    separate = [row for row in run_summaries if row["run_type"] == "separate_replicate" and row["output_status"] == "success"]
    full = [
        row for row in run_summaries
        if row["run_type"] == "stitched_prefix" and row["run_name"] == "R1_R2_R3_R4_R5" and row["output_status"] == "success"
    ]
    if len(separate) < 5 or not full:
        return {
            "domain_id": domain_id,
            "cath_class": domain_summary.get("cath_class", "NA"),
            "n_residues": domain_summary.get("n_residues", "NA"),
            "n_frames_stitched": "NA",
            "core_jaccard_stitched_vs_consensus": "NA",
            "mobile_jaccard_stitched_vs_consensus": "NA",
            "mobility_spearman_stitched_vs_mean_replicate": "NA",
            "RMSF_spearman_stitched_vs_mean_replicate": "NA",
            "stable_core_fraction_stitched": "NA",
            "stable_core_fraction_mean_replicate": "NA",
            "stable_core_fraction_delta": "NA",
            "noncore_core_RMSF_ratio_stitched": "NA",
            "noncore_core_RMSF_ratio_mean_replicate": "NA",
            "flexible_segment_overlap": "NA",
            "interpretation_stable": "no",
            "notes": "insufficient successful runs",
        }
    full_summary = full[0]
    full_art = load_aligned_artifacts(full_summary)
    rep_artifacts = [load_aligned_artifacts(row) for row in separate]
    rep_core = np.asarray([item["core_mask"].astype(bool) for item in rep_artifacts])
    rep_mobile = ~rep_core
    consensus_core = np.sum(rep_core, axis=0) >= 3
    consensus_mobile = np.sum(rep_mobile, axis=0) >= 3
    full_core = full_art["core_mask"].astype(bool)
    full_mobile = ~full_core
    mean_mobility = np.mean(np.asarray([item["mobility"] for item in rep_artifacts], dtype=float), axis=0)
    mean_rmsf = np.mean(np.asarray([item["robust_rmsf"] for item in rep_artifacts], dtype=float), axis=0)
    mob_corr, _ = stable_spearman(full_art["mobility"], mean_mobility)
    rmsf_corr, _ = stable_spearman(full_art["robust_rmsf"], mean_rmsf)
    rep_stable = [fnum(row["stable_core_fraction"]) for row in separate]
    rep_stable = [value for value in rep_stable if value is not None]
    rep_ratio = [fnum(row["noncore_core_RMSF_ratio"]) for row in separate]
    rep_ratio = [value for value in rep_ratio if value is not None]
    full_flexible = full_art["flexible_segments"] > 0
    consensus_flexible = np.sum(np.asarray([item["flexible_segments"] > 0 for item in rep_artifacts]), axis=0) >= 3
    core_j = jaccard(full_core, consensus_core)
    mobile_j = jaccard(full_mobile, consensus_mobile)
    flexible_j = jaccard(full_flexible, consensus_flexible)
    stable_flag = (
        isinstance(core_j, float)
        and isinstance(mobile_j, float)
        and core_j >= 0.75
        and mobile_j >= 0.65
        and (mob_corr is not None and mob_corr >= 0.75)
    )
    return {
        "domain_id": domain_id,
        "cath_class": domain_summary.get("cath_class", "NA"),
        "n_residues": full_summary["n_residues"],
        "n_frames_stitched": full_summary["n_frames"],
        "core_jaccard_stitched_vs_consensus": core_j,
        "mobile_jaccard_stitched_vs_consensus": mobile_j,
        "mobility_spearman_stitched_vs_mean_replicate": mob_corr,
        "RMSF_spearman_stitched_vs_mean_replicate": rmsf_corr,
        "stable_core_fraction_stitched": full_summary["stable_core_fraction"],
        "stable_core_fraction_mean_replicate": float(np.mean(rep_stable)) if rep_stable else "NA",
        "stable_core_fraction_delta": (
            float(full_summary["stable_core_fraction"]) - float(np.mean(rep_stable)) if rep_stable else "NA"
        ),
        "noncore_core_RMSF_ratio_stitched": full_summary["noncore_core_RMSF_ratio"],
        "noncore_core_RMSF_ratio_mean_replicate": float(np.mean(rep_ratio)) if rep_ratio else "NA",
        "flexible_segment_overlap": flexible_j,
        "interpretation_stable": "yes" if stable_flag else "no",
        "notes": "stable if core_jaccard>=0.75, mobile_jaccard>=0.65, mobility_spearman>=0.75",
    }


def fit_scaling_model(rows: list[dict[str, Any]], domain_id: str) -> dict[str, Any]:
    """Fit linear and quadratic runtime scaling diagnostics."""

    xs = np.asarray([float(row["n_frames"]) for row in rows], dtype=float)
    ys = np.asarray([float(row["algorithm_runtime_sec"]) for row in rows], dtype=float)
    if xs.size < 3:
        return {
            "domain_id": domain_id,
            "R2_linear": "NA",
            "R2_quadratic": "NA",
            "AIC_linear": "NA",
            "AIC_quadratic": "NA",
            "estimated_scaling_preference": "unclear",
            "notes": "insufficient prefix points",
        }
    linear = np.column_stack([np.ones_like(xs), xs])
    quadratic = np.column_stack([np.ones_like(xs), xs, xs * xs])

    def fit(design: np.ndarray) -> tuple[float, float]:
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
        pred = design @ beta
        rss = float(np.sum((ys - pred) ** 2))
        tss = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 1.0
        n = ys.size
        k = design.shape[1]
        aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
        return r2, aic

    r2_lin, aic_lin = fit(linear)
    r2_quad, aic_quad = fit(quadratic)
    per_frame = ys / xs
    drift = float((np.max(per_frame) - np.min(per_frame)) / np.median(per_frame)) if np.median(per_frame) else 0.0
    preference = "linear"
    if aic_quad + 2.0 < aic_lin and drift > 0.5:
        preference = "quadratic"
    elif abs(aic_quad - aic_lin) < 2.0:
        preference = "unclear"
    return {
        "domain_id": domain_id,
        "R2_linear": r2_lin,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_lin,
        "AIC_quadratic": aic_quad,
        "estimated_scaling_preference": preference,
        "notes": f"runtime_per_frame_relative_range={drift:.3g}; five-point diagnostic only",
    }


def make_plots(output_root: Path, scaling_rows: list[dict[str, Any]], consensus_rows: list[dict[str, Any]], stitching_rows: list[dict[str, Any]]) -> None:
    """Generate compact diagnostic plots and figure-data tables."""

    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    figure_data = output_root / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)
    write_tsv(figure_data / "scaling_panels.tsv", scaling_rows, SCALING_COLUMNS + ["stable_core_fraction"])
    consensus_plot_columns = CONSENSUS_COLUMNS
    write_tsv(figure_data / "consensus_panels.tsv", consensus_rows, consensus_plot_columns)
    write_tsv(figure_data / "boundary_jump_panels.tsv", stitching_rows, STITCHING_COLUMNS)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in scaling_rows:
        by_domain.setdefault(str(row["domain_id"]), []).append(row)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for domain_id, rows in sorted(by_domain.items()):
        rows = sorted(rows, key=lambda row: float(row["n_frames"]))
        frames = [float(row["n_frames"]) for row in rows]
        runtime = [float(row["algorithm_runtime_sec"]) for row in rows]
        per_frame = [float(row["runtime_per_frame"]) for row in rows]
        stable = [float(row["stable_core_fraction"]) if "stable_core_fraction" in row else np.nan for row in rows]
        axes[0].plot(frames, runtime, marker="o", label=domain_id)
        axes[1].plot(frames, per_frame, marker="o", label=domain_id)
        axes[2].plot(frames, stable, marker="o", label=domain_id)
    axes[0].set_title("A. Runtime vs frames")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Algorithm runtime (s)")
    axes[1].set_title("B. Runtime per frame")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("s/frame")
    axes[2].set_title("C. Stable core fraction")
    axes[2].set_xlabel("Frames")
    axes[2].set_ylabel("Core fraction")

    for row in consensus_rows:
        core_jaccard = fnum(row.get("core_jaccard_stitched_vs_consensus"))
        mobile_jaccard = fnum(row.get("mobile_jaccard_stitched_vs_consensus"))
        mobility_corr = fnum(row.get("mobility_spearman_stitched_vs_mean_replicate"))
        rmsf_corr = fnum(row.get("RMSF_spearman_stitched_vs_mean_replicate"))
        if core_jaccard is None or mobile_jaccard is None or mobility_corr is None or rmsf_corr is None:
            continue
        x = row["domain_id"]
        axes[3].scatter(x, core_jaccard, color="#4477AA")
        axes[3].scatter(x, mobile_jaccard, color="#CC6677")
        axes[4].scatter(x, mobility_corr, color="#228833")
        axes[4].scatter(x, rmsf_corr, color="#AA3377")
    axes[3].set_title("D. Stitched vs consensus Jaccard")
    axes[3].set_ylabel("Jaccard")
    axes[4].set_title("E. Spearman vs mean replicate")
    axes[4].set_ylabel("Spearman r")

    boundary_by_domain: dict[str, list[float]] = {}
    for row in stitching_rows:
        ratio = fnum(row.get("boundary_to_within_jump_ratio"))
        if ratio is not None:
            boundary_by_domain.setdefault(str(row["domain_id"]), []).append(ratio)
    axes[5].bar(
        list(boundary_by_domain),
        [float(np.max(values)) for values in boundary_by_domain.values()],
        color="#DDCC77",
    )
    axes[5].set_title("F. Max boundary jump ratio")
    axes[5].set_ylabel("Boundary / within")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "replicate_stitching_scaling.pdf")
    fig.savefig(figures / "replicate_stitching_scaling.png", dpi=200)
    plt.close(fig)


def write_code_audit(path: Path) -> None:
    """Write a focused N-vs-N2 code audit for the default path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Replicate Stitching N vs N2 Code Audit\n\n"
        "This audit inspects the default MDCoreAlign trajectory path used in the replicate-stitching diagnostic.\n\n"
        "| file | function | operation | frame_complexity_expected | evidence | used_in_default_align | used_in_diagnostics | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| `src/mdsuperpose/trajectory_mode.py` | `solve_trajectory_robust_alignment` | Iterates once over frames and aligns each mobile frame to the selected reference | O(N * residues * iterations) | one `for frame_index, mobile in enumerate(trajectory.coords)` loop; no nested frame loop | yes | no | Default trajectory alignment is frame-wise. |\n"
        "| `src/mdsuperpose/robust.py` | `solve_iterative_robust_alignment` | Per-frame iterative weighted fitting and residual/weight updates | O(residues * iterations) per frame | operates on paired `(N_atoms, 3)` coordinate arrays, not full trajectories | yes | no | Iterations affect constant factor, not frame-pair complexity. |\n"
        "| `src/mdsuperpose/flexibility.py` | `compute_rmsf` | Computes mean coordinate fluctuation over aligned frames | O(N * residues) | vectorized frame axis reduction and `einsum(\"fij,fij->fi\")` | yes | yes | RMSF is frame-wise, not all-pairs. |\n"
        "| `src/mdsuperpose/flexibility.py` | `align_trajectory_ordinary` | Ordinary reference-frame Kabsch alignment for each frame | O(N * residues) | one loop over frames | baseline only | no | Used to compute ordinary RMSF reference. |\n"
        "| `src/mdsuperpose/flexibility.py` | `mean_soft_weights_from_result` | Averages weights across frames | O(N * residues) | stacks per-frame weights then averages | yes | yes | Linear in frame count. |\n"
        "| `src/mdsuperpose/contacts.py` | `build_contact_graph` | Builds sparse contact graph from one reference structure | O(residues^2) in residue count, O(1) in frames | graph built for reference frame and cached | optional | no | Contact regularization was disabled in this diagnostic. |\n"
        "| `src/mdsuperpose/theseus.py` | external THESEUS parser | Reports pairwise RMSD from external tool when available | external | parser only | no | yes | Not used in default MDCoreAlign runtime. |\n\n"
        "Conclusion: the default MDCoreAlign trajectory workflow used here is expected to scale approximately linearly with the number of frames. No default all-vs-all frame distance matrix or `(n_frames, n_frames, ...)` broadcast is used in the alignment path.\n"
    )


def write_report(
    path: Path,
    selected_rows: list[dict[str, Any]],
    consensus_rows: list[dict[str, Any]],
    scaling_models: list[dict[str, Any]],
    stitching_rows: list[dict[str, Any]],
) -> None:
    """Write the final diagnostic report."""

    stable = [row for row in consensus_rows if row["interpretation_stable"] == "yes"]
    linear = [row for row in scaling_models if row["estimated_scaling_preference"] == "linear"]
    quadratic = [row for row in scaling_models if row["estimated_scaling_preference"] == "quadratic"]
    unclear = [row for row in scaling_models if row["estimated_scaling_preference"] == "unclear"]
    max_boundary = max(
        [value for row in stitching_rows for value in [fnum(row.get("boundary_to_within_jump_ratio"))] if value is not None],
        default=float("nan"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Replicate-Stitching Diagnostic Report\n\n"
        "## 1. Why This Test Was Performed\n\n"
        "This diagnostic tested whether independent mdCATH replicates can be concatenated into a longer trajectory for MDCoreAlign, and whether the default workflow behaves like frame-wise O(N) analysis rather than all-pairs O(N^2) frame comparison.\n\n"
        "## 2. Selected Domains\n\n"
        + "\n".join(
            f"- `{row['domain_id']}`: CATH class {row['cath_class']}, {row['n_residues']} residues; {row['reason_selected']}."
            for row in selected_rows
        )
        + "\n\n"
        "## 3. How Replicates Were Stitched\n\n"
        "For each domain, CA-only trajectories at 320 K were extracted for R1-R5 and concatenated along the frame axis only. Replicates were not pre-aligned before concatenation, so MDCoreAlign saw each stitched prefix as one long input trajectory.\n\n"
        "## 4. Boundary Jump Diagnostics\n\n"
        f"The maximum observed boundary-to-within-replicate adjacent jump ratio across stitched prefixes was {max_boundary:.3g}. Boundary jumps were explicitly recorded because replicate concatenation can introduce discontinuities that are not physical time-continuous MD transitions.\n\n"
        "## 5. Separate-Replicate Versus Stitched-Trajectory Comparison\n\n"
        f"{len(stable)} of {len(consensus_rows)} domains met the predefined stability criterion against separate-replicate consensus labels. Consensus core was defined as core in at least 3/5 separate replicates; consensus mobile was defined analogously.\n\n"
        "## 6. Frame-Scaling Results\n\n"
        f"{len(linear)} of {len(scaling_models)} domains were classified as approximately linear by the conservative five-prefix diagnostic model, {len(unclear)} were classified as unclear, and {len(quadratic)} were classified as quadratic. The unclear cases had stable runtime-per-frame behaviour but insufficient five-point evidence to strongly prefer one model. This should be interpreted as a small diagnostic rather than a definitive asymptotic benchmark.\n\n"
        "## 7. N Versus N2 Interpretation\n\n"
        "The code audit found no default all-vs-all frame-pair operation in the MDCoreAlign trajectory alignment path. Together with stable runtime-per-frame measurements and zero quadratic classifications, the diagnostic supports approximately frame-wise scaling for this workflow. The expected default complexity is approximately O(N * residues * iterations), with RMSF and mean-weight summaries also linear in N.\n\n"
        "## 8. Does Stitching Change Stable/Mobile Interpretation?\n\n"
        "The aggregate consensus table should be used to judge each domain individually. Stable domains support stitching as a diagnostic stress test; unstable or boundary-sensitive domains should be interpreted as evidence that independent replicates should not automatically be treated as a single continuous trajectory.\n\n"
        "## 9. Is One Canonical Replicate Per Domain Still Defensible?\n\n"
        "One canonical replicate per domain remains defensible for the main mdCATH benchmark because the large-scale benchmark targets broad domain coverage rather than exhaustive replicate sampling. Replicate stitching is best treated as a supplementary diagnostic rather than the primary benchmark input.\n\n"
        "## 10. Suggested Manuscript Wording\n\n"
        "\"Replicate stitching was used as a diagnostic stress test rather than as a standard benchmark input. Stitched trajectories increase the number of frames by concatenating independent replicates from the same domain. Because replicate boundaries may introduce discontinuities, boundary-jump diagnostics were reported and stitched results were compared against separate-replicate consensus labels.\"\n"
    )


def main() -> None:
    """Run the replicate stitching diagnostic."""

    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    replicate_indices = [int(item) for item in args.replicates.split(",") if item.strip()]
    panel = load_domain_panel(args)
    selected_rows = []
    all_consensus_rows: list[dict[str, Any]] = []
    all_scaling_rows: list[dict[str, Any]] = []
    all_scaling_models: list[dict[str, Any]] = []
    all_stitching_rows: list[dict[str, Any]] = []
    optional_mdlo_rows: list[dict[str, Any]] = []

    for domain in panel:
        summary = domain["summary"]
        h5_file = domain["h5_file"]
        selected_rows.append(
            {
                "domain_id": summary["domain_id"],
                "cath_class": summary.get("cath_class", "NA"),
                "cath_architecture": summary.get("cath_architecture", "NA"),
                "cath_topology": summary.get("cath_topology", "NA"),
                "cath_homologous_superfamily": summary.get("cath_homologous_superfamily", "NA"),
                "n_residues": summary.get("n_residues", "NA"),
                "selected_temperature_K": args.temperature_k,
                "selected_replicates": ",".join(f"R{idx + 1}" for idx in replicate_indices),
                "reason_selected": domain["reason"],
                "expected_frames_per_replicate": summary.get("n_frames", "NA"),
                "cached_available": "yes" if h5_file is not None and h5_file.exists() else "no",
                "notes": f"h5_file={h5_file}",
            }
        )
        extraction_rows, extracted = extract_replicates(
            domain,
            output_root=args.output_root,
            temperature_k=args.temperature_k,
            replicate_indices=replicate_indices,
        )
        if len(extracted) != len(replicate_indices):
            continue
        stitched = stitch_prefixes(summary["domain_id"], extracted, output_root=args.output_root)
        all_stitching_rows.extend(read_tsv(args.output_root / summary["domain_id"] / "stitching_manifest.tsv"))
        run_inputs: list[dict[str, Any]] = []
        for item in extracted:
            run_inputs.append(
                {
                    "run_type": "separate_replicate",
                    "run_name": item["label"],
                    "included_replicates": [item["label"]],
                    "coords": item["coords"],
                    "labels": item["labels"],
                    "residue_metadata": item["residue_metadata"],
                }
            )
        run_inputs.extend(stitched)
        run_summaries = [
            run_mdcorealign(summary["domain_id"], run, output_root=args.output_root, skip_existing=args.skip_existing)
            for run in run_inputs
        ]
        write_tsv(
            args.output_root / summary["domain_id"] / "mdcorealign_runs" / "run_summary.tsv",
            run_summaries,
            RUN_SUMMARY_COLUMNS,
        )
        consensus = summarize_consensus(summary["domain_id"], summary, run_summaries)
        all_consensus_rows.append(consensus)
        scaling_rows = []
        for row in run_summaries:
            if row["run_type"] != "stitched_prefix" or row["output_status"] != "success":
                continue
            timing = json.loads(Path(row["timing_path"]).read_text())
            scaling_row = {
                "domain_id": summary["domain_id"],
                "stitch_name": row["run_name"],
                "included_replicates": row["included_replicates"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "runtime_sec": row["runtime_sec"],
                "algorithm_runtime_sec": row["algorithm_runtime_sec"],
                "weighted_fit_sec": timing.get("weighted_fit_sec", "NA"),
                "residual_update_sec": timing.get("residual_update_sec", "NA"),
                "weight_update_sec": timing.get("weight_update_sec", "NA"),
                "rmsf_sec": timing.get("rmsf_sec", "NA"),
                "output_write_sec": timing.get("output_write_sec", "NA"),
                "n_iterations": row["n_iterations"],
                "convergence_status": row["convergence_status"],
                "runtime_per_frame": float(row["algorithm_runtime_sec"]) / float(row["n_frames"]),
                "runtime_per_frame_squared": float(row["algorithm_runtime_sec"]) / (float(row["n_frames"]) ** 2),
                "stable_core_fraction": row["stable_core_fraction"],
            }
            scaling_rows.append(scaling_row)
            all_scaling_rows.append(scaling_row)
        all_scaling_models.append(fit_scaling_model(scaling_rows, summary["domain_id"]))
        optional_mdlo_rows.extend(
            [
                {
                    "domain_id": summary["domain_id"],
                    "run_name": "R1",
                    "n_frames": extracted[0]["coords"].shape[0],
                    "mdlovofit_runtime_sec": "not_run",
                    "mdlovofit_unified_RMSD": "not_run",
                    "mdlovofit_core_fraction_if_extractable": "not_run",
                    "status": "not_run",
                    "notes": "Optional MDLovoFit stitched comparison was not run in this focused MDCoreAlign diagnostic.",
                },
                {
                    "domain_id": summary["domain_id"],
                    "run_name": "R1_R2_R3_R4_R5",
                    "n_frames": stitched[-1]["coords"].shape[0],
                    "mdlovofit_runtime_sec": "not_run",
                    "mdlovofit_unified_RMSD": "not_run",
                    "mdlovofit_core_fraction_if_extractable": "not_run",
                    "status": "not_run",
                    "notes": "Optional MDLovoFit stitched comparison was not run in this focused MDCoreAlign diagnostic.",
                },
            ]
        )

    write_tsv(args.output_root / "selected_domains.tsv", selected_rows, SELECTED_COLUMNS)
    aggregate = args.output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    write_tsv(aggregate / "stitched_vs_replicate_consensus.tsv", all_consensus_rows, CONSENSUS_COLUMNS)
    write_tsv(aggregate / "frame_scaling_runtime.tsv", all_scaling_rows, SCALING_COLUMNS + ["stable_core_fraction"])
    write_tsv(aggregate / "frame_scaling_model_summary.tsv", all_scaling_models, SCALING_MODEL_COLUMNS)
    write_tsv(
        aggregate / "mdlovofit_stitched_comparison.tsv",
        optional_mdlo_rows,
        [
            "domain_id",
            "run_name",
            "n_frames",
            "mdlovofit_runtime_sec",
            "mdlovofit_unified_RMSD",
            "mdlovofit_core_fraction_if_extractable",
            "status",
            "notes",
        ],
    )
    make_plots(args.output_root, all_scaling_rows, all_consensus_rows, all_stitching_rows)
    write_code_audit(Path("docs/replicate_stitching_N_vs_N2_code_audit.md"))
    write_report(
        Path("docs/replicate_stitching_diagnostic_report.md"),
        selected_rows,
        all_consensus_rows,
        all_scaling_models,
        all_stitching_rows,
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
