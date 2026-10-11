#!/usr/bin/env python
"""Run a transition-aligned synthetic long-trajectory scaling benchmark.

This benchmark intentionally uses synthetic transition-aligned mdCATH
replicate stitching. It is a computational scaling diagnostic, not evidence of
physical long-time biological continuity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.alignment import apply_transform, solve_weighted_rigid_alignment
from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.flexibility import (
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
    spearman_correlation,
)
from mdsuperpose.long_mode import build_long_flexibility_profile, solve_long_trajectory_alignment
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_coords,
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.metrics import mean_rmsd
from mdsuperpose.theseus import parse_theseus_stdout, parse_theseus_variances
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment
from scripts.mdcath_benchmark.build_transition_aligned_stitched_trajectory import (
    adjacent_displacements,
    coord_path,
    h5_path_from_notes,
    load_raw_replicates,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory
from scripts.mdcath_benchmark.mdcath_common import write_json, write_tsv
from scripts.mdcath_benchmark.run_replicate_stitching_external_comparison import (
    check_command,
    percentile_iqr,
    sha256_file,
    theseus_output_path,
)


REPLICATE_LABELS = ("A", "B", "C", "D", "E")
REPLICATE_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
MDLOVOFIT_EXE = Path("/usr/local/bin/mdlovofit")
THESEUS_EXE = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")

SELECTED_COLUMNS = [
    "domain_id",
    "cath_class",
    "n_residues",
    "runtime_ratio_mdcorealign_over_mdlovofit",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "selection_bucket",
    "h5_path",
    "notes",
]

BOUNDARY_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_segments",
    "n_total_frames",
    "n_residues",
    "median_raw_boundary_jump",
    "median_post_boundary_jump",
    "max_post_boundary_jump",
    "median_within_adjacent_jump",
    "median_post_boundary_to_within_ratio",
    "max_post_boundary_to_within_ratio",
    "boundary_smoothing_success",
    "notes",
]

REPEAT_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_frames",
    "n_residues",
    "method",
    "mode",
    "repeat_id",
    "runtime_scope",
    "runtime_sec",
    "status",
    "notes",
]

SUMMARY_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_frames",
    "n_residues",
    "mdcorealign_default_runtime",
    "mdcorealign_long_runtime",
    "mdcorealign_long_fast_runtime",
    "mdlovofit_runtime",
    "theseus_runtime_if_available",
    "long_vs_mdlovofit_speedup",
    "default_vs_mdlovofit_speedup",
    "long_vs_default_speedup",
    "long_vs_theseus_speedup_if_available",
    "mdcorealign_long_faster_than_mdlovofit",
    "notes",
]

DEFAULT_LONG_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_frames",
    "n_residues",
    "default_runtime_sec",
    "long_runtime_sec",
    "long_fast_runtime_sec",
    "long_vs_default_speedup",
    "default_unified_RMSD",
    "long_unified_RMSD",
    "long_fast_unified_RMSD",
    "core_jaccard_default_vs_long",
    "mobile_jaccard_default_vs_long",
    "mobility_spearman_default_vs_long",
    "RMSF_spearman_default_vs_long",
    "stable_core_fraction_diff",
    "flexible_segment_overlap",
    "status",
    "notes",
]

LONG_MDLOVO_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_frames",
    "n_residues",
    "long_runtime_sec",
    "mdlovofit_runtime_sec",
    "long_vs_mdlovofit_speedup",
    "long_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "long_lower_RMSD_than_mdlovofit",
    "mdlovofit_status",
    "notes",
]

RMSD_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_frames",
    "n_residues",
    "mdcorealign_default_RMSD",
    "mdcorealign_long_RMSD",
    "mdcorealign_long_fast_RMSD",
    "mdlovofit_RMSD",
    "theseus_RMSD_if_available",
    "long_lower_RMSD_than_mdlovofit",
    "long_RMSD_diff_vs_default",
    "evaluation_status",
    "notes",
]

SCALING_COLUMNS = [
    "domain_id",
    "method",
    "mode",
    "n_prefixes",
    "R2_linear",
    "R2_quadratic",
    "AIC_linear",
    "AIC_quadratic",
    "scaling_label",
    "notes",
]

MEMORY_COLUMNS = [
    "domain_id",
    "target_frames",
    "method",
    "mode",
    "peak_memory_mb",
    "runtime_sec",
    "status",
    "notes",
]


@dataclass(slots=True)
class MethodResult:
    """Compact method result used for aggregation."""

    status: str
    runtime_sec: float | str
    full_workflow_runtime_sec: float | str
    unified_rmsd: float | str
    weighted_rmsd: float | str = "NA"
    core_mask: np.ndarray | None = None
    mobile_mask: np.ndarray | None = None
    mobility_score: np.ndarray | None = None
    robust_rmsf: np.ndarray | None = None
    flexible_mask: np.ndarray | None = None
    peak_memory_mb: float | str = "NA"
    notes: str = "NA"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/mdcath_benchmark/transition_aligned_long_mode_benchmark"))
    parser.add_argument(
        "--all-domain-summary",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate/all_domain_canonical_per_domain_summary.tsv"),
    )
    parser.add_argument("--n-domains", type=int, default=28)
    parser.add_argument("--targets", default="10000,25000,50000,100000")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-default-frames", type=int, default=10000)
    parser.add_argument("--max-mdlovofit-frame-atoms", type=int, default=3_500_000)
    parser.add_argument("--max-theseus-frame-atoms", type=int, default=750_000)
    parser.add_argument("--theseus-max-frames", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-weight-fit-frames", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mdlovofit-exe", type=Path, default=MDLOVOFIT_EXE)
    parser.add_argument("--theseus-exe", type=Path, default=THESEUS_EXE)
    parser.add_argument("--keep-external-pdbs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def split_ints(value: str) -> list[int]:
    """Parse comma-separated integers."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def finite_float(value: object) -> float | None:
    """Return a finite float or None."""

    try:
        if value in ("", "NA", None, "not_run", "skipped"):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def rss_mb(children: bool = False) -> float:
    """Return max resident set size in MB for this process or children."""

    usage = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    return value / 1024.0


def jaccard(a: np.ndarray | None, b: np.ndarray | None) -> float | str:
    """Return Jaccard overlap for two masks."""

    if a is None or b is None:
        return "NA"
    first = np.asarray(a, dtype=bool)
    second = np.asarray(b, dtype=bool)
    union = np.logical_or(first, second)
    if not np.any(union):
        return 1.0
    return float(np.sum(np.logical_and(first, second)) / np.sum(union))


def fit_scaling(rows: list[dict[str, Any]], method: str, mode: str) -> dict[str, Any]:
    """Fit linear and quadratic runtime models for one method/domain."""

    xs = []
    ys = []
    for row in rows:
        if row.get("method") != method or row.get("mode") != mode or row.get("status") != "success":
            continue
        runtime = finite_float(row.get("runtime_sec"))
        frames = finite_float(row.get("target_frames"))
        if runtime is None or frames is None:
            continue
        xs.append(frames)
        ys.append(runtime)
    if len(xs) < 3:
        return {
            "n_prefixes": len(xs),
            "R2_linear": "NA",
            "R2_quadratic": "NA",
            "AIC_linear": "NA",
            "AIC_quadratic": "NA",
            "scaling_label": "insufficient_data",
        }
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    def fit(design: np.ndarray) -> tuple[float, float]:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 1.0
        aic = len(y) * math.log(max(rss / len(y), 1e-12)) + 2 * design.shape[1]
        return r2, aic

    r2_linear, aic_linear = fit(np.column_stack([np.ones_like(x), x]))
    r2_quad, aic_quad = fit(np.column_stack([np.ones_like(x), x, x * x]))
    per_frame = y / x
    variation = float((np.max(per_frame) - np.min(per_frame)) / np.median(per_frame)) if np.median(per_frame) else float("nan")
    label = "linear"
    if aic_quad + 2.0 < aic_linear and variation > 0.5:
        label = "possible_quadratic"
    elif aic_quad + 2.0 < aic_linear and variation > 0.25:
        label = "possible_superlinear"
    elif abs(aic_quad - aic_linear) < 2.0:
        label = "unclear"
    return {
        "n_prefixes": len(xs),
        "R2_linear": r2_linear,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_linear,
        "AIC_quadratic": aic_quad,
        "scaling_label": label,
    }


def load_summary(path: Path) -> pd.DataFrame:
    """Load all-domain summary with numeric helper columns."""

    df = pd.read_csv(path, sep="\t")
    numeric = [
        "cath_class",
        "n_frames",
        "n_residues",
        "mdcorealign_runtime_sec",
        "mdlovofit_runtime_sec",
        "mdcorealign_unified_RMSD",
        "mdlovofit_unified_RMSD",
        "runtime_ratio_mdcorealign_over_mdlovofit",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["h5_path"] = df["notes"].map(h5_path_from_notes)
    df["difficulty_rmsd"] = df[["mdcorealign_unified_RMSD", "mdlovofit_unified_RMSD"]].max(axis=1)
    return df


def add_unique(selected: list[dict[str, Any]], rows: pd.DataFrame, bucket: str, limit: int) -> None:
    """Append unique domain rows from a bucket."""

    existing = {item["domain_id"] for item in selected}
    added = 0
    for _, row in rows.iterrows():
        domain_id = str(row["domain_id"])
        if domain_id in existing:
            continue
        h5_path = row["h5_path"]
        if h5_path is None or not Path(h5_path).exists():
            continue
        selected.append(
            {
                "domain_id": domain_id,
                "cath_class": int(row["cath_class"]) if pd.notna(row["cath_class"]) else "NA",
                "n_residues": int(row["n_residues"]),
                "runtime_ratio_mdcorealign_over_mdlovofit": float(row["runtime_ratio_mdcorealign_over_mdlovofit"]),
                "mdcorealign_unified_RMSD": float(row["mdcorealign_unified_RMSD"]),
                "mdlovofit_unified_RMSD": float(row["mdlovofit_unified_RMSD"]),
                "selection_bucket": bucket,
                "h5_path": str(h5_path),
                "notes": "Selected for transition-aligned synthetic scaling coverage.",
            }
        )
        existing.add(domain_id)
        added += 1
        if added >= limit:
            break


def select_domain_panel(summary: pd.DataFrame, n_domains: int) -> list[dict[str, Any]]:
    """Select a balanced 20-50 domain panel."""

    base = summary[
        summary["mdcorealign_status"].eq("success")
        & summary["mdlovofit_status"].eq("success")
        & summary["runtime_ratio_mdcorealign_over_mdlovofit"].notna()
        & summary["n_residues"].between(50, 180)
    ].copy()
    selected: list[dict[str, Any]] = []
    if n_domains <= 20:
        quotas = {
            "fast": 4,
            "slow": 4,
            "class4": 3,
            "class6": 3,
            "long": 3,
            "rmsd": max(3, n_domains - 17),
        }
    else:
        quotas = {
            "fast": 5,
            "slow": 5,
            "class4": 4,
            "class6": 4,
            "long": max(4, (n_domains - 18) // 2),
            "rmsd": max(6, n_domains - 22),
        }
    add_unique(selected, base.sort_values("runtime_ratio_mdcorealign_over_mdlovofit"), "mdcorealign_fastest_vs_mdlovofit", quotas["fast"])
    add_unique(selected, base.sort_values("runtime_ratio_mdcorealign_over_mdlovofit", ascending=False), "mdcorealign_slowest_vs_mdlovofit", quotas["slow"])
    add_unique(selected, base[base["cath_class"] == 4].sort_values("runtime_ratio_mdcorealign_over_mdlovofit", ascending=False), "cath_class_4", quotas["class4"])
    add_unique(selected, base[base["cath_class"] == 6].sort_values("runtime_ratio_mdcorealign_over_mdlovofit", ascending=False), "cath_class_6", quotas["class6"])
    add_unique(selected, base.sort_values("n_residues", ascending=False), "longer_domains", quotas["long"])
    add_unique(selected, base.sort_values("difficulty_rmsd", ascending=False), "high_rmsd_difficulty", quotas["rmsd"])
    if len(selected) < n_domains:
        add_unique(selected, base.sample(frac=1.0, random_state=42), "deterministic_filler", n_domains - len(selected))
    return selected[:n_domains]


def extract_replicates(domain_row: dict[str, Any], root: Path) -> None:
    """Extract A-E raw CA replicates to local compressed arrays."""

    domain_id = str(domain_row["domain_id"])
    domain_root = root / "domains" / domain_id
    raw_root = domain_root / "raw_replicates"
    raw_root.mkdir(parents=True, exist_ok=True)
    reference_labels: list[str] | None = None
    rows = []
    for label in REPLICATE_LABELS:
        replicate_index = REPLICATE_INDEX[label]
        out_npz = raw_root / f"{label}_ca_coords.npz"
        out_json = raw_root / f"{label}_metadata.json"
        status = "success"
        failure = "NA"
        try:
            if out_npz.exists() and out_npz.stat().st_size == 0:
                out_npz.unlink()
            if out_npz.exists():
                payload = np.load(out_npz, allow_pickle=True)
                coords = np.asarray(payload["coords"], dtype=float)
                labels = [str(item) for item in payload["labels"].tolist()]
                residue_metadata = payload["residue_metadata"].tolist() if "residue_metadata" in payload.files else []
            else:
                coords, labels, residue_metadata, metadata = extract_ca_trajectory(
                    Path(domain_row["h5_path"]),
                    coordinate_path=coord_path(domain_id, replicate_index),
                    min_ca_atoms=30,
                    min_frames=20,
                )
                coords = coords - np.mean(coords, axis=1, keepdims=True)
                metadata["coordinate_preprocessing"] = "per-frame CA centroid removal before transition-aligned stitching"
                np.savez_compressed(
                    out_npz,
                    coords=coords.astype(np.float32, copy=False),
                    labels=np.asarray(labels, dtype=object),
                    residue_metadata=np.asarray(residue_metadata, dtype=object),
                )
                write_json(out_json, metadata)
            mapping_ok = reference_labels is None or labels == reference_labels
            if reference_labels is None:
                reference_labels = labels
            if not mapping_ok:
                raise ValueError("replicate residue labels do not match")
            if not np.isfinite(coords).all():
                raise ValueError("non-finite coordinates")
        except Exception as exc:
            coords = np.zeros((0, 0, 3), dtype=float)
            residue_metadata = []
            mapping_ok = False
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "domain_id": domain_id,
                "replicate_label": label,
                "replicate_index": replicate_index,
                "n_frames": int(coords.shape[0]),
                "n_residues": int(coords.shape[1]) if coords.ndim == 3 else 0,
                "residue_mapping_ok": int(mapping_ok),
                "extraction_status": status,
                "failure_reason": failure,
                "npz": str(out_npz) if out_npz.exists() else "NA",
                "n_metadata_rows": len(residue_metadata),
            }
        )
    write_tsv(
        domain_root / "replicate_manifest.tsv",
        rows,
        [
            "domain_id",
            "replicate_label",
            "replicate_index",
            "n_frames",
            "n_residues",
            "residue_mapping_ok",
            "extraction_status",
            "failure_reason",
            "npz",
            "n_metadata_rows",
        ],
    )
    if any(row["extraction_status"] != "success" or not row["residue_mapping_ok"] for row in rows):
        raise RuntimeError(f"replicate extraction failed for {domain_id}")


def canonical_sequence(raw_lengths: dict[str, int], target_frames: int) -> list[str]:
    """Build the canonical R1-R5 cycle sequence needed to reach target frames."""

    labels: list[str] = []
    total = 0
    index = 0
    while total < target_frames:
        label = REPLICATE_LABELS[index % len(REPLICATE_LABELS)]
        labels.append(label)
        total += raw_lengths[label]
        index += 1
    return labels


def rmsd_between(first: np.ndarray, second: np.ndarray) -> float:
    """Return per-frame CA RMSD without fitting."""

    diff = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def build_transition_aligned_fast(
    domain_id: str,
    raw: dict[str, tuple[np.ndarray, list[str], list[dict[str, str]]]],
    target_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    """Build a transition-aligned canonical R1-R5 cycle trajectory."""

    raw_lengths = {label: raw[label][0].shape[0] for label in REPLICATE_LABELS}
    sequence = canonical_sequence(raw_lengths, target_frames)
    chunks = [np.asarray(raw[sequence[0]][0], dtype=np.float32).copy()]
    current_last = np.asarray(chunks[0][-1], dtype=float)
    boundary_records: list[dict[str, Any]] = []
    for segment_index, label in enumerate(sequence[1:], start=2):
        raw_coords = np.asarray(raw[label][0], dtype=float)
        source_first = raw_coords[0]
        pre = rmsd_between(source_first, current_last)
        transform = solve_weighted_rigid_alignment(source_first, current_last, np.ones(source_first.shape[0], dtype=float))
        transformed = apply_transform(raw_coords, transform).astype(np.float32, copy=False)
        post = rmsd_between(transformed[0], current_last)
        chunks.append(transformed)
        raw_within = adjacent_displacements(raw_coords)
        boundary_records.append(
            {
                "domain_id": domain_id,
                "target_frames": target_frames,
                "segment_index": segment_index,
                "raw_replicate_label": label,
                "transformed_segment_label": f"{label}'_{segment_index}",
                "pre_alignment_boundary_rmsd": pre,
                "post_alignment_boundary_rmsd": post,
                "raw_within_median": float(np.median(raw_within)) if raw_within.size else "NA",
                "determinant_rotation": float(np.linalg.det(transform.rotation)),
                "notes": "Fresh Kabsch transform from raw replicate first frame to current terminal frame.",
            }
        )
        current_last = np.asarray(transformed[-1], dtype=float)
        if sum(chunk.shape[0] for chunk in chunks) >= target_frames:
            break
    coords = np.concatenate(chunks, axis=0)[:target_frames]
    coords = coords - np.mean(coords[-1], axis=0, keepdims=True)
    return coords.astype(np.float32, copy=False), boundary_records, sequence


def summarize_boundary(coords: np.ndarray, boundary_records: list[dict[str, Any]], domain_id: str, target_frames: int) -> dict[str, Any]:
    """Return boundary smoothing summary for one synthetic trajectory."""

    raw_jumps = [float(item["pre_alignment_boundary_rmsd"]) for item in boundary_records]
    post_jumps = [float(item["post_alignment_boundary_rmsd"]) for item in boundary_records]
    within = adjacent_displacements(coords)
    within_median = float(np.median(within)) if within.size else float("nan")
    ratios = [jump / within_median for jump in post_jumps if math.isfinite(within_median) and within_median > 0.0]
    return {
        "domain_id": domain_id,
        "target_frames": target_frames,
        "n_segments": len(boundary_records) + 1,
        "n_total_frames": int(coords.shape[0]),
        "n_residues": int(coords.shape[1]),
        "median_raw_boundary_jump": float(np.median(raw_jumps)) if raw_jumps else "NA",
        "median_post_boundary_jump": float(np.median(post_jumps)) if post_jumps else "NA",
        "max_post_boundary_jump": float(np.max(post_jumps)) if post_jumps else "NA",
        "median_within_adjacent_jump": within_median,
        "median_post_boundary_to_within_ratio": float(np.median(ratios)) if ratios else "NA",
        "max_post_boundary_to_within_ratio": float(np.max(ratios)) if ratios else "NA",
        "boundary_smoothing_success": int(bool(raw_jumps) and np.median(post_jumps) < np.median(raw_jumps)),
        "notes": "Transition-aligned synthetic long trajectory; not physically continuous MD.",
    }


def run_default(coords: np.ndarray) -> MethodResult:
    """Run default MDCoreAlign trajectory mode."""

    start = time.perf_counter()
    try:
        result = solve_trajectory_robust_alignment(
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
        aligned = robust_aligned_coords_from_result(result)
        weights = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
        rmsf = compute_rmsf(aligned)
        mobility = 1.0 - weights
        segments = detect_flexible_segments(mobility, percentile_cutoff=0.8, min_segment_length=3, merge_gap=1)
        runtime = time.perf_counter() - start
        core = select_core_by_target_fraction(weights, 0.6).core_mask
        mobile = ~core
        return MethodResult(
            status="success",
            runtime_sec=runtime,
            full_workflow_runtime_sec=runtime,
            unified_rmsd=mean_rmsd(aligned, coords[0]),
            weighted_rmsd=result.mean_weighted_rmsd,
            core_mask=core,
            mobile_mask=mobile,
            mobility_score=mobility,
            robust_rmsf=rmsf,
            flexible_mask=segments > 0,
            peak_memory_mb=rss_mb(),
            notes="MDCoreAlign default per-frame iterative robust alignment.",
        )
    except Exception as exc:
        return MethodResult(
            status="failed",
            runtime_sec=time.perf_counter() - start,
            full_workflow_runtime_sec=time.perf_counter() - start,
            unified_rmsd="NA",
            peak_memory_mb=rss_mb(),
            notes=f"{type(exc).__name__}: {exc}",
        )


def run_long(coords: np.ndarray, *, mode: str, max_weight_fit_frames: int, chunk_size: int) -> MethodResult:
    """Run MDCoreAlign long or long-fast mode."""

    start = time.perf_counter()
    try:
        max_frames = max_weight_fit_frames if mode == "long" else min(max_weight_fit_frames, 500)
        result = solve_long_trajectory_alignment(
            coords,
            mode=mode,
            max_weight_fit_frames=max_frames,
            frame_selection="uniform",
            random_seed=42,
            chunk_size=chunk_size,
            store_aligned=False,
            rule="huber",
            max_iterations=20,
            tolerance=1e-6,
            objective_tolerance=1e-5,
            objective_patience=2,
            sequence_continuity_strength=0.05,
            contact_consistency_strength=0.0,
            record_history=False,
        )
        profile = build_long_flexibility_profile(result, min_segment_length=3, merge_gap=1)
        runtime = time.perf_counter() - start
        core = select_core_by_target_fraction(result.frozen_weights, 0.6).core_mask
        return MethodResult(
            status="success",
            runtime_sec=runtime,
            full_workflow_runtime_sec=runtime,
            unified_rmsd=result.mean_rmsd,
            weighted_rmsd=result.mean_weighted_rmsd,
            core_mask=core,
            mobile_mask=~core,
            mobility_score=profile.mobility_score,
            robust_rmsf=profile.robust_aligned_rmsf,
            flexible_mask=profile.flexible_segment_id > 0,
            peak_memory_mb=rss_mb(),
            notes=(
                f"MDCoreAlign {mode}; representative_frames={len(result.selected_frame_indices)}; "
                f"chunk_size={chunk_size}"
            ),
        )
    except Exception as exc:
        return MethodResult(
            status="failed",
            runtime_sec=time.perf_counter() - start,
            full_workflow_runtime_sec=time.perf_counter() - start,
            unified_rmsd="NA",
            peak_memory_mb=rss_mb(),
            notes=f"{type(exc).__name__}: {exc}",
        )


def run_mdlovofit_external(
    coords: np.ndarray,
    run_dir: Path,
    executable: Path,
    timeout: float,
    *,
    keep_pdbs: bool,
) -> MethodResult:
    """Run external MDLovoFit on temporary PDB trajectory input."""

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp_name:
            tmp = Path(tmp_name)
            input_pdb = tmp / "input_ca_multimodel.pdb"
            output_pdb = tmp / "mdlovofit_aligned.pdb"
            write_multimodel_pdb(coords, input_pdb)
            run = run_mdlovofit_on_files([input_pdb], fraction=0.6, output_path=output_pdb, executable=executable, timeout_seconds=timeout)
            aligned, _ = read_multimodel_pdb_with_occupancy(output_pdb, n_atoms=coords.shape[1])
            runtime = time.perf_counter() - start
            (run_dir / "stdout.txt").write_text(run.stdout)
            (run_dir / "stderr.txt").write_text(run.stderr)
            (run_dir / "command.txt").write_text(" ".join(run.command) + "\n")
            checksum = sha256_file(output_pdb)
            if keep_pdbs:
                shutil.copy2(input_pdb, run_dir / "input_ca_multimodel.pdb")
                shutil.copy2(output_pdb, run_dir / "mdlovofit_aligned.pdb")
            return MethodResult(
                status="success",
                runtime_sec=run.subprocess_runtime_seconds,
                full_workflow_runtime_sec=runtime,
                unified_rmsd=mean_rmsd(aligned, coords[0]),
                peak_memory_mb=rss_mb(children=True),
                notes=f"MDLovoFit external subprocess; output_sha256={checksum}",
            )
    except Exception as exc:
        return MethodResult(
            status="failed",
            runtime_sec=time.perf_counter() - start,
            full_workflow_runtime_sec=time.perf_counter() - start,
            unified_rmsd="NA",
            peak_memory_mb=rss_mb(children=True),
            notes=f"{type(exc).__name__}: {exc}",
        )


def run_theseus_external(
    coords: np.ndarray,
    run_dir: Path,
    executable: Path,
    timeout: float,
    *,
    keep_pdbs: bool,
) -> MethodResult:
    """Run external THESEUS on temporary PDB trajectory input."""

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp_name:
            tmp = Path(tmp_name)
            input_pdb = tmp / "input_ca_multimodel.pdb"
            write_multimodel_pdb(coords, input_pdb)
            command = [str(executable), "-a0", str(input_pdb.resolve())]
            env = dict(os.environ)
            libdir = executable.parent.parent / "lib"
            if libdir.exists():
                existing = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = f"{libdir}{os.pathsep}{existing}" if existing else str(libdir)
            completed = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout, check=False)
            runtime = time.perf_counter() - start
            (run_dir / "stdout.txt").write_text(completed.stdout)
            (run_dir / "stderr.txt").write_text(completed.stderr)
            (run_dir / "command.txt").write_text(" ".join(command) + "\n")
            if completed.returncode != 0:
                raise RuntimeError(f"THESEUS return code {completed.returncode}")
            aligned_path = theseus_output_path(tmp, input_pdb)
            if aligned_path is None:
                raise FileNotFoundError("No THESEUS *_sup.pdb output found")
            aligned = read_multimodel_pdb_coords(aligned_path, n_atoms=coords.shape[1])
            _ = parse_theseus_stdout(completed.stdout)
            variance_paths = sorted(tmp.glob("*_variances.txt"))
            if variance_paths:
                _ = parse_theseus_variances(variance_paths[0])
            checksum = sha256_file(aligned_path)
            if keep_pdbs:
                shutil.copy2(input_pdb, run_dir / "input_ca_multimodel.pdb")
                shutil.copy2(aligned_path, run_dir / "theseus_aligned.pdb")
            return MethodResult(
                status="success",
                runtime_sec=runtime,
                full_workflow_runtime_sec=runtime,
                unified_rmsd=mean_rmsd(aligned, coords[0]),
                peak_memory_mb=rss_mb(children=True),
                notes=f"THESEUS external subprocess; output_sha256={checksum}",
            )
    except Exception as exc:
        return MethodResult(
            status="failed",
            runtime_sec=time.perf_counter() - start,
            full_workflow_runtime_sec=time.perf_counter() - start,
            unified_rmsd="NA",
            peak_memory_mb=rss_mb(children=True),
            notes=f"{type(exc).__name__}: {exc}",
        )


def skipped(reason: str) -> MethodResult:
    """Return a skipped result."""

    return MethodResult(status="skipped", runtime_sec="NA", full_workflow_runtime_sec="NA", unified_rmsd="NA", notes=reason)


def median(values: list[float]) -> float | str:
    """Return median or NA."""

    return float(np.median(np.asarray(values, dtype=float))) if values else "NA"


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark and write all required outputs."""

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "aggregate").mkdir(exist_ok=True)

    summary = load_summary(args.all_domain_summary)
    selected = select_domain_panel(summary, args.n_domains)
    write_tsv(root / "selected_domains.tsv", selected, SELECTED_COLUMNS)
    print(f"[start] selected_domains={len(selected)} root={root}", flush=True)

    tool_rows = []
    for name, exe in [("MDLovoFit", args.mdlovofit_exe), ("THESEUS", args.theseus_exe)]:
        ok = exe.exists() and os.access(exe, os.X_OK)
        header_ok, header = check_command([str(exe), "-h"]) if ok else (False, "missing executable")
        tool_rows.append({"tool": name, "executable": str(exe), "available": int(ok and header_ok), "header": header})
    write_tsv(root / "external_tool_status.tsv", tool_rows)

    targets = split_ints(args.targets)
    repeats: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    default_long_rows: list[dict[str, Any]] = []
    long_mdlovo_rows: list[dict[str, Any]] = []
    rmsd_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []

    for domain in selected:
        domain_id = str(domain["domain_id"])
        print(f"[domain] {domain_id}: extracting R1-R5", flush=True)
        extract_replicates(domain, root)
        raw = load_raw_replicates(root / "domains" / domain_id)
        for target in targets:
            print(f"[case] {domain_id} target={target}: building transition-aligned trajectory", flush=True)
            case_dir = root / "runs" / domain_id / str(target)
            case_dir.mkdir(parents=True, exist_ok=True)
            coords, boundary_records, sequence = build_transition_aligned_fast(domain_id, raw, target)
            np.savez_compressed(
                case_dir / "transition_aligned_coords.npz",
                coords=coords.astype(np.float32, copy=False),
                replicate_sequence=np.asarray(sequence, dtype=object),
                labels=np.asarray(raw[sequence[0]][1], dtype=object),
                residue_metadata=np.asarray(raw[sequence[0]][2], dtype=object),
            )
            write_tsv(case_dir / "boundary_transform_log.tsv", boundary_records)
            boundary_rows.append(summarize_boundary(coords, boundary_records, domain_id, target))
            write_tsv(root / "boundary_smoothing_summary.tsv", boundary_rows, BOUNDARY_COLUMNS)

            default_result = skipped(f"default skipped above max_default_frames={args.max_default_frames}")
            if target <= args.max_default_frames and not args.build_only:
                print(f"[method] {domain_id} target={target}: MDCoreAlign default", flush=True)
                default_result = run_default(coords)

            long_result = skipped("build_only")
            long_fast_result = skipped("build_only")
            if not args.build_only:
                print(f"[method] {domain_id} target={target}: MDCoreAlign long", flush=True)
                long_result = run_long(coords, mode="long", max_weight_fit_frames=args.max_weight_fit_frames, chunk_size=args.chunk_size)
                print(f"[method] {domain_id} target={target}: MDCoreAlign long-fast", flush=True)
                long_fast_result = run_long(coords, mode="long-fast", max_weight_fit_frames=args.max_weight_fit_frames, chunk_size=args.chunk_size)

            frame_atoms = int(coords.shape[0] * coords.shape[1])
            mdlovo_result = skipped(f"skipped because frame_atoms={frame_atoms} exceeds {args.max_mdlovofit_frame_atoms}")
            if (
                not args.build_only
                and args.mdlovofit_exe.exists()
                and frame_atoms <= args.max_mdlovofit_frame_atoms
            ):
                print(f"[method] {domain_id} target={target}: MDLovoFit frame_atoms={frame_atoms}", flush=True)
                mdlovo_result = run_mdlovofit_external(
                    coords,
                    case_dir / "mdlovofit",
                    args.mdlovofit_exe,
                    args.timeout_seconds,
                    keep_pdbs=args.keep_external_pdbs,
                )

            theseus_result = skipped(
                f"THESEUS skipped above theseus limits: target={target}, frame_atoms={frame_atoms}"
            )
            if (
                not args.build_only
                and args.theseus_exe.exists()
                and target <= args.theseus_max_frames
                and frame_atoms <= args.max_theseus_frame_atoms
            ):
                print(f"[method] {domain_id} target={target}: THESEUS frame_atoms={frame_atoms}", flush=True)
                theseus_result = run_theseus_external(
                    coords,
                    case_dir / "theseus",
                    args.theseus_exe,
                    min(args.timeout_seconds, 600.0),
                    keep_pdbs=args.keep_external_pdbs,
                )

            results = {
                ("mdcorealign", "default"): default_result,
                ("mdcorealign", "long"): long_result,
                ("mdcorealign", "long-fast"): long_fast_result,
                ("mdlovofit", "external"): mdlovo_result,
                ("theseus", "external"): theseus_result,
            }
            for (method, mode), result in results.items():
                runtime_scope = "external_runtime" if method in {"mdlovofit", "theseus"} else "alignment_or_mode_runtime"
                repeats.append(
                    {
                        "domain_id": domain_id,
                        "target_frames": target,
                        "n_frames": int(coords.shape[0]),
                        "n_residues": int(coords.shape[1]),
                        "method": method,
                        "mode": mode,
                        "repeat_id": 1,
                        "runtime_scope": runtime_scope,
                        "runtime_sec": result.runtime_sec,
                        "status": result.status,
                        "notes": result.notes,
                    }
                )
                memory_rows.append(
                    {
                        "domain_id": domain_id,
                        "target_frames": target,
                        "method": method,
                        "mode": mode,
                        "peak_memory_mb": result.peak_memory_mb,
                        "runtime_sec": result.runtime_sec,
                        "status": result.status,
                        "notes": "ru_maxrss is process-level approximate peak memory.",
                    }
                )
            write_tsv(root / "aggregate" / "warmcache_repeats.tsv", repeats, REPEAT_COLUMNS)
            write_tsv(root / "memory_summary.tsv", memory_rows, MEMORY_COLUMNS)
            print(f"[done] {domain_id} target={target}", flush=True)

            default_rt = default_result.runtime_sec
            long_rt = long_result.runtime_sec
            long_fast_rt = long_fast_result.runtime_sec
            mdlovo_rt = mdlovo_result.runtime_sec
            theseus_rt = theseus_result.runtime_sec
            long_vs_mdlovo = (
                float(mdlovo_rt) / float(long_rt)
                if finite_float(mdlovo_rt) is not None and finite_float(long_rt) not in (None, 0.0)
                else "NA"
            )
            default_vs_mdlovo = (
                float(mdlovo_rt) / float(default_rt)
                if finite_float(mdlovo_rt) is not None and finite_float(default_rt) not in (None, 0.0)
                else "NA"
            )
            long_vs_default = (
                float(default_rt) / float(long_rt)
                if finite_float(default_rt) is not None and finite_float(long_rt) not in (None, 0.0)
                else "NA"
            )
            long_vs_theseus = (
                float(theseus_rt) / float(long_rt)
                if finite_float(theseus_rt) is not None and finite_float(long_rt) not in (None, 0.0)
                else "NA"
            )

            method_rows.append(
                {
                    "domain_id": domain_id,
                    "target_frames": target,
                    "n_frames": int(coords.shape[0]),
                    "n_residues": int(coords.shape[1]),
                    "mdcorealign_default_runtime": default_rt,
                    "mdcorealign_long_runtime": long_rt,
                    "mdcorealign_long_fast_runtime": long_fast_rt,
                    "mdlovofit_runtime": mdlovo_rt,
                    "theseus_runtime_if_available": theseus_rt,
                    "long_vs_mdlovofit_speedup": long_vs_mdlovo,
                    "default_vs_mdlovofit_speedup": default_vs_mdlovo,
                    "long_vs_default_speedup": long_vs_default,
                    "long_vs_theseus_speedup_if_available": long_vs_theseus,
                    "mdcorealign_long_faster_than_mdlovofit": int(
                        finite_float(long_rt) is not None
                        and finite_float(mdlovo_rt) is not None
                        and float(long_rt) < float(mdlovo_rt)
                    ),
                    "notes": "Transition-aligned synthetic long-trajectory scaling benchmark.",
                }
            )

            default_long_rows.append(
                {
                    "domain_id": domain_id,
                    "target_frames": target,
                    "n_frames": int(coords.shape[0]),
                    "n_residues": int(coords.shape[1]),
                    "default_runtime_sec": default_rt,
                    "long_runtime_sec": long_rt,
                    "long_fast_runtime_sec": long_fast_rt,
                    "long_vs_default_speedup": long_vs_default,
                    "default_unified_RMSD": default_result.unified_rmsd,
                    "long_unified_RMSD": long_result.unified_rmsd,
                    "long_fast_unified_RMSD": long_fast_result.unified_rmsd,
                    "core_jaccard_default_vs_long": jaccard(default_result.core_mask, long_result.core_mask),
                    "mobile_jaccard_default_vs_long": jaccard(default_result.mobile_mask, long_result.mobile_mask),
                    "mobility_spearman_default_vs_long": (
                        spearman_correlation(default_result.mobility_score, long_result.mobility_score)
                        if default_result.mobility_score is not None and long_result.mobility_score is not None
                        else "NA"
                    ),
                    "RMSF_spearman_default_vs_long": (
                        spearman_correlation(default_result.robust_rmsf, long_result.robust_rmsf)
                        if default_result.robust_rmsf is not None and long_result.robust_rmsf is not None
                        else "NA"
                    ),
                    "stable_core_fraction_diff": (
                        float(np.mean(long_result.core_mask) - np.mean(default_result.core_mask))
                        if default_result.core_mask is not None and long_result.core_mask is not None
                        else "NA"
                    ),
                    "flexible_segment_overlap": jaccard(default_result.flexible_mask, long_result.flexible_mask),
                    "status": "success" if default_result.status == "success" and long_result.status == "success" else "partial",
                    "notes": "Default comparison is available only for prefixes within max-default-frames.",
                }
            )

            long_mdlovo_rows.append(
                {
                    "domain_id": domain_id,
                    "target_frames": target,
                    "n_frames": int(coords.shape[0]),
                    "n_residues": int(coords.shape[1]),
                    "long_runtime_sec": long_rt,
                    "mdlovofit_runtime_sec": mdlovo_rt,
                    "long_vs_mdlovofit_speedup": long_vs_mdlovo,
                    "long_unified_RMSD": long_result.unified_rmsd,
                    "mdlovofit_unified_RMSD": mdlovo_result.unified_rmsd,
                    "long_lower_RMSD_than_mdlovofit": int(
                        finite_float(long_result.unified_rmsd) is not None
                        and finite_float(mdlovo_result.unified_rmsd) is not None
                        and float(long_result.unified_rmsd) < float(mdlovo_result.unified_rmsd)
                    ),
                    "mdlovofit_status": mdlovo_result.status,
                    "notes": mdlovo_result.notes,
                }
            )

            rmsd_rows.append(
                {
                    "domain_id": domain_id,
                    "target_frames": target,
                    "n_frames": int(coords.shape[0]),
                    "n_residues": int(coords.shape[1]),
                    "mdcorealign_default_RMSD": default_result.unified_rmsd,
                    "mdcorealign_long_RMSD": long_result.unified_rmsd,
                    "mdcorealign_long_fast_RMSD": long_fast_result.unified_rmsd,
                    "mdlovofit_RMSD": mdlovo_result.unified_rmsd,
                    "theseus_RMSD_if_available": theseus_result.unified_rmsd,
                    "long_lower_RMSD_than_mdlovofit": int(
                        finite_float(long_result.unified_rmsd) is not None
                        and finite_float(mdlovo_result.unified_rmsd) is not None
                        and float(long_result.unified_rmsd) < float(mdlovo_result.unified_rmsd)
                    ),
                    "long_RMSD_diff_vs_default": (
                        float(long_result.unified_rmsd) - float(default_result.unified_rmsd)
                        if finite_float(long_result.unified_rmsd) is not None
                        and finite_float(default_result.unified_rmsd) is not None
                        else "NA"
                    ),
                    "evaluation_status": "success" if long_result.status == "success" else "partial",
                    "notes": "Unified RMSD against first synthetic trajectory frame under shared CA selection.",
                }
            )
            write_tsv(root / "speedup_vs_target_frames.tsv", aggregate_speedup(method_rows, rmsd_rows), [
                "target_frames",
                "n_cases",
                "n_mdlovofit_success",
                "long_median_runtime",
                "long_fast_median_runtime",
                "mdlovofit_median_runtime",
                "long_vs_mdlovofit_median_speedup",
                "long_faster_than_mdlovofit_count",
                "long_faster_than_mdlovofit_percent",
                "long_lower_RMSD_than_mdlovofit_percent",
                "notes",
            ])
            write_tsv(root / "mdcorealign_default_vs_long_mode.tsv", default_long_rows, DEFAULT_LONG_COLUMNS)
            write_tsv(root / "long_mode_vs_mdlovofit.tsv", long_mdlovo_rows, LONG_MDLOVO_COLUMNS)
            write_tsv(root / "unified_rmsd_comparison.tsv", rmsd_rows, RMSD_COLUMNS)
            write_tsv(root / "boundary_smoothing_summary.tsv", boundary_rows, BOUNDARY_COLUMNS)
            write_tsv(root / "memory_summary.tsv", memory_rows, MEMORY_COLUMNS)

    scaling_rows = []
    for domain_id in sorted({row["domain_id"] for row in repeats}):
        domain_method_rows = [row for row in repeats if row["domain_id"] == domain_id]
        for method, mode in [
            ("mdcorealign", "default"),
            ("mdcorealign", "long"),
            ("mdcorealign", "long-fast"),
            ("mdlovofit", "external"),
            ("theseus", "external"),
        ]:
            fitted = fit_scaling(domain_method_rows, method, mode)
            scaling_rows.append(
                {
                    "domain_id": domain_id,
                    "method": method,
                    "mode": mode,
                    **fitted,
                    "notes": "Diagnostic fit across available target-frame points.",
                }
            )
    write_tsv(root / "scaling_model_summary.tsv", scaling_rows, SCALING_COLUMNS)
    write_report(root, selected, method_rows, default_long_rows, long_mdlovo_rows, boundary_rows, rmsd_rows, scaling_rows)


def aggregate_speedup(method_rows: list[dict[str, Any]], rmsd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate speedup by target frame count."""

    out = []
    for target in sorted({int(row["target_frames"]) for row in method_rows}):
        group = [row for row in method_rows if int(row["target_frames"]) == target]
        long_rt = [float(row["mdcorealign_long_runtime"]) for row in group if finite_float(row["mdcorealign_long_runtime"]) is not None]
        long_fast_rt = [float(row["mdcorealign_long_fast_runtime"]) for row in group if finite_float(row["mdcorealign_long_fast_runtime"]) is not None]
        mdlo_rt = [float(row["mdlovofit_runtime"]) for row in group if finite_float(row["mdlovofit_runtime"]) is not None]
        speed = [float(row["long_vs_mdlovofit_speedup"]) for row in group if finite_float(row["long_vs_mdlovofit_speedup"]) is not None]
        faster = [int(row["mdcorealign_long_faster_than_mdlovofit"]) for row in group if finite_float(row["mdlovofit_runtime"]) is not None]
        if rmsd_rows:
            rg = [row for row in rmsd_rows if int(row["target_frames"]) == target and finite_float(row["mdlovofit_RMSD"]) is not None]
            rmsd_percent = 100.0 * float(np.mean([int(row["long_lower_RMSD_than_mdlovofit"]) for row in rg])) if rg else "NA"
        else:
            rmsd_percent = "NA"
        out.append(
            {
                "target_frames": target,
                "n_cases": len(group),
                "n_mdlovofit_success": len(mdlo_rt),
                "long_median_runtime": median(long_rt),
                "long_fast_median_runtime": median(long_fast_rt),
                "mdlovofit_median_runtime": median(mdlo_rt),
                "long_vs_mdlovofit_median_speedup": median(speed),
                "long_faster_than_mdlovofit_count": int(sum(faster)) if faster else 0,
                "long_faster_than_mdlovofit_percent": 100.0 * float(np.mean(faster)) if faster else "NA",
                "long_lower_RMSD_than_mdlovofit_percent": rmsd_percent,
                "notes": "Only successful MDLovoFit rows contribute to speedup percentages.",
            }
        )
    return out


def write_report(
    root: Path,
    selected: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    default_long_rows: list[dict[str, Any]],
    long_mdlovo_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    rmsd_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
) -> None:
    """Write the requested benchmark report."""

    speed = aggregate_speedup(method_rows, rmsd_rows)
    mdlo_success = [row for row in long_mdlovo_rows if finite_float(row["mdlovofit_runtime_sec"]) is not None]
    mdlo_faster_count = sum(1 for row in mdlo_success if int(row["long_vs_mdlovofit_speedup"] != "NA" and float(row["long_vs_mdlovofit_speedup"]) > 1.0))
    rmsd_success = [row for row in rmsd_rows if finite_float(row["mdlovofit_RMSD"]) is not None and finite_float(row["mdcorealign_long_RMSD"]) is not None]
    rmsd_better = sum(int(row["long_lower_RMSD_than_mdlovofit"]) for row in rmsd_success)
    boundary_df = pd.DataFrame(boundary_rows)
    scaling_counts = pd.DataFrame(scaling_rows).groupby(["method", "mode", "scaling_label"]).size().to_dict() if scaling_rows else {}

    lines = [
        "# Transition-Aligned Synthetic Long-Trajectory Scaling Benchmark",
        "",
        "## Scope",
        "",
        "This benchmark uses transition-aligned synthetic long trajectories built from mdCATH 320 K R1-R5 replicates. It is not a real continuous MD benchmark and does not claim biological continuity.",
        "",
        "## Domain Panel",
        "",
        f"- selected domains: {len(selected)}",
        "- selection covered MDCoreAlign-faster cases, MDCoreAlign-slower cases, CATH class 4/6, longer domains and high-RMSD difficulty cases.",
        "",
        "## Boundary Smoothing",
        "",
        f"- median raw boundary jump: {pd.to_numeric(boundary_df['median_raw_boundary_jump'], errors='coerce').median():.4f}" if len(boundary_df) else "- median raw boundary jump: NA",
        f"- median post-transform boundary jump: {pd.to_numeric(boundary_df['median_post_boundary_jump'], errors='coerce').median():.4f}" if len(boundary_df) else "- median post-transform boundary jump: NA",
        f"- median post-boundary / within-adjacent ratio: {pd.to_numeric(boundary_df['median_post_boundary_to_within_ratio'], errors='coerce').median():.4f}" if len(boundary_df) else "- median post-boundary / within-adjacent ratio: NA",
        "",
        "## Long-Mode Versus MDLovoFit",
        "",
        f"- successful MDLovoFit comparisons: {len(mdlo_success)}",
        f"- MDCoreAlign long-mode faster than MDLovoFit in successful comparisons: {mdlo_faster_count}/{len(mdlo_success)}" if mdlo_success else "- MDCoreAlign long-mode faster than MDLovoFit: NA",
        f"- MDCoreAlign long-mode lower unified RMSD than MDLovoFit: {rmsd_better}/{len(rmsd_success)}" if rmsd_success else "- MDCoreAlign long-mode lower unified RMSD than MDLovoFit: NA",
        "",
        "## Speedup By Target Frames",
        "",
    ]
    for row in speed:
        lines.append(
            f"- {row['target_frames']} frames: n_cases={row['n_cases']}; "
            f"n_mdlovofit_success={row['n_mdlovofit_success']}; "
            f"median long-vs-MDLovoFit speedup={row['long_vs_mdlovofit_median_speedup']}"
        )
    lines.extend(
        [
            "",
            "## Default Versus Long Mode",
            "",
            "Default per-frame robust alignment is only run for prefixes within the configured maximum default frame count. Larger targets are marked as skipped rather than inferred.",
            "",
            "## THESEUS",
            "",
            "THESEUS is run only where target size is within configured frame/atom limits. Larger rows are skipped or timeout-limited so that THESEUS does not block the scaling benchmark.",
            "",
            "## Scaling Model Summary",
            "",
            f"Scaling-label counts: `{scaling_counts}`.",
            "",
            "## Recommended Wording",
            "",
            "Use: transition-aligned synthetic long-trajectory scaling benchmark. Do not call these inputs physically continuous MD trajectories. The benchmark supports computational scaling and long-mode performance interpretation only.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (root / "transition_aligned_long_mode_benchmark_report.md").write_text(report)
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "transition_aligned_long_mode_benchmark_report.md").write_text(report)


def main() -> None:
    """Entrypoint."""

    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    run_benchmark(args)
    print(args.root)


if __name__ == "__main__":
    main()
