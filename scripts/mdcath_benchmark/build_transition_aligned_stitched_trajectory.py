#!/usr/bin/env python
"""Build transition-aligned mdCATH replicate trajectories and benchmark tools.

This benchmark is a synthetic long-trajectory scaling diagnostic. It aligns
each appended replicate segment to the current terminal frame using one rigid
transform per segment. It is smoother than raw replicate concatenation, but it
is still not a physically continuous MD simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.alignment import apply_transform, solve_weighted_rigid_alignment
from mdsuperpose.flexibility import (
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
)
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_coords,
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.metrics import mean_rmsd
from mdsuperpose.theseus import parse_theseus_stdout, parse_theseus_variances
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory
from scripts.mdcath_benchmark.mdcath_common import write_json, write_tsv
from scripts.mdcath_benchmark.run_replicate_stitching_external_comparison import (
    check_command,
    percentile_iqr,
    sha256_file,
    theseus_output_path,
)


DEFAULT_DOMAINS = ("1j0tA00", "1vs5F01", "1e8rA00", "3hinA02", "1vq8L01")
REPLICATE_LABELS = ("A", "B", "C", "D", "E")
REPLICATE_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
SEED = 42
MDLOVOFIT_EXE = Path("/usr/local/bin/mdlovofit")
THESEUS_EXE = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")

SELECTED_COLUMNS = [
    "domain_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "n_residues",
    "replicate_labels_available",
    "selected_reason",
    "all_5_replicates_available",
    "cached_available",
    "notes",
]

REPLICATE_COLUMNS = [
    "domain_id",
    "replicate_label",
    "replicate_index",
    "n_frames",
    "n_residues",
    "residue_mapping_ok",
    "checksum",
    "extraction_status",
    "notes",
]

SEQUENCE_COLUMNS = [
    "domain_id",
    "sequence_name",
    "seed",
    "target_frames",
    "replicate_label_sequence",
    "n_segments",
    "expected_frames",
    "boundary_policy",
    "notes",
]

TRANSFORM_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "segment_index",
    "raw_replicate_label",
    "transformed_segment_label",
    "source_first_frame",
    "target_previous_last_frame",
    "rotation_matrix_flat",
    "translation_vector",
    "pre_alignment_boundary_rmsd",
    "post_alignment_boundary_rmsd",
    "determinant_rotation",
    "notes",
]

BOUNDARY_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "n_segments",
    "n_total_frames",
    "median_raw_boundary_jump",
    "median_post_boundary_jump",
    "max_post_boundary_jump",
    "median_within_adjacent_jump",
    "median_post_boundary_to_within_ratio",
    "max_post_boundary_to_within_ratio",
    "boundary_smoothing_success",
    "notes",
]

WARM_REPEAT_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "method",
    "repeat_id",
    "n_frames",
    "n_residues",
    "runtime_scope",
    "runtime_sec",
    "status",
    "notes",
]

WARM_SUMMARY_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "method",
    "runtime_scope",
    "n_repeats",
    "median_runtime_sec",
    "iqr_runtime_sec",
    "n_frames",
    "n_residues",
    "runtime_per_frame",
    "status",
    "notes",
]

RMSD_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "n_frames",
    "n_residues",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_theseus",
    "evaluation_status",
    "notes",
]

TOOL_COMPARISON_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "boundary_policy",
    "n_frames",
    "n_residues",
    "mdcorealign_median_runtime_sec",
    "mdlovofit_median_runtime_sec",
    "theseus_median_runtime_sec",
    "mdcorealign_vs_mdlovofit_speedup",
    "mdcorealign_vs_theseus_speedup",
    "mdcorealign_faster_than_mdlovofit",
    "mdcorealign_faster_than_theseus",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_theseus",
    "status",
    "notes",
]

SPEEDUP_COLUMNS = [
    "target_frames",
    "sequence_name",
    "boundary_policy",
    "n_cases",
    "mdcorealign_median_runtime",
    "mdlovofit_median_runtime",
    "theseus_median_runtime",
    "mdcorealign_vs_mdlovofit_median_speedup",
    "mdcorealign_faster_than_mdlovofit_count",
    "mdcorealign_faster_than_mdlovofit_percent",
    "mdcorealign_vs_theseus_median_speedup",
    "mdcorealign_faster_than_theseus_count",
    "mdcorealign_faster_than_theseus_percent",
    "mdcorealign_lower_RMSD_than_mdlovofit_percent",
    "mdcorealign_lower_RMSD_than_theseus_percent",
]

SCALING_COLUMNS = [
    "domain_id",
    "sequence_name",
    "boundary_policy",
    "method",
    "n_frame_points",
    "R2_linear",
    "R2_quadratic",
    "AIC_linear",
    "AIC_quadratic",
    "estimated_scaling_preference",
    "runtime_per_frame_trend",
    "notes",
]

OVERHEAD_COLUMNS = [
    "domain_id",
    "sequence_name",
    "target_frames",
    "method",
    "n_frames",
    "n_residues",
    "alignment_only_runtime_sec",
    "output_parse_runtime_sec",
    "RMSF_runtime_sec",
    "flexible_segment_runtime_sec",
    "profile_write_runtime_sec",
    "full_workflow_runtime_sec",
    "workflow_overhead_sec",
    "workflow_overhead_percent",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/mdcath_benchmark/transition_aligned_stitching"))
    parser.add_argument(
        "--all-domain-summary",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate/all_domain_canonical_per_domain_summary.tsv"),
    )
    parser.add_argument("--domain-ids", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--targets", default="2000,5000,10000")
    parser.add_argument("--sequence-names", default="canonical_cycle,seeded_random_42,block_motif_42")
    parser.add_argument(
        "--benchmark-sequence-names",
        default=None,
        help="Optional comma-separated sequence subset for external benchmarking. Defaults to --sequence-names.",
    )
    parser.add_argument(
        "--build-boundary-policies",
        default="keep_boundary_first_frame,drop_boundary_duplicate",
    )
    parser.add_argument(
        "--benchmark-boundary-policies",
        default="keep_boundary_first_frame",
        help="Comma-separated policies to benchmark. Both policies are still built by default.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--mdlovofit-exe", type=Path, default=MDLOVOFIT_EXE)
    parser.add_argument("--theseus-exe", type=Path, default=THESEUS_EXE)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    """Split a comma-separated CLI value."""

    return [item.strip() for item in str(value).split(",") if item.strip()]


def finite_float(value: object) -> float | None:
    """Parse a finite float value."""

    try:
        if value in ("", "NA", None, "not_run"):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def median(values: list[float]) -> float | str:
    """Return median or NA."""

    return float(np.median(np.asarray(values, dtype=float))) if values else "NA"


def h5_path_from_notes(notes: str) -> Path | None:
    """Extract cached HDF5 path from all-domain summary notes."""

    match = re.search(r"h5_file=([^;]+)", str(notes or ""))
    return Path(match.group(1)) if match else None


def coord_path(domain_id: str, replicate_index: int) -> str:
    """Return the mdCATH 320 K coordinate path."""

    return f"/{domain_id}/320/{replicate_index}/coords"


def load_summary(summary_path: Path) -> pd.DataFrame:
    """Load the canonical all-domain summary."""

    df = pd.read_csv(summary_path, sep="\t")
    for col in [
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "n_residues",
        "n_frames",
        "mdcorealign_runtime_sec",
        "mdlovofit_runtime_sec",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["h5_path"] = df["notes"].map(h5_path_from_notes)
    return df


def select_domains(df: pd.DataFrame, domain_ids: list[str], root: Path) -> list[dict[str, Any]]:
    """Create and write the selected domain panel."""

    rows = []
    for domain_id in domain_ids:
        match = df[df["domain_id"] == domain_id]
        if match.empty:
            raise ValueError(f"domain_id not found in all-domain summary: {domain_id}")
        row = match.iloc[0]
        h5_path = Path(row["h5_path"]) if row["h5_path"] is not None else Path("NA")
        if not h5_path.exists():
            raise FileNotFoundError(f"cached HDF5 missing for {domain_id}: {h5_path}")
        faster = bool(float(row["mdcorealign_runtime_sec"]) < float(row["mdlovofit_runtime_sec"]))
        if domain_id == "1vq8L01":
            selected_reason = "CATH class 4 domain and MDCoreAlign slower-than-MDLovoFit case"
        elif faster:
            selected_reason = "MDCoreAlign faster than MDLovoFit in all-domain benchmark"
        else:
            selected_reason = "MDCoreAlign slower than MDLovoFit in all-domain benchmark"
        rows.append(
            {
                "domain_id": domain_id,
                "cath_class": int(row["cath_class"]) if pd.notna(row["cath_class"]) else "NA",
                "cath_architecture": int(row["cath_architecture"]) if pd.notna(row["cath_architecture"]) else "NA",
                "cath_topology": int(row["cath_topology"]) if pd.notna(row["cath_topology"]) else "NA",
                "cath_homologous_superfamily": int(row["cath_homologous_superfamily"]) if pd.notna(row["cath_homologous_superfamily"]) else "NA",
                "n_residues": int(row["n_residues"]),
                "replicate_labels_available": "A,B,C,D,E",
                "selected_reason": selected_reason,
                "all_5_replicates_available": "yes",
                "cached_available": "yes",
                "notes": (
                    f"h5_file={h5_path}; mdcorealign_runtime={row['mdcorealign_runtime_sec']}; "
                    f"mdlovofit_runtime={row['mdlovofit_runtime_sec']}"
                ),
            }
        )
    write_tsv(root / "selected_domains.tsv", rows, SELECTED_COLUMNS)
    return rows


def load_raw_replicates(domain_root: Path) -> dict[str, tuple[np.ndarray, list[str], list[dict[str, str]]]]:
    """Load cached raw replicate arrays."""

    reps = {}
    for label in REPLICATE_LABELS:
        path = domain_root / "raw_replicates" / f"{label}_ca_coords.npz"
        payload = np.load(path, allow_pickle=True)
        coords = np.asarray(payload["coords"], dtype=float)
        labels = [str(item) for item in payload["labels"].tolist()]
        residue_metadata = payload["residue_metadata"].tolist() if "residue_metadata" in payload.files else []
        reps[label] = (coords, labels, residue_metadata)
    return reps


def extract_replicates(domain_row: dict[str, Any], summary_df: pd.DataFrame, root: Path) -> None:
    """Extract A-E raw CA replicate arrays for one domain."""

    domain_id = str(domain_row["domain_id"])
    domain_root = root / domain_id
    raw_root = domain_root / "raw_replicates"
    raw_root.mkdir(parents=True, exist_ok=True)
    h5_path = h5_path_from_notes(str(summary_df[summary_df["domain_id"] == domain_id].iloc[0]["notes"]))
    if h5_path is None:
        raise ValueError(f"missing h5 path for {domain_id}")

    rows = []
    reference_labels: list[str] | None = None
    for label in REPLICATE_LABELS:
        index = REPLICATE_INDEX[label]
        out_npz = raw_root / f"{label}_ca_coords.npz"
        out_json = raw_root / f"{label}_metadata.json"
        status = "success"
        notes = "NA"
        try:
            if out_npz.exists():
                payload = np.load(out_npz, allow_pickle=True)
                coords = np.asarray(payload["coords"], dtype=float)
                labels = [str(item) for item in payload["labels"].tolist()]
                residue_metadata = payload["residue_metadata"].tolist() if "residue_metadata" in payload.files else []
            else:
                coords, labels, residue_metadata, metadata = extract_ca_trajectory(
                    h5_path,
                    coordinate_path=coord_path(domain_id, index),
                    min_ca_atoms=30,
                    min_frames=20,
                )
                # Remove per-frame global translation only. This keeps internal
                # conformations intact while avoiding PDB coordinate overflow
                # from mdCATH absolute/PBC-imaged coordinates.
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
            if not np.isfinite(coords).all():
                raise ValueError("non-finite coordinates")
        except Exception as exc:
            coords = np.zeros((0, 0, 3), dtype=float)
            mapping_ok = False
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "domain_id": domain_id,
                "replicate_label": label,
                "replicate_index": index,
                "n_frames": int(coords.shape[0]),
                "n_residues": int(coords.shape[1]) if coords.ndim == 3 else 0,
                "residue_mapping_ok": int(mapping_ok),
                "checksum": sha256_file(out_npz) if out_npz.exists() else "NA",
                "extraction_status": status,
                "notes": notes if notes != "NA" else "per-frame CA centroid removal applied before caching",
            }
        )
    write_tsv(domain_root / "replicate_manifest.tsv", rows, REPLICATE_COLUMNS)
    if any(row["extraction_status"] != "success" or not row["residue_mapping_ok"] for row in rows):
        raise RuntimeError(f"replicate extraction/mapping failed for {domain_id}")


def sequence_labels(sequence_name: str, target_frames: int, raw_lengths: dict[str, int], boundary_policy: str) -> list[str]:
    """Create a deterministic replicate-label sequence."""

    labels: list[str] = []
    if sequence_name == "canonical_cycle":
        generator = (REPLICATE_LABELS[i % len(REPLICATE_LABELS)] for i in range(10_000))
    elif sequence_name == "seeded_random_42":
        rng = np.random.default_rng(SEED)
        random_labels = rng.choice(REPLICATE_LABELS, size=10_000, replace=True)
        generator = (str(label) for label in random_labels)
    elif sequence_name == "block_motif_42":
        motif = ("A", "A", "C", "C", "B", "E", "D", "D", "A")
        generator = (motif[i % len(motif)] for i in range(10_000))
    else:
        raise ValueError(f"unknown sequence_name: {sequence_name}")

    total = 0
    for label in generator:
        labels.append(label)
        if len(labels) == 1 or boundary_policy == "keep_boundary_first_frame":
            total += raw_lengths[label]
        else:
            total += max(0, raw_lengths[label] - 1)
        if total >= target_frames:
            break
    return labels


def rmsd_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return RMSD between two same-shape coordinate frames."""

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def adjacent_displacements(coords: np.ndarray) -> np.ndarray:
    """Compute adjacent-frame CA RMSD-like displacement without fitting."""

    diff = coords[1:] - coords[:-1]
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))


def build_transition_aligned(
    domain_id: str,
    raw: dict[str, tuple[np.ndarray, list[str], list[dict[str, str]]]],
    labels: list[str],
    target_frames: int,
    sequence_name: str,
    boundary_policy: str,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Build one transition-aligned stitched trajectory."""

    transform_rows = []
    boundary_records = []
    boundary_indices = []
    first_label = labels[0]
    long_coords = np.asarray(raw[first_label][0], dtype=float).copy()
    current_last = long_coords[-1].copy()
    initial_shift = np.mean(current_last, axis=0)
    long_coords = long_coords - initial_shift
    current_last = current_last - initial_shift

    for segment_index, label in enumerate(labels[1:], start=2):
        raw_coords = np.asarray(raw[label][0], dtype=float)
        source_first = raw_coords[0]
        pre = rmsd_between(source_first, current_last)
        solve = solve_weighted_rigid_alignment(source_first, current_last, np.ones(source_first.shape[0], dtype=float))
        transformed = apply_transform(raw_coords, solve)
        post = rmsd_between(transformed[0], current_last)
        appended = transformed if boundary_policy == "keep_boundary_first_frame" else transformed[1:]
        boundary_frame_index = long_coords.shape[0]
        boundary_indices.append(boundary_frame_index)
        actual_boundary_jump = rmsd_between(appended[0], current_last) if appended.shape[0] else float("nan")
        raw_within = adjacent_displacements(raw_coords)
        boundary_records.append(
            {
                "pre_boundary": pre,
                "post_boundary": actual_boundary_jump,
                "raw_within_median": float(np.median(raw_within)) if raw_within.size else float("nan"),
            }
        )
        long_coords = np.concatenate([long_coords, appended], axis=0)
        current_last = long_coords[-1].copy()
        # A global translation of all frames preserves internal geometry and
        # boundary jumps, while preventing synthetic repeated segments from
        # drifting outside fixed-width PDB coordinate fields.
        recenter_shift = np.mean(current_last, axis=0)
        long_coords = long_coords - recenter_shift
        current_last = current_last - recenter_shift
        transform_rows.append(
            {
                "domain_id": domain_id,
                "sequence_name": sequence_name,
                "target_frames": target_frames,
                "boundary_policy": boundary_policy,
                "segment_index": segment_index,
                "raw_replicate_label": label,
                "transformed_segment_label": f"{label}'_{segment_index}",
                "source_first_frame": "raw_X[0]",
                "target_previous_last_frame": "current_last",
                "rotation_matrix_flat": ",".join(f"{value:.10g}" for value in solve.rotation.reshape(-1)),
                "translation_vector": ",".join(f"{value:.10g}" for value in solve.translation),
                "pre_alignment_boundary_rmsd": pre,
                "post_alignment_boundary_rmsd": post,
                "determinant_rotation": float(np.linalg.det(solve.rotation)),
                "notes": "One rigid transform applied to the entire raw replicate segment; global recentering prevents coordinate drift.",
            }
        )
        if long_coords.shape[0] >= target_frames:
            break
    return long_coords[:target_frames], transform_rows, boundary_records, boundary_indices


def build_all_stitched(
    selected: list[dict[str, Any]],
    root: Path,
    targets: list[int],
    sequence_names: list[str],
    boundary_policies: list[str],
) -> None:
    """Build all transition-aligned stitched trajectories and diagnostics."""

    aggregate_boundary = []
    for row in selected:
        domain_id = str(row["domain_id"])
        domain_root = root / domain_id
        raw = load_raw_replicates(domain_root)
        raw_lengths = {label: raw[label][0].shape[0] for label in REPLICATE_LABELS}
        sequence_rows = []
        transform_rows_all = []
        stitched_root = domain_root / "stitched_long"
        stitched_root.mkdir(parents=True, exist_ok=True)
        for sequence_name in sequence_names:
            for target_frames in targets:
                for policy in boundary_policies:
                    labels = sequence_labels(sequence_name, target_frames, raw_lengths, policy)
                    coords, transform_rows, boundary_records, _boundary_indices = build_transition_aligned(
                        domain_id,
                        raw,
                        labels,
                        target_frames,
                        sequence_name,
                        policy,
                    )
                    out_npz = stitched_root / f"{sequence_name}_{target_frames}_{policy}.npz"
                    np.savez_compressed(
                        out_npz,
                        coords=coords.astype(np.float32, copy=False),
                        labels=np.asarray(raw[labels[0]][1], dtype=object),
                        residue_metadata=np.asarray(raw[labels[0]][2], dtype=object),
                        replicate_sequence=np.asarray(labels, dtype=object),
                    )
                    expected_frames = sum(
                        raw_lengths[label] if i == 0 or policy == "keep_boundary_first_frame" else max(0, raw_lengths[label] - 1)
                        for i, label in enumerate(labels)
                    )
                    sequence_rows.append(
                        {
                            "domain_id": domain_id,
                            "sequence_name": sequence_name,
                            "seed": SEED,
                            "target_frames": target_frames,
                            "replicate_label_sequence": ",".join(labels),
                            "n_segments": len(labels),
                            "expected_frames": expected_frames,
                            "boundary_policy": policy,
                            "notes": f"output_npz={out_npz}; final_frames={coords.shape[0]}",
                        }
                    )
                    transform_rows_all.extend(transform_rows)
                    raw_jumps = [item["pre_boundary"] for item in boundary_records]
                    post_jumps = [item["post_boundary"] for item in boundary_records]
                    within = adjacent_displacements(coords)
                    boundary_mask = np.ones(within.shape[0], dtype=bool)
                    # For this aggregate diagnostic, use all adjacent pairs as
                    # a conservative within-trajectory reference.
                    within_median = float(np.median(within[boundary_mask])) if within.size else float("nan")
                    ratios = [jump / within_median for jump in post_jumps if within_median and math.isfinite(within_median)]
                    aggregate_boundary.append(
                        {
                            "domain_id": domain_id,
                            "sequence_name": sequence_name,
                            "target_frames": target_frames,
                            "boundary_policy": policy,
                            "n_segments": len(labels),
                            "n_total_frames": int(coords.shape[0]),
                            "median_raw_boundary_jump": float(np.median(raw_jumps)) if raw_jumps else "NA",
                            "median_post_boundary_jump": float(np.median(post_jumps)) if post_jumps else "NA",
                            "max_post_boundary_jump": float(np.max(post_jumps)) if post_jumps else "NA",
                            "median_within_adjacent_jump": within_median,
                            "median_post_boundary_to_within_ratio": float(np.median(ratios)) if ratios else "NA",
                            "max_post_boundary_to_within_ratio": float(np.max(ratios)) if ratios else "NA",
                            "boundary_smoothing_success": int(bool(raw_jumps) and np.median(post_jumps) < np.median(raw_jumps)),
                            "notes": "Transition-aligned synthetic long trajectory; not physically continuous MD.",
                        }
                    )
        write_tsv(domain_root / "sequence_manifest.tsv", sequence_rows, SEQUENCE_COLUMNS)
        write_tsv(domain_root / "transform_log.tsv", transform_rows_all, TRANSFORM_COLUMNS)
    write_tsv(root / "aggregate" / "boundary_smoothing_summary.tsv", aggregate_boundary, BOUNDARY_COLUMNS)


def write_profile(path: Path, residue_metadata: list[dict[str, str]], rmsf: np.ndarray, weights: np.ndarray, segments: np.ndarray) -> None:
    """Write a minimal residue profile for MDCoreAlign full workflow timing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "atom_index",
                "chain",
                "residue_id",
                "resname",
                "robust_RMSF",
                "soft_stability_weight",
                "mobility_score",
                "flexible_segment_id",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(len(rmsf)):
            meta = residue_metadata[index] if index < len(residue_metadata) and isinstance(residue_metadata[index], dict) else {}
            weight = float(weights[index])
            writer.writerow(
                {
                    "atom_index": index,
                    "chain": meta.get("chain", "_"),
                    "residue_id": meta.get("residue_id", index + 1),
                    "resname": meta.get("resname", "UNK"),
                    "robust_RMSF": float(rmsf[index]),
                    "soft_stability_weight": weight,
                    "mobility_score": float(1.0 - weight),
                    "flexible_segment_id": int(segments[index]),
                }
            )


def run_mdcorealign(coords: np.ndarray, residue_metadata: list[dict[str, str]], outdir: Path) -> dict[str, Any]:
    """Run MDCoreAlign on cached coordinates and separate workflow timing."""

    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
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
    alignment_sec = time.perf_counter() - start

    rmsf_start = time.perf_counter()
    rmsf = compute_rmsf(aligned)
    rmsf_sec = time.perf_counter() - rmsf_start

    weights = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
    segment_start = time.perf_counter()
    chain = [item.get("chain", "_") if isinstance(item, dict) else "_" for item in residue_metadata]
    segments = detect_flexible_segments(1.0 - weights, percentile_cutoff=0.80, min_segment_length=3, merge_gap=1, chain=chain)
    segment_sec = time.perf_counter() - segment_start

    profile_start = time.perf_counter()
    write_profile(outdir / "residue_profile.tsv", residue_metadata, rmsf, weights, segments)
    profile_sec = time.perf_counter() - profile_start

    summary = {
        "status": "success",
        "alignment_only_runtime_sec": alignment_sec,
        "RMSF_runtime_sec": rmsf_sec,
        "flexible_segment_runtime_sec": segment_sec,
        "profile_write_runtime_sec": profile_sec,
        "full_workflow_runtime_sec": alignment_sec + rmsf_sec + segment_sec + profile_sec,
        "unified_RMSD": mean_rmsd(aligned, coords[0]),
        "failure_reason": "NA",
    }
    np.savez_compressed(outdir / "aligned_coords.npz", coords=aligned.astype(np.float32, copy=False))
    return summary


def run_mdlovofit(input_path: Path, coords: np.ndarray, outdir: Path, executable: Path, timeout: float) -> dict[str, Any]:
    """Run MDLovoFit on cached PDB input."""

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "trajectory_aligned.pdb"
    wall_start = time.perf_counter()
    try:
        run = run_mdlovofit_on_files([input_path], fraction=0.6, output_path=output_path, executable=executable, timeout_seconds=timeout)
        parse_start = time.perf_counter()
        aligned, _occupancy = read_multimodel_pdb_with_occupancy(output_path, n_atoms=coords.shape[1])
        parse_sec = time.perf_counter() - parse_start
        total_sec = time.perf_counter() - wall_start
        (outdir / "stdout.txt").write_text(run.stdout)
        (outdir / "stderr.txt").write_text(run.stderr)
        (outdir / "command.txt").write_text(" ".join(run.command) + "\n")
        return {
            "status": "success",
            "external_runtime_sec": run.subprocess_runtime_seconds,
            "parse_runtime_sec": parse_sec,
            "external_plus_parse_sec": total_sec,
            "unified_RMSD": mean_rmsd(aligned, coords[0]),
            "failure_reason": "NA",
            "command": " ".join(run.command),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
        }
    except Exception as exc:
        (outdir / "stderr.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        return {
            "status": "failed",
            "external_runtime_sec": "NA",
            "parse_runtime_sec": "NA",
            "external_plus_parse_sec": time.perf_counter() - wall_start,
            "unified_RMSD": "NA",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "command": f"{executable} {input_path}",
            "output_path": "NA",
            "output_sha256": "NA",
        }


def run_theseus(input_path: Path, coords: np.ndarray, outdir: Path, executable: Path, timeout: float) -> dict[str, Any]:
    """Run THESEUS on cached PDB input."""

    outdir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-a0", str(input_path.resolve())]
    env = dict(os.environ)
    candidate_lib = executable.parent.parent / "lib"
    if candidate_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{candidate_lib}{os.pathsep}{existing}" if existing else str(candidate_lib)
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, cwd=outdir, env=env, timeout=timeout, check=False)
        external_sec = time.perf_counter() - start
        (outdir / "stdout.txt").write_text(completed.stdout)
        (outdir / "stderr.txt").write_text(completed.stderr)
        (outdir / "command.txt").write_text(" ".join(command) + "\n")
        if completed.returncode != 0:
            raise RuntimeError(f"THESEUS return code {completed.returncode}")
        aligned_path = theseus_output_path(outdir, input_path)
        if aligned_path is None:
            raise FileNotFoundError("No THESEUS *_sup.pdb output found")
        parse_start = time.perf_counter()
        aligned = read_multimodel_pdb_coords(aligned_path, n_atoms=coords.shape[1])
        _parsed = parse_theseus_stdout(completed.stdout)
        variance_paths = sorted(outdir.glob("*_variances.txt"))
        if variance_paths:
            _ = parse_theseus_variances(variance_paths[0])
        parse_sec = time.perf_counter() - parse_start
        return {
            "status": "success",
            "external_runtime_sec": external_sec,
            "parse_runtime_sec": parse_sec,
            "external_plus_parse_sec": external_sec + parse_sec,
            "unified_RMSD": mean_rmsd(aligned, coords[0]),
            "failure_reason": "NA",
            "command": " ".join(command),
            "output_path": str(aligned_path),
            "output_sha256": sha256_file(aligned_path),
        }
    except Exception as exc:
        old = (outdir / "stderr.txt").read_text() if (outdir / "stderr.txt").exists() else ""
        (outdir / "stderr.txt").write_text(old + f"\n{type(exc).__name__}: {exc}\n")
        return {
            "status": "failed",
            "external_runtime_sec": time.perf_counter() - start,
            "parse_runtime_sec": "NA",
            "external_plus_parse_sec": time.perf_counter() - start,
            "unified_RMSD": "NA",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "command": " ".join(command),
            "output_path": "NA",
            "output_sha256": "NA",
        }


def export_input_pdb(root: Path, domain_id: str, sequence_name: str, target_frames: int, policy: str, coords: np.ndarray) -> Path:
    """Write or reuse a CA-only multi-model PDB input."""

    path = root / "external_inputs" / f"{domain_id}_{sequence_name}_{target_frames}_{policy}_ca_multimodel.pdb"
    if not path.exists():
        write_multimodel_pdb(coords, path)
    return path


def load_stitched_npz(path: Path) -> tuple[np.ndarray, list[str], list[dict[str, str]], list[str]]:
    """Load one stitched-long coordinate NPZ."""

    payload = np.load(path, allow_pickle=True)
    coords = np.asarray(payload["coords"], dtype=float)
    labels = [str(item) for item in payload["labels"].tolist()]
    residue_metadata = payload["residue_metadata"].tolist() if "residue_metadata" in payload.files else []
    seq = [str(item) for item in payload["replicate_sequence"].tolist()] if "replicate_sequence" in payload.files else []
    return coords, labels, residue_metadata, seq


def benchmark_inputs(
    selected: list[dict[str, Any]],
    root: Path,
    sequence_names: list[str],
    targets: list[int],
    boundary_policies: list[str],
    repeats: int,
    mdlovofit_exe: Path,
    theseus_exe: Path,
    timeout: float,
) -> None:
    """Run warm-cache repeats for all requested stitched trajectories."""

    warm_rows: list[dict[str, Any]] = []
    rmsd_records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    overhead_rows: list[dict[str, Any]] = []
    root.joinpath("runs").mkdir(parents=True, exist_ok=True)
    root.joinpath("external_inputs").mkdir(parents=True, exist_ok=True)

    for domain in selected:
        domain_id = str(domain["domain_id"])
        for sequence_name in sequence_names:
            for target in targets:
                for policy in boundary_policies:
                    npz = root / domain_id / "stitched_long" / f"{sequence_name}_{target}_{policy}.npz"
                    coords, _labels, residue_metadata, _seq = load_stitched_npz(npz)
                    input_pdb = export_input_pdb(root, domain_id, sequence_name, target, policy, coords)
                    key = (domain_id, sequence_name, target, policy)
                    rmsd_records[key] = {
                        "domain_id": domain_id,
                        "sequence_name": sequence_name,
                        "target_frames": target,
                        "boundary_policy": policy,
                        "n_frames": int(coords.shape[0]),
                        "n_residues": int(coords.shape[1]),
                        "mdcorealign_unified_RMSD": "NA",
                        "mdlovofit_unified_RMSD": "NA",
                        "theseus_unified_RMSD": "NA",
                        "mdcorealign_lower_RMSD_than_mdlovofit": "NA",
                        "mdcorealign_lower_RMSD_than_theseus": "NA",
                        "evaluation_status": "pending",
                        "notes": "Unified post-hoc RMSD against first frame under shared CA selection.",
                    }
                    for repeat in range(1, repeats + 1):
                        run_base = root / "runs" / domain_id / sequence_name / str(target) / policy / f"repeat_{repeat}"
                        common = {
                            "domain_id": domain_id,
                            "sequence_name": sequence_name,
                            "target_frames": target,
                            "boundary_policy": policy,
                            "repeat_id": repeat,
                            "n_frames": int(coords.shape[0]),
                            "n_residues": int(coords.shape[1]),
                        }
                        mdca_dir = run_base / "mdcorealign"
                        mdca_start = time.perf_counter()
                        try:
                            mdca = run_mdcorealign(coords, residue_metadata, mdca_dir)
                        except Exception as exc:
                            mdca = {
                                "status": "failed",
                                "alignment_only_runtime_sec": time.perf_counter() - mdca_start,
                                "RMSF_runtime_sec": "NA",
                                "flexible_segment_runtime_sec": "NA",
                                "profile_write_runtime_sec": "NA",
                                "full_workflow_runtime_sec": "NA",
                                "unified_RMSD": "NA",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                            }
                        write_json(mdca_dir / "runtime.json", mdca)
                        write_json(mdca_dir / "summary.json", mdca)
                        if rmsd_records[key]["mdcorealign_unified_RMSD"] == "NA" and mdca["status"] == "success":
                            rmsd_records[key]["mdcorealign_unified_RMSD"] = mdca["unified_RMSD"]
                        for scope, value in (
                            ("alignment_only", mdca["alignment_only_runtime_sec"]),
                            ("full_workflow", mdca["full_workflow_runtime_sec"]),
                        ):
                            warm_rows.append(
                                {
                                    **common,
                                    "method": "mdcorealign",
                                    "runtime_scope": scope,
                                    "runtime_sec": value,
                                    "status": mdca["status"],
                                    "notes": mdca["failure_reason"] if mdca["status"] != "success" else "cached coordinate timing",
                                }
                            )
                        overhead_rows.append(
                            {
                                "domain_id": domain_id,
                                "sequence_name": sequence_name,
                                "target_frames": target,
                                "method": "mdcorealign",
                                "n_frames": int(coords.shape[0]),
                                "n_residues": int(coords.shape[1]),
                                "alignment_only_runtime_sec": mdca["alignment_only_runtime_sec"],
                                "output_parse_runtime_sec": "NA",
                                "RMSF_runtime_sec": mdca["RMSF_runtime_sec"],
                                "flexible_segment_runtime_sec": mdca["flexible_segment_runtime_sec"],
                                "profile_write_runtime_sec": mdca["profile_write_runtime_sec"],
                                "full_workflow_runtime_sec": mdca["full_workflow_runtime_sec"],
                                "workflow_overhead_sec": (
                                    float(mdca["full_workflow_runtime_sec"]) - float(mdca["alignment_only_runtime_sec"])
                                    if finite_float(mdca["full_workflow_runtime_sec"]) is not None
                                    and finite_float(mdca["alignment_only_runtime_sec"]) is not None
                                    else "NA"
                                ),
                                "workflow_overhead_percent": (
                                    100.0
                                    * (float(mdca["full_workflow_runtime_sec"]) - float(mdca["alignment_only_runtime_sec"]))
                                    / float(mdca["alignment_only_runtime_sec"])
                                    if finite_float(mdca["full_workflow_runtime_sec"]) is not None
                                    and finite_float(mdca["alignment_only_runtime_sec"]) not in (None, 0.0)
                                    else "NA"
                                ),
                                "notes": "MDCoreAlign stable-core alignment plus RMSF/flexible-segment profile output.",
                            }
                        )

                        mdlo_dir = run_base / "mdlovofit"
                        mdlo = run_mdlovofit(input_pdb, coords, mdlo_dir, mdlovofit_exe, timeout)
                        write_json(mdlo_dir / "runtime.json", mdlo)
                        write_json(mdlo_dir / "summary.json", mdlo)
                        if rmsd_records[key]["mdlovofit_unified_RMSD"] == "NA" and mdlo["status"] == "success":
                            rmsd_records[key]["mdlovofit_unified_RMSD"] = mdlo["unified_RMSD"]
                        for scope, value in (
                            ("external_tool_runtime", mdlo["external_runtime_sec"]),
                            ("external_tool_plus_parse", mdlo["external_plus_parse_sec"]),
                        ):
                            warm_rows.append(
                                {
                                    **common,
                                    "method": "mdlovofit",
                                    "runtime_scope": scope,
                                    "runtime_sec": value,
                                    "status": mdlo["status"],
                                    "notes": mdlo["failure_reason"] if mdlo["status"] != "success" else "cached PDB subprocess timing",
                                }
                            )
                        overhead_rows.append(
                            {
                                "domain_id": domain_id,
                                "sequence_name": sequence_name,
                                "target_frames": target,
                                "method": "mdlovofit",
                                "n_frames": int(coords.shape[0]),
                                "n_residues": int(coords.shape[1]),
                                "alignment_only_runtime_sec": mdlo["external_runtime_sec"],
                                "output_parse_runtime_sec": mdlo["parse_runtime_sec"],
                                "RMSF_runtime_sec": "NA",
                                "flexible_segment_runtime_sec": "NA",
                                "profile_write_runtime_sec": "NA",
                                "full_workflow_runtime_sec": mdlo["external_plus_parse_sec"],
                                "workflow_overhead_sec": mdlo["parse_runtime_sec"],
                                "workflow_overhead_percent": (
                                    100.0 * float(mdlo["parse_runtime_sec"]) / float(mdlo["external_runtime_sec"])
                                    if finite_float(mdlo["parse_runtime_sec"]) is not None
                                    and finite_float(mdlo["external_runtime_sec"]) not in (None, 0.0)
                                    else "NA"
                                ),
                                "notes": "External runtime plus output parse for unified RMSD audit.",
                            }
                        )

                        th_dir = run_base / "theseus"
                        th = run_theseus(input_pdb, coords, th_dir, theseus_exe, timeout)
                        write_json(th_dir / "runtime.json", th)
                        write_json(th_dir / "summary.json", th)
                        if rmsd_records[key]["theseus_unified_RMSD"] == "NA" and th["status"] == "success":
                            rmsd_records[key]["theseus_unified_RMSD"] = th["unified_RMSD"]
                        for scope, value in (
                            ("external_tool_runtime", th["external_runtime_sec"]),
                            ("external_tool_plus_parse", th["external_plus_parse_sec"]),
                        ):
                            warm_rows.append(
                                {
                                    **common,
                                    "method": "theseus",
                                    "runtime_scope": scope,
                                    "runtime_sec": value,
                                    "status": th["status"],
                                    "notes": th["failure_reason"] if th["status"] != "success" else "cached PDB subprocess timing",
                                }
                            )
                        overhead_rows.append(
                            {
                                "domain_id": domain_id,
                                "sequence_name": sequence_name,
                                "target_frames": target,
                                "method": "theseus",
                                "n_frames": int(coords.shape[0]),
                                "n_residues": int(coords.shape[1]),
                                "alignment_only_runtime_sec": th["external_runtime_sec"],
                                "output_parse_runtime_sec": th["parse_runtime_sec"],
                                "RMSF_runtime_sec": "NA",
                                "flexible_segment_runtime_sec": "NA",
                                "profile_write_runtime_sec": "NA",
                                "full_workflow_runtime_sec": th["external_plus_parse_sec"],
                                "workflow_overhead_sec": th["parse_runtime_sec"],
                                "workflow_overhead_percent": (
                                    100.0 * float(th["parse_runtime_sec"]) / float(th["external_runtime_sec"])
                                    if finite_float(th["parse_runtime_sec"]) is not None
                                    and finite_float(th["external_runtime_sec"]) not in (None, 0.0)
                                    else "NA"
                                ),
                                "notes": "External runtime plus output parse for unified RMSD audit.",
                            }
                        )
                        write_tsv(root / "aggregate" / "warmcache_repeats.tsv", warm_rows, WARM_REPEAT_COLUMNS)
    write_tsv(root / "aggregate" / "warmcache_repeats.tsv", warm_rows, WARM_REPEAT_COLUMNS)
    write_tsv(root / "aggregate" / "alignment_only_vs_full_workflow_timing.tsv", overhead_rows, OVERHEAD_COLUMNS)
    finalize_aggregates(root, rmsd_records)


def summarize_warmcache(root: Path) -> pd.DataFrame:
    """Summarize warm-cache repeat table."""

    repeats = pd.read_csv(root / "aggregate" / "warmcache_repeats.tsv", sep="\t")
    rows = []
    for key, group in repeats.groupby(["domain_id", "sequence_name", "target_frames", "boundary_policy", "method", "runtime_scope"], sort=False):
        values = pd.to_numeric(group[group["status"] == "success"]["runtime_sec"], errors="coerce").dropna().to_list()
        first = group.iloc[0]
        rows.append(
            {
                "domain_id": key[0],
                "sequence_name": key[1],
                "target_frames": key[2],
                "boundary_policy": key[3],
                "method": key[4],
                "runtime_scope": key[5],
                "n_repeats": len(values),
                "median_runtime_sec": median(values),
                "iqr_runtime_sec": percentile_iqr(values),
                "n_frames": first["n_frames"],
                "n_residues": first["n_residues"],
                "runtime_per_frame": float(np.median(values) / first["n_frames"]) if values else "NA",
                "status": "success" if values else "failed",
                "notes": "Median over warm-cache repeats.",
            }
        )
    out = pd.DataFrame(rows, columns=WARM_SUMMARY_COLUMNS)
    out.to_csv(root / "aggregate" / "warmcache_summary.tsv", sep="\t", index=False)
    return out


def finalize_aggregates(root: Path, rmsd_records: dict[tuple[str, str, int, str], dict[str, Any]]) -> None:
    """Create RMSD, speedup, scaling and report outputs."""

    warm = summarize_warmcache(root)
    lookup = {
        (row.domain_id, row.sequence_name, int(row.target_frames), row.boundary_policy, row.method, row.runtime_scope): row
        for row in warm.itertuples(index=False)
    }

    rmsd_rows = []
    comparison_rows = []
    for key, row in rmsd_records.items():
        domain_id, sequence_name, target, policy = key
        mdca = lookup.get((domain_id, sequence_name, target, policy, "mdcorealign", "alignment_only"))
        mdlo = lookup.get((domain_id, sequence_name, target, policy, "mdlovofit", "external_tool_runtime"))
        th = lookup.get((domain_id, sequence_name, target, policy, "theseus", "external_tool_runtime"))
        mdca_rt = finite_float(mdca.median_runtime_sec if mdca is not None else None)
        mdlo_rt = finite_float(mdlo.median_runtime_sec if mdlo is not None else None)
        th_rt = finite_float(th.median_runtime_sec if th is not None else None)
        mdca_r = finite_float(row["mdcorealign_unified_RMSD"])
        mdlo_r = finite_float(row["mdlovofit_unified_RMSD"])
        th_r = finite_float(row["theseus_unified_RMSD"])
        row["mdcorealign_lower_RMSD_than_mdlovofit"] = int(mdca_r is not None and mdlo_r is not None and mdca_r < mdlo_r)
        row["mdcorealign_lower_RMSD_than_theseus"] = int(mdca_r is not None and th_r is not None and mdca_r < th_r)
        row["evaluation_status"] = "success" if mdca_r is not None and mdlo_r is not None and th_r is not None else "partial"
        rmsd_rows.append(row)
        comparison_rows.append(
            {
                "domain_id": domain_id,
                "sequence_name": sequence_name,
                "target_frames": target,
                "boundary_policy": policy,
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "mdcorealign_median_runtime_sec": mdca_rt if mdca_rt is not None else "NA",
                "mdlovofit_median_runtime_sec": mdlo_rt if mdlo_rt is not None else "NA",
                "theseus_median_runtime_sec": th_rt if th_rt is not None else "NA",
                "mdcorealign_vs_mdlovofit_speedup": float(mdlo_rt / mdca_rt) if mdlo_rt is not None and mdca_rt not in (None, 0.0) else "NA",
                "mdcorealign_vs_theseus_speedup": float(th_rt / mdca_rt) if th_rt is not None and mdca_rt not in (None, 0.0) else "NA",
                "mdcorealign_faster_than_mdlovofit": int(mdca_rt is not None and mdlo_rt is not None and mdca_rt < mdlo_rt),
                "mdcorealign_faster_than_theseus": int(mdca_rt is not None and th_rt is not None and mdca_rt < th_rt),
                "mdcorealign_unified_RMSD": row["mdcorealign_unified_RMSD"],
                "mdlovofit_unified_RMSD": row["mdlovofit_unified_RMSD"],
                "theseus_unified_RMSD": row["theseus_unified_RMSD"],
                "mdcorealign_lower_RMSD_than_mdlovofit": row["mdcorealign_lower_RMSD_than_mdlovofit"],
                "mdcorealign_lower_RMSD_than_theseus": row["mdcorealign_lower_RMSD_than_theseus"],
                "status": row["evaluation_status"],
                "notes": "Transition-aligned synthetic long trajectory; not physically continuous MD.",
            }
        )

    rmsd = pd.DataFrame(rmsd_rows, columns=RMSD_COLUMNS)
    comparison = pd.DataFrame(comparison_rows, columns=TOOL_COMPARISON_COLUMNS)
    rmsd.to_csv(root / "aggregate" / "unified_rmsd_comparison.tsv", sep="\t", index=False)
    comparison.to_csv(root / "aggregate" / "long_transition_aligned_tool_comparison.tsv", sep="\t", index=False)
    create_speedup_summary(root, comparison)
    create_scaling_summary(root, comparison)
    create_smoothed_vs_raw(root, comparison)
    create_figures(root)
    write_report(root)


def create_speedup_summary(root: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create speedup versus target-frame table."""

    rows = []
    for key, group in comparison.groupby(["target_frames", "sequence_name", "boundary_policy"], sort=False):
        n = len(group)
        rows.append(
            {
                "target_frames": key[0],
                "sequence_name": key[1],
                "boundary_policy": key[2],
                "n_cases": n,
                "mdcorealign_median_runtime": median(pd.to_numeric(group["mdcorealign_median_runtime_sec"], errors="coerce").dropna().to_list()),
                "mdlovofit_median_runtime": median(pd.to_numeric(group["mdlovofit_median_runtime_sec"], errors="coerce").dropna().to_list()),
                "theseus_median_runtime": median(pd.to_numeric(group["theseus_median_runtime_sec"], errors="coerce").dropna().to_list()),
                "mdcorealign_vs_mdlovofit_median_speedup": median(pd.to_numeric(group["mdcorealign_vs_mdlovofit_speedup"], errors="coerce").dropna().to_list()),
                "mdcorealign_faster_than_mdlovofit_count": int(pd.to_numeric(group["mdcorealign_faster_than_mdlovofit"], errors="coerce").fillna(0).sum()),
                "mdcorealign_faster_than_mdlovofit_percent": 100.0 * pd.to_numeric(group["mdcorealign_faster_than_mdlovofit"], errors="coerce").fillna(0).mean(),
                "mdcorealign_vs_theseus_median_speedup": median(pd.to_numeric(group["mdcorealign_vs_theseus_speedup"], errors="coerce").dropna().to_list()),
                "mdcorealign_faster_than_theseus_count": int(pd.to_numeric(group["mdcorealign_faster_than_theseus"], errors="coerce").fillna(0).sum()),
                "mdcorealign_faster_than_theseus_percent": 100.0 * pd.to_numeric(group["mdcorealign_faster_than_theseus"], errors="coerce").fillna(0).mean(),
                "mdcorealign_lower_RMSD_than_mdlovofit_percent": 100.0 * pd.to_numeric(group["mdcorealign_lower_RMSD_than_mdlovofit"], errors="coerce").fillna(0).mean(),
                "mdcorealign_lower_RMSD_than_theseus_percent": 100.0 * pd.to_numeric(group["mdcorealign_lower_RMSD_than_theseus"], errors="coerce").fillna(0).mean(),
            }
        )
    out = pd.DataFrame(rows, columns=SPEEDUP_COLUMNS)
    out.to_csv(root / "aggregate" / "speedup_vs_target_frames.tsv", sep="\t", index=False)
    return out


def fit_scaling(group: pd.DataFrame, runtime_col: str) -> dict[str, Any]:
    """Fit linear and quadratic runtime models."""

    ok = group.copy()
    ok["x"] = pd.to_numeric(ok["target_frames"], errors="coerce")
    ok["y"] = pd.to_numeric(ok[runtime_col], errors="coerce")
    ok = ok[ok["x"].notna() & ok["y"].notna()]
    if len(ok) < 3:
        return {
            "n_frame_points": len(ok),
            "R2_linear": "NA",
            "R2_quadratic": "NA",
            "AIC_linear": "NA",
            "AIC_quadratic": "NA",
            "estimated_scaling_preference": "unclear",
            "runtime_per_frame_trend": "NA",
        }
    x = ok["x"].to_numpy(float)
    y = ok["y"].to_numpy(float)

    def fit(design: np.ndarray) -> tuple[float, float]:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 1.0
        aic = len(y) * math.log(max(rss / len(y), 1e-12)) + 2 * design.shape[1]
        return r2, aic

    r2_lin, aic_lin = fit(np.column_stack([np.ones_like(x), x]))
    r2_quad, aic_quad = fit(np.column_stack([np.ones_like(x), x, x * x]))
    per_frame = y / x
    stability = float((np.max(per_frame) - np.min(per_frame)) / np.median(per_frame)) if np.median(per_frame) else float("nan")
    preference = "linear"
    if aic_quad + 2.0 < aic_lin and stability > 0.5:
        preference = "possible_quadratic"
    elif aic_quad + 2.0 < aic_lin and stability > 0.25:
        preference = "possible_superlinear"
    elif abs(aic_quad - aic_lin) < 2.0:
        preference = "unclear"
    return {
        "n_frame_points": len(ok),
        "R2_linear": r2_lin,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_lin,
        "AIC_quadratic": aic_quad,
        "estimated_scaling_preference": preference,
        "runtime_per_frame_trend": stability,
    }


def create_scaling_summary(root: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create scaling model summary."""

    specs = [
        ("mdcorealign", "mdcorealign_median_runtime_sec"),
        ("mdlovofit", "mdlovofit_median_runtime_sec"),
        ("theseus", "theseus_median_runtime_sec"),
    ]
    rows = []
    for (domain_id, sequence_name, policy), group in comparison.groupby(["domain_id", "sequence_name", "boundary_policy"], sort=False):
        for method, col in specs:
            fitted = fit_scaling(group, col)
            rows.append(
                {
                    "domain_id": domain_id,
                    "sequence_name": sequence_name,
                    "boundary_policy": policy,
                    "method": method,
                    **fitted,
                    "notes": "Three target-frame points; diagnostic scaling fit only.",
                }
            )
    out = pd.DataFrame(rows, columns=SCALING_COLUMNS)
    out.to_csv(root / "aggregate" / "scaling_model_summary.tsv", sep="\t", index=False)
    return out


def create_smoothed_vs_raw(root: Path, comparison: pd.DataFrame) -> None:
    """Create optional smoothed-vs-raw comparison if raw results overlap."""

    raw_path = PROJECT_ROOT / "results/mdcath_benchmark/long_trajectory_stitching/aggregate/long_trajectory_tool_comparison.tsv"
    rows = []
    boundary = pd.read_csv(root / "aggregate" / "boundary_smoothing_summary.tsv", sep="\t")
    if raw_path.exists():
        raw = pd.read_csv(raw_path, sep="\t")
    else:
        raw = pd.DataFrame()
    for _, row in comparison.iterrows():
        bmatch = boundary[
            (boundary["domain_id"] == row["domain_id"])
            & (boundary["sequence_name"] == row["sequence_name"])
            & (boundary["target_frames"] == row["target_frames"])
            & (boundary["boundary_policy"] == row["boundary_policy"])
        ]
        smoothed_ratio = bmatch.iloc[0]["median_post_boundary_to_within_ratio"] if len(bmatch) else "NA"
        raw_match = raw[(raw.get("domain_id", pd.Series(dtype=str)) == row["domain_id"]) & (raw.get("n_frames", pd.Series(dtype=float)).astype(float).sub(float(row["n_frames"])).abs() < 100)]
        for method, runtime_col, rmsd_col in (
            ("mdcorealign", "mdcorealign_median_runtime_sec", "mdcorealign_unified_RMSD"),
            ("mdlovofit", "mdlovofit_median_runtime_sec", "mdlovofit_unified_RMSD"),
            ("theseus", "theseus_median_runtime_sec", "theseus_unified_RMSD"),
        ):
            raw_runtime = "NA"
            raw_rmsd = "NA"
            raw_boundary = "NA"
            if len(raw_match):
                raw_row = raw_match.iloc[0]
                raw_runtime = raw_row.get(f"{method}_runtime_sec", raw_row.get(f"{method}_median_runtime_sec", "NA"))
                raw_rmsd = raw_row.get(f"{method}_unified_RMSD", "NA")
                raw_boundary = raw_row.get("boundary_to_within_jump_ratio", "NA")
            rows.append(
                {
                    "domain_id": row["domain_id"],
                    "target_frames": row["target_frames"],
                    "method": method,
                    "raw_boundary_jump_ratio": raw_boundary,
                    "smoothed_boundary_jump_ratio": smoothed_ratio,
                    "raw_runtime_sec": raw_runtime,
                    "smoothed_runtime_sec": row[runtime_col],
                    "raw_unified_RMSD": raw_rmsd,
                    "smoothed_unified_RMSD": row[rmsd_col],
                    "interpretation": "Raw comparison is approximate when only nearest previous raw-stitched prefix exists.",
                }
            )
    pd.DataFrame(rows).to_csv(root / "aggregate" / "smoothed_vs_raw_stitching_comparison.tsv", sep="\t", index=False)


def create_figures(root: Path) -> None:
    """Create supervisor-facing diagnostic figure."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    boundary = pd.read_csv(root / "aggregate" / "boundary_smoothing_summary.tsv", sep="\t")
    comparison = pd.read_csv(root / "aggregate" / "long_transition_aligned_tool_comparison.tsv", sep="\t")
    scaling = pd.read_csv(root / "aggregate" / "scaling_model_summary.tsv", sep="\t")
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.ravel()

    axes[0].scatter(pd.to_numeric(boundary["median_raw_boundary_jump"], errors="coerce"), pd.to_numeric(boundary["median_post_boundary_jump"], errors="coerce"), alpha=0.7)
    axes[0].set_xlabel("Raw boundary jump RMSD")
    axes[0].set_ylabel("Post-transform boundary jump RMSD")
    axes[0].set_title("A. Boundary smoothing")

    for method, col in (
        ("MDCoreAlign", "mdcorealign_median_runtime_sec"),
        ("MDLovoFit", "mdlovofit_median_runtime_sec"),
        ("THESEUS", "theseus_median_runtime_sec"),
    ):
        grouped = comparison.groupby("target_frames")[col].median(numeric_only=True)
        axes[1].plot(grouped.index, grouped.values, marker="o", label=method)
        axes[2].plot(grouped.index, grouped.values / grouped.index.astype(float), marker="o", label=method)
    axes[1].set_title("B. Runtime vs frames")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].legend()
    axes[2].set_title("C. Runtime per frame")
    axes[2].set_ylabel("s/frame")

    for col, title, ax in (
        ("mdcorealign_vs_mdlovofit_speedup", "D. MDCoreAlign vs MDLovoFit speedup", axes[3]),
        ("mdcorealign_vs_theseus_speedup", "E. MDCoreAlign vs THESEUS speedup", axes[4]),
    ):
        grouped = comparison.groupby("target_frames")[col].median(numeric_only=True)
        ax.plot(grouped.index, grouped.values, marker="o")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Speedup")

    rmsd_melt = comparison.melt(
        id_vars=["target_frames"],
        value_vars=["mdcorealign_unified_RMSD", "mdlovofit_unified_RMSD", "theseus_unified_RMSD"],
        var_name="method",
        value_name="rmsd",
    )
    labels = []
    data = []
    for method, group in rmsd_melt.groupby("method"):
        labels.append(method.replace("_unified_RMSD", ""))
        data.append(pd.to_numeric(group["rmsd"], errors="coerce").dropna().to_numpy())
    axes[5].boxplot(data, labels=labels, showfliers=False)
    axes[5].set_title("F. Unified RMSD")
    axes[5].tick_params(axis="x", rotation=20)

    counts = scaling.groupby(["method", "estimated_scaling_preference"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[6])
    axes[6].set_title("G. Scaling model labels")
    axes[6].set_ylabel("Count")

    axes[7].axis("off")
    axes[7].text(0.0, 0.8, "Transition-aligned stitching is a synthetic scaling diagnostic, not physically continuous MD.", wrap=True)
    fig.tight_layout()
    fig.savefig(figdir / "transition_aligned_long_trajectory_benchmark.pdf")
    fig.savefig(figdir / "transition_aligned_long_trajectory_benchmark.png", dpi=200)
    comparison.to_csv(root / "figures" / "figure_data_tool_comparison.tsv", sep="\t", index=False)
    boundary.to_csv(root / "figures" / "figure_data_boundary_smoothing.tsv", sep="\t", index=False)


def write_report(root: Path) -> None:
    """Write the supervisor report."""

    selected = pd.read_csv(root / "selected_domains.tsv", sep="\t")
    boundary = pd.read_csv(root / "aggregate" / "boundary_smoothing_summary.tsv", sep="\t")
    comparison = pd.read_csv(root / "aggregate" / "long_transition_aligned_tool_comparison.tsv", sep="\t")
    speed = pd.read_csv(root / "aggregate" / "speedup_vs_target_frames.tsv", sep="\t")
    scaling = pd.read_csv(root / "aggregate" / "scaling_model_summary.tsv", sep="\t")

    mdlo_faster = int(pd.to_numeric(comparison["mdcorealign_faster_than_mdlovofit"], errors="coerce").fillna(0).sum())
    th_faster = int(pd.to_numeric(comparison["mdcorealign_faster_than_theseus"], errors="coerce").fillna(0).sum())
    total = len(comparison)
    mdlo_rmsd = int(pd.to_numeric(comparison["mdcorealign_lower_RMSD_than_mdlovofit"], errors="coerce").fillna(0).sum())
    th_rmsd = int(pd.to_numeric(comparison["mdcorealign_lower_RMSD_than_theseus"], errors="coerce").fillna(0).sum())
    speed_lines = []
    for _, row in speed.iterrows():
        speed_lines.append(
            f"- {row['sequence_name']} / {row['boundary_policy']} / {int(row['target_frames'])} frames: "
            f"MDCoreAlign-vs-MDLovoFit speedup={row['mdcorealign_vs_mdlovofit_median_speedup']}; "
            f"MDCoreAlign-vs-THESEUS speedup={row['mdcorealign_vs_theseus_median_speedup']}"
        )
    scaling_counts = scaling.groupby(["method", "estimated_scaling_preference"]).size()
    report = f"""# Transition-Aligned Stitching Supervisor Report

## Why This Test Was Done

The previous raw stitching benchmark concatenated independent mdCATH replicates directly. That created large artificial boundary jumps. This test applies one rigid transform to each appended replicate segment so that the segment's first frame is aligned to the current terminal frame before appending.

## Important Caveat

These are synthetic long trajectories for scaling diagnostics. They are smoother than raw stitched replicates, but they are not physically continuous MD simulations. A repeated sequence such as `A, A, A` means `A, A', A''`, where each appended copy is freshly transformed from the raw replicate to the current terminal frame.

## Domains Selected

{selected[['domain_id', 'cath_class', 'n_residues', 'selected_reason']].to_markdown(index=False)}

## Sequence Generation

Fixed seed: 42. Sequence types: canonical cycle, seeded random, and block motif. Targets: 2,000, 5,000 and 10,000 frames. Primary benchmark policy: rows present in the tool-comparison table. Both boundary policies were built for boundary diagnostics when generated.

## Boundary Smoothing Results

- Median raw boundary jump across constructed inputs: {pd.to_numeric(boundary['median_raw_boundary_jump'], errors='coerce').median():.4f}
- Median post-transform boundary jump: {pd.to_numeric(boundary['median_post_boundary_jump'], errors='coerce').median():.4f}
- Median post-boundary / within-adjacent ratio: {pd.to_numeric(boundary['median_post_boundary_to_within_ratio'], errors='coerce').median():.4f}

## Runtime Comparison

- MDCoreAlign faster than MDLovoFit: {mdlo_faster}/{total}
- MDCoreAlign faster than THESEUS: {th_faster}/{total}

## Speedup Versus Frame Count

{chr(10).join(speed_lines)}

## RMSD Audit

- MDCoreAlign lower unified RMSD than MDLovoFit: {mdlo_rmsd}/{total}
- MDCoreAlign lower unified RMSD than THESEUS: {th_rmsd}/{total}

Unified RMSD is a geometric audit under a common CA reference convention and should not be interpreted as the only quality metric.

## Scaling Model Results

Scaling labels: {dict(scaling_counts)}. These three-point fits are diagnostic, not formal complexity proofs.

## Recommendation

Use this as an internal scaling stress test or supplementary diagnostic only. It is useful for assessing whether smoother synthetic long inputs change the performance conclusion, but it should not be presented as evidence from physically continuous long MD trajectories.
"""
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "transition_aligned_stitching_supervisor_report.md").write_text(report)
    (root / "transition_aligned_stitching_supervisor_report.md").write_text(report)


def check_tools(root: Path, mdlovofit_exe: Path, theseus_exe: Path) -> None:
    """Record external tool availability."""

    rows = []
    for name, exe, commands in (
        ("MDLovoFit", mdlovofit_exe, [[str(mdlovofit_exe), "-h"], [str(mdlovofit_exe), "--help"]]),
        ("THESEUS", theseus_exe, [[str(theseus_exe), "-h"], [str(theseus_exe), "--version"]]),
    ):
        exists = exe.exists() and os.access(exe, os.X_OK)
        ok = False
        header = "NA"
        if exists:
            for command in commands:
                ok, header = check_command(command)
                if ok:
                    break
        rows.append({"tool": name, "executable": exe, "available": int(exists and ok), "header": header})
    write_tsv(root / "aggregate" / "external_tool_status.tsv", rows)


def main() -> None:
    """Build and run the transition-aligned stitching benchmark."""

    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "aggregate").mkdir(parents=True, exist_ok=True)
    check_tools(args.root, args.mdlovofit_exe, args.theseus_exe)

    domains = split_csv(args.domain_ids)
    targets = [int(item) for item in split_csv(args.targets)]
    sequence_names = split_csv(args.sequence_names)
    benchmark_sequence_names = split_csv(args.benchmark_sequence_names) if args.benchmark_sequence_names else sequence_names
    build_policies = split_csv(args.build_boundary_policies)
    benchmark_policies = split_csv(args.benchmark_boundary_policies)

    summary_df = load_summary(args.all_domain_summary)
    selected = select_domains(summary_df, domains, args.root)
    for domain in selected:
        extract_replicates(domain, summary_df, args.root)
    build_all_stitched(selected, args.root, targets, sequence_names, build_policies)

    if not args.build_only:
        benchmark_inputs(
            selected,
            args.root,
            benchmark_sequence_names,
            targets,
            benchmark_policies,
            args.repeats,
            args.mdlovofit_exe,
            args.theseus_exe,
            args.timeout_seconds,
        )
    print(args.root)


if __name__ == "__main__":
    main()
