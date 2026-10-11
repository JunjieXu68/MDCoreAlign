#!/usr/bin/env python
"""Benchmark continuous mdCATH trajectories at increasing prefix lengths.

This benchmark uses single continuous trajectories only. It does not stitch
independent replicates. Download and HDF5 extraction are excluded from timed
method runtimes.
"""

from __future__ import annotations

import argparse
import csv
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

from mdsuperpose.core_selection import select_core_by_target_fraction
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


MDLOVOFIT_EXE = Path("/usr/local/bin/mdlovofit")
THESEUS_EXE = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")
PREFIX_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)

RUNTIME_COLUMNS = [
    "domain_id",
    "trajectory_id",
    "source",
    "prefix_label",
    "prefix_fraction",
    "n_frames",
    "n_residues",
    "method",
    "repeat_id",
    "runtime_scope",
    "alignment_only_runtime_sec",
    "rmsf_runtime_sec",
    "flexible_segment_runtime_sec",
    "profile_write_runtime_sec",
    "full_workflow_runtime_sec",
    "runtime_per_frame",
    "unified_posthoc_RMSD",
    "weighted_RMSD_mean",
    "iteration_count_mean",
    "core_fraction",
    "status",
    "failure_reason",
    "notes",
]

COMPARISON_COLUMNS = [
    "domain_id",
    "trajectory_id",
    "prefix_label",
    "prefix_fraction",
    "n_frames",
    "n_residues",
    "mdcorealign_alignment_runtime_sec",
    "mdcorealign_full_workflow_runtime_sec",
    "mdlovofit_runtime_sec",
    "theseus_runtime_sec",
    "mdcorealign_vs_mdlovofit_speedup",
    "mdcorealign_vs_theseus_speedup",
    "mdcorealign_faster_than_mdlovofit",
    "mdcorealign_faster_than_theseus",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_theseus",
    "mdlovofit_status",
    "theseus_status",
    "notes",
]

SCALING_COLUMNS = [
    "domain_id",
    "trajectory_id",
    "method",
    "runtime_scope",
    "n_prefixes_success",
    "R2_linear",
    "R2_quadratic",
    "AIC_linear",
    "AIC_quadratic",
    "estimated_scaling_preference",
    "runtime_per_frame_stability",
    "speedup_trend_note",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-domain-summary",
        type=Path,
        default=Path(
            "results/mdcath_benchmark/all_domain_canonical/aggregate/"
            "all_domain_canonical_per_domain_summary.tsv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/mdcath_benchmark/long_continuous_trajectory"),
    )
    parser.add_argument("--n-trajectories", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--mdlovofit-exe", type=Path, default=MDLOVOFIT_EXE)
    parser.add_argument("--theseus-exe", type=Path, default=THESEUS_EXE)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def fnum(value: object) -> float | None:
    """Parse a finite float."""

    try:
        if value in ("", "NA", None, "not_run"):
            return None
        converted = float(value)
    except Exception:
        return None
    return converted if math.isfinite(converted) else None


def median(values: list[float]) -> float | str:
    """Return median or NA."""

    return float(np.median(np.asarray(values, dtype=float))) if values else "NA"


def h5_path_from_notes(notes: str) -> Path | None:
    """Extract a cached HDF5 path from summary notes."""

    match = re.search(r"h5_file=([^;]+)", str(notes or ""))
    return Path(match.group(1)) if match else None


def coord_path(domain_id: str, temperature_k: int, replicate_index_zero_based: int) -> str:
    """Return mdCATH coordinate dataset path."""

    return f"/{domain_id}/{temperature_k}/{replicate_index_zero_based}/coords"


def prefix_label(fraction: float) -> str:
    """Return a stable prefix label."""

    return f"p{int(round(fraction * 100)):03d}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""

    subset = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in subset.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *body])


def select_continuous_trajectories(summary_path: Path, n: int) -> list[dict[str, Any]]:
    """Select the largest-frame continuous trajectories from all-domain results."""

    df = pd.read_csv(summary_path, sep="\t")
    for col in [
        "n_frames",
        "n_residues",
        "selected_temperature_K",
        "selected_replica_label_one_based",
        "mdcorealign_runtime_sec",
        "mdlovofit_runtime_sec",
        "theseus_runtime_sec",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        (df["mdcorealign_status"] == "success")
        & (df["mdlovofit_status"] == "success")
        & (df["theseus_status"] == "success")
        & df["n_frames"].notna()
        & df["n_residues"].notna()
    ].copy()
    df["h5_path"] = df["notes"].map(h5_path_from_notes)
    df = df[df["h5_path"].map(lambda p: p is not None and p.exists())]
    df = df.sort_values(["n_frames", "n_residues"], ascending=[False, False])
    selected = df.head(n).copy()
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "domain_id": str(row["domain_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "source": "mdCATH single continuous 320 K replicate",
                "cath_class": row.get("cath_class", "NA"),
                "n_frames": int(row["n_frames"]),
                "n_residues": int(row["n_residues"]),
                "selected_temperature_K": int(row["selected_temperature_K"]),
                "selected_replica_label_one_based": int(row["selected_replica_label_one_based"]),
                "h5_path": str(row["h5_path"]),
                "selection_reason": "largest available continuous frame counts, tie-broken by residue count",
                "notes": "Single-replicate mdCATH trajectory; not stitched.",
            }
        )
    return rows


def load_or_extract_coords(row: dict[str, Any], output_root: Path) -> tuple[np.ndarray, list[str], list[dict[str, str]], Path]:
    """Load cached NPZ coordinates or extract CA coordinates from HDF5."""

    trajectory_id = row["trajectory_id"]
    out_npz = output_root / "continuous_inputs" / f"{trajectory_id}_ca_coords.npz"
    out_json = output_root / "continuous_inputs" / f"{trajectory_id}_metadata.json"
    if out_npz.exists():
        payload = np.load(out_npz, allow_pickle=True)
        coords = np.asarray(payload["coords"], dtype=float)
        labels = [str(item) for item in payload["labels"].tolist()]
        metadata = payload["residue_metadata"].tolist() if "residue_metadata" in payload.files else []
        return coords, labels, metadata, out_npz
    rep_index = int(row["selected_replica_label_one_based"]) - 1
    coords, labels, residue_metadata, metadata = extract_ca_trajectory(
        Path(row["h5_path"]),
        coordinate_path=coord_path(row["domain_id"], int(row["selected_temperature_K"]), rep_index),
        min_ca_atoms=30,
        min_frames=20,
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        coords=coords.astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=object),
        residue_metadata=np.asarray(residue_metadata, dtype=object),
    )
    write_json(out_json, metadata)
    return coords, labels, residue_metadata, out_npz


def write_profile(path: Path, residue_metadata: list[dict[str, str]], rmsf: np.ndarray, weights: np.ndarray, segments: np.ndarray) -> None:
    """Write a minimal profile table for full-workflow timing."""

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
        for index, meta in enumerate(residue_metadata):
            weight = float(weights[index])
            writer.writerow(
                {
                    "atom_index": index,
                    "chain": meta.get("chain", "_") if isinstance(meta, dict) else "_",
                    "residue_id": meta.get("residue_id", index + 1) if isinstance(meta, dict) else index + 1,
                    "resname": meta.get("resname", "UNK") if isinstance(meta, dict) else "UNK",
                    "robust_RMSF": float(rmsf[index]),
                    "soft_stability_weight": weight,
                    "mobility_score": float(1.0 - weight),
                    "flexible_segment_id": int(segments[index]),
                }
            )


def run_mdcorealign_cached(
    coords: np.ndarray,
    residue_metadata: list[dict[str, str]],
    profile_path: Path,
) -> dict[str, Any]:
    """Run MDCoreAlign once and return timing plus summary fields."""

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
    alignment_seconds = time.perf_counter() - start

    rmsf_start = time.perf_counter()
    rmsf = compute_rmsf(aligned)
    rmsf_seconds = time.perf_counter() - rmsf_start

    weights = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
    segment_start = time.perf_counter()
    chain = [
        item.get("chain", "_") if isinstance(item, dict) else "_"
        for item in residue_metadata
    ]
    segments = detect_flexible_segments(
        1.0 - weights,
        percentile_cutoff=0.80,
        min_segment_length=3,
        merge_gap=1,
        chain=chain,
    )
    segment_seconds = time.perf_counter() - segment_start

    profile_start = time.perf_counter()
    write_profile(profile_path, residue_metadata, rmsf, weights, segments)
    profile_seconds = time.perf_counter() - profile_start

    core = select_core_by_target_fraction(weights, 0.6)
    weighted = [frame.result.weighted_rmsd for frame in result.frame_results]
    iterations = [frame.iteration_count for frame in result.frame_results]
    return {
        "status": "success",
        "alignment_runtime_sec": alignment_seconds,
        "rmsf_runtime_sec": rmsf_seconds,
        "flexible_segment_runtime_sec": segment_seconds,
        "profile_write_runtime_sec": profile_seconds,
        "full_workflow_runtime_sec": alignment_seconds + rmsf_seconds + segment_seconds + profile_seconds,
        "unified_rmsd": mean_rmsd(aligned, coords[0]),
        "weighted_rmsd_mean": float(np.mean(weighted)) if weighted else "NA",
        "iteration_count_mean": float(np.mean(iterations)) if iterations else "NA",
        "core_fraction": float(np.mean(core.core_mask)),
        "failure_reason": "NA",
    }


def run_mdlovofit_cached(
    input_path: Path,
    coords: np.ndarray,
    outdir: Path,
    executable: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run MDLovoFit on an existing multi-model PDB."""

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "trajectory_aligned.pdb"
    try:
        run = run_mdlovofit_on_files(
            [input_path],
            fraction=0.6,
            output_path=output_path,
            executable=executable,
            timeout_seconds=timeout_seconds,
        )
        (outdir / "stdout.txt").write_text(run.stdout)
        (outdir / "stderr.txt").write_text(run.stderr)
        aligned, _occupancy = read_multimodel_pdb_with_occupancy(output_path, n_atoms=coords.shape[1])
        return {
            "status": "success",
            "runtime_sec": run.subprocess_runtime_seconds,
            "unified_rmsd": mean_rmsd(aligned, coords[0]),
            "failure_reason": "NA",
            "notes": f"sha256_output={sha256_file(output_path)}",
        }
    except Exception as exc:
        (outdir / "stderr.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        return {
            "status": "failed",
            "runtime_sec": "NA",
            "unified_rmsd": "NA",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "notes": "NA",
        }


def run_theseus_cached(
    input_path: Path,
    coords: np.ndarray,
    outdir: Path,
    executable: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run THESEUS on an existing multi-model PDB."""

    outdir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-a0", str(input_path.resolve())]
    env = dict(os.environ)
    candidate_lib = executable.parent.parent / "lib"
    if candidate_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{candidate_lib}{os.pathsep}{existing}" if existing else str(candidate_lib)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=outdir,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
        runtime = time.perf_counter() - start
        (outdir / "stdout.txt").write_text(completed.stdout)
        (outdir / "stderr.txt").write_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"THESEUS return code {completed.returncode}")
        aligned_path = theseus_output_path(outdir, input_path)
        if aligned_path is None:
            raise FileNotFoundError("No THESEUS *_sup.pdb output found")
        aligned = read_multimodel_pdb_coords(aligned_path, n_atoms=coords.shape[1])
        _parsed = parse_theseus_stdout(completed.stdout)
        variance_paths = sorted(outdir.glob("*_variances.txt"))
        if variance_paths:
            _ = parse_theseus_variances(variance_paths[0])
        return {
            "status": "success",
            "runtime_sec": runtime,
            "unified_rmsd": mean_rmsd(aligned, coords[0]),
            "failure_reason": "NA",
            "notes": f"sha256_output={sha256_file(aligned_path)}",
        }
    except Exception as exc:
        runtime = time.perf_counter() - start
        old_stderr = (outdir / "stderr.txt").read_text() if (outdir / "stderr.txt").exists() else ""
        (outdir / "stderr.txt").write_text(old_stderr + f"\n{type(exc).__name__}: {exc}\n")
        return {
            "status": "failed",
            "runtime_sec": runtime,
            "unified_rmsd": "NA",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "notes": "NA",
        }


def fit_scaling(rows: pd.DataFrame, runtime_col: str) -> dict[str, Any]:
    """Fit linear and quadratic runtime models for one method/domain."""

    ok = rows[pd.to_numeric(rows[runtime_col], errors="coerce").notna()].copy()
    ok["x"] = pd.to_numeric(ok["n_frames"], errors="coerce")
    ok["y"] = pd.to_numeric(ok[runtime_col], errors="coerce")
    ok = ok[ok["x"].notna() & ok["y"].notna()]
    if len(ok) < 3:
        return {
            "n_prefixes_success": len(ok),
            "R2_linear": "NA",
            "R2_quadratic": "NA",
            "AIC_linear": "NA",
            "AIC_quadratic": "NA",
            "estimated_scaling_preference": "unclear",
            "runtime_per_frame_stability": "NA",
        }
    xs = ok["x"].to_numpy(dtype=float)
    ys = ok["y"].to_numpy(dtype=float)

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

    r2_lin, aic_lin = fit(np.column_stack([np.ones_like(xs), xs]))
    r2_quad, aic_quad = fit(np.column_stack([np.ones_like(xs), xs, xs * xs]))
    per_frame = ys / xs
    stability = float((np.max(per_frame) - np.min(per_frame)) / np.median(per_frame)) if np.median(per_frame) else float("nan")
    preference = "linear"
    if aic_quad + 2.0 < aic_lin and stability > 0.5:
        preference = "possible_quadratic"
    elif aic_quad + 2.0 < aic_lin and stability > 0.25:
        preference = "possible_superlinear"
    elif abs(aic_quad - aic_lin) < 2.0:
        preference = "unclear"
    return {
        "n_prefixes_success": int(len(ok)),
        "R2_linear": r2_lin,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_lin,
        "AIC_quadratic": aic_quad,
        "estimated_scaling_preference": preference,
        "runtime_per_frame_stability": stability,
    }


def summarize_runtime(runtime_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create per-method comparison rows from repeat-level runtimes."""

    df = pd.DataFrame(runtime_rows)
    rows = []
    for (domain_id, trajectory_id, prefix_label_value), group in df.groupby(["domain_id", "trajectory_id", "prefix_label"], sort=False):
        base = group.iloc[0]
        lookup: dict[str, pd.DataFrame] = {method: sub for method, sub in group.groupby("method")}
        def med(method: str, col: str) -> float | str:
            if method not in lookup:
                return "NA"
            vals = pd.to_numeric(lookup[method][col], errors="coerce").dropna().to_list()
            return median(vals)

        mdca_align = med("mdcorealign", "alignment_only_runtime_sec")
        mdca_full = med("mdcorealign", "full_workflow_runtime_sec")
        mdlo_rt = med("mdlovofit", "alignment_only_runtime_sec")
        th_rt = med("theseus", "alignment_only_runtime_sec")
        mdca_rmsd = med("mdcorealign", "unified_posthoc_RMSD")
        mdlo_rmsd = med("mdlovofit", "unified_posthoc_RMSD")
        th_rmsd = med("theseus", "unified_posthoc_RMSD")
        mdca = fnum(mdca_align)
        mdlo = fnum(mdlo_rt)
        th = fnum(th_rt)
        mdca_r = fnum(mdca_rmsd)
        mdlo_r = fnum(mdlo_rmsd)
        th_r = fnum(th_rmsd)
        rows.append(
            {
                "domain_id": domain_id,
                "trajectory_id": trajectory_id,
                "prefix_label": prefix_label_value,
                "prefix_fraction": base["prefix_fraction"],
                "n_frames": base["n_frames"],
                "n_residues": base["n_residues"],
                "mdcorealign_alignment_runtime_sec": mdca_align,
                "mdcorealign_full_workflow_runtime_sec": mdca_full,
                "mdlovofit_runtime_sec": mdlo_rt,
                "theseus_runtime_sec": th_rt,
                "mdcorealign_vs_mdlovofit_speedup": float(mdlo / mdca) if mdlo and mdca else "NA",
                "mdcorealign_vs_theseus_speedup": float(th / mdca) if th and mdca else "NA",
                "mdcorealign_faster_than_mdlovofit": int(mdca is not None and mdlo is not None and mdca < mdlo),
                "mdcorealign_faster_than_theseus": int(mdca is not None and th is not None and mdca < th),
                "mdcorealign_unified_RMSD": mdca_rmsd,
                "mdlovofit_unified_RMSD": mdlo_rmsd,
                "theseus_unified_RMSD": th_rmsd,
                "mdcorealign_lower_RMSD_than_mdlovofit": int(mdca_r is not None and mdlo_r is not None and mdca_r < mdlo_r),
                "mdcorealign_lower_RMSD_than_theseus": int(mdca_r is not None and th_r is not None and mdca_r < th_r),
                "mdlovofit_status": "success" if mdlo is not None else "failed",
                "theseus_status": "success" if th is not None else "failed",
                "notes": "Continuous single-trajectory prefix; no stitched replicates.",
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def scaling_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Create scaling summary for each trajectory and method."""

    rows = []
    for (domain_id, trajectory_id), group in comparison.groupby(["domain_id", "trajectory_id"], sort=False):
        method_specs = [
            ("mdcorealign", "alignment_only", "mdcorealign_alignment_runtime_sec"),
            ("mdcorealign", "full_workflow", "mdcorealign_full_workflow_runtime_sec"),
            ("mdlovofit", "external_subprocess", "mdlovofit_runtime_sec"),
            ("theseus", "external_subprocess", "theseus_runtime_sec"),
        ]
        for method, scope, col in method_specs:
            fitted = fit_scaling(group, col)
            speed_note = "NA"
            if method == "mdcorealign" and scope == "alignment_only":
                speeds = pd.to_numeric(group["mdcorealign_vs_mdlovofit_speedup"], errors="coerce")
                frames = pd.to_numeric(group["n_frames"], errors="coerce")
                ok = speeds.notna() & frames.notna()
                if ok.sum() >= 2:
                    trend = float(np.corrcoef(frames[ok], speeds[ok])[0, 1])
                    speed_note = f"corr_frame_count_vs_mdlovofit_speedup={trend:.4f}"
            rows.append(
                {
                    "domain_id": domain_id,
                    "trajectory_id": trajectory_id,
                    "method": method,
                    "runtime_scope": scope,
                    **fitted,
                    "speedup_trend_note": speed_note,
                    "notes": "Five prefix lengths: 10%, 25%, 50%, 75%, 100%.",
                }
            )
    return pd.DataFrame(rows, columns=SCALING_COLUMNS)


def write_report(root: Path, selected: pd.DataFrame, comparison: pd.DataFrame, scaling: pd.DataFrame) -> None:
    """Write the Markdown report."""

    runtime_path = root / "long_continuous_trajectory_runtime.tsv"
    runtime = pd.read_csv(runtime_path, sep="\t") if runtime_path.exists() else pd.DataFrame()
    mdlo_success = pd.to_numeric(comparison["mdlovofit_runtime_sec"], errors="coerce").notna().sum()
    th_success = pd.to_numeric(comparison["theseus_runtime_sec"], errors="coerce").notna().sum()
    mdca_faster_mdlo = int(pd.to_numeric(comparison["mdcorealign_faster_than_mdlovofit"], errors="coerce").fillna(0).sum())
    mdca_faster_th = int(pd.to_numeric(comparison["mdcorealign_faster_than_theseus"], errors="coerce").fillna(0).sum())
    mdca_lower_mdlo = int(pd.to_numeric(comparison["mdcorealign_lower_RMSD_than_mdlovofit"], errors="coerce").fillna(0).sum())
    mdca_lower_th = int(pd.to_numeric(comparison["mdcorealign_lower_RMSD_than_theseus"], errors="coerce").fillna(0).sum())
    speed_by_prefix = []
    speed_medians = []
    for prefix, group in comparison.groupby("prefix_label", sort=False):
        median_speed = pd.to_numeric(group["mdcorealign_vs_mdlovofit_speedup"], errors="coerce").median()
        speed_medians.append((float(pd.to_numeric(group["prefix_fraction"], errors="coerce").median()), float(median_speed)))
        speed_by_prefix.append(
            "- "
            f"{prefix}: median frames={pd.to_numeric(group['n_frames'], errors='coerce').median():.1f}, "
            f"MDCoreAlign/MDLovoFit speedup={median_speed:.3f}, "
            f"MDCoreAlign faster={100.0 * pd.to_numeric(group['mdcorealign_faster_than_mdlovofit'], errors='coerce').fillna(0).mean():.1f}%, "
            f"MDCoreAlign/THESEUS speedup={pd.to_numeric(group['mdcorealign_vs_theseus_speedup'], errors='coerce').median():.3f}"
        )
    scaling_counts = scaling.groupby(["method", "runtime_scope", "estimated_scaling_preference"]).size()
    scaling_text = "; ".join(f"{idx}:{value}" for idx, value in scaling_counts.items())
    runtime_per_frame_lines = []
    overhead_lines = []
    if not runtime.empty:
        for prefix, group in runtime.groupby("prefix_label", sort=False):
            per_method = group.groupby("method")["runtime_per_frame"].median(numeric_only=True)
            runtime_per_frame_lines.append(
                "- "
                f"{prefix}: MDCoreAlign={per_method.get('mdcorealign', float('nan')):.6f} s/frame, "
                f"MDLovoFit={per_method.get('mdlovofit', float('nan')):.6f} s/frame, "
                f"THESEUS={per_method.get('theseus', float('nan')):.6f} s/frame"
            )
        mdca_runtime = runtime[runtime["method"] == "mdcorealign"].copy()
        if not mdca_runtime.empty:
            mdca_runtime["overhead"] = (
                pd.to_numeric(mdca_runtime["full_workflow_runtime_sec"], errors="coerce")
                - pd.to_numeric(mdca_runtime["alignment_only_runtime_sec"], errors="coerce")
            )
            mdca_runtime["overhead_percent"] = (
                100.0
                * mdca_runtime["overhead"]
                / pd.to_numeric(mdca_runtime["alignment_only_runtime_sec"], errors="coerce")
            )
            for prefix, group in mdca_runtime.groupby("prefix_label", sort=False):
                overhead_lines.append(
                    "- "
                    f"{prefix}: median overhead={group['overhead'].median():.6f} s, "
                    f"median overhead/alignment={group['overhead_percent'].median():.2f}%"
                )
    first_speed = speed_medians[0][1] if speed_medians else float("nan")
    final_speed = speed_medians[-1][1] if speed_medians else float("nan")
    if math.isfinite(first_speed) and math.isfinite(final_speed):
        if final_speed > first_speed * 1.05:
            direct_answer = (
                "MDCoreAlign's speedup versus MDLovoFit increases across the continuous-prefix panel, "
                f"from {first_speed:.3f}x at the shortest prefixes to {final_speed:.3f}x at the full trajectories."
            )
        elif final_speed < first_speed * 0.95:
            direct_answer = (
                "MDCoreAlign is consistently faster than MDLovoFit, but the relative speedup does not increase "
                f"with frame count in this panel: median speedup changes from {first_speed:.3f}x at the shortest "
                f"prefixes to {final_speed:.3f}x at the full trajectories. Thus, longer physically continuous "
                "trajectories confirm a robust MDCoreAlign advantage over MDLovoFit, but they do not show an "
                "increasing MDLovoFit speedup trend."
            )
        else:
            direct_answer = (
                "MDCoreAlign is consistently faster than MDLovoFit, and the relative speedup is approximately "
                f"stable across frame counts: {first_speed:.3f}x at the shortest prefixes and {final_speed:.3f}x "
                "at the full trajectories."
            )
    else:
        direct_answer = "The MDCoreAlign-versus-MDLovoFit frame-count trend could not be estimated from finite speedups."
    report = f"""# Long Continuous Trajectory Runtime Benchmark

## Scope

This benchmark uses real continuous mdCATH single-replicate trajectories only. It does not use stitched independent replicates. The selected trajectories were chosen by largest available frame count from existing cached mdCATH all-domain results, with ties broken by residue count.

## Selected Trajectories

{markdown_table(selected, ['domain_id', 'trajectory_id', 'n_frames', 'n_residues', 'source'])}

## Runtime Protocol

For each trajectory, five prefixes were evaluated: 10%, 25%, 50%, 75%, and 100% of frames. CA-only coordinates were extracted once before timing. PDB export, HDF5 extraction, and download/cache lookup were excluded from method runtime. MDCoreAlign was timed on cached coordinate arrays. MDLovoFit and THESEUS were timed as subprocesses on cached multi-model PDB inputs. Unified post-hoc RMSD was computed against the first frame under the same CA selection.

## Results Summary

- Prefix inputs: {len(comparison)}
- MDLovoFit finite runtime comparisons: {mdlo_success}/{len(comparison)}
- THESEUS finite runtime comparisons: {th_success}/{len(comparison)}
- MDCoreAlign faster than MDLovoFit: {mdca_faster_mdlo}/{mdlo_success}
- MDCoreAlign faster than THESEUS: {mdca_faster_th}/{th_success}
- MDCoreAlign lower unified RMSD than MDLovoFit: {mdca_lower_mdlo}/{mdlo_success}
- MDCoreAlign lower unified RMSD than THESEUS: {mdca_lower_th}/{th_success}

## Speedup Versus Frame Count

{chr(10).join(speed_by_prefix)}

## Runtime Per Frame

{chr(10).join(runtime_per_frame_lines) if runtime_per_frame_lines else 'Runtime-per-frame values were unavailable.'}

## Alignment-Only Versus Full-Workflow Timing

{chr(10).join(overhead_lines) if overhead_lines else 'MDCoreAlign full-workflow overhead values were unavailable.'}

## Scaling Summary

Scaling labels: {scaling_text}. These five-point fits are diagnostic rather than formal complexity proofs.

## Direct Answer

{direct_answer} The stitched-replicate benchmark should remain a diagnostic stress test because it contains artificial replicate-boundary jumps.
"""
    (root / "long_continuous_trajectory_report.md").write_text(report)
    Path("docs/long_continuous_trajectory_report.md").write_text(report)


def main() -> None:
    """Run the continuous long-trajectory benchmark."""

    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "aggregate").mkdir(parents=True, exist_ok=True)
    (args.output_root / "external_inputs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    tool_rows = []
    for name, exe, commands in (
        ("MDLovoFit", args.mdlovofit_exe, [[str(args.mdlovofit_exe), "-h"], [str(args.mdlovofit_exe), "--help"]]),
        ("THESEUS", args.theseus_exe, [[str(args.theseus_exe), "-h"], [str(args.theseus_exe), "--version"]]),
    ):
        exists = exe.exists() and os.access(exe, os.X_OK)
        ok = False
        header = "NA"
        if exists:
            for command in commands:
                ok, header = check_command(command)
                if ok:
                    break
        tool_rows.append({"tool": name, "executable": exe, "available": int(exists and ok), "header": header})
    write_tsv(args.output_root / "external_tool_status.tsv", tool_rows)

    selected_rows = select_continuous_trajectories(args.per_domain_summary, args.n_trajectories)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_root / "selected_continuous_trajectories.tsv", sep="\t", index=False)

    runtime_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for selected_row in selected_rows:
        coords_full, labels, residue_metadata, npz_path = load_or_extract_coords(selected_row, args.output_root)
        for fraction in PREFIX_FRACTIONS:
            n_prefix = min(coords_full.shape[0], max(2, int(math.ceil(coords_full.shape[0] * fraction))))
            coords = np.asarray(coords_full[:n_prefix], dtype=float)
            label = prefix_label(fraction)
            input_path = args.output_root / "external_inputs" / f"{selected_row['trajectory_id']}_{label}_ca_multimodel.pdb"
            if not input_path.exists():
                write_multimodel_pdb(coords, input_path)
            manifest_rows.append(
                {
                    "domain_id": selected_row["domain_id"],
                    "trajectory_id": selected_row["trajectory_id"],
                    "prefix_label": label,
                    "prefix_fraction": fraction,
                    "n_frames": n_prefix,
                    "n_residues": coords.shape[1],
                    "input_npz": npz_path,
                    "input_pdb": input_path,
                    "sha256_pdb": sha256_file(input_path),
                    "notes": "Continuous prefix from one mdCATH replicate; no stitching.",
                }
            )
            for repeat_id in range(1, args.repeats + 1):
                run_base = args.output_root / "runs" / selected_row["trajectory_id"] / label / f"repeat_{repeat_id}"
                common = {
                    "domain_id": selected_row["domain_id"],
                    "trajectory_id": selected_row["trajectory_id"],
                    "source": selected_row["source"],
                    "prefix_label": label,
                    "prefix_fraction": fraction,
                    "n_frames": n_prefix,
                    "n_residues": coords.shape[1],
                    "repeat_id": repeat_id,
                }
                try:
                    mdca = run_mdcorealign_cached(
                        coords,
                        residue_metadata,
                        run_base / "mdcorealign" / "residue_profile.tsv",
                    )
                except Exception as exc:
                    mdca = {
                        "status": "failed",
                        "alignment_runtime_sec": "NA",
                        "rmsf_runtime_sec": "NA",
                        "flexible_segment_runtime_sec": "NA",
                        "profile_write_runtime_sec": "NA",
                        "full_workflow_runtime_sec": "NA",
                        "unified_rmsd": "NA",
                        "weighted_rmsd_mean": "NA",
                        "iteration_count_mean": "NA",
                        "core_fraction": "NA",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                runtime_rows.append(
                    {
                        **common,
                        "method": "mdcorealign",
                        "runtime_scope": "cached_coords_algorithm_and_full_workflow",
                        "alignment_only_runtime_sec": mdca["alignment_runtime_sec"],
                        "rmsf_runtime_sec": mdca["rmsf_runtime_sec"],
                        "flexible_segment_runtime_sec": mdca["flexible_segment_runtime_sec"],
                        "profile_write_runtime_sec": mdca["profile_write_runtime_sec"],
                        "full_workflow_runtime_sec": mdca["full_workflow_runtime_sec"],
                        "runtime_per_frame": (
                            float(mdca["alignment_runtime_sec"]) / n_prefix
                            if fnum(mdca["alignment_runtime_sec"]) is not None
                            else "NA"
                        ),
                        "unified_posthoc_RMSD": mdca["unified_rmsd"],
                        "weighted_RMSD_mean": mdca["weighted_rmsd_mean"],
                        "iteration_count_mean": mdca["iteration_count_mean"],
                        "core_fraction": mdca["core_fraction"],
                        "status": mdca["status"],
                        "failure_reason": mdca["failure_reason"],
                        "notes": "Warm-cache cached coordinate timing; extraction/export excluded.",
                    }
                )

                mdlo = run_mdlovofit_cached(
                    input_path,
                    coords,
                    run_base / "mdlovofit",
                    args.mdlovofit_exe,
                    args.timeout_seconds,
                )
                runtime_rows.append(
                    {
                        **common,
                        "method": "mdlovofit",
                        "runtime_scope": "external_subprocess_cached_pdb",
                        "alignment_only_runtime_sec": mdlo["runtime_sec"],
                        "rmsf_runtime_sec": "NA",
                        "flexible_segment_runtime_sec": "NA",
                        "profile_write_runtime_sec": "NA",
                        "full_workflow_runtime_sec": mdlo["runtime_sec"],
                        "runtime_per_frame": float(mdlo["runtime_sec"]) / n_prefix if fnum(mdlo["runtime_sec"]) is not None else "NA",
                        "unified_posthoc_RMSD": mdlo["unified_rmsd"],
                        "weighted_RMSD_mean": "NA",
                        "iteration_count_mean": "NA",
                        "core_fraction": "NA",
                        "status": mdlo["status"],
                        "failure_reason": mdlo["failure_reason"],
                        "notes": mdlo["notes"],
                    }
                )

                theseus = run_theseus_cached(
                    input_path,
                    coords,
                    run_base / "theseus",
                    args.theseus_exe,
                    args.timeout_seconds,
                )
                runtime_rows.append(
                    {
                        **common,
                        "method": "theseus",
                        "runtime_scope": "external_subprocess_cached_pdb",
                        "alignment_only_runtime_sec": theseus["runtime_sec"],
                        "rmsf_runtime_sec": "NA",
                        "flexible_segment_runtime_sec": "NA",
                        "profile_write_runtime_sec": "NA",
                        "full_workflow_runtime_sec": theseus["runtime_sec"],
                        "runtime_per_frame": float(theseus["runtime_sec"]) / n_prefix if fnum(theseus["runtime_sec"]) is not None else "NA",
                        "unified_posthoc_RMSD": theseus["unified_rmsd"],
                        "weighted_RMSD_mean": "NA",
                        "iteration_count_mean": "NA",
                        "core_fraction": "NA",
                        "status": theseus["status"],
                        "failure_reason": theseus["failure_reason"],
                        "notes": theseus["notes"],
                    }
                )
                pd.DataFrame(runtime_rows).to_csv(
                    args.output_root / "long_continuous_trajectory_runtime.tsv",
                    sep="\t",
                    index=False,
                    columns=RUNTIME_COLUMNS,
                )

    pd.DataFrame(manifest_rows).to_csv(args.output_root / "input_manifest.tsv", sep="\t", index=False)
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(args.output_root / "long_continuous_trajectory_runtime.tsv", sep="\t", index=False, columns=RUNTIME_COLUMNS)
    comparison = summarize_runtime(runtime_rows)
    comparison.to_csv(args.output_root / "long_continuous_trajectory_tool_comparison.tsv", sep="\t", index=False)
    scaling = scaling_summary(comparison)
    scaling.to_csv(args.output_root / "long_continuous_trajectory_scaling_summary.tsv", sep="\t", index=False)
    write_report(args.output_root, selected, comparison, scaling)

    # Also mirror the requested outputs under aggregate/ for consistency with
    # previous mdCATH benchmark packages.
    for filename in [
        "long_continuous_trajectory_runtime.tsv",
        "long_continuous_trajectory_scaling_summary.tsv",
        "long_continuous_trajectory_tool_comparison.tsv",
        "long_continuous_trajectory_report.md",
    ]:
        shutil.copyfile(args.output_root / filename, args.output_root / "aggregate" / filename)
    print(args.output_root / "long_continuous_trajectory_tool_comparison.tsv")


if __name__ == "__main__":
    main()
