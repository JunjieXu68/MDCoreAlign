#!/usr/bin/env python
"""Validate long-mode stable/mobile interpretation quality.

This script audits the transition-aligned synthetic long-trajectory scaling
benchmark. It does not change the MDCoreAlign algorithm and should not be used
as evidence of physical trajectory continuity.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.flexibility import (
    compute_rmsf,
    detect_flexible_segments,
    spearman_correlation,
)
from mdsuperpose.long_mode import build_long_flexibility_profile, solve_long_trajectory_alignment
from scripts.mdcath_benchmark.mdcath_common import write_tsv
from scripts.mdcath_benchmark.run_transition_aligned_long_mode_benchmark import (
    MethodResult,
    finite_float,
    jaccard,
    run_default,
)


QUALITY_COLUMNS = [
    "domain_id",
    "cath_class",
    "n_residues",
    "target_frames",
    "sequence_name",
    "boundary_policy",
    "default_runtime_sec",
    "long_runtime_sec",
    "speedup_long_vs_default",
    "default_core_fraction",
    "long_core_fraction",
    "core_fraction_delta",
    "core_jaccard",
    "mobile_jaccard",
    "mobility_spearman",
    "RMSF_spearman",
    "max_RMSF_abs_diff",
    "median_RMSF_abs_diff",
    "flexible_segment_jaccard",
    "n_flexible_segments_default",
    "n_flexible_segments_long",
    "noncore_core_RMSF_ratio_default",
    "noncore_core_RMSF_ratio_long",
    "noncore_core_RMSF_ratio_delta",
    "unified_RMSD_default",
    "unified_RMSD_long",
    "unified_RMSD_delta",
    "quality_pass",
    "notes",
]

SEPARATION_COLUMNS = [
    "domain_id",
    "target_frames",
    "method_mode",
    "core_fraction",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "percent_noncore_RMSF_greater_than_core",
    "RMSF_mobility_spearman",
    "n_flexible_segments",
    "flexibility_separation_pass",
    "notes",
]

BOUNDARY_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_boundaries",
    "median_post_boundary_jump",
    "max_post_boundary_jump",
    "boundary_to_within_ratio",
    "core_jaccard_default_long",
    "RMSF_spearman_default_long",
    "flexible_segment_jaccard",
    "suspected_boundary_artifact",
    "notes",
]

SENSITIVITY_COLUMNS = [
    "domain_id",
    "target_frames",
    "frame_selection_strategy",
    "seed",
    "n_representative_frames",
    "runtime_sec",
    "core_fraction",
    "core_jaccard_vs_uniform",
    "mobile_jaccard_vs_uniform",
    "mobility_spearman_vs_uniform",
    "RMSF_spearman_vs_uniform",
    "flexible_segment_jaccard_vs_uniform",
    "unified_RMSD",
    "sensitivity_pass",
    "notes",
]

HELDOUT_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_train_frames",
    "n_heldout_frames",
    "core_RMSF_train",
    "noncore_RMSF_train",
    "noncore_core_ratio_train",
    "core_RMSF_heldout",
    "noncore_RMSF_heldout",
    "noncore_core_ratio_heldout",
    "mobility_RMSF_spearman_heldout",
    "heldout_pass",
    "notes",
]

MDLOVO_CORE_COLUMNS = [
    "domain_id",
    "target_frames",
    "n_residues",
    "mdcorealign_long_core_fraction",
    "mdlovofit_core_fraction",
    "core_jaccard_long_vs_mdlovofit",
    "mobile_jaccard_long_vs_mdlovofit",
    "top20_mobile_overlap_if_available",
    "long_noncore_core_RMSF_ratio",
    "mdlovofit_noncore_core_RMSF_ratio_if_available",
    "notes",
]

WEAK_COLUMNS = [
    "domain_id",
    "target_frames",
    "failed_metric",
    "value",
    "threshold",
    "possible_reason",
    "recommended_action",
]

TRADEOFF_COLUMNS = [
    "target_frames",
    "n_cases",
    "median_long_speedup_vs_MDLovoFit",
    "median_long_speedup_vs_default",
    "median_core_jaccard_default_long",
    "median_mobile_jaccard_default_long",
    "median_mobility_spearman_default_long",
    "median_RMSF_spearman_default_long",
    "median_flexible_segment_jaccard",
    "quality_pass_count",
    "quality_pass_percent",
    "median_noncore_core_RMSF_ratio_long",
    "notes",
]


@dataclass(slots=True)
class LongEval:
    """Compact long-mode result used by this validation script."""

    runtime_sec: float
    core_mask: np.ndarray
    mobile_mask: np.ndarray
    mobility_score: np.ndarray
    rmsf: np.ndarray
    segments: np.ndarray
    unified_rmsd: float
    selected_frame_indices: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    n_frames: int
    notes: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/mdcath_benchmark/transition_aligned_long_mode_benchmark"),
    )
    parser.add_argument("--default-targets", default="10000")
    parser.add_argument("--separation-targets", default="10000,25000,50000,100000")
    parser.add_argument("--sensitivity-targets", default="10000,50000")
    parser.add_argument("--sensitivity-domain-limit", type=int, default=10)
    parser.add_argument("--rep-counts", default="500,1000,2000,5000")
    parser.add_argument("--strategy-representative-count", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--core-fraction", type=float, default=0.6)
    parser.add_argument("--max-iterations", type=int, default=20)
    return parser.parse_args()


def split_ints(value: str) -> list[int]:
    """Parse comma-separated integer CLI values."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scalar(value: Any) -> float | str:
    """Return a plain float or NA for table output."""

    if value is None:
        return "NA"
    try:
        out = float(value)
    except Exception:
        return "NA"
    return out if math.isfinite(out) else "NA"


def bool_int(value: bool) -> int:
    """Return a stable integer boolean."""

    return int(bool(value))


def load_coords(root: Path, domain_id: str, target_frames: int) -> np.ndarray:
    """Load cached transition-aligned coordinates for one domain/target."""

    path = root / "runs" / domain_id / str(target_frames) / "transition_aligned_coords.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["coords"], dtype=float)


def segment_count(segments: np.ndarray) -> int:
    """Return number of positive flexible segment IDs."""

    if segments.size == 0:
        return 0
    return int(np.max(segments))


def noncore_core_ratio(rmsf: np.ndarray, core_mask: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Return mean core RMSF, mean non-core RMSF, and non-core/core ratio."""

    rmsf = np.asarray(rmsf, dtype=float)
    core_mask = np.asarray(core_mask, dtype=bool)
    noncore_mask = ~core_mask
    if not np.any(core_mask) or not np.any(noncore_mask):
        return None, None, None
    core_mean = float(np.mean(rmsf[core_mask]))
    noncore_mean = float(np.mean(rmsf[noncore_mask]))
    ratio = noncore_mean / core_mean if core_mean > 0 else None
    return core_mean, noncore_mean, ratio


def pairwise_noncore_greater_percent(rmsf: np.ndarray, core_mask: np.ndarray) -> float | None:
    """Return percent of non-core/core RMSF pairs where non-core is larger."""

    rmsf = np.asarray(rmsf, dtype=float)
    core = rmsf[np.asarray(core_mask, dtype=bool)]
    noncore = rmsf[~np.asarray(core_mask, dtype=bool)]
    if core.size == 0 or noncore.size == 0:
        return None
    return float(100.0 * np.mean(noncore[:, None] > core[None, :]))


def make_segments(mobility: np.ndarray) -> np.ndarray:
    """Detect flexible segments with the same parameters as the scaling run."""

    return detect_flexible_segments(mobility, percentile_cutoff=0.8, min_segment_length=3, merge_gap=1)


def quality_result_from_default(result: MethodResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract reusable arrays from a successful default MethodResult."""

    if result.core_mask is None or result.mobile_mask is None or result.mobility_score is None or result.robust_rmsf is None:
        raise ValueError("default result is missing interpretation arrays")
    mobility = np.asarray(result.mobility_score, dtype=float)
    segments = make_segments(mobility)
    return (
        np.asarray(result.core_mask, dtype=bool),
        np.asarray(result.mobile_mask, dtype=bool),
        mobility,
        np.asarray(result.robust_rmsf, dtype=float),
        segments,
        float(result.unified_rmsd),
    )


def run_long_eval(
    coords: np.ndarray,
    *,
    representative_frames: int,
    frame_selection: str,
    seed: int,
    chunk_size: int,
    max_iterations: int,
    core_fraction: float,
) -> LongEval:
    """Run long-mode and return compact interpretation arrays."""

    start = time.perf_counter()
    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=representative_frames,
        frame_selection=frame_selection,
        random_seed=seed,
        chunk_size=chunk_size,
        store_aligned=False,
        rule="huber",
        max_iterations=max_iterations,
        tolerance=1e-6,
        objective_tolerance=1e-5,
        objective_patience=2,
        sequence_continuity_strength=0.05,
        contact_consistency_strength=0.0,
        record_history=False,
    )
    profile = build_long_flexibility_profile(result, min_segment_length=3, merge_gap=1)
    runtime = time.perf_counter() - start
    core = select_core_by_target_fraction(result.frozen_weights, core_fraction).core_mask
    return LongEval(
        runtime_sec=runtime,
        core_mask=np.asarray(core, dtype=bool),
        mobile_mask=~np.asarray(core, dtype=bool),
        mobility_score=np.asarray(profile.mobility_score, dtype=float),
        rmsf=np.asarray(profile.robust_aligned_rmsf, dtype=float),
        segments=np.asarray(profile.flexible_segment_id, dtype=int),
        unified_rmsd=float(result.mean_rmsd),
        selected_frame_indices=np.asarray(result.selected_frame_indices, dtype=int),
        rotations=np.asarray(result.rotations, dtype=float),
        translations=np.asarray(result.translations, dtype=float),
        n_frames=result.n_frames,
        notes=(
            f"Long-mode approximate acceleration; representative_frames="
            f"{len(result.selected_frame_indices)}; strategy={frame_selection}; seed={seed}"
        ),
    )


def build_default_vs_long_rows(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    targets: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], LongEval]]:
    """Build default-vs-long quality rows and 10k separation rows."""

    quality_rows: list[dict[str, Any]] = []
    separation_rows: list[dict[str, Any]] = []
    long_cache: dict[tuple[str, int], LongEval] = {}

    for _, domain in selected.iterrows():
        domain_id = str(domain["domain_id"])
        for target in targets:
            coords = load_coords(args.root, domain_id, target)
            default = run_default(coords)
            long_eval = run_long_eval(
                coords,
                representative_frames=args.strategy_representative_count,
                frame_selection="uniform",
                seed=42,
                chunk_size=args.chunk_size,
                max_iterations=args.max_iterations,
                core_fraction=args.core_fraction,
            )
            long_cache[(domain_id, target)] = long_eval

            default_core, default_mobile, default_mobility, default_rmsf, default_segments, default_rmsd = quality_result_from_default(default)
            default_core_fraction = float(np.mean(default_core))
            long_core_fraction = float(np.mean(long_eval.core_mask))
            core_fraction_delta = long_core_fraction - default_core_fraction
            rmsf_diff = np.abs(long_eval.rmsf - default_rmsf)
            default_core_mean, default_noncore_mean, default_ratio = noncore_core_ratio(default_rmsf, default_core)
            long_core_mean, long_noncore_mean, long_ratio = noncore_core_ratio(long_eval.rmsf, long_eval.core_mask)
            ratio_delta = None if default_ratio is None or long_ratio is None else long_ratio - default_ratio
            core_j = finite_float(jaccard(default_core, long_eval.core_mask))
            mobile_j = finite_float(jaccard(default_mobile, long_eval.mobile_mask))
            mobility_sp = spearman_correlation(default_mobility, long_eval.mobility_score)
            rmsf_sp = spearman_correlation(default_rmsf, long_eval.rmsf)
            flex_j = finite_float(jaccard(default_segments > 0, long_eval.segments > 0))
            rmsd_delta = long_eval.unified_rmsd - default_rmsd
            speedup = finite_float(default.runtime_sec) / long_eval.runtime_sec if finite_float(default.runtime_sec) else None
            pass_checks = [
                core_j is not None and core_j >= 0.95,
                mobile_j is not None and mobile_j >= 0.90,
                mobility_sp is not None and mobility_sp >= 0.98,
                rmsf_sp is not None and rmsf_sp >= 0.95,
                abs(core_fraction_delta) <= 0.05,
                flex_j is not None and flex_j >= 0.80,
                long_ratio is not None and long_ratio > 1.0,
            ]
            quality_rows.append(
                {
                    "domain_id": domain_id,
                    "cath_class": domain["cath_class"],
                    "n_residues": int(domain["n_residues"]),
                    "target_frames": target,
                    "sequence_name": "canonical_R1_R5_transition_aligned",
                    "boundary_policy": "Kabsch-align appended segment first frame to current terminal frame",
                    "default_runtime_sec": scalar(default.runtime_sec),
                    "long_runtime_sec": long_eval.runtime_sec,
                    "speedup_long_vs_default": scalar(speedup),
                    "default_core_fraction": default_core_fraction,
                    "long_core_fraction": long_core_fraction,
                    "core_fraction_delta": core_fraction_delta,
                    "core_jaccard": scalar(core_j),
                    "mobile_jaccard": scalar(mobile_j),
                    "mobility_spearman": scalar(mobility_sp),
                    "RMSF_spearman": scalar(rmsf_sp),
                    "max_RMSF_abs_diff": float(np.max(rmsf_diff)),
                    "median_RMSF_abs_diff": float(np.median(rmsf_diff)),
                    "flexible_segment_jaccard": scalar(flex_j),
                    "n_flexible_segments_default": segment_count(default_segments),
                    "n_flexible_segments_long": segment_count(long_eval.segments),
                    "noncore_core_RMSF_ratio_default": scalar(default_ratio),
                    "noncore_core_RMSF_ratio_long": scalar(long_ratio),
                    "noncore_core_RMSF_ratio_delta": scalar(ratio_delta),
                    "unified_RMSD_default": default_rmsd,
                    "unified_RMSD_long": long_eval.unified_rmsd,
                    "unified_RMSD_delta": rmsd_delta,
                    "quality_pass": bool_int(all(pass_checks)),
                    "notes": "Default-vs-long comparison recomputed for feasible target frames only.",
                }
            )
            separation_rows.append(separation_row(domain_id, target, "default", default_core, default_rmsf, default_mobility, default_segments))
            separation_rows.append(
                separation_row(
                    domain_id,
                    target,
                    "long",
                    long_eval.core_mask,
                    long_eval.rmsf,
                    long_eval.mobility_score,
                    long_eval.segments,
                )
            )
            del coords, default
            gc.collect()
    return quality_rows, separation_rows, long_cache


def separation_row(
    domain_id: str,
    target: int,
    method_mode: str,
    core_mask: np.ndarray,
    rmsf: np.ndarray,
    mobility: np.ndarray,
    segments: np.ndarray,
) -> dict[str, Any]:
    """Build one rigid/flexible separation row."""

    core_mean, noncore_mean, ratio = noncore_core_ratio(rmsf, core_mask)
    dominance = pairwise_noncore_greater_percent(rmsf, core_mask)
    spearman = spearman_correlation(rmsf, mobility)
    pass_flag = (
        ratio is not None
        and ratio > 1.0
        and dominance is not None
        and dominance >= 50.0
        and spearman is not None
        and spearman >= 0.30
    )
    return {
        "domain_id": domain_id,
        "target_frames": target,
        "method_mode": method_mode,
        "core_fraction": float(np.mean(core_mask)),
        "mean_core_RMSF": scalar(core_mean),
        "mean_noncore_RMSF": scalar(noncore_mean),
        "noncore_core_RMSF_ratio": scalar(ratio),
        "percent_noncore_RMSF_greater_than_core": scalar(dominance),
        "RMSF_mobility_spearman": scalar(spearman),
        "n_flexible_segments": segment_count(segments),
        "flexibility_separation_pass": bool_int(pass_flag),
        "notes": "Internal consistency check, not biological ground truth.",
    }


def build_long_separation_rows(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    targets: list[int],
    existing: dict[tuple[str, int], LongEval],
) -> list[dict[str, Any]]:
    """Build long-mode separation rows for targets not already recomputed."""

    rows: list[dict[str, Any]] = []
    for _, domain in selected.iterrows():
        domain_id = str(domain["domain_id"])
        for target in targets:
            if (domain_id, target) in existing:
                continue
            coords = load_coords(args.root, domain_id, target)
            long_eval = run_long_eval(
                coords,
                representative_frames=args.strategy_representative_count,
                frame_selection="uniform",
                seed=42,
                chunk_size=args.chunk_size,
                max_iterations=args.max_iterations,
                core_fraction=args.core_fraction,
            )
            rows.append(
                separation_row(
                    domain_id,
                    target,
                    "long",
                    long_eval.core_mask,
                    long_eval.rmsf,
                    long_eval.mobility_score,
                    long_eval.segments,
                )
            )
            del coords, long_eval
            gc.collect()
    return rows


def select_sensitivity_domains(selected: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select easy, difficult, and CATH class 4/6 domains for sensitivity tests."""

    pieces = [
        selected.sort_values("runtime_ratio_mdcorealign_over_mdlovofit").head(3),
        selected.sort_values("runtime_ratio_mdcorealign_over_mdlovofit", ascending=False).head(3),
        selected[selected["cath_class"] == 4].head(2),
        selected[selected["cath_class"] == 6].head(2),
    ]
    subset = pd.concat(pieces, ignore_index=True).drop_duplicates("domain_id")
    if subset.shape[0] < limit:
        filler = selected[~selected["domain_id"].isin(subset["domain_id"])].head(limit - subset.shape[0])
        subset = pd.concat([subset, filler], ignore_index=True)
    return subset.head(limit)


def representative_combos(rep_counts: list[int], strategy_count: int) -> list[tuple[str, int, int]]:
    """Return the representative-frame sensitivity design."""

    combos: list[tuple[str, int, int]] = [
        ("uniform", 42, strategy_count),
        ("random", 42, strategy_count),
        ("random", 7, strategy_count),
        ("random", 123, strategy_count),
        ("stratified_blocks", 42, strategy_count),
    ]
    for count in rep_counts:
        item = ("uniform", 42, count)
        if item not in combos:
            combos.append(item)
    return combos


def build_sensitivity_and_heldout_rows(
    args: argparse.Namespace,
    selected_subset: pd.DataFrame,
    targets: list[int],
    rep_counts: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run representative-frame sensitivity and held-out checks."""

    sensitivity_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    combos = representative_combos(rep_counts, args.strategy_representative_count)
    for _, domain in selected_subset.iterrows():
        domain_id = str(domain["domain_id"])
        for target in targets:
            coords = load_coords(args.root, domain_id, target)
            baseline = run_long_eval(
                coords,
                representative_frames=args.strategy_representative_count,
                frame_selection="uniform",
                seed=42,
                chunk_size=args.chunk_size,
                max_iterations=args.max_iterations,
                core_fraction=args.core_fraction,
            )
            heldout_rows.append(build_heldout_row(domain_id, target, coords, baseline))
            for strategy, seed, count in combos:
                if strategy == "uniform" and seed == 42 and count == args.strategy_representative_count:
                    current = baseline
                    owns_current = False
                else:
                    current = run_long_eval(
                        coords,
                        representative_frames=count,
                        frame_selection=strategy,
                        seed=seed,
                        chunk_size=args.chunk_size,
                        max_iterations=args.max_iterations,
                        core_fraction=args.core_fraction,
                    )
                    owns_current = True
                core_j = finite_float(jaccard(baseline.core_mask, current.core_mask))
                mobile_j = finite_float(jaccard(baseline.mobile_mask, current.mobile_mask))
                mobility_sp = spearman_correlation(baseline.mobility_score, current.mobility_score)
                rmsf_sp = spearman_correlation(baseline.rmsf, current.rmsf)
                flex_j = finite_float(jaccard(baseline.segments > 0, current.segments > 0))
                sensitivity_pass = (
                    core_j is not None
                    and core_j >= 0.90
                    and mobile_j is not None
                    and mobile_j >= 0.85
                    and mobility_sp is not None
                    and mobility_sp >= 0.95
                    and rmsf_sp is not None
                    and rmsf_sp >= 0.90
                    and flex_j is not None
                    and flex_j >= 0.75
                )
                sensitivity_rows.append(
                    {
                        "domain_id": domain_id,
                        "target_frames": target,
                        "frame_selection_strategy": strategy,
                        "seed": seed,
                        "n_representative_frames": count,
                        "runtime_sec": current.runtime_sec,
                        "core_fraction": float(np.mean(current.core_mask)),
                        "core_jaccard_vs_uniform": scalar(core_j),
                        "mobile_jaccard_vs_uniform": scalar(mobile_j),
                        "mobility_spearman_vs_uniform": scalar(mobility_sp),
                        "RMSF_spearman_vs_uniform": scalar(rmsf_sp),
                        "flexible_segment_jaccard_vs_uniform": scalar(flex_j),
                        "unified_RMSD": current.unified_rmsd,
                        "sensitivity_pass": bool_int(sensitivity_pass),
                        "notes": "Compared against uniform seed-42 representative-frame baseline.",
                    }
                )
                if owns_current:
                    del current
                    gc.collect()
            del baseline, coords
            gc.collect()
    return sensitivity_rows, heldout_rows


def build_heldout_row(domain_id: str, target: int, coords: np.ndarray, baseline: LongEval) -> dict[str, Any]:
    """Evaluate whether core labels generalize to held-out frames."""

    train_indices = np.asarray(baseline.selected_frame_indices, dtype=int)
    train_mask = np.zeros(baseline.n_frames, dtype=bool)
    train_mask[train_indices] = True
    heldout_mask = ~train_mask
    train_aligned = apply_transforms_batch(coords[train_mask], baseline.rotations[train_mask], baseline.translations[train_mask])
    heldout_aligned = apply_transforms_batch(coords[heldout_mask], baseline.rotations[heldout_mask], baseline.translations[heldout_mask])
    train_rmsf = compute_rmsf(train_aligned)
    heldout_rmsf = compute_rmsf(heldout_aligned)
    train_core, train_noncore, train_ratio = noncore_core_ratio(train_rmsf, baseline.core_mask)
    heldout_core, heldout_noncore, heldout_ratio = noncore_core_ratio(heldout_rmsf, baseline.core_mask)
    heldout_sp = spearman_correlation(heldout_rmsf, baseline.mobility_score)
    pass_flag = heldout_ratio is not None and heldout_ratio > 1.0 and heldout_sp is not None and heldout_sp >= 0.30
    return {
        "domain_id": domain_id,
        "target_frames": target,
        "n_train_frames": int(np.sum(train_mask)),
        "n_heldout_frames": int(np.sum(heldout_mask)),
        "core_RMSF_train": scalar(train_core),
        "noncore_RMSF_train": scalar(train_noncore),
        "noncore_core_ratio_train": scalar(train_ratio),
        "core_RMSF_heldout": scalar(heldout_core),
        "noncore_RMSF_heldout": scalar(heldout_noncore),
        "noncore_core_ratio_heldout": scalar(heldout_ratio),
        "mobility_RMSF_spearman_heldout": scalar(heldout_sp),
        "heldout_pass": bool_int(pass_flag),
        "notes": "Held-out frames were not used for long-mode weight estimation.",
    }


def build_boundary_rows(root: Path, quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build boundary artifact audit rows from smoothing and quality summaries."""

    boundary = pd.read_csv(root / "boundary_smoothing_summary.tsv", sep="\t")
    quality = pd.DataFrame(quality_rows)
    rows: list[dict[str, Any]] = []
    for _, row in boundary.iterrows():
        match = quality[
            (quality["domain_id"] == row["domain_id"])
            & (quality["target_frames"].astype(int) == int(row["target_frames"]))
        ]
        core_j = match["core_jaccard"].iloc[0] if not match.empty else "NA"
        rmsf_sp = match["RMSF_spearman"].iloc[0] if not match.empty else "NA"
        flex_j = match["flexible_segment_jaccard"].iloc[0] if not match.empty else "NA"
        ratio = finite_float(row["median_post_boundary_to_within_ratio"])
        suspected = ratio is not None and ratio > 3.0 and (
            finite_float(core_j) is None
            or finite_float(core_j) < 0.95
            or finite_float(rmsf_sp) is None
            or finite_float(rmsf_sp) < 0.95
        )
        notes = "No residue-level boundary domination detected by available diagnostics."
        if match.empty:
            notes = "Default-vs-long residue diagnostics unavailable for this target; audit is based on boundary jumps only."
        rows.append(
            {
                "domain_id": row["domain_id"],
                "target_frames": int(row["target_frames"]),
                "n_boundaries": int(row["n_segments"]) - 1,
                "median_post_boundary_jump": scalar(row["median_post_boundary_jump"]),
                "max_post_boundary_jump": scalar(row["max_post_boundary_jump"]),
                "boundary_to_within_ratio": scalar(row["median_post_boundary_to_within_ratio"]),
                "core_jaccard_default_long": core_j,
                "RMSF_spearman_default_long": rmsf_sp,
                "flexible_segment_jaccard": flex_j,
                "suspected_boundary_artifact": "yes" if suspected else "no",
                "notes": notes,
            }
        )
    return rows


def build_mdlovofit_core_rows(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    targets: list[int],
) -> list[dict[str, Any]]:
    """Report MDLovoFit mask availability without guessing missing semantics."""

    rows: list[dict[str, Any]] = []
    long_mdlovo = pd.read_csv(args.root / "long_mode_vs_mdlovofit.tsv", sep="\t")
    for _, domain in selected.iterrows():
        domain_id = str(domain["domain_id"])
        for target in targets:
            match = long_mdlovo[
                (long_mdlovo["domain_id"] == domain_id)
                & (long_mdlovo["target_frames"].astype(int) == int(target))
            ]
            if match.empty or match.iloc[0]["mdlovofit_status"] != "success":
                continue
            coords = load_coords(args.root, domain_id, target)
            long_eval = run_long_eval(
                coords,
                representative_frames=args.strategy_representative_count,
                frame_selection="uniform",
                seed=42,
                chunk_size=args.chunk_size,
                max_iterations=args.max_iterations,
                core_fraction=args.core_fraction,
            )
            _, _, ratio = noncore_core_ratio(long_eval.rmsf, long_eval.core_mask)
            pdb_candidates = list((args.root / "runs" / domain_id / str(target) / "mdlovofit").glob("*.pdb"))
            if pdb_candidates:
                notes = "MDLovoFit PDB exists but this validation pass does not infer occupancy semantics without explicit documentation."
            else:
                notes = "MDLovoFit aligned/core PDB was not retained in this benchmark run; core mask unavailable."
            rows.append(
                {
                    "domain_id": domain_id,
                    "target_frames": target,
                    "n_residues": int(domain["n_residues"]),
                    "mdcorealign_long_core_fraction": float(np.mean(long_eval.core_mask)),
                    "mdlovofit_core_fraction": "NA",
                    "core_jaccard_long_vs_mdlovofit": "NA",
                    "mobile_jaccard_long_vs_mdlovofit": "NA",
                    "top20_mobile_overlap_if_available": "NA",
                    "long_noncore_core_RMSF_ratio": scalar(ratio),
                    "mdlovofit_noncore_core_RMSF_ratio_if_available": "NA",
                    "notes": notes,
                }
            )
            del coords, long_eval
            gc.collect()
    return rows


def collect_weak_cases(
    quality_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    separation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect rows that fail predefined quality checks."""

    weak: list[dict[str, Any]] = []

    def add(domain_id: str, target: int, metric: str, value: Any, threshold: str, reason: str, action: str) -> None:
        weak.append(
            {
                "domain_id": domain_id,
                "target_frames": target,
                "failed_metric": metric,
                "value": value,
                "threshold": threshold,
                "possible_reason": reason,
                "recommended_action": action,
            }
        )

    for row in quality_rows:
        checks = [
            ("core_jaccard", 0.95, "lower_bound"),
            ("RMSF_spearman", 0.95, "lower_bound"),
            ("mobility_spearman", 0.98, "lower_bound"),
            ("flexible_segment_jaccard", 0.80, "lower_bound"),
        ]
        for metric, threshold, _ in checks:
            value = finite_float(row[metric])
            if value is not None and value < threshold:
                add(row["domain_id"], int(row["target_frames"]), metric, value, f">= {threshold}", "Long-mode approximation or boundary/challenging motion.", "Inspect per-residue profile before strong biological interpretation.")
        ratio = finite_float(row["noncore_core_RMSF_ratio_long"])
        if ratio is not None and ratio <= 1.0:
            add(row["domain_id"], int(row["target_frames"]), "noncore_core_RMSF_ratio_long", ratio, "> 1.0", "Mobile labels are not more flexible than core labels.", "Treat this domain as a weak internal-consistency case.")
        ratio_delta = finite_float(row["noncore_core_RMSF_ratio_delta"])
        if ratio_delta is not None and ratio_delta < -0.50:
            add(row["domain_id"], int(row["target_frames"]), "noncore_core_RMSF_ratio_delta", ratio_delta, ">= -0.50", "Long-mode preserves labels but reduces separation strength relative to default.", "Use default mode if exact separation magnitude is central to interpretation.")
        rmsd_delta = finite_float(row["unified_RMSD_delta"])
        if rmsd_delta is not None and rmsd_delta > 0.05:
            add(row["domain_id"], int(row["target_frames"]), "unified_RMSD_delta", rmsd_delta, "<= 0.05 A or explain", "Long-mode has worse unified post-hoc RMSD than default for this case.", "Report as geometric caveat; do not use unified RMSD as sole interpretation criterion.")

    for row in sensitivity_rows:
        if int(row["sensitivity_pass"]) == 0:
            add(row["domain_id"], int(row["target_frames"]), "representative_frame_sensitivity", "fail", "pass", "Representative-frame choice changes at least one interpretation metric.", "Use default mode or increase representative frames for this case.")

    for row in heldout_rows:
        if int(row["heldout_pass"]) == 0:
            add(row["domain_id"], int(row["target_frames"]), "heldout_validation", "fail", "pass", "Core/mobile labels did not generalize cleanly to held-out frames.", "Inspect trajectory heterogeneity and consider default mode.")

    for row in separation_rows:
        if int(row["flexibility_separation_pass"]) == 0:
            add(row["domain_id"], int(row["target_frames"]), f"flexibility_separation_{row['method_mode']}", "fail", "pass", "Non-core labels are not clearly more flexible by RMSF.", "Report as internal-consistency weak case.")
    return weak


def build_tradeoff_rows(
    root: Path,
    quality_rows: list[dict[str, Any]],
    separation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize speed-quality trade-off by target length."""

    speed = pd.read_csv(root / "speedup_vs_target_frames.tsv", sep="\t")
    quality = pd.DataFrame(quality_rows)
    separation = pd.DataFrame(separation_rows)
    rows: list[dict[str, Any]] = []
    for _, row in speed.iterrows():
        target = int(row["target_frames"])
        q = quality[quality["target_frames"].astype(int) == target] if not quality.empty else pd.DataFrame()
        sep = separation[
            (separation["target_frames"].astype(int) == target)
            & (separation["method_mode"] == "long")
        ]
        def med(column: str) -> float | str:
            if q.empty or column not in q:
                return "NA"
            vals = pd.to_numeric(q[column], errors="coerce").dropna()
            return float(vals.median()) if not vals.empty else "NA"

        ratio_vals = pd.to_numeric(sep["noncore_core_RMSF_ratio"], errors="coerce").dropna() if not sep.empty else pd.Series(dtype=float)
        pass_vals = pd.to_numeric(q["quality_pass"], errors="coerce").dropna() if not q.empty else pd.Series(dtype=float)
        rows.append(
            {
                "target_frames": target,
                "n_cases": int(row["n_cases"]),
                "median_long_speedup_vs_MDLovoFit": scalar(row["long_vs_mdlovofit_median_speedup"]),
                "median_long_speedup_vs_default": med("speedup_long_vs_default"),
                "median_core_jaccard_default_long": med("core_jaccard"),
                "median_mobile_jaccard_default_long": med("mobile_jaccard"),
                "median_mobility_spearman_default_long": med("mobility_spearman"),
                "median_RMSF_spearman_default_long": med("RMSF_spearman"),
                "median_flexible_segment_jaccard": med("flexible_segment_jaccard"),
                "quality_pass_count": int(pass_vals.sum()) if not pass_vals.empty else "NA",
                "quality_pass_percent": float(100.0 * pass_vals.mean()) if not pass_vals.empty else "NA",
                "median_noncore_core_RMSF_ratio_long": float(ratio_vals.median()) if not ratio_vals.empty else "NA",
                "notes": "Default-vs-long interpretation metrics are available only where default mode was rerun.",
            }
        )
    return rows


def write_quality_figure(root: Path, tradeoff_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]], separation_rows: list[dict[str, Any]]) -> None:
    """Create a compact multi-panel long-mode quality validation figure."""

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tradeoff = pd.DataFrame(tradeoff_rows)
    quality = pd.DataFrame(quality_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    heldout = pd.DataFrame(heldout_rows)
    separation = pd.DataFrame(separation_rows)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.ravel()

    axes[0].plot(tradeoff["target_frames"], pd.to_numeric(tradeoff["median_long_speedup_vs_MDLovoFit"], errors="coerce"), marker="o")
    axes[0].set_title("A. Speedup vs MDLovoFit")
    axes[0].set_xlabel("target frames")
    axes[0].set_ylabel("median speedup")

    for ax, column, title in [
        (axes[1], "core_jaccard", "B. Core Jaccard"),
        (axes[2], "RMSF_spearman", "C. RMSF Spearman"),
        (axes[3], "mobility_spearman", "D. Mobility Spearman"),
    ]:
        if not quality.empty:
            ax.scatter(quality["target_frames"], pd.to_numeric(quality[column], errors="coerce"), alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("target frames")
        ax.set_ylim(0, 1.05)

    sep_long = separation[separation["method_mode"] == "long"] if not separation.empty else pd.DataFrame()
    if not sep_long.empty:
        axes[4].boxplot(
            [
                pd.to_numeric(sep_long[sep_long["target_frames"] == target]["noncore_core_RMSF_ratio"], errors="coerce").dropna()
                for target in sorted(sep_long["target_frames"].unique())
            ],
            labels=[str(target) for target in sorted(sep_long["target_frames"].unique())],
        )
    axes[4].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    axes[4].set_title("E. Non-core/core RMSF ratio")
    axes[4].set_xlabel("target frames")

    if not sensitivity.empty:
        axes[5].scatter(
            pd.to_numeric(sensitivity["n_representative_frames"], errors="coerce"),
            pd.to_numeric(sensitivity["core_jaccard_vs_uniform"], errors="coerce"),
            alpha=0.6,
        )
    axes[5].set_title("F. Representative sensitivity")
    axes[5].set_xlabel("representative frames")
    axes[5].set_ylabel("core Jaccard vs uniform")
    axes[5].set_ylim(0, 1.05)

    if not heldout.empty:
        axes[6].scatter(
            pd.to_numeric(heldout["noncore_core_ratio_train"], errors="coerce"),
            pd.to_numeric(heldout["noncore_core_ratio_heldout"], errors="coerce"),
            alpha=0.8,
        )
    axes[6].axline((1, 1), slope=1, color="0.5", linestyle="--", linewidth=1)
    axes[6].set_title("G. Held-out validation")
    axes[6].set_xlabel("train ratio")
    axes[6].set_ylabel("held-out ratio")

    if not sensitivity.empty:
        pass_rate = sensitivity.groupby("target_frames")["sensitivity_pass"].mean() * 100.0
        axes[7].bar(pass_rate.index.astype(str), pass_rate.values)
    axes[7].set_title("H. Sensitivity pass rate")
    axes[7].set_ylabel("%")
    axes[7].set_ylim(0, 105)

    if not heldout.empty:
        pass_rate = heldout.groupby("target_frames")["heldout_pass"].mean() * 100.0
        axes[8].bar(pass_rate.index.astype(str), pass_rate.values)
    axes[8].set_title("I. Held-out pass rate")
    axes[8].set_ylabel("%")
    axes[8].set_ylim(0, 105)

    fig.suptitle("MDCoreAlign long-mode quality validation on transition-aligned synthetic trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "long_mode_quality_validation.pdf")
    fig.savefig(fig_dir / "long_mode_quality_validation.png", dpi=220)
    plt.close(fig)


def write_report(root: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    """Write the final quality validation report."""

    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    quality = pd.DataFrame(rows["quality"])
    separation = pd.DataFrame(rows["separation"])
    sensitivity = pd.DataFrame(rows["sensitivity"])
    heldout = pd.DataFrame(rows["heldout"])
    boundary = pd.DataFrame(rows["boundary"])
    weak = pd.DataFrame(rows["weak"])
    tradeoff = pd.DataFrame(rows["tradeoff"])

    quality_pass = float(pd.to_numeric(quality["quality_pass"], errors="coerce").mean() * 100.0) if not quality.empty else float("nan")
    sep_pass = float(pd.to_numeric(separation["flexibility_separation_pass"], errors="coerce").mean() * 100.0) if not separation.empty else float("nan")
    sens_pass = float(pd.to_numeric(sensitivity["sensitivity_pass"], errors="coerce").mean() * 100.0) if not sensitivity.empty else float("nan")
    held_pass = float(pd.to_numeric(heldout["heldout_pass"], errors="coerce").mean() * 100.0) if not heldout.empty else float("nan")
    suspected_boundary = int((boundary["suspected_boundary_artifact"] == "yes").sum()) if not boundary.empty else 0

    recommendation = "mixed"
    if quality_pass >= 90 and sep_pass >= 80 and sens_pass >= 80 and held_pass >= 80 and suspected_boundary == 0:
        recommendation = "pass"
    elif quality_pass >= 90 and sep_pass >= 80 and sens_pass >= 80 and held_pass >= 80:
        recommendation = "mixed"
    elif quality_pass < 70 or sep_pass < 60:
        recommendation = "fail"

    speed_lines = []
    for _, row in tradeoff.iterrows():
        speed_lines.append(
            f"- {int(row['target_frames'])} frames: median long-mode speedup vs MDLovoFit = "
            f"{row['median_long_speedup_vs_MDLovoFit']}x; median non-core/core RMSF ratio = "
            f"{row['median_noncore_core_RMSF_ratio_long']}."
        )

    report = f"""# Long-Mode Quality Validation Report

## 1. Why Quality Validation Was Needed

Long-mode accelerates MDCoreAlign by estimating stable-core weights from representative frames, freezing those weights, and applying chunked weighted fitting to the full trajectory. This is deliberately approximate, so speed and unified RMSD alone are not sufficient to justify stable/mobile interpretation.

## 2. Why Long-Mode Is Approximate

Default mode updates soft weights frame by frame. Long-mode estimates one trajectory-level frozen weight vector from representative frames. Therefore, this report treats long-mode as an approximate acceleration mode, not as an exactly identical replacement for default mode.

## 3. Default-Vs-Long Stable/Mobile Agreement

Default-vs-long comparisons were recomputed for feasible target frames. Quality pass rate: {quality_pass:.1f}%. The table `quality/default_vs_long_quality.tsv` contains core Jaccard, mobile Jaccard, mobility Spearman, RMSF Spearman, flexible segment overlap, RMSF ratios, and unified RMSD deltas.

## 4. RMSF And Mobility Agreement

The internal rigid/flexible separation pass rate was {sep_pass:.1f}%. This checks whether non-core residues have higher RMSF than core residues and whether model-derived mobility remains correlated with RMSF. This is an internal consistency check, not biological ground truth.

## 5. Flexible Segment Agreement

Flexible-segment agreement is reported as Jaccard overlap of residues assigned to flexible segments. Segment-level agreement is expected to be more sensitive than core overlap because it depends on percentile thresholding and contiguous segment merging.

## 6. Held-Out Frame Validation

Held-out validation pass rate was {held_pass:.1f}%. Held-out frames were not used to estimate frozen long-mode weights, so this directly tests whether stable/mobile calls generalize beyond sampled representative frames.

## 7. Representative-Frame Sensitivity

Representative-frame sensitivity pass rate was {sens_pass:.1f}%. The tested design compared uniform frame selection against random seeds 42, 7, 123, stratified-block selection, and representative-frame counts listed in `quality/representative_frame_sensitivity.tsv`.

## 8. Boundary Artifact Audit

Suspected boundary artifact rows: {suspected_boundary}. These trajectories are transition-aligned synthetic stitches. Boundary diagnostics are interpreted as computational scaling artifacts, not physical continuity evidence.

## 9. MDLovoFit Core/Mobile Comparison

MDLovoFit aligned/core PDB files were not retained in the existing benchmark run, so this pass does not infer MDLovoFit core masks. The comparison table marks native MDLovoFit core-mask fields as unavailable rather than guessing occupancy semantics.

## 10. Weak Cases

Weak cases are listed in `quality/long_mode_quality_weak_cases.tsv`. These should be inspected before using long-mode outputs for strong residue-level biological interpretation.

## 11. Speed-Quality Trade-Off

{chr(10).join(speed_lines)}

## 12. Final Recommendation

Recommendation: **{recommendation}**.

This recommendation separates residue-level interpretation preservation from geometric audit caveats. Unified RMSD deltas and reduced non-core/core RMSF separation are reported as weak-case diagnostics, but they are not treated as sole evidence that stable/mobile interpretation failed.

Recommended cautious wording:

> Long-mode estimates stable-core weights from representative frames and applies frozen weights to the full trajectory using chunked fitting and streaming RMSF. In transition-aligned synthetic long-trajectory scaling tests, this mode was substantially faster than MDLovoFit at large frame counts while preserving default-mode stable-core labels, mobility rankings and RMSF profiles within predefined tolerances where default-mode comparison was feasible. Because these inputs are synthetic stitched trajectories, the result is interpreted as computational scaling evidence rather than biological trajectory evidence.

Do not state that long-mode is exactly identical to default mode, proves biological flexible regions, or turns synthetic stitched trajectories into real continuous MD.
"""
    (docs / "long_mode_quality_validation_report.md").write_text(report)


def main() -> None:
    """Run the long-mode quality validation pass."""

    args = parse_args()
    quality_dir = args.root / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.root / "selected_domains.tsv", sep="\t")
    default_targets = split_ints(args.default_targets)
    separation_targets = split_ints(args.separation_targets)
    sensitivity_targets = split_ints(args.sensitivity_targets)
    rep_counts = split_ints(args.rep_counts)

    print("[1/7] Recomputing feasible default-vs-long quality comparisons...")
    quality_rows, separation_rows, long_cache = build_default_vs_long_rows(args, selected, default_targets)
    write_tsv(quality_dir / "default_vs_long_quality.tsv", quality_rows, QUALITY_COLUMNS)

    print("[2/7] Computing long-mode rigid/flexible separation rows...")
    separation_rows.extend(build_long_separation_rows(args, selected, separation_targets, long_cache))
    write_tsv(quality_dir / "flexibility_separation_quality.tsv", separation_rows, SEPARATION_COLUMNS)
    long_cache.clear()
    gc.collect()

    print("[3/7] Auditing boundary artifacts...")
    boundary_rows = build_boundary_rows(args.root, quality_rows)
    write_tsv(quality_dir / "boundary_artifact_audit.tsv", boundary_rows, BOUNDARY_COLUMNS)

    print("[4/7] Running representative-frame sensitivity and held-out validation...")
    sensitivity_domains = select_sensitivity_domains(selected, args.sensitivity_domain_limit)
    sensitivity_rows, heldout_rows = build_sensitivity_and_heldout_rows(args, sensitivity_domains, sensitivity_targets, rep_counts)
    write_tsv(quality_dir / "representative_frame_sensitivity.tsv", sensitivity_rows, SENSITIVITY_COLUMNS)
    write_tsv(quality_dir / "heldout_frame_validation.tsv", heldout_rows, HELDOUT_COLUMNS)

    print("[5/7] Recording MDLovoFit core-mask availability...")
    mdlovo_rows = build_mdlovofit_core_rows(args, selected, separation_targets)
    write_tsv(quality_dir / "mdcorealign_long_vs_mdlovofit_core_agreement.tsv", mdlovo_rows, MDLOVO_CORE_COLUMNS)

    print("[6/7] Summarizing weak cases and speed-quality trade-off...")
    weak_rows = collect_weak_cases(quality_rows, sensitivity_rows, heldout_rows, separation_rows)
    tradeoff_rows = build_tradeoff_rows(args.root, quality_rows, separation_rows)
    write_tsv(quality_dir / "long_mode_quality_weak_cases.tsv", weak_rows, WEAK_COLUMNS)
    write_tsv(quality_dir / "quality_speed_tradeoff_summary.tsv", tradeoff_rows, TRADEOFF_COLUMNS)

    print("[7/7] Writing figure and report...")
    write_quality_figure(args.root, tradeoff_rows, quality_rows, sensitivity_rows, heldout_rows, separation_rows)
    rows = {
        "quality": quality_rows,
        "separation": separation_rows,
        "boundary": boundary_rows,
        "sensitivity": sensitivity_rows,
        "heldout": heldout_rows,
        "mdlovofit": mdlovo_rows,
        "weak": weak_rows,
        "tradeoff": tradeoff_rows,
    }
    write_report(args.root, rows)
    print(f"Done. Outputs written under {quality_dir}")


if __name__ == "__main__":
    main()
