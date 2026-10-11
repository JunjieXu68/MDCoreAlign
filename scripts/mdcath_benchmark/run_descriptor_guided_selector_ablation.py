#!/usr/bin/env python
"""Ablate descriptor-guided representative-frame selectors for long-mode.

This is a prototype optimization benchmark. Descriptor/ANN methods only select
representative frames for long-mode weight estimation; exact weighted Kabsch
alignment is still used for all fitted frames.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.flexibility import compute_rmsf, detect_flexible_segments, spearman_correlation
from mdsuperpose.long_mode import build_long_flexibility_profile, solve_long_trajectory_alignment
from scripts.mdcath_benchmark.mdcath_common import append_tsv, write_json, write_tsv
from scripts.mdcath_benchmark.run_transition_aligned_long_mode_benchmark import finite_float, jaccard


ABLATION_COLUMNS = [
    "domain_id",
    "cath_class",
    "n_residues",
    "target_frames",
    "selector",
    "representative_frame_count",
    "boundary_exclusion",
    "boundary_exclusion_window",
    "n_stitched_boundaries",
    "n_boundary_frames_excluded",
    "status",
    "runtime_sec",
    "total_runtime_including_selector_sec",
    "selector_runtime_sec",
    "weight_estimation_runtime_sec",
    "full_long_mode_runtime_sec",
    "baseline_runtime_sec",
    "speedup_vs_current_long_mode",
    "core_jaccard_vs_long_baseline",
    "mobile_jaccard_vs_long_baseline",
    "mobility_spearman_vs_long_baseline",
    "RMSF_spearman_vs_long_baseline",
    "flexible_segment_jaccard_vs_long_baseline",
    "noncore_core_RMSF_ratio",
    "heldout_noncore_core_RMSF_ratio",
    "heldout_RMSF_mobility_spearman",
    "heldout_pass",
    "unified_RMSD",
    "baseline_unified_RMSD",
    "unified_RMSD_delta",
    "quality_pass",
    "selected_frame_indices_path",
    "descriptor_metadata_path",
    "notes",
]

METADATA_COLUMNS = [
    "domain_id",
    "target_frames",
    "selector",
    "representative_frame_count",
    "boundary_exclusion",
    "boundary_exclusion_window",
    "status",
    "strategy",
    "selector_kind",
    "n_frames",
    "n_atoms",
    "requested_frames",
    "selected_frames",
    "candidate_pool_size",
    "raw_descriptor_size",
    "compressed_dim",
    "compression",
    "compression_method",
    "descriptor_runtime_seconds",
    "selection_runtime_seconds",
    "n_possible_long_range_pairs",
    "n_contact_changing_pairs",
    "pair_sampling_mode",
    "uses_raw_xyz_descriptor",
    "boundary_exclusion_enabled",
    "n_excluded_frames",
    "notes",
]

SUMMARY_COLUMNS = [
    "selector",
    "representative_frame_count",
    "boundary_exclusion",
    "target_frames",
    "n_cases",
    "n_success",
    "median_runtime_sec",
    "median_selector_runtime_sec",
    "median_weight_estimation_runtime_sec",
    "median_full_long_mode_runtime_sec",
    "median_speedup_vs_current_long_mode",
    "median_core_jaccard",
    "median_mobile_jaccard",
    "median_mobility_spearman",
    "median_RMSF_spearman",
    "median_flexible_segment_jaccard",
    "median_noncore_core_RMSF_ratio",
    "quality_pass_percent",
    "recommendation",
    "notes",
]


class LongArrays:
    """Compact long-mode arrays used for selector quality comparison."""

    def __init__(
        self,
        *,
        runtime_sec: float,
        core_mask: np.ndarray,
        mobility: np.ndarray,
        rmsf: np.ndarray,
        segments: np.ndarray,
        unified_rmsd: float,
        selected: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray,
        metadata: dict[str, Any],
        selector_runtime_sec: float,
        weight_estimation_runtime_sec: float,
        full_long_mode_runtime_sec: float,
    ) -> None:
        self.runtime_sec = runtime_sec
        self.core_mask = core_mask
        self.mobile_mask = ~core_mask
        self.mobility = mobility
        self.rmsf = rmsf
        self.segments = segments
        self.unified_rmsd = unified_rmsd
        self.selected = selected
        self.rotations = rotations
        self.translations = translations
        self.metadata = metadata
        self.selector_runtime_sec = selector_runtime_sec
        self.weight_estimation_runtime_sec = weight_estimation_runtime_sec
        self.full_long_mode_runtime_sec = full_long_mode_runtime_sec


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-root", type=Path, default=Path("results/mdcath_benchmark/transition_aligned_long_mode_benchmark"))
    parser.add_argument("--output-root", type=Path, default=Path("results/long_mode_optimization_v2"))
    parser.add_argument("--selectors", default="uniform,random,distance_sketch_farthest_first,pca_kmedoids,annoy,faiss")
    parser.add_argument("--targets", default="10000,50000,100000")
    parser.add_argument("--representative-counts", default="250,500,1000,2000")
    parser.add_argument("--domain-limit", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=0, help="Optional safety cap on selector runs; 0 means no cap.")
    parser.add_argument("--baseline-representative-count", type=int, default=2000)
    parser.add_argument("--raw-descriptor-size", type=int, default=1024)
    parser.add_argument("--compressed-dim", type=int, default=96)
    parser.add_argument("--candidate-pool-size", type=int, default=3000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-exclusion-window", type=int, default=2)
    parser.add_argument("--enable-boundary-exclusion", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    """Split comma-separated strings."""

    return [item.strip() for item in value.split(",") if item.strip()]


def split_ints(value: str) -> list[int]:
    """Split comma-separated integers."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_coords(root: Path, domain_id: str, target_frames: int) -> np.ndarray:
    """Load cached transition-aligned synthetic trajectory coordinates."""

    path = root / "runs" / domain_id / str(target_frames) / "transition_aligned_coords.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["coords"], dtype=float)


def run_long_arrays(
    coords: np.ndarray,
    *,
    selector: str,
    representative_count: int,
    excluded_frame_indices: np.ndarray | None,
    args: argparse.Namespace,
) -> LongArrays:
    """Run long-mode and extract comparison arrays."""

    start = time.perf_counter()
    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=representative_count,
        frame_selection=selector,
        random_seed=args.seed,
        chunk_size=args.chunk_size,
        raw_descriptor_size=args.raw_descriptor_size,
        compressed_descriptor_dim=args.compressed_dim,
        candidate_pool_size=args.candidate_pool_size,
        excluded_representative_frame_indices=excluded_frame_indices,
        store_aligned=False,
        rule="huber",
        max_iterations=args.max_iterations,
        tolerance=1e-6,
        objective_tolerance=1e-5,
        objective_patience=2,
        sequence_continuity_strength=0.05,
        contact_consistency_strength=0.0,
        record_history=False,
    )
    profile = build_long_flexibility_profile(result, min_segment_length=3, merge_gap=1)
    core = select_core_by_target_fraction(result.frozen_weights, 0.6).core_mask
    return LongArrays(
        runtime_sec=time.perf_counter() - start,
        core_mask=np.asarray(core, dtype=bool),
        mobility=np.asarray(profile.mobility_score, dtype=float),
        rmsf=np.asarray(profile.robust_aligned_rmsf, dtype=float),
        segments=np.asarray(profile.flexible_segment_id, dtype=int),
        unified_rmsd=float(result.mean_rmsd),
        selected=np.asarray(result.selected_frame_indices, dtype=int),
        rotations=np.asarray(result.rotations, dtype=float),
        translations=np.asarray(result.translations, dtype=float),
        metadata=dict(result.selection_metadata),
        selector_runtime_sec=float(result.timing.get("representative_selection", result.timing.get("initialization", 0.0))),
        weight_estimation_runtime_sec=float(result.timing.get("iterative_weight_estimation", 0.0)),
        full_long_mode_runtime_sec=float(result.total_runtime_seconds),
    )


def scalar(value: Any) -> float | str:
    """Return finite scalar or NA."""

    try:
        out = float(value)
    except Exception:
        return "NA"
    return out if math.isfinite(out) else "NA"


def noncore_core_ratio(rmsf: np.ndarray, core_mask: np.ndarray) -> float | None:
    """Return non-core/core RMSF ratio."""

    core = np.asarray(core_mask, dtype=bool)
    if not np.any(core) or not np.any(~core):
        return None
    core_mean = float(np.mean(rmsf[core]))
    noncore_mean = float(np.mean(rmsf[~core]))
    return noncore_mean / core_mean if core_mean > 0 else None


def heldout_validation(coords: np.ndarray, result: LongArrays) -> tuple[float | None, float | None, int]:
    """Validate long-mode labels on frames not used for representative fitting."""

    mask = np.ones(coords.shape[0], dtype=bool)
    mask[result.selected] = False
    if not np.any(mask):
        return None, None, 0
    aligned = apply_transforms_batch(coords[mask], result.rotations[mask], result.translations[mask])
    heldout_rmsf = compute_rmsf(aligned)
    ratio = noncore_core_ratio(heldout_rmsf, result.core_mask)
    spearman = spearman_correlation(heldout_rmsf, result.mobility)
    pass_flag = ratio is not None and ratio > 1.0 and spearman is not None and spearman >= 0.30
    return ratio, spearman, int(pass_flag)


def write_selection_artifacts(
    root: Path,
    domain_id: str,
    target: int,
    selector: str,
    count: int,
    boundary_exclusion: bool,
    result: LongArrays,
) -> tuple[Path, Path]:
    """Write selected frame indices and descriptor metadata for one run."""

    boundary_label = "boundary_excluded" if boundary_exclusion else "no_boundary_exclusion"
    run_dir = root / "selections" / domain_id / str(target) / f"{selector}_{count}_{boundary_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    indices_path = run_dir / "selected_frame_indices.tsv"
    metadata_path = run_dir / "descriptor_metadata.json"
    pd.DataFrame({"rank": np.arange(1, result.selected.shape[0] + 1), "frame_index": result.selected}).to_csv(
        indices_path,
        sep="\t",
        index=False,
    )
    write_json(metadata_path, result.metadata)
    return indices_path, metadata_path


def boundary_exclusion_indices(root: Path, domain_id: str, target: int, window: int) -> tuple[np.ndarray, int]:
    """Reconstruct stitched segment boundary frame indices and exclusion mask."""

    run_npz = root / "runs" / domain_id / str(target) / "transition_aligned_coords.npz"
    manifest_path = root / "domains" / domain_id / "replicate_manifest.tsv"
    if not run_npz.exists() or not manifest_path.exists():
        return np.asarray([], dtype=int), 0
    with np.load(run_npz, allow_pickle=True) as data:
        sequence = [str(item) for item in data["replicate_sequence"].tolist()]
        n_frames = int(data["coords"].shape[0])
    manifest = pd.read_csv(manifest_path, sep="\t")
    lengths = {str(row["replicate_label"]): int(row["n_frames"]) for _, row in manifest.iterrows()}
    boundary_positions: list[int] = []
    cumulative = 0
    for label in sequence[:-1]:
        cumulative += int(lengths[label])
        if 0 < cumulative < n_frames:
            boundary_positions.append(cumulative)
    excluded: set[int] = set()
    window = int(window)
    for boundary in boundary_positions:
        for frame_index in range(boundary - window, boundary + window + 1):
            if 0 <= frame_index < n_frames:
                excluded.add(frame_index)
    excluded.discard(0)
    return np.asarray(sorted(excluded), dtype=int), len(boundary_positions)


def optional_selector_available(selector: str) -> tuple[bool, str]:
    """Return whether optional ANN dependencies are available."""

    if selector == "annoy" and importlib.util.find_spec("annoy") is None:
        return False, "annoy optional dependency is not installed"
    if selector == "faiss" and importlib.util.find_spec("faiss") is None and importlib.util.find_spec("faiss_cpu") is None:
        return False, "faiss optional dependency is not installed"
    return True, "available"


def quality_pass(row: dict[str, Any]) -> int:
    """Apply the descriptor-selector decision quality rule per case."""

    checks = [
        finite_float(row["core_jaccard_vs_long_baseline"]) is not None
        and finite_float(row["core_jaccard_vs_long_baseline"]) >= 0.98,
        finite_float(row["mobility_spearman_vs_long_baseline"]) is not None
        and finite_float(row["mobility_spearman_vs_long_baseline"]) >= 0.99,
        finite_float(row["RMSF_spearman_vs_long_baseline"]) is not None
        and finite_float(row["RMSF_spearman_vs_long_baseline"]) >= 0.98,
        finite_float(row["noncore_core_RMSF_ratio"]) is not None
        and finite_float(row["noncore_core_RMSF_ratio"]) > 1.0,
        int(row["heldout_pass"]) == 1,
    ]
    return int(all(checks))


def run_ablation(args: argparse.Namespace) -> None:
    """Run selector ablation and write requested outputs."""

    args.output_root.mkdir(parents=True, exist_ok=True)
    ablation_path = args.output_root / "rep_frame_selector_ablation.tsv"
    metadata_path = args.output_root / "descriptor_metadata.tsv"
    for path in (ablation_path, metadata_path):
        if path.exists() and not args.skip_existing:
            path.unlink()

    selected = pd.read_csv(args.transition_root / "selected_domains.tsv", sep="\t").head(args.domain_limit)
    selectors = split_csv(args.selectors)
    targets = split_ints(args.targets)
    counts = split_ints(args.representative_counts)
    rows_written = 0
    baseline_cache: dict[tuple[str, int], LongArrays] = {}

    for _, domain in selected.iterrows():
        domain_id = str(domain["domain_id"])
        cath_class = domain.get("cath_class", "NA")
        n_residues = int(domain["n_residues"])
        for target in targets:
            coords_path = args.transition_root / "runs" / domain_id / str(target) / "transition_aligned_coords.npz"
            if not coords_path.exists():
                continue
            coords = load_coords(args.transition_root, domain_id, target)
            baseline_key = (domain_id, target)
            if baseline_key not in baseline_cache:
                print(f"[baseline] {domain_id} {target}", flush=True)
                baseline_cache[baseline_key] = run_long_arrays(
                    coords,
                    selector="uniform",
                    representative_count=args.baseline_representative_count,
                    excluded_frame_indices=None,
                    args=args,
                )
            baseline = baseline_cache[baseline_key]
            for selector in selectors:
                available, availability_note = optional_selector_available(selector)
                for count in counts:
                    boundary_modes = [False]
                    if args.enable_boundary_exclusion and selector in {"distance_sketch_farthest_first", "pca_kmedoids", "annoy", "faiss"}:
                        boundary_modes = [False, True]
                    for boundary_exclusion in boundary_modes:
                        if args.max_cases and rows_written >= args.max_cases:
                            write_summary(args.output_root)
                            write_report(args.output_root)
                            return
                        excluded, n_boundaries = (
                            boundary_exclusion_indices(args.transition_root, domain_id, target, args.boundary_exclusion_window)
                            if boundary_exclusion
                            else (np.asarray([], dtype=int), 0)
                        )
                        if not available:
                            row = skipped_row(
                                domain_id,
                                cath_class,
                                n_residues,
                                target,
                                selector,
                                count,
                                boundary_exclusion,
                                args.boundary_exclusion_window,
                                n_boundaries,
                                int(excluded.shape[0]),
                                availability_note,
                            )
                            append_tsv(ablation_path, row, ABLATION_COLUMNS)
                            append_tsv(metadata_path, metadata_row_from_skip(row), METADATA_COLUMNS)
                            rows_written += 1
                            continue
                        print(
                            f"[selector] {domain_id} target={target} selector={selector} "
                            f"count={count} boundary_exclusion={int(boundary_exclusion)}",
                            flush=True,
                        )
                        try:
                            result = run_long_arrays(
                                coords,
                                selector=selector,
                                representative_count=count,
                                excluded_frame_indices=excluded if boundary_exclusion else None,
                                args=args,
                            )
                            indices_path, descriptor_path = write_selection_artifacts(
                                args.output_root,
                                domain_id,
                                target,
                                selector,
                                count,
                                boundary_exclusion,
                                result,
                            )
                            ratio = noncore_core_ratio(result.rmsf, result.core_mask)
                            held_ratio, held_sp, held_pass = heldout_validation(coords, result)
                            row = {
                                "domain_id": domain_id,
                                "cath_class": cath_class,
                                "n_residues": n_residues,
                                "target_frames": target,
                                "selector": selector,
                                "representative_frame_count": count,
                                "boundary_exclusion": int(boundary_exclusion),
                                "boundary_exclusion_window": args.boundary_exclusion_window if boundary_exclusion else 0,
                                "n_stitched_boundaries": n_boundaries,
                                "n_boundary_frames_excluded": int(excluded.shape[0]),
                                "status": "success",
                                "runtime_sec": result.runtime_sec,
                                "total_runtime_including_selector_sec": result.runtime_sec,
                                "selector_runtime_sec": result.selector_runtime_sec,
                                "weight_estimation_runtime_sec": result.weight_estimation_runtime_sec,
                                "full_long_mode_runtime_sec": result.full_long_mode_runtime_sec,
                                "baseline_runtime_sec": baseline.runtime_sec,
                                "speedup_vs_current_long_mode": baseline.runtime_sec / result.runtime_sec if result.runtime_sec > 0 else "NA",
                                "core_jaccard_vs_long_baseline": scalar(jaccard(baseline.core_mask, result.core_mask)),
                                "mobile_jaccard_vs_long_baseline": scalar(jaccard(baseline.mobile_mask, result.mobile_mask)),
                                "mobility_spearman_vs_long_baseline": scalar(spearman_correlation(baseline.mobility, result.mobility)),
                                "RMSF_spearman_vs_long_baseline": scalar(spearman_correlation(baseline.rmsf, result.rmsf)),
                                "flexible_segment_jaccard_vs_long_baseline": scalar(jaccard(baseline.segments > 0, result.segments > 0)),
                                "noncore_core_RMSF_ratio": scalar(ratio),
                                "heldout_noncore_core_RMSF_ratio": scalar(held_ratio),
                                "heldout_RMSF_mobility_spearman": scalar(held_sp),
                                "heldout_pass": held_pass,
                                "unified_RMSD": result.unified_rmsd,
                                "baseline_unified_RMSD": baseline.unified_rmsd,
                                "unified_RMSD_delta": result.unified_rmsd - baseline.unified_rmsd,
                                "quality_pass": 0,
                                "selected_frame_indices_path": str(indices_path),
                                "descriptor_metadata_path": str(descriptor_path),
                                "notes": "Transition-aligned synthetic long trajectory; selector only chooses representative frames.",
                            }
                            row["quality_pass"] = quality_pass(row)
                            append_tsv(ablation_path, row, ABLATION_COLUMNS)
                            append_tsv(
                                metadata_path,
                                metadata_row(
                                    domain_id,
                                    target,
                                    selector,
                                    count,
                                    boundary_exclusion,
                                    args.boundary_exclusion_window if boundary_exclusion else 0,
                                    "success",
                                    result.metadata,
                                ),
                                METADATA_COLUMNS,
                            )
                        except Exception as exc:
                            row = skipped_row(
                                domain_id,
                                cath_class,
                                n_residues,
                                target,
                                selector,
                                count,
                                boundary_exclusion,
                                args.boundary_exclusion_window if boundary_exclusion else 0,
                                n_boundaries,
                                int(excluded.shape[0]),
                                f"{type(exc).__name__}: {exc}",
                                status="failed",
                            )
                            append_tsv(ablation_path, row, ABLATION_COLUMNS)
                            append_tsv(metadata_path, metadata_row_from_skip(row), METADATA_COLUMNS)
                        rows_written += 1
                        gc.collect()
            del coords
            gc.collect()
    write_summary(args.output_root)
    write_report(args.output_root)


def skipped_row(
    domain_id: str,
    cath_class: Any,
    n_residues: int,
    target: int,
    selector: str,
    count: int,
    boundary_exclusion: bool,
    boundary_exclusion_window: int,
    n_boundaries: int,
    n_excluded: int,
    notes: str,
    *,
    status: str = "skipped_optional_dependency_missing",
) -> dict[str, Any]:
    """Build a skipped/failed ablation row."""

    return {
        "domain_id": domain_id,
        "cath_class": cath_class,
        "n_residues": n_residues,
        "target_frames": target,
        "selector": selector,
        "representative_frame_count": count,
        "boundary_exclusion": int(boundary_exclusion),
        "boundary_exclusion_window": boundary_exclusion_window,
        "n_stitched_boundaries": n_boundaries,
        "n_boundary_frames_excluded": n_excluded,
        "status": status,
        "runtime_sec": "NA",
        "total_runtime_including_selector_sec": "NA",
        "selector_runtime_sec": "NA",
        "weight_estimation_runtime_sec": "NA",
        "full_long_mode_runtime_sec": "NA",
        "baseline_runtime_sec": "NA",
        "speedup_vs_current_long_mode": "NA",
        "core_jaccard_vs_long_baseline": "NA",
        "mobile_jaccard_vs_long_baseline": "NA",
        "mobility_spearman_vs_long_baseline": "NA",
        "RMSF_spearman_vs_long_baseline": "NA",
        "flexible_segment_jaccard_vs_long_baseline": "NA",
        "noncore_core_RMSF_ratio": "NA",
        "heldout_noncore_core_RMSF_ratio": "NA",
        "heldout_RMSF_mobility_spearman": "NA",
        "heldout_pass": 0,
        "unified_RMSD": "NA",
        "baseline_unified_RMSD": "NA",
        "unified_RMSD_delta": "NA",
        "quality_pass": 0,
        "selected_frame_indices_path": "NA",
        "descriptor_metadata_path": "NA",
        "notes": notes,
    }


def metadata_row(
    domain_id: str,
    target: int,
    selector: str,
    count: int,
    boundary_exclusion: bool,
    boundary_exclusion_window: int,
    status: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build aggregate descriptor metadata row."""

    row = {
        "domain_id": domain_id,
        "target_frames": target,
        "selector": selector,
        "representative_frame_count": count,
        "boundary_exclusion": int(boundary_exclusion),
        "boundary_exclusion_window": boundary_exclusion_window,
        "status": status,
        "notes": "Descriptor metadata for representative-frame selector.",
    }
    row.update(metadata)
    return row


def metadata_row_from_skip(row: dict[str, Any]) -> dict[str, Any]:
    """Build metadata row for skipped/failed selectors."""

    return {
        "domain_id": row["domain_id"],
        "target_frames": row["target_frames"],
        "selector": row["selector"],
        "representative_frame_count": row["representative_frame_count"],
        "boundary_exclusion": row["boundary_exclusion"],
        "boundary_exclusion_window": row["boundary_exclusion_window"],
        "status": row["status"],
        "notes": row["notes"],
    }


def write_summary(output_root: Path) -> None:
    """Write selector-level quality/speed summary."""

    path = output_root / "rep_frame_selector_ablation.tsv"
    if not path.exists():
        write_tsv(output_root / "selector_quality_speed_summary.tsv", [], SUMMARY_COLUMNS)
        return
    df = pd.read_csv(path, sep="\t")
    success = df[df["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["selector", "representative_frame_count", "boundary_exclusion", "target_frames"]
    for (selector, count, boundary_exclusion, target), group in success.groupby(group_cols, dropna=False):
        def med(column: str) -> float | str:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            return float(values.median()) if not values.empty else "NA"

        quality = pd.to_numeric(group["quality_pass"], errors="coerce")
        speedup = med("speedup_vs_current_long_mode")
        core = med("core_jaccard_vs_long_baseline")
        mobility = med("mobility_spearman_vs_long_baseline")
        rmsf = med("RMSF_spearman_vs_long_baseline")
        pass_percent = float(100.0 * quality.mean()) if not quality.empty else "NA"
        recommended = (
            group.shape[0] >= 20
            and
            finite_float(speedup) is not None
            and finite_float(speedup) >= 1.25
            and finite_float(core) is not None
            and finite_float(core) >= 0.98
            and finite_float(mobility) is not None
            and finite_float(mobility) >= 0.99
            and finite_float(rmsf) is not None
            and finite_float(rmsf) >= 0.98
            and finite_float(pass_percent) is not None
            and finite_float(pass_percent) >= 95.0
        )
        rows.append(
            {
                "selector": selector,
                "representative_frame_count": int(count),
                "boundary_exclusion": int(boundary_exclusion),
                "target_frames": int(target),
                "n_cases": int(group.shape[0]),
                "n_success": int(group.shape[0]),
                "median_runtime_sec": med("runtime_sec"),
                "median_selector_runtime_sec": med("selector_runtime_sec"),
                "median_weight_estimation_runtime_sec": med("weight_estimation_runtime_sec"),
                "median_full_long_mode_runtime_sec": med("full_long_mode_runtime_sec"),
                "median_speedup_vs_current_long_mode": speedup,
                "median_core_jaccard": core,
                "median_mobile_jaccard": med("mobile_jaccard_vs_long_baseline"),
                "median_mobility_spearman": mobility,
                "median_RMSF_spearman": rmsf,
                "median_flexible_segment_jaccard": med("flexible_segment_jaccard_vs_long_baseline"),
                "median_noncore_core_RMSF_ratio": med("noncore_core_RMSF_ratio"),
                "quality_pass_percent": pass_percent,
                "recommendation": "recommend_new_selector" if recommended else "keep_experimental",
                "notes": (
                    "Decision rule compares against current uniform-2000 long-mode baseline; "
                    "recommendation requires at least 20 successful cases."
                ),
            }
        )
    write_tsv(output_root / "selector_quality_speed_summary.tsv", rows, SUMMARY_COLUMNS)


def write_report(output_root: Path) -> None:
    """Write concise descriptor-guided selector report."""

    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    summary_path = output_root / "selector_quality_speed_summary.tsv"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        text = "# Descriptor-Guided Long-Mode Report\n\nNo successful selector rows were available.\n"
    else:
        summary = pd.read_csv(summary_path, sep="\t")
        recommended = summary[summary["recommendation"] == "recommend_new_selector"]
        summary["_quality_numeric"] = pd.to_numeric(summary["quality_pass_percent"], errors="coerce")
        summary["_speedup_numeric"] = pd.to_numeric(summary["median_speedup_vs_current_long_mode"], errors="coerce")
        best = summary.sort_values(
            ["_quality_numeric", "_speedup_numeric"],
            ascending=[False, False],
        ).drop(columns=["_quality_numeric", "_speedup_numeric"]).head(10)
        text = "# Descriptor-Guided Long-Mode Report\n\n"
        text += "This prototype tests invariant C-alpha pairwise-distance sketches for selecting representative frames in MDCoreAlign long-mode. The selector only chooses frames used for soft stable-core weight estimation; exact weighted Kabsch alignment is unchanged.\n\n"
        text += "The benchmark uses transition-aligned synthetic long trajectories for computational scaling diagnostics. These inputs are not real continuous MD trajectories and should not be interpreted as biological continuity evidence.\n\n"
        text += "## Decision Rule\n\n"
        text += "A selector is recommended only if it reaches at least 1.25x median speedup over current uniform-2000 long-mode while preserving median core Jaccard >= 0.98, median mobility Spearman >= 0.99, median RMSF Spearman >= 0.98, and quality pass >= 95%.\n\n"
        text += "## Recommendation\n\n"
        if recommended.empty:
            text += "No descriptor-guided selector met the full decision rule with at least 20 successful cases in the current run. Keep descriptor-guided selection experimental.\n\n"
        else:
            text += "Recommended selector configurations:\n\n"
            for _, row in recommended.iterrows():
                text += f"- {row['selector']} with {int(row['representative_frame_count'])} frames at target {int(row['target_frames'])}: speedup {row['median_speedup_vs_current_long_mode']}, quality pass {row['quality_pass_percent']}%.\n"
            text += "\n"
        text += "## Top Observed Configurations\n\n"
        text += "```text\n"
        text += best.to_string(index=False)
        text += "\n```\n"
        text += "\n\n## Interpretation\n\n"
        text += "The 20-domain 10k benchmark is the main completed ablation block. Some 5-domain 50k exploratory rows met the numerical thresholds, but they are not sufficient to change the default long-mode selector. Uniform 1000-frame selection was also competitive in those exploratory rows, so descriptor-guided selection is not yet clearly better than a simpler lower-count uniform strategy.\n"
        text += "\n## Optional ANN Backends\n\nAnnoy and Faiss are optional. If missing or slower than distance-sketch farthest-first/PCA-kmedoids, they should remain experimental and not become required dependencies.\n"
    (docs / "descriptor_guided_long_mode_report.md").write_text(text)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
