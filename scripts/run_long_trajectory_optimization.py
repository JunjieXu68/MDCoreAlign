#!/usr/bin/env python
"""Profile, validate and benchmark MDCoreAlign long mode.

This script intentionally reuses cached MDLovoFit/THESEUS baseline tables when
available. It reruns only MDCoreAlign default/long-mode calculations needed for
equivalence and timing, keeping this optimization pass focused and repeatable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (  # noqa: E402
    Trajectory,
    build_flexibility_profile,
    build_long_flexibility_profile,
    select_core_by_target_fraction,
    solve_long_trajectory_alignment,
    solve_trajectory_robust_alignment,
)
from mdsuperpose.flexibility import spearman_correlation, write_table  # noqa: E402


OUT_ROOT = ROOT / "results" / "long_optimization"
PROFILE_DIR = OUT_ROOT / "profile"
VALIDATION_DIR = OUT_ROOT / "validation"
BENCHMARK_DIR = OUT_ROOT / "benchmark"
DOCS_REPORT = ROOT / "docs" / "long_trajectory_optimization_report.md"


def load_npz_coords(path: Path, *, n_frames: int | None = None) -> np.ndarray:
    """Load a coordinate array from an mdCATH-style NPZ file."""

    with np.load(path) as archive:
        key = "coords" if "coords" in archive.files else archive.files[0]
        coords = np.asarray(archive[key], dtype=float)
    if n_frames is not None:
        coords = coords[:n_frames]
    return coords


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    """Write TSV rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Return Jaccard overlap for two boolean masks."""

    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second)
    if not np.any(union):
        return 1.0
    return float(np.sum(np.logical_and(first, second)) / np.sum(union))


def continuous_cases() -> list[dict[str, object]]:
    """Return available real continuous mdCATH NPZ cases."""

    root = ROOT / "results" / "mdcath_benchmark" / "long_continuous_trajectory"
    selected_path = root / "selected_continuous_trajectories.tsv"
    if not selected_path.exists():
        return []
    selected = pd.read_csv(selected_path, sep="\t")
    cases: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        path = root / "continuous_inputs" / f"{row['trajectory_id']}_ca_coords.npz"
        if path.exists():
            cases.append(
                {
                    "benchmark_set": "real_continuous_500",
                    "domain_id": row["domain_id"],
                    "trajectory_id": row["trajectory_id"],
                    "input_type": "real_continuous_mdCATH_single_replicate",
                    "path": path,
                    "n_frames": int(row["n_frames"]),
                    "n_residues": int(row["n_residues"]),
                    "notes": "Single 320 K mdCATH replicate; not stitched.",
                }
            )
    return cases


def transition_cases() -> list[dict[str, object]]:
    """Return primary transition-aligned synthetic stress-test NPZ cases."""

    root = ROOT / "results" / "mdcath_benchmark" / "transition_aligned_stitching"
    cases: list[dict[str, object]] = []
    for path in sorted(root.glob("*/stitched_long/canonical_cycle_*_keep_boundary_first_frame.npz")):
        domain_id = path.parts[-3]
        match = re.match(r"canonical_cycle_(?P<frames>\d+)_keep_boundary_first_frame$", path.stem)
        if match is None:
            continue
        target_frames = int(match.group("frames"))
        with np.load(path) as archive:
            key = "coords" if "coords" in archive.files else archive.files[0]
            shape = archive[key].shape
        cases.append(
            {
                "benchmark_set": "transition_aligned_stress",
                "domain_id": domain_id,
                "trajectory_id": f"{domain_id}_canonical_cycle_{target_frames}",
                "input_type": "synthetic_transition_aligned_replicates",
                "path": path,
                "n_frames": int(shape[0]),
                "n_residues": int(shape[1]),
                "notes": "Synthetic transition-aligned scaling stress test; not physically continuous MD.",
            }
        )
    return cases


def profile_default_runtime(case: dict[str, object]) -> list[dict[str, object]]:
    """Profile current default runtime stages for one case."""

    coords = load_npz_coords(Path(case["path"]))
    trajectory = Trajectory(coords=coords)
    timing_path = PROFILE_DIR / f"{case['trajectory_id']}_default_solver_timing.json"
    profile_start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode="first_frame",
        warm_start=True,
        collect_timing=True,
        timing_output_path=timing_path,
        max_iterations=20,
        delta=0.2,
    )
    solver_seconds = time.perf_counter() - profile_start

    flex_start = time.perf_counter()
    profile = build_flexibility_profile(
        trajectory,
        trajectory_result=result,
        max_iterations=20,
        delta=0.2,
    )
    profile_seconds = time.perf_counter() - flex_start
    output_start = time.perf_counter()
    write_table(PROFILE_DIR / f"{case['trajectory_id']}_profile_rows.tsv", profile.to_rows())
    output_seconds = time.perf_counter() - output_start
    with timing_path.open() as handle:
        timing = json.load(handle)
    stage_values = {
        "input_load": 0.0,
        "initialization": timing.get("prepare_input_sec", 0.0) + timing.get("initialize_alignment_sec", 0.0),
        "iterative_weight_estimation": (
            timing.get("weighted_fit_sec", 0.0)
            + timing.get("residual_update_sec", 0.0)
            + timing.get("weight_update_sec", 0.0)
            + timing.get("regularization_sec", 0.0)
            + timing.get("convergence_check_sec", 0.0)
        ),
        "weighted_fit": timing.get("weighted_fit_sec", 0.0),
        "residual_update": timing.get("residual_update_sec", 0.0),
        "RMSF": profile_seconds,
        "flexible_segment_detection": 0.0,
        "profile_table_build": profile_seconds,
        "output_write": output_seconds,
        "unified_RMSD_eval": 0.0,
    }
    total = solver_seconds + profile_seconds + output_seconds
    rows = []
    for stage, seconds in stage_values.items():
        rows.append(
            {
                "benchmark_set": case["benchmark_set"],
                "domain_id": case["domain_id"],
                "trajectory_id": case["trajectory_id"],
                "n_frames": case["n_frames"],
                "n_residues": case["n_residues"],
                "mode": "default",
                "stage": stage,
                "runtime_seconds": seconds,
                "percent_total": 100.0 * seconds / total if total > 0 else 0.0,
                "total_profiled_runtime_seconds": total,
                "notes": "Current default mode profile; stage percentages are lightweight wall-clock estimates.",
            }
        )
    return rows


def solve_default_and_profile(coords: np.ndarray):
    """Run default mode and build a flexibility profile."""

    trajectory = Trajectory(coords=coords)
    start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode="first_frame",
        warm_start=True,
        max_iterations=20,
        delta=0.2,
    )
    runtime = time.perf_counter() - start
    profile = build_flexibility_profile(
        trajectory,
        trajectory_result=result,
        max_iterations=20,
        delta=0.2,
    )
    return result, profile, runtime


def solve_long_and_profile(coords: np.ndarray, *, max_weight_fit_frames: int = 2000):
    """Run long mode and build its flexibility profile."""

    start = time.perf_counter()
    result = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=max_weight_fit_frames,
        frame_selection="uniform",
        random_seed=42,
        chunk_size=2000,
        max_iterations=20,
        delta=0.2,
    )
    runtime = time.perf_counter() - start
    profile = build_long_flexibility_profile(result)
    return result, profile, runtime


def validate_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compare default and long mode on selected cases."""

    rows: list[dict[str, object]] = []
    for case in cases:
        coords = load_npz_coords(Path(case["path"]))
        default_result, default_profile, default_runtime = solve_default_and_profile(coords)
        long_result, long_profile, long_runtime = solve_long_and_profile(coords)
        default_core = select_core_by_target_fraction(default_profile.soft_stability_weight, 0.6).core_mask
        long_core = select_core_by_target_fraction(long_profile.soft_stability_weight, 0.6).core_mask
        default_flexible = default_profile.flexible_segment_id > 0
        long_flexible = long_profile.flexible_segment_id > 0
        core_jaccard = jaccard(default_core, long_core)
        mobility_spearman = spearman_correlation(default_profile.mobility_score, long_profile.mobility_score)
        rmsf_spearman = spearman_correlation(default_profile.robust_aligned_rmsf, long_profile.robust_aligned_rmsf)
        unified_diff = abs(default_result.mean_rmsd - long_result.mean_rmsd)
        stable_fraction_diff = abs(float(np.mean(default_core)) - float(np.mean(long_core)))
        pass_flag = (
            core_jaccard >= 0.95
            and (mobility_spearman is not None and mobility_spearman >= 0.98)
            and (rmsf_spearman is not None and rmsf_spearman >= 0.98)
            and unified_diff <= 0.02
            and stable_fraction_diff <= 0.03
            and jaccard(default_flexible, long_flexible) >= 0.85
        )
        rows.append(
            {
                "benchmark_set": case["benchmark_set"],
                "domain_id": case["domain_id"],
                "trajectory_id": case["trajectory_id"],
                "input_type": case["input_type"],
                "n_frames": case["n_frames"],
                "n_residues": case["n_residues"],
                "unified_RMSD_diff": unified_diff,
                "weighted_RMSD_diff": abs(default_result.mean_weighted_rmsd - long_result.mean_weighted_rmsd),
                "core_Jaccard": core_jaccard,
                "mobile_Jaccard": jaccard(~default_core, ~long_core),
                "mobility_Spearman": mobility_spearman,
                "RMSF_Spearman": rmsf_spearman,
                "max_RMSF_abs_diff": float(np.max(np.abs(default_profile.robust_aligned_rmsf - long_profile.robust_aligned_rmsf))),
                "stable_core_fraction_diff": stable_fraction_diff,
                "flexible_segment_Jaccard": jaccard(default_flexible, long_flexible),
                "runtime_speedup_long_vs_default": default_runtime / long_runtime if long_runtime > 0 else "",
                "long_mode_equivalence_pass": int(pass_flag),
                "notes": "Long mode is approximate; failures indicate cases where default remains primary.",
            }
        )
    return rows


def cached_baseline_rows() -> pd.DataFrame:
    """Load cached default/MDLovoFit/THESEUS baseline summaries."""

    frames = []
    continuous = ROOT / "results" / "mdcath_benchmark" / "long_continuous_trajectory" / "long_continuous_trajectory_tool_comparison.tsv"
    if continuous.exists():
        df = pd.read_csv(continuous, sep="\t")
        df["benchmark_set"] = "real_continuous_500"
        df["input_type"] = "real_continuous_mdCATH_single_replicate"
        frames.append(df)
    transition = ROOT / "results" / "mdcath_benchmark" / "transition_aligned_stitching" / "aggregate" / "long_transition_aligned_tool_comparison.tsv"
    if transition.exists():
        df = pd.read_csv(transition, sep="\t")
        df["trajectory_id"] = df["domain_id"] + "_canonical_cycle_" + df["target_frames"].astype(str)
        df["prefix_label"] = df["target_frames"].astype(str)
        df["benchmark_set"] = "transition_aligned_stress"
        df["input_type"] = "synthetic_transition_aligned_replicates"
        df = df.rename(
            columns={
                "mdcorealign_median_runtime_sec": "mdcorealign_alignment_runtime_sec",
                "mdlovofit_median_runtime_sec": "mdlovofit_runtime_sec",
                "theseus_median_runtime_sec": "theseus_runtime_sec",
            }
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def benchmark_long_mode(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Run long mode and merge with cached baseline runtimes where possible."""

    baseline = cached_baseline_rows()
    rows: list[dict[str, object]] = []
    for case in cases:
        coords = load_npz_coords(Path(case["path"]))
        result, profile, runtime = solve_long_and_profile(coords)
        baseline_match = baseline[
            (baseline["domain_id"] == case["domain_id"])
            & (baseline["n_frames"].astype(int) == int(case["n_frames"]))
            & (baseline["benchmark_set"] == case["benchmark_set"])
        ]
        baseline_row = baseline_match.iloc[0].to_dict() if not baseline_match.empty else {}
        rmsf_mobility = spearman_correlation(profile.robust_aligned_rmsf, profile.mobility_score)
        rows.append(
            {
                "benchmark_set": case["benchmark_set"],
                "domain_id": case["domain_id"],
                "input_type": case["input_type"],
                "n_frames": case["n_frames"],
                "n_residues": case["n_residues"],
                "method": "mdcorealign_long",
                "runtime_scope": "alignment_plus_RMSF_profile",
                "median_runtime_sec": runtime,
                "iqr_runtime_sec": 0.0,
                "unified_RMSD": result.mean_rmsd,
                "core_fraction": 0.6,
                "RMSF_mobility_spearman": rmsf_mobility,
                "status": "success",
                "notes": "Long mode rerun in current environment; stable-core weights estimated from representative frames.",
            }
        )
        if baseline_row:
            rows.extend(
                [
                    {
                        "benchmark_set": case["benchmark_set"],
                        "domain_id": case["domain_id"],
                        "input_type": case["input_type"],
                        "n_frames": case["n_frames"],
                        "n_residues": case["n_residues"],
                        "method": "mdcorealign_default_cached",
                        "runtime_scope": "alignment_only_cached",
                        "median_runtime_sec": baseline_row.get("mdcorealign_alignment_runtime_sec", ""),
                        "iqr_runtime_sec": "",
                        "unified_RMSD": baseline_row.get("mdcorealign_unified_RMSD", ""),
                        "core_fraction": "",
                        "RMSF_mobility_spearman": "",
                        "status": "cached",
                        "notes": "Cached baseline table; not rerun by this optimization script.",
                    },
                    {
                        "benchmark_set": case["benchmark_set"],
                        "domain_id": case["domain_id"],
                        "input_type": case["input_type"],
                        "n_frames": case["n_frames"],
                        "n_residues": case["n_residues"],
                        "method": "mdlovofit_cached",
                        "runtime_scope": "external_runtime_cached",
                        "median_runtime_sec": baseline_row.get("mdlovofit_runtime_sec", ""),
                        "iqr_runtime_sec": "",
                        "unified_RMSD": baseline_row.get("mdlovofit_unified_RMSD", ""),
                        "core_fraction": "",
                        "RMSF_mobility_spearman": "",
                        "status": baseline_row.get("mdlovofit_status", "cached"),
                        "notes": "Cached MDLovoFit baseline.",
                    },
                    {
                        "benchmark_set": case["benchmark_set"],
                        "domain_id": case["domain_id"],
                        "input_type": case["input_type"],
                        "n_frames": case["n_frames"],
                        "n_residues": case["n_residues"],
                        "method": "theseus_cached",
                        "runtime_scope": "external_runtime_cached",
                        "median_runtime_sec": baseline_row.get("theseus_runtime_sec", ""),
                        "iqr_runtime_sec": "",
                        "unified_RMSD": baseline_row.get("theseus_unified_RMSD", ""),
                        "core_fraction": "",
                        "RMSF_mobility_spearman": "",
                        "status": baseline_row.get("theseus_status", "cached"),
                        "notes": "Cached THESEUS baseline.",
                    },
                ]
            )
    return rows


def speedup_summary(benchmark_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize speedups by benchmark set and frame count."""

    df = pd.DataFrame(benchmark_rows)
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    for (benchmark_set, n_frames), group in df.groupby(["benchmark_set", "n_frames"]):
        pivot = group.pivot_table(index=["domain_id"], columns="method", values="median_runtime_sec", aggfunc="first")
        validation = pd.DataFrame(validation_rows)
        val_subset = validation[(validation["benchmark_set"] == benchmark_set) & (validation["n_frames"].astype(int) == int(n_frames))]
        def median_ratio(num: str, den: str) -> float | str:
            if num not in pivot or den not in pivot:
                return ""
            ratios = pd.to_numeric(pivot[num], errors="coerce") / pd.to_numeric(pivot[den], errors="coerce")
            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
            if ratios.empty:
                return ""
            return float(np.median(ratios))
        rmsd_pivot = group.pivot_table(index=["domain_id"], columns="method", values="unified_RMSD", aggfunc="first")
        if "mdcorealign_long" in rmsd_pivot and "mdlovofit_cached" in rmsd_pivot:
            long_rmsd = pd.to_numeric(rmsd_pivot["mdcorealign_long"], errors="coerce")
            mdlovo_rmsd = pd.to_numeric(rmsd_pivot["mdlovofit_cached"], errors="coerce")
            valid = long_rmsd.notna() & mdlovo_rmsd.notna()
            lower_rmsd_percent = float(100.0 * (long_rmsd[valid] < mdlovo_rmsd[valid]).mean()) if valid.any() else ""
        else:
            lower_rmsd_percent = ""
        rows.append(
            {
                "benchmark_set": benchmark_set,
                "n_frames": int(n_frames),
                "n_cases": int(pivot.shape[0]),
                "mdcorealign_default_vs_mdlovofit_speedup": median_ratio("mdlovofit_cached", "mdcorealign_default_cached"),
                "mdcorealign_long_vs_mdlovofit_speedup": median_ratio("mdlovofit_cached", "mdcorealign_long"),
                "mdcorealign_long_vs_default_speedup": median_ratio("mdcorealign_default_cached", "mdcorealign_long"),
                "mdcorealign_long_vs_theseus_speedup": median_ratio("theseus_cached", "mdcorealign_long"),
                "mdcorealign_long_lower_RMSD_than_mdlovofit_percent": lower_rmsd_percent,
                "long_mode_equivalence_pass_percent": (
                    float(100.0 * val_subset["long_mode_equivalence_pass"].mean()) if not val_subset.empty else ""
                ),
            }
        )
    return rows


def slow_case_rows(benchmark_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a slow-case rescue table for known MDLovoFit-faster domains."""

    slow_domains = {"1e8rA00", "3hinA02", "1vq8L01"}
    df = pd.DataFrame(benchmark_rows)
    val = pd.DataFrame(validation_rows)
    rows: list[dict[str, object]] = []
    for (domain_id, n_frames), group in df[df["domain_id"].isin(slow_domains)].groupby(["domain_id", "n_frames"]):
        pivot = group.set_index("method")
        if "mdcorealign_long" not in pivot.index or "mdlovofit_cached" not in pivot.index:
            continue
        default_runtime = pivot.loc["mdcorealign_default_cached", "median_runtime_sec"] if "mdcorealign_default_cached" in pivot.index else ""
        long_runtime = float(pivot.loc["mdcorealign_long", "median_runtime_sec"])
        mdlovo_runtime = float(pivot.loc["mdlovofit_cached", "median_runtime_sec"])
        match = val[(val["domain_id"] == domain_id) & (val["n_frames"].astype(int) == int(n_frames))]
        pass_value = int(match["long_mode_equivalence_pass"].iloc[0]) if not match.empty else ""
        rows.append(
            {
                "domain_id": domain_id,
                "n_residues": int(group["n_residues"].iloc[0]),
                "n_frames": int(n_frames),
                "default_runtime": default_runtime,
                "long_mode_runtime": long_runtime,
                "mdlovofit_runtime": mdlovo_runtime,
                "default_vs_mdlovofit_speedup": (mdlovo_runtime / float(default_runtime)) if default_runtime != "" else "",
                "long_vs_mdlovofit_speedup": mdlovo_runtime / long_runtime if long_runtime > 0 else "",
                "long_mode_equivalence_pass": pass_value,
                "bottleneck_stage": "representative_weight_estimation" if long_runtime > mdlovo_runtime else "external_baseline_overhead_reduced",
                "notes": "Known synthetic slow case; use as stress-test diagnosis, not primary real continuous MD evidence.",
            }
        )
    return rows


def build_report(profile_rows: list[dict[str, object]], validation_rows: list[dict[str, object]], benchmark_rows: list[dict[str, object]], speed_rows: list[dict[str, object]]) -> str:
    """Build the long-trajectory optimization report."""

    validation = pd.DataFrame(validation_rows)
    benchmark = pd.DataFrame(benchmark_rows)
    speed = pd.DataFrame(speed_rows)
    pass_rate = float(100.0 * validation["long_mode_equivalence_pass"].mean()) if not validation.empty else 0.0
    real_speed = speed[speed["benchmark_set"] == "real_continuous_500"] if not speed.empty else pd.DataFrame()
    stress_speed = speed[speed["benchmark_set"] == "transition_aligned_stress"] if not speed.empty else pd.DataFrame()
    return "\n".join(
        [
            "# Long Trajectory Optimization Report",
            "",
            "## Scope",
            "",
            "This pass adds and evaluates an explicit MDCoreAlign long mode. It is optimized for real long continuous MD trajectories, not for raw or transition-aligned replicate stitching. Local real continuous mdCATH inputs currently contain single 320 K replicates of about 500 frames; transition-aligned 2k/5k/10k inputs are retained only as synthetic scaling stress tests.",
            "",
            "## Long-Mode Algorithm",
            "",
            "Long mode estimates stable-core weights from representative frames, freezes those weights, applies weighted Kabsch to the full trajectory in chunks, and accumulates robust/ordinary RMSF using streaming statistics. The output explicitly records selected frame indices, timing, and the fact that RMSF uses all frames.",
            "",
            "## Why the Synthetic Stitching Test Was Mixed",
            "",
            "MDLovoFit can remain faster on small synthetic repeated-replicate inputs because its compiled hard-trimming workflow has low overhead, while MDCoreAlign still performs soft-weight estimation and full workflow output generation. The transition-aligned stress panel therefore remains useful for scaling diagnostics, not as evidence from physically continuous MD.",
            "",
            "## Default vs Long-Mode Equivalence",
            "",
            f"Validation rows: {len(validation_rows)}. Strict equivalence pass rate under the predeclared tolerances: {pass_rate:.1f}%. Failures should be interpreted as cases where default mode remains the primary scientific workflow.",
            "",
            "## Runtime Summary",
            "",
            f"Benchmark rows written: {len(benchmark_rows)}. Real continuous speed rows: {len(real_speed)}. Synthetic stress speed rows: {len(stress_speed)}.",
            "",
            "## Selling-Point Assessment",
            "",
            "The supported wording is conservative: MDCoreAlign now provides a default full-workflow mode and an approximate long-trajectory mode. Long mode should be claimed only where equivalence checks pass or differences are explicitly reported. Do not claim that MDCoreAlign is always faster than MDLovoFit.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "MDCoreAlign provides a default full-workflow mode and a long-trajectory mode. In long mode, stable-core weights are estimated from representative frames and applied to the full trajectory using chunked weighted fitting and streaming RMSF. This mode is designed for large continuous trajectories and is reported with explicit equivalence diagnostics against the default solver.",
            "",
            "## Output Files",
            "",
            "- `results/long_optimization/profile/current_long_runtime_profile.tsv`",
            "- `results/long_optimization/validation/default_vs_long_mode_equivalence.tsv`",
            "- `results/long_optimization/benchmark/long_mode_vs_baselines.tsv`",
            "- `results/long_optimization/benchmark/speedup_vs_frame_count.tsv`",
            "- `results/long_optimization/slow_case_rescue_analysis.tsv`",
        ]
    ) + "\n"


def main() -> int:
    """Run the focused long-trajectory optimization pass."""

    global OUT_ROOT, PROFILE_DIR, VALIDATION_DIR, BENCHMARK_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    parser.add_argument("--skip-transition-10k", action="store_true", help="Skip 10k synthetic stress inputs for quick local checks")
    parser.add_argument("--validation-transition-frames", type=int, default=2000, help="Transition stress length used for default-vs-long validation")
    args = parser.parse_args()

    OUT_ROOT = Path(args.output_root)
    PROFILE_DIR = OUT_ROOT / "profile"
    VALIDATION_DIR = OUT_ROOT / "validation"
    BENCHMARK_DIR = OUT_ROOT / "benchmark"
    for path in [PROFILE_DIR, VALIDATION_DIR, BENCHMARK_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    continuous = continuous_cases()
    transition = transition_cases()
    if args.skip_transition_10k:
        transition = [case for case in transition if int(case["n_frames"]) < 10000]

    profile_cases = []
    if continuous:
        profile_cases.append(continuous[0])
    profile_cases.extend([case for case in transition if int(case["n_frames"]) in {2000, 10000}][:2])
    profile_rows: list[dict[str, object]] = []
    for case in profile_cases:
        profile_rows.extend(profile_default_runtime(case))
    write_tsv(PROFILE_DIR / "current_long_runtime_profile.tsv", profile_rows)

    validation_cases = list(continuous)
    validation_cases.extend([case for case in transition if int(case["n_frames"]) == int(args.validation_transition_frames)])
    validation_rows = validate_cases(validation_cases)
    validation_fields = list(validation_rows[0]) if validation_rows else []
    write_tsv(VALIDATION_DIR / "default_vs_long_mode_equivalence.tsv", validation_rows, validation_fields)

    benchmark_cases = list(continuous)
    benchmark_cases.extend(transition)
    benchmark_rows = benchmark_long_mode(benchmark_cases)
    benchmark_fields = list(benchmark_rows[0]) if benchmark_rows else []
    write_tsv(BENCHMARK_DIR / "long_mode_vs_baselines.tsv", benchmark_rows, benchmark_fields)

    speed_rows = speedup_summary(benchmark_rows, validation_rows)
    write_tsv(BENCHMARK_DIR / "speedup_vs_frame_count.tsv", speed_rows, list(speed_rows[0]) if speed_rows else [])

    slow_rows = slow_case_rows(benchmark_rows, validation_rows)
    write_tsv(OUT_ROOT / "slow_case_rescue_analysis.tsv", slow_rows, list(slow_rows[0]) if slow_rows else [])

    DOCS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_REPORT.write_text(build_report(profile_rows, validation_rows, benchmark_rows, speed_rows))
    (OUT_ROOT / "long_trajectory_optimization_report.md").write_text(DOCS_REPORT.read_text())
    print(f"Wrote long optimization outputs to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
