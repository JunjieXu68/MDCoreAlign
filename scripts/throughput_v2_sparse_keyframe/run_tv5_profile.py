#!/usr/bin/env python3
"""Run TV5 sampled stable-RMSF/profile development calibration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdcorealign.experimental_throughput_v2 import (  # noqa: E402
    build_frame_signature,
    fit_sampled_profile,
    select_keyframes,
    select_weighted_landmarks,
    temporal_sample_indices,
)
from scripts.throughput_v2_sparse_keyframe.run_tv4_keyframes import development_cases  # noqa: E402
from scripts.throughput_v2_sparse_keyframe.study_utils import (  # noqa: E402
    OUT,
    frozen_current_throughput,
    load_study_case,
    profile_quality,
    write_tsv,
)


PHASE = OUT / "05_profile_sampling"
FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
STRATEGIES = (
    "uniform",
    "stratified_blocks",
    "keyframe_interval",
    "transition_oversampled",
    "motion_weighted",
    "adaptive_variance",
)


def top_fraction_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    count = max(1, int(np.ceil(values.size * fraction)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def run() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, object]] = []
    ablation: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    uncertainty_rows: list[dict[str, object]] = []
    for case_number, row in enumerate(development_cases(), start=1):
        coords, chains = load_study_case(row)
        baseline = frozen_current_throughput(coords, chains)
        landmarks = select_weighted_landmarks(
            coords[0],
            baseline.weights,
            count=min(64, coords.shape[1]),
            method="spatial_farthest",
            pool_fraction=1.0,
            chain_ids=chains,
        )
        signature, signature_metadata = build_frame_signature(
            coords, landmarks.indices, method="pair_distances"
        )
        keyframes, _ = select_keyframes(
            signature,
            method="max_gap_novelty",
            fraction=0.05,
            maximum_gap=50,
        )
        increments = np.concatenate(
            ([0.0], np.linalg.norm(np.diff(signature, axis=0), axis=1))
        )
        curvature = np.zeros(coords.shape[0], dtype=np.float64)
        if coords.shape[0] > 2:
            curvature[1:-1] = np.linalg.norm(
                signature[2:] - 2.0 * signature[1:-1] + signature[:-2],
                axis=1,
            )
        rolling_variance = np.zeros(coords.shape[0], dtype=np.float64)
        if coords.shape[0] > 2:
            rolling_variance[1:-1] = np.mean(
                (signature[1:-1] - 0.5 * (signature[:-2] + signature[2:])) ** 2,
                axis=1,
            )
        for fraction in FRACTIONS:
            for strategy in STRATEGIES:
                scores = None
                if strategy == "transition_oversampled":
                    scores = curvature
                elif strategy == "motion_weighted":
                    scores = increments
                elif strategy == "adaptive_variance":
                    scores = rolling_variance
                sample = temporal_sample_indices(
                    coords.shape[0],
                    fraction=fraction,
                    strategy=strategy,
                    keyframes=keyframes,
                    transition_scores=scores,
                )
                started = time.perf_counter()
                profile = fit_sampled_profile(
                    coords,
                    coords[0],
                    baseline.weights,
                    landmarks,
                    sample,
                )
                profile_seconds = time.perf_counter() - started
                quality = profile_quality(
                    profile.robust_rmsf,
                    profile.rmsf_ci_lower,
                    profile.rmsf_ci_upper,
                    baseline.fit.robust_rmsf,
                )
                peak_overlap = jaccard(
                    top_fraction_mask(profile.robust_rmsf),
                    top_fraction_mask(baseline.fit.robust_rmsf),
                )
                config = f"{strategy}__F{fraction:g}"
                common = {
                    "case_id": row["case_id"],
                    "config_id": config,
                    "strategy": strategy,
                    "target_fraction": fraction,
                    "actual_fraction": sample.size / coords.shape[0],
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "sample_count": sample.size,
                    "effective_sample_size": profile.effective_sample_size,
                    "signature_seconds": signature_metadata["construction_seconds"],
                    "profile_seconds": profile_seconds,
                    "baseline_total_seconds": baseline.total_seconds,
                    "candidate_total_seconds": baseline.weight_seconds
                    + float(signature_metadata["construction_seconds"])
                    + profile_seconds,
                    "speedup_vs_current_throughput": baseline.total_seconds
                    / (
                        baseline.weight_seconds
                        + float(signature_metadata["construction_seconds"])
                        + profile_seconds
                    ),
                    "peak_10pct_jaccard": peak_overlap,
                    **quality,
                }
                registry.append(
                    {
                        "config_id": config,
                        "strategy": strategy,
                        "target_fraction": fraction,
                        "uncertainty_method": profile.metadata["uncertainty_method"],
                        "profile_contract": "v2_profile",
                    }
                )
                ablation.append(common)
                quality_rows.append(
                    {
                        key: value
                        for key, value in common.items()
                        if key
                        in {
                            "case_id",
                            "config_id",
                            "strategy",
                            "target_fraction",
                            "stable_rmsf_spearman",
                            "stable_rmsf_mae_A",
                            "peak_10pct_jaccard",
                            "catastrophic_mismatch",
                        }
                    }
                )
                uncertainty_rows.append(
                    {
                        "case_id": row["case_id"],
                        "config_id": config,
                        "strategy": strategy,
                        "target_fraction": fraction,
                        "effective_sample_size": profile.effective_sample_size,
                        "interval_coverage": quality["interval_coverage"],
                        "calibration_slope": quality["calibration_slope"],
                        "undercoverage": float(quality["interval_coverage"]) < 0.90,
                    }
                )
        write_tsv(PHASE / "sampling_registry.tsv", list({row["config_id"]: row for row in registry}.values()))
        write_tsv(PHASE / "development_ablation.tsv", ablation)
        write_tsv(PHASE / "rmsf_quality.tsv", quality_rows)
        write_tsv(PHASE / "uncertainty_calibration.tsv", uncertainty_rows)
        print(f"[TV5] {case_number}/{len(development_cases())} {row['case_id']}", flush=True)
    lock(ablation)


def lock(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_id"]), []).append(row)
    summary: list[dict[str, object]] = []
    for config, values in grouped.items():
        summary.append(
            {
                "config_id": config,
                "strategy": values[0]["strategy"],
                "target_fraction": values[0]["target_fraction"],
                "n_cases": len(values),
                "catastrophic_count": sum(bool(row["catastrophic_mismatch"]) for row in values),
                "median_speedup": float(np.median([float(row["speedup_vs_current_throughput"]) for row in values])),
                "median_rmsf_spearman": float(np.median([float(row["stable_rmsf_spearman"]) for row in values])),
                "p5_rmsf_spearman": float(np.quantile([float(row["stable_rmsf_spearman"]) for row in values], 0.05)),
                "median_rmsf_mae_A": float(np.median([float(row["stable_rmsf_mae_A"]) for row in values])),
                "median_interval_coverage": float(np.median([float(row["interval_coverage"]) for row in values])),
                "median_peak_10pct_jaccard": float(np.median([float(row["peak_10pct_jaccard"]) for row in values])),
            }
        )
    for row in summary:
        row["balanced_pass"] = bool(
            row["catastrophic_count"] == 0
            and row["median_rmsf_spearman"] >= 0.995
            and row["p5_rmsf_spearman"] >= 0.98
            and row["median_rmsf_mae_A"] <= 0.10
            and row["median_interval_coverage"] >= 0.90
        )
        row["fast_pass"] = bool(
            row["catastrophic_count"] == 0
            and row["median_rmsf_spearman"] >= 0.98
            and row["p5_rmsf_spearman"] >= 0.95
            and row["median_rmsf_mae_A"] <= 0.25
            and row["median_interval_coverage"] >= 0.80
        )
    write_tsv(PHASE / "profile_quality_speed_summary.tsv", summary)
    locks: list[dict[str, object]] = []
    for label, gate in (("profile_balanced", "balanced_pass"), ("profile_fast", "fast_pass")):
        eligible = [row for row in summary if row[gate]]
        if eligible:
            selected = max(eligible, key=lambda row: row["median_speedup"])
            status = f"{gate}_development"
        else:
            selected = min(
                summary,
                key=lambda row: (
                    row["catastrophic_count"],
                    -row["median_rmsf_spearman"],
                    row["median_rmsf_mae_A"],
                ),
            )
            status = "rejected_uncertainty_or_quality_gate"
        locks.append(
            {
                "candidate": label,
                **selected,
                "lock_status": status,
                "fresh_results_seen": False,
            }
        )
    write_tsv(PHASE / "locked_profile_configs.tsv", locks)


def main() -> None:
    run()
    print(json.dumps({"phase": "TV5", "status": "complete"}))


if __name__ == "__main__":
    main()
