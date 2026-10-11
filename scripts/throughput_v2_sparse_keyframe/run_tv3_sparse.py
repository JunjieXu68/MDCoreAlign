#!/usr/bin/env python3
"""Run TV3 sparse-landmark development ablation and lock two candidates."""

from __future__ import annotations

import json
import os
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
    LANDMARK_COUNTS,
    fit_sparse_all_frames,
    select_weighted_landmarks,
)
from mdsuperpose.chronology_study import representative_indices  # noqa: E402
from mdsuperpose.window_kinematics import scan_local_kinematics  # noqa: E402
from scripts.throughput_v2_sparse_keyframe.study_utils import (  # noqa: E402
    OUT,
    compare_alignment,
    frozen_current_throughput,
    load_study_case,
    read_tsv,
    write_tsv,
)


PHASE = OUT / "03_sparse_landmarks"
PANEL = OUT / "07_fresh_validation/development_panel.tsv"
METHODS = (
    ("top_stable", 0),
    ("sequence_spaced", 2),
    ("sequence_spaced", 4),
    ("sequence_spaced", 8),
    ("spatial_farthest", 0),
    ("chain_balanced", 0),
    ("conditioning_aware", 0),
    ("w5_low_deformation", 0),
)
POOL_FRACTIONS = (0.30, 0.40, 0.50, 0.60, 1.00)


def screening_panel() -> list[dict[str, str]]:
    rows = read_tsv(PANEL)
    atlas = [row for row in rows if row["source_class"].startswith("ATLAS")]
    mdcath = [row for row in rows if row["source_class"].startswith("mdCATH")]
    selected = atlas[:2] + mdcath[:12]
    for frames in (10_000, 50_000, 100_000):
        selected.append(
            {
                "case_id": f"transition_control_4oelA02_{frames}",
                "family_id": "MDCATH:4oelA02",
                "domain_id": "4oelA02",
                "source_class": "transition_aligned_synthetic_scaling_control",
                "input_path": "registered_raw_replicates",
                "input_key": "transition_aligned_cycle",
                "n_frames": str(frames),
                "n_residues": "153",
                "panel_role": "development_scaling_control",
            }
        )
    return selected


def representative_w5(coords: np.ndarray, chains: np.ndarray) -> tuple[np.ndarray, float]:
    selected = representative_indices(coords.shape[0], min(256, coords.shape[0]))
    started = time.perf_counter()
    result = scan_local_kinematics(
        np.asarray(coords[selected], dtype=np.float64),
        chain_ids=chains,
        evaluate_blocks=False,
    )
    return np.asarray(result.local_deformation_index, dtype=np.float64), time.perf_counter() - started


def config_id(method: str, spacing: int, count: int, pool: float) -> str:
    suffix = f"_s{spacing}" if spacing else ""
    return f"{method}{suffix}_M{count}_P{int(round(pool * 100)):03d}"


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    existing = read_tsv(path) if path.exists() else []
    existing.extend(rows)
    write_tsv(path, existing)


def run_screen() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    registry_path = PHASE / "landmark_registry.tsv"
    ablation_path = PHASE / "development_ablation.tsv"
    conditioning_path = PHASE / "conditioning_results.tsv"
    completed = {
        row["case_id"]
        for row in read_tsv(ablation_path)
    } if ablation_path.exists() else set()
    panel = screening_panel()
    for case_number, row in enumerate(panel, start=1):
        if row["case_id"] in completed:
            continue
        decode_start = time.perf_counter()
        coords, chains = load_study_case(row)
        decode_seconds = time.perf_counter() - decode_start
        baseline = frozen_current_throughput(coords, chains)
        w5, w5_seconds = representative_w5(coords, chains)
        registry_rows: list[dict[str, object]] = []
        conditioning_rows: list[dict[str, object]] = []
        ablation_rows: list[dict[str, object]] = []
        for count in LANDMARK_COUNTS:
            if count > coords.shape[1]:
                continue
            for method, spacing in METHODS:
                for pool in POOL_FRACTIONS:
                    candidate_id = config_id(method, spacing, count, pool)
                    try:
                        landmarks = select_weighted_landmarks(
                            coords[0],
                            baseline.weights,
                            count=count,
                            method=method,
                            pool_fraction=pool,
                            chain_ids=chains,
                            minimum_sequence_spacing=max(2, spacing),
                            representative_w5=w5 if method == "w5_low_deformation" else None,
                        )
                    except Exception as error:
                        registry_rows.append(
                            {
                                "case_id": row["case_id"],
                                "config_id": candidate_id,
                                "method": method,
                                "minimum_sequence_spacing": spacing,
                                "landmark_count": count,
                                "pool_fraction": pool,
                                "status": "selection_failure",
                                "failure": type(error).__name__,
                            }
                        )
                        continue
                    common = {
                        "case_id": row["case_id"],
                        "source_class": row["source_class"],
                        "config_id": candidate_id,
                        "method": method,
                        "minimum_sequence_spacing": spacing,
                        "landmark_count_requested": count,
                        "landmark_count_actual": landmarks.actual_count,
                        "pool_fraction": pool,
                        "selection_runtime_seconds": landmarks.runtime_seconds,
                        "representative_w5_seconds_untimed_selection_evidence": w5_seconds,
                        "status": "valid" if landmarks.valid else "invalid",
                        "warning": landmarks.warning or "",
                    }
                    registry_rows.append(common)
                    conditioning_rows.append(
                        {
                            **common,
                            "covariance_rank": landmarks.covariance_rank,
                            "smallest_singular_value": landmarks.smallest_singular_value,
                            "condition_number": landmarks.condition_number,
                            "spatial_radius": landmarks.spatial_radius,
                            "spatial_coverage": landmarks.spatial_coverage,
                            "chain_coverage": landmarks.chain_coverage,
                            "sequence_coverage": landmarks.sequence_coverage,
                            "deterministic_indices": ",".join(map(str, landmarks.indices)),
                        }
                    )
                    if not landmarks.valid:
                        continue
                    candidate_start = time.perf_counter()
                    candidate = fit_sparse_all_frames(
                        coords,
                        coords[0],
                        baseline.weights,
                        landmarks,
                    )
                    candidate_fit_seconds = time.perf_counter() - candidate_start
                    quality = compare_alignment(candidate, baseline, coords)
                    candidate_total = (
                        baseline.weight_seconds
                        + landmarks.runtime_seconds
                        + candidate_fit_seconds
                    )
                    ablation_rows.append(
                        {
                            **common,
                            "n_frames": coords.shape[0],
                            "n_residues": coords.shape[1],
                            "decode_seconds_untimed": decode_seconds,
                            "baseline_weight_seconds": baseline.weight_seconds,
                            "baseline_all_frame_seconds": baseline.fit_seconds,
                            "baseline_total_seconds": baseline.total_seconds,
                            "candidate_sparse_fit_and_metrics_seconds": candidate_fit_seconds,
                            "candidate_total_seconds": candidate_total,
                            "speedup_vs_current_throughput": baseline.total_seconds / candidate_total,
                            **quality,
                        }
                    )
        append_rows(registry_path, registry_rows)
        append_rows(conditioning_path, conditioning_rows)
        append_rows(ablation_path, ablation_rows)
        print(f"[TV3 screen] {case_number}/{len(panel)} {row['case_id']} rows={len(ablation_rows)}", flush=True)


def summarize_and_lock() -> None:
    rows = read_tsv(PHASE / "development_ablation.tsv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)
    summary: list[dict[str, object]] = []
    for candidate_id, values in grouped.items():
        catastrophic = sum(row["catastrophic_mismatch"] == "True" for row in values)
        summary.append(
            {
                "config_id": candidate_id,
                "method": values[0]["method"],
                "minimum_sequence_spacing": values[0]["minimum_sequence_spacing"],
                "landmark_count": values[0]["landmark_count_requested"],
                "pool_fraction": values[0]["pool_fraction"],
                "n_cases": len(values),
                "catastrophic_count": catastrophic,
                "median_speedup": float(np.median([float(row["speedup_vs_current_throughput"]) for row in values])),
                "median_rmsd_abs_diff_A": float(np.median([float(row["rmsd_abs_diff_median_A"]) for row in values])),
                "p95_of_case_p95_rmsd_abs_diff_A": float(np.quantile([float(row["rmsd_abs_diff_p95_A"]) for row in values], 0.95)),
                "median_rmsf_spearman": float(np.median([float(row["stable_rmsf_spearman"]) for row in values])),
                "p5_rmsf_spearman": float(np.quantile([float(row["stable_rmsf_spearman"]) for row in values], 0.05)),
                "median_rmsf_mae_A": float(np.median([float(row["stable_rmsf_mae_A"]) for row in values])),
            }
        )
    for row in summary:
        row["balanced_quality_screen_pass"] = bool(
            row["catastrophic_count"] == 0
            and row["median_rmsd_abs_diff_A"] <= 0.10
            and row["p95_of_case_p95_rmsd_abs_diff_A"] <= 0.50
            and row["median_rmsf_spearman"] >= 0.995
            and row["p5_rmsf_spearman"] >= 0.98
            and row["median_rmsf_mae_A"] <= 0.10
        )
        row["fast_quality_screen_pass"] = bool(
            row["catastrophic_count"] == 0
            and row["median_rmsd_abs_diff_A"] <= 0.25
            and row["p95_of_case_p95_rmsd_abs_diff_A"] <= 1.0
            and row["median_rmsf_spearman"] >= 0.98
            and row["p5_rmsf_spearman"] >= 0.95
            and row["median_rmsf_mae_A"] <= 0.25
        )
    write_tsv(PHASE / "quality_speed_pareto.tsv", summary)

    eligible = [row for row in summary if row["catastrophic_count"] == 0 and row["n_cases"] >= 8]
    balanced_pool = [row for row in eligible if row["balanced_quality_screen_pass"]]
    fast_pool = [row for row in eligible if row["fast_quality_screen_pass"]]
    if not balanced_pool:
        balanced_pool = sorted(
            eligible,
            key=lambda row: (
                row["median_rmsd_abs_diff_A"],
                row["median_rmsf_mae_A"],
                -row["median_speedup"],
            ),
        )[:1]
        balanced_status = "locked_for_safety_validation_not_balanced_gate_pass"
    else:
        balanced_status = "balanced_screen_pass"
    if not fast_pool:
        fast_pool = sorted(
            eligible,
            key=lambda row: (
                -row["median_speedup"],
                row["median_rmsd_abs_diff_A"],
            ),
        )[:1]
        fast_status = "locked_for_tail_validation_not_fast_gate_pass"
    else:
        fast_status = "fast_screen_pass"
    balanced = max(balanced_pool, key=lambda row: row["median_speedup"])
    fast = max(fast_pool, key=lambda row: row["median_speedup"])
    locks = []
    for label, value, status in (
        ("sparse_balanced", balanced, balanced_status),
        ("sparse_fast", fast, fast_status),
    ):
        locks.append(
            {
                "candidate": label,
                **value,
                "lock_status": status,
                "fresh_results_seen": False,
            }
        )
    write_tsv(PHASE / "locked_sparse_configs.tsv", locks)
    report = f"""# Sparse Landmark Method Report

The broad development screen evaluated {len(summary)} registered configurations across {len({row['case_id'] for row in rows})} development/stress cases. Stable weights were identical to current throughput; only the per-frame transform fit used sparse landmarks.

- Balanced lock: `{balanced['config_id']}` ({balanced_status}).
- Fast lock: `{fast['config_id']}` ({fast_status}).

Candidate-pool fractions 30%, 40%, 50%, 60%, and an all-residue spatial control were tested. Representative-frame W5 was used only as untimed selection evidence for L6 and never added as a full-trajectory timed pass. The locked candidates remain internal and require full development/challenge validation before TV4.
"""
    (PHASE / "landmark_method_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    run_screen()
    summarize_and_lock()
    print(json.dumps({"status": "complete", "phase": "TV3_screen"}))


if __name__ == "__main__":
    main()
