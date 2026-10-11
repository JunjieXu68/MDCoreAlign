#!/usr/bin/env python3
"""Run locked Throughput-v2 candidates on the prospectively frozen TV7 panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdcorealign.experimental_throughput_v2 import (  # noqa: E402
    deterministic_audit_indices,
    deterministic_core_mask,
    fit_sampled_profile,
    fit_sparse_all_frames,
    select_weighted_landmarks,
    temporal_sample_indices,
)
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights  # noqa: E402
from mdsuperpose.chronology_study import representative_indices  # noqa: E402
from mdsuperpose.fast_kabsch import apply_transforms_batch  # noqa: E402
from mdsuperpose.parallel_weights import estimate_shared_weight  # noqa: E402
from scripts.throughput_v2_sparse_keyframe.study_utils import (  # noqa: E402
    OUT,
    build_transition_control,
    jaccard,
    load_study_case,
    profile_quality,
    read_tsv,
    spearman,
    write_tsv,
)


PHASE = OUT / "08_performance"
PANELS = OUT / "07_fresh_validation"
FROZEN_LONG_TIMING = ROOT / "results/cpu_chronology_v3/07_performance/fresh_long_repeats.tsv"
FROZEN_MDCATH_TIMING = ROOT / "results/cpu_chronology_v3/07_performance/fresh_mdcath_128_repeats.tsv"
TV2_TIMING = OUT / "02_current_throughput_audit/memory_timing.tsv"
TV3_DEVELOPMENT = OUT / "03_sparse_landmarks/development_ablation.tsv"
REPEATED_IDS = {
    row["case_id"]
    for row in read_tsv(
        ROOT / "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_performance_128.tsv"
    )
}

SPARSE_CONFIG = {
    "config_id": "spatial_farthest_M64_P100",
    "method": "spatial_farthest",
    "count": 64,
    "pool_fraction": 1.0,
}
PROFILE_CONFIGS = {
    "profile_balanced": {
        "config_id": "stratified_blocks__F0.2",
        "strategy": "stratified_blocks",
        "fraction": 0.20,
    },
    "profile_fast": {
        "config_id": "motion_weighted__F0.05",
        "strategy": "motion_weighted",
        "fraction": 0.05,
    },
}


def append_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    existing = read_tsv(path) if path.exists() else []
    existing.extend(rows)
    write_tsv(path, existing)


def median_map(path: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    if not path.exists():
        return {}
    for row in read_tsv(path):
        if row.get("method") != "throughput_compact_f64":
            continue
        value = row.get("loaded_runtime_seconds", "")
        if value:
            grouped.setdefault(row["case_id"], []).append(float(value))
    return {case_id: float(np.median(values)) for case_id, values in grouped.items()}


def control_baseline_map() -> dict[str, float]:
    output: dict[str, float] = {}
    if not TV2_TIMING.exists():
        return output
    for row in read_tsv(TV2_TIMING):
        case_id = row.get("case_id", "")
        if case_id.startswith("transition_control_4oelA02_"):
            output[case_id] = float(row["algorithm_total_seconds"])
    if TV3_DEVELOPMENT.exists():
        grouped: dict[str, list[float]] = {}
        for row in read_tsv(TV3_DEVELOPMENT):
            case_id = row.get("case_id", "")
            if case_id.startswith("transition_control_4oelA02_"):
                grouped.setdefault(case_id, []).append(float(row["baseline_total_seconds"]))
        for case_id, values in grouped.items():
            output.setdefault(case_id, float(np.median(values)))
    return output


FROZEN_RUNTIME = {
    **median_map(FROZEN_LONG_TIMING),
    **median_map(FROZEN_MDCATH_TIMING),
    **control_baseline_map(),
}


def current_weights(coords: np.ndarray, chains: np.ndarray) -> tuple[np.ndarray, float]:
    selected = representative_indices(coords.shape[0])
    started = time.perf_counter()
    estimate = estimate_shared_weight(
        np.asarray(coords[selected], dtype=np.float64),
        np.asarray(coords[0], dtype=np.float64),
        chain_ids=chains,
    )
    return np.asarray(estimate.mean_weights, dtype=np.float64), time.perf_counter() - started


def motion_scores(coords: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    values = np.asarray(coords[:, landmarks], dtype=np.float64)
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float64), time.perf_counter() - started
    left, right = np.triu_indices(values.shape[1], k=1)
    separation = np.abs(landmarks[right] - landmarks[left])
    order = np.lexsort((right, left, -separation))
    order = order[np.rint(np.linspace(0, max(0, order.size - 1), min(128, order.size))).astype(int)]
    distances = np.linalg.norm(values[:, left[order]] - values[:, right[order]], axis=2)
    scale = np.maximum(distances[0], 1e-12)
    signature = distances / scale[None, :] - 1.0
    scores = np.concatenate(([0.0], np.linalg.norm(np.diff(signature, axis=0), axis=1)))
    return scores, time.perf_counter() - started


def audit_reference(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    audit: np.ndarray,
) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fit_all_frames_fixed_weights(
        np.asarray(coords[audit], dtype=np.float64),
        reference,
        weights,
        chunk_size=512,
        return_aligned=False,
        compute_ordinary_rmsf=False,
    )
    return result, time.perf_counter() - started


def sparse_audit_quality(
    coords: np.ndarray,
    weights: np.ndarray,
    candidate: Any,
    audit: np.ndarray,
    baseline: Any,
) -> dict[str, object]:
    candidate_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64),
        candidate.rotations[audit],
        candidate.translations[audit],
    )
    baseline_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64),
        baseline.rotations,
        baseline.translations,
    )
    coordinate = np.sqrt(np.mean(np.sum((candidate_aligned - baseline_aligned) ** 2, axis=2), axis=1))
    rmsd_delta = np.abs(candidate.rmsd[audit] - baseline.rmsd)
    relative = candidate.rotations[audit] @ np.swapaxes(baseline.rotations, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation = np.degrees(np.arccos(cosine))
    audit_rmsf = np.sqrt(np.sum(np.var(candidate_aligned, axis=0), axis=1))
    audit_rmsf_delta = np.abs(audit_rmsf - baseline.robust_rmsf)
    audit_rho = spearman(audit_rmsf, baseline.robust_rmsf)
    nonfinite = not all(
        np.isfinite(item).all()
        for item in (candidate.rotations, candidate.translations, candidate.rmsd, candidate.robust_rmsf)
    )
    reflected = bool(np.any(np.linalg.det(candidate.rotations) <= 0.0))
    catastrophic = bool(
        nonfinite
        or reflected
        or float(np.median(coordinate)) > 2.0
        or float(np.quantile(rmsd_delta, 0.95)) > 2.0
        or audit_rho < 0.90
    )
    return {
        "stable_weight_spearman": 1.0,
        "core_jaccard": 1.0,
        "rotation_geodesic_median_degrees": float(np.median(rotation)),
        "rotation_geodesic_p95_degrees": float(np.quantile(rotation, 0.95)),
        "aligned_coordinate_audit_median_A": float(np.median(coordinate)),
        "aligned_coordinate_audit_p95_A": float(np.quantile(coordinate, 0.95)),
        "rmsd_abs_diff_median_A": float(np.median(rmsd_delta)),
        "rmsd_abs_diff_p95_A": float(np.quantile(rmsd_delta, 0.95)),
        "stable_rmsf_spearman": audit_rho,
        "stable_rmsf_mae_A": float(np.mean(audit_rmsf_delta)),
        "uncertainty_coverage": "not_applicable_contract_A",
        "quality_reference_scope": "deterministic_audit_frames_only",
        "non_finite": nonfinite,
        "reflected_rotation": reflected,
        "catastrophic_mismatch": catastrophic,
    }


def run_sparse(coords: np.ndarray, chains: np.ndarray) -> tuple[Any, dict[str, float], np.ndarray]:
    total_start = time.perf_counter()
    weights, weight_seconds = current_weights(coords, chains)
    select_start = time.perf_counter()
    landmarks = select_weighted_landmarks(
        coords[0],
        weights,
        count=min(SPARSE_CONFIG["count"], coords.shape[1]),
        method=SPARSE_CONFIG["method"],
        pool_fraction=SPARSE_CONFIG["pool_fraction"],
        chain_ids=chains,
    )
    selection_seconds = time.perf_counter() - select_start
    fit_start = time.perf_counter()
    result = fit_sparse_all_frames(coords, coords[0], weights, landmarks)
    fit_seconds = time.perf_counter() - fit_start
    stages = {
        "weight_estimation_seconds": weight_seconds,
        "landmark_selection_seconds": selection_seconds,
        "signature_construction_seconds": 0.0,
        "explicit_keyframe_fit_seconds": float(result.timing["explicit_frame_fits"]),
        "transform_interpolation_seconds": 0.0,
        "full_frame_rmsd_seconds": float(result.timing["RMSD"]),
        "rmsf_profile_seconds": float(result.timing["RMSF"]),
        "audit_seconds_untimed": 0.0,
        "refinement_seconds": 0.0,
        "result_construction_seconds": max(
            0.0,
            fit_seconds
            - float(result.timing["explicit_frame_fits"])
            - float(result.timing["transform_application"])
            - float(result.timing["RMSD"])
            - float(result.timing["RMSF"]),
        ),
        "end_to_end_total_seconds": time.perf_counter() - total_start,
    }
    return result, stages, weights


def run_profile(
    coords: np.ndarray,
    chains: np.ndarray,
    config: dict[str, object],
) -> tuple[Any, dict[str, float], np.ndarray]:
    total_start = time.perf_counter()
    weights, weight_seconds = current_weights(coords, chains)
    select_start = time.perf_counter()
    landmarks = select_weighted_landmarks(
        coords[0],
        weights,
        count=min(64, coords.shape[1]),
        method="spatial_farthest",
        pool_fraction=1.0,
        chain_ids=chains,
    )
    selection_seconds = time.perf_counter() - select_start
    signature_seconds = 0.0
    scores = None
    if config["strategy"] == "motion_weighted":
        scores, signature_seconds = motion_scores(coords, landmarks.indices)
    sample = temporal_sample_indices(
        coords.shape[0],
        fraction=float(config["fraction"]),
        strategy=str(config["strategy"]),
        transition_scores=scores,
    )
    profile_start = time.perf_counter()
    result = fit_sampled_profile(coords, coords[0], weights, landmarks, sample)
    profile_seconds = time.perf_counter() - profile_start
    stages = {
        "weight_estimation_seconds": weight_seconds,
        "landmark_selection_seconds": selection_seconds,
        "signature_construction_seconds": signature_seconds,
        "explicit_keyframe_fit_seconds": float(result.timing["sampled_frame_fits"]),
        "transform_interpolation_seconds": 0.0,
        "full_frame_rmsd_seconds": 0.0,
        "rmsf_profile_seconds": float(result.timing["sampled_profile_statistics"]),
        "audit_seconds_untimed": 0.0,
        "refinement_seconds": 0.0,
        "result_construction_seconds": max(
            0.0,
            profile_seconds
            - float(result.timing["sampled_frame_fits"])
            - float(result.timing["sampled_profile_statistics"]),
        ),
        "end_to_end_total_seconds": time.perf_counter() - total_start,
    }
    return result, stages, weights


def profile_audit_quality(result: Any, weights: np.ndarray, baseline: Any) -> dict[str, object]:
    quality = profile_quality(
        result.robust_rmsf,
        result.rmsf_ci_lower,
        result.rmsf_ci_upper,
        baseline.robust_rmsf,
    )
    return {
        "stable_weight_spearman": 1.0,
        "core_jaccard": jaccard(deterministic_core_mask(weights), deterministic_core_mask(weights)),
        "rotation_geodesic_median_degrees": "not_applicable_contract_C",
        "rotation_geodesic_p95_degrees": "not_applicable_contract_C",
        "aligned_coordinate_audit_median_A": "not_applicable_contract_C",
        "aligned_coordinate_audit_p95_A": "not_applicable_contract_C",
        "rmsd_abs_diff_median_A": "not_applicable_contract_C",
        "rmsd_abs_diff_p95_A": "not_applicable_contract_C",
        "stable_rmsf_spearman": quality["stable_rmsf_spearman"],
        "stable_rmsf_mae_A": quality["stable_rmsf_mae_A"],
        "uncertainty_coverage": quality["interval_coverage"],
        "quality_reference_scope": "deterministic_audit_frames_only",
        "non_finite": not np.isfinite(result.robust_rmsf).all(),
        "reflected_rotation": "not_applicable_contract_C",
        "catastrophic_mismatch": quality["catastrophic_mismatch"],
    }


def runtime_row(
    row: dict[str, str],
    candidate: str,
    config_id: str,
    repeat: int,
    stages: dict[str, float],
    *,
    warmup: bool,
    shared_config: bool = False,
) -> dict[str, object]:
    baseline = FROZEN_RUNTIME.get(row["case_id"])
    total = stages["end_to_end_total_seconds"]
    return {
        "case_id": row["case_id"],
        "family_id": row.get("family_id", ""),
        "panel": row.get("panel_role", row.get("source_class", "")),
        "candidate": candidate,
        "config_id": config_id,
        "repeat": repeat,
        "warmup": warmup,
        "n_frames": row["n_frames"],
        "n_residues": row["n_residues"],
        "candidate_runtime_seconds": total,
        "frozen_current_runtime_seconds": baseline if baseline is not None else "",
        "speedup_vs_frozen_current": baseline / total if baseline is not None else "",
        "baseline_reuse_status": "matched_frozen_timing" if baseline is not None else "unavailable_not_rerun_on_fresh",
        "shared_locked_configuration": shared_config,
        "cpu_threads": 1,
        "loaded_coordinates": True,
        "aligned_coordinates_materialised": False,
        "ordinary_rmsf": False,
    }


def process_case(
    row: dict[str, str],
    *,
    warmup: bool,
    repeats: int,
    panel_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    decode_start = time.perf_counter()
    coords, chains = load_study_case(row)
    decode_seconds = time.perf_counter() - decode_start
    reference = np.asarray(coords[0], dtype=np.float64)
    audit = deterministic_audit_indices(coords.shape[0])

    quality_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []

    configs = ["sparse_balanced", "profile_balanced", "profile_fast"]
    for candidate in configs:
        runner = (
            (lambda: run_sparse(coords, chains))
            if candidate == "sparse_balanced"
            else (lambda candidate=candidate: run_profile(coords, chains, PROFILE_CONFIGS[candidate]))
        )
        if warmup:
            runner()
        final_result = None
        final_weights = None
        final_stages = None
        for repeat in range(1, repeats + 1):
            before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result, stages, weights = runner()
            after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            config_id = SPARSE_CONFIG["config_id"] if candidate == "sparse_balanced" else str(PROFILE_CONFIGS[candidate]["config_id"])
            runtime_rows.append(runtime_row(row, candidate, config_id, repeat, stages, warmup=warmup))
            stage_rows.append(
                {
                    "case_id": row["case_id"],
                    "panel": panel_name,
                    "candidate": candidate,
                    "config_id": config_id,
                    "repeat": repeat,
                    **stages,
                    "decode_seconds_untimed_shared_upstream": decode_seconds,
                }
            )
            memory_rows.append(
                {
                    "case_id": row["case_id"],
                    "panel": panel_name,
                    "candidate": candidate,
                    "repeat": repeat,
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "peak_rss_mib": after_rss / 1024.0,
                    "peak_rss_increment_mib_lower_bound": max(0.0, after_rss - before_rss) / 1024.0,
                    "vram_mib": "not_applicable_CPU_only",
                }
            )
            if candidate == "sparse_balanced":
                runtime_rows.append(
                    runtime_row(
                        row,
                        "sparse_fast",
                        config_id,
                        repeat,
                        stages,
                        warmup=warmup,
                        shared_config=True,
                    )
                )
                stage_rows.append(
                    {
                        **stage_rows[-1],
                        "candidate": "sparse_fast",
                        "shared_locked_configuration": True,
                    }
                )
                memory_rows.append(
                    {
                        **memory_rows[-1],
                        "candidate": "sparse_fast",
                        "shared_locked_configuration": True,
                    }
                )
            final_result, final_weights, final_stages = result, weights, stages

        assert final_result is not None and final_weights is not None and final_stages is not None
        audit_fit, audit_seconds = audit_reference(coords, reference, final_weights, audit)
        final_stages["audit_seconds_untimed"] = audit_seconds
        if candidate == "sparse_balanced":
            quality = sparse_audit_quality(coords, final_weights, final_result, audit, audit_fit)
        else:
            quality = profile_audit_quality(final_result, final_weights, audit_fit)
        config_id = SPARSE_CONFIG["config_id"] if candidate == "sparse_balanced" else str(PROFILE_CONFIGS[candidate]["config_id"])
        common = {
            "case_id": row["case_id"],
            "family_id": row.get("family_id", ""),
            "panel": panel_name,
            "candidate": candidate,
            "config_id": config_id,
            "n_frames": row["n_frames"],
            "n_residues": row["n_residues"],
            "audit_frame_count": audit.size,
            "audit_fraction": audit.size / coords.shape[0],
            "audit_reference_seconds_untimed": audit_seconds,
            "stable_weights_source": "candidate_current_throughput_estimator_identity",
            "refinement_rate": 0.0,
            "unresolved_interval_rate": 0.0,
            **quality,
        }
        quality_rows.append(common)
        if candidate == "sparse_balanced":
            quality_rows.append(
                {
                    **common,
                    "candidate": "sparse_fast",
                    "shared_locked_configuration": True,
                }
            )

    return quality_rows, runtime_rows, stage_rows, memory_rows


def panel_rows(panel: str) -> list[dict[str, str]]:
    if panel == "mdcath":
        rows = read_tsv(PANELS / "fresh_mdcath_512.tsv")
    elif panel == "atlas":
        rows = read_tsv(PANELS / "fresh_atlas_r2_r3.tsv")
    elif panel == "scaling":
        rows = []
        for n_frames in (10_000, 50_000, 100_000):
            rows.append(
                {
                    "case_id": f"transition_control_4oelA02_{n_frames}",
                    "family_id": "MDCATH:4oelA02",
                    "source_class": "transition_aligned_synthetic_scaling_control",
                    "panel_role": "scaling_control",
                    "n_frames": str(n_frames),
                    "n_residues": "153",
                }
            )
    elif panel == "challenge":
        rows = read_tsv(PANELS / "challenge_panel.tsv")
    else:
        raise ValueError(panel)
    return rows


def run_panel(panel: str, limit: int | None = None) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    rows = panel_rows(panel)
    if limit is not None:
        rows = rows[:limit]
    quality_path = PHASE / f".{panel}_quality_checkpoint.tsv"
    runtime_path = PHASE / f".{panel}_runtime_checkpoint.tsv"
    stage_path = PHASE / f".{panel}_stage_checkpoint.tsv"
    memory_path = PHASE / f".{panel}_memory_checkpoint.tsv"
    completed = {row["case_id"] for row in read_tsv(quality_path)} if quality_path.exists() else set()
    for position, row in enumerate(rows, start=1):
        if row["case_id"] in completed:
            continue
        if panel == "mdcath":
            repeated = row["case_id"] in REPEATED_IDS
            repeats, warmup = (5, True) if repeated else (1, False)
        elif panel == "atlas":
            repeats, warmup = 5, True
        elif panel == "scaling":
            repeats, warmup = 3, True
        else:
            repeats, warmup = 1, False
        try:
            quality, runtime, stages, memory = process_case(
                row,
                warmup=warmup,
                repeats=repeats,
                panel_name=panel,
            )
            append_tsv(quality_path, quality)
            append_tsv(runtime_path, runtime)
            append_tsv(stage_path, stages)
            append_tsv(memory_path, memory)
            status = "ok"
        except Exception as error:
            append_tsv(
                PHASE / ".failures_checkpoint.tsv",
                [{
                    "case_id": row["case_id"],
                    "panel": panel,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "stage": "candidate_only_validation",
                }],
            )
            status = f"failed:{type(error).__name__}"
        print(f"[TV8 {panel}] {position}/{len(rows)} {row['case_id']} {status}", flush=True)


def finalise() -> None:
    quality: list[dict[str, str]] = []
    runtime: list[dict[str, str]] = []
    stages: list[dict[str, str]] = []
    memory: list[dict[str, str]] = []
    for panel in ("mdcath", "atlas", "scaling", "challenge"):
        for target, name in (
            (quality, "quality"),
            (runtime, "runtime"),
            (stages, "stage"),
            (memory, "memory"),
        ):
            path = PHASE / f".{panel}_{name}_checkpoint.tsv"
            if path.exists():
                target.extend(read_tsv(path))
    write_tsv(PHASE / "candidate_quality.tsv", quality)
    write_tsv(PHASE / "candidate_runtime.tsv", runtime)
    write_tsv(PHASE / "stage_decomposition.tsv", stages)
    write_tsv(PHASE / "memory.tsv", memory)
    write_tsv(
        PHASE / "atlas_repeats.tsv",
        [row for row in runtime if row["case_id"].startswith("ATLAS_")],
    )
    write_tsv(
        PHASE / "mdcath_all_domain.tsv",
        [row for row in runtime if row.get("panel") == "fresh_locked_validation" and row["repeat"] == "1"],
    )
    write_tsv(
        PHASE / "mdcath_repeated_subset.tsv",
        [row for row in runtime if row["case_id"] in REPEATED_IDS],
    )
    write_tsv(
        PHASE / "scaling_10k_50k_100k.tsv",
        [row for row in runtime if row["case_id"].startswith("transition_control_")],
    )
    failures = read_tsv(PHASE / ".failures_checkpoint.tsv") if (PHASE / ".failures_checkpoint.tsv").exists() else []
    for row in quality:
        if row.get("catastrophic_mismatch") == "True":
            failures.append(
                {
                    "case_id": row["case_id"],
                    "panel": row["panel"],
                    "candidate": row["candidate"],
                    "failure_type": "catastrophic_mismatch",
                    "failure_message": "locked catastrophic definition triggered on audit reference",
                    "stage": "quality_evaluation",
                }
            )
    for family, reason in (
        ("keyframe_balanced/keyframe_fast", "rejected_development_catastrophic"),
        ("adaptive_balanced/adaptive_fast", "rejected_development_quality_or_unresolved_gate"),
    ):
        failures.append(
            {
                "case_id": "not_run_fresh",
                "panel": "fresh_validation",
                "candidate": family,
                "failure_type": "family_skipped_by_registered_rule",
                "failure_message": reason,
                "stage": "TV8_method_selection",
            }
        )
    write_tsv(PHASE / "failures_warnings.tsv", failures)

    summary: list[dict[str, object]] = []
    for candidate in ("sparse_balanced", "sparse_fast", "profile_balanced", "profile_fast"):
        q = [row for row in quality if row["candidate"] == candidate]
        r = [row for row in runtime if row["candidate"] == candidate and row["warmup"] == "False"]
        speedups = [float(row["speedup_vs_frozen_current"]) for row in r if row.get("speedup_vs_frozen_current")]
        rmsf_rho = [float(row["stable_rmsf_spearman"]) for row in q if row.get("stable_rmsf_spearman")]
        rmsf_mae = [float(row["stable_rmsf_mae_A"]) for row in q if row.get("stable_rmsf_mae_A")]
        summary.append(
            {
                "candidate": candidate,
                "quality_cases": len(q),
                "timed_runs": len(r),
                "catastrophic_count": sum(row.get("catastrophic_mismatch") == "True" for row in q),
                "median_speedup_where_frozen_baseline_available": float(np.median(speedups)) if speedups else "",
                "median_audit_rmsf_spearman": float(np.median(rmsf_rho)) if rmsf_rho else "",
                "p5_audit_rmsf_spearman": float(np.quantile(rmsf_rho, 0.05)) if rmsf_rho else "",
                "median_audit_rmsf_mae_A": float(np.median(rmsf_mae)) if rmsf_mae else "",
                "fresh_reference_limitation": "current-throughput full outputs not rerun; deterministic audit frames only",
            }
        )
    write_tsv(PHASE / "quality_speed_pareto.tsv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("mdcath", "atlas", "scaling", "challenge", "all", "finalise"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    panels = ("mdcath", "atlas", "scaling", "challenge") if args.panel == "all" else (args.panel,)
    if args.panel != "finalise":
        for panel in panels:
            run_panel(panel, args.limit)
    finalise()
    print(json.dumps({"phase": "TV8", "status": "complete", "panel": args.panel}))


if __name__ == "__main__":
    main()
