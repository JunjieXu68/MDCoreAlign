#!/usr/bin/env python3
"""Run S2-S6 using only the new stream algorithms and frozen references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import spearmanr

from mdcorealign.experimental_stream import (
    LandmarkSet,
    apply_sampled_transforms,
    build_landmark_evidence,
    fit_sparse_landmarks,
    run_stream_contract,
    sampled_rmsf,
    select_sparse_landmarks,
    temporally_stratified_indices,
)
from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.stream_align_v1.common import (
    ATLAS_SYSTEMS,
    KEYFRAME_FRACTIONS,
    LANDMARK_COUNTS,
    LANDMARK_METHODS,
    OUT,
    REFINEMENT_ROUNDS,
    REPRESENTATIVE_COUNTS,
    ROOT,
    SYNTHETIC,
    frozen_gpu_runtime,
    load_atlas,
    read_tsv,
    write_json,
    write_tsv,
)


FINAL_REPRESENTATIVES = 128
FINAL_LANDMARKS = 128
FINAL_METHOD = "chain_balanced_spatial"
FINAL_KEYFRAME_FRACTION = 0.05
FINAL_REFINEMENT_ROUNDS = 1
NEW_BACKEND = "cpu"


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def _rotation_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.asarray(left) @ np.swapaxes(np.asarray(right), 1, 2)
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _apply(coords: np.ndarray, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    return np.einsum("fni,fji->fnj", coords, rotations) + translations[:, None, :]


def _reference_audit(
    coords: np.ndarray,
    exact_weights: np.ndarray,
    indices: np.ndarray,
):
    """Reconstruct sampled evaluation transforms; never include this in timing."""

    return weighted_kabsch_batch(
        np.asarray(coords)[indices],
        np.asarray(coords)[0],
        exact_weights,
        validate_inputs=False,
        compute_metrics=True,
    )


def _quality(
    coords: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    exact_weights: np.ndarray,
    exact_rmsd: np.ndarray,
    exact_rmsf: np.ndarray,
    *,
    audit_indices: np.ndarray,
    profile_indices: np.ndarray,
) -> dict[str, float | bool]:
    reference = _reference_audit(coords, exact_weights, audit_indices)
    rotation = _rotation_error(rotations[audit_indices], reference.rotations)
    translation = np.linalg.norm(
        translations[audit_indices] - reference.translations,
        axis=1,
    )
    stream_aligned = _apply(
        np.asarray(coords)[audit_indices],
        rotations[audit_indices],
        translations[audit_indices],
    )
    reference_aligned = reference.aligned_coords
    displacement = np.sqrt(
        np.mean(
            np.sum((stream_aligned - reference_aligned) ** 2, axis=2),
            axis=1,
        )
    )
    stream_rmsd = np.sqrt(
        np.mean(
            np.sum((stream_aligned - np.asarray(coords)[0][None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    rmsd_delta = np.abs(stream_rmsd - exact_rmsd[audit_indices])
    approximate_rmsf = sampled_rmsf(
        apply_sampled_transforms(
            coords,
            rotations,
            translations,
            profile_indices,
        )
    )
    return {
        "median_rotation_error_degrees": float(np.median(rotation)),
        "P95_rotation_error_degrees": float(np.quantile(rotation, 0.95)),
        "median_translation_error_A": float(np.median(translation)),
        "P95_translation_error_A": float(np.quantile(translation, 0.95)),
        "median_coordinate_displacement_A": float(np.median(displacement)),
        "P95_coordinate_displacement_A": float(np.quantile(displacement, 0.95)),
        "median_RMSD_difference_A": float(np.median(rmsd_delta)),
        "P95_RMSD_difference_A": float(np.quantile(rmsd_delta, 0.95)),
        "RMSF_Spearman": _spearman(approximate_rmsf, exact_rmsf),
        "RMSF_MAE_A": float(np.mean(np.abs(approximate_rmsf - exact_rmsf))),
        "finite_outputs": bool(
            np.isfinite(rotations).all()
            and np.isfinite(translations).all()
            and np.isfinite(stream_rmsd).all()
            and np.isfinite(approximate_rmsf).all()
        ),
        "valid_transforms": bool(
            np.max(np.abs(np.linalg.det(rotations[audit_indices]) - 1.0)) < 5e-3
        ),
    }


def _sample_indices(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    audit = temporally_stratified_indices(n_frames, min(256, max(64, int(math.ceil(n_frames * 0.0075)))))
    profile = temporally_stratified_indices(n_frames, min(512, n_frames))
    return audit, profile


def _sparse_row(
    system: str,
    coords: np.ndarray,
    cache,
    landmarks: LandmarkSet,
    *,
    evidence_seconds: float,
    representative_count: int,
    landmark_count: int,
    method: str,
) -> dict[str, object]:
    fit = fit_sparse_landmarks(
        coords,
        np.asarray(coords)[0],
        landmarks.indices,
        backend=NEW_BACKEND,
    )
    audit, profile = _sample_indices(coords.shape[0])
    quality_start = time.perf_counter()
    quality = _quality(
        coords,
        fit.rotations,
        fit.translations,
        np.asarray(cache["exact_weights"]),
        np.asarray(cache["exact_rmsd"]),
        np.asarray(cache["exact_rmsf"]),
        audit_indices=audit,
        profile_indices=profile,
    )
    quality_seconds = time.perf_counter() - quality_start
    algorithm_seconds = evidence_seconds + landmarks.timing["landmark_selection_seconds"] + fit.runtime_seconds
    frozen = frozen_gpu_runtime(system)
    return {
        "system": system,
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "representative_frames": representative_count,
        "representative_fraction": representative_count / coords.shape[0],
        "landmark_count": landmark_count,
        "landmark_fraction": landmark_count / coords.shape[1],
        "landmark_method": method,
        "backend": NEW_BACKEND,
        "evidence_seconds": evidence_seconds,
        "selection_seconds": landmarks.timing["landmark_selection_seconds"],
        "all_frame_sparse_fit_seconds": fit.runtime_seconds,
        "algorithm_runtime_seconds": algorithm_seconds,
        "sampled_quality_evaluation_seconds_excluded": quality_seconds,
        "peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_VRAM_bytes": fit.peak_vram_bytes,
        "frozen_GPU_throughput_seconds": frozen,
        "historical_runtime_ratio_frozen_GPU_over_stream": (
            frozen / algorithm_seconds if frozen else ""
        ),
        "speedup_comparison_status": "same_input_hardware_but_approximate_profile_contract",
        "audit_frame_count": audit.size,
        "profile_frame_count": profile.size,
        **quality,
    }


def run_landmark_ablation() -> None:
    output = OUT / "landmark_ablation.tsv"
    rows: list[dict[str, object]] = read_tsv(output) if output.exists() else []
    completed = {
        (row["system"], int(row["representative_frames"]), int(row["landmark_count"]), row["landmark_method"])
        for row in rows
    }
    landmark_dir = OUT / "frozen_landmarks"
    landmark_dir.mkdir(parents=True, exist_ok=True)
    for system in ATLAS_SYSTEMS:
        coords, chains, cache, _ = load_atlas(system)
        for representative_count in REPRESENTATIVE_COUNTS:
            evidence = build_landmark_evidence(
                coords,
                chain_ids=chains,
                representative_count=representative_count,
            )
            for landmark_count in LANDMARK_COUNTS:
                if landmark_count > coords.shape[1]:
                    continue
                for method in LANDMARK_METHODS:
                    key = (system, representative_count, landmark_count, method)
                    if key in completed:
                        continue
                    landmarks = select_sparse_landmarks(
                        np.asarray(coords)[0],
                        evidence,
                        chain_ids=chains,
                        landmark_count=landmark_count,
                        method=method,
                    )
                    rows.append(
                        _sparse_row(
                            system,
                            coords,
                            cache,
                            landmarks,
                            evidence_seconds=evidence.timing["total_evidence_seconds"],
                            representative_count=representative_count,
                            landmark_count=landmark_count,
                            method=method,
                        )
                    )
                    write_tsv(output, rows)
            final_landmarks = select_sparse_landmarks(
                np.asarray(coords)[0],
                evidence,
                chain_ids=chains,
                landmark_count=min(FINAL_LANDMARKS, coords.shape[1]),
                method=FINAL_METHOD,
            )
            if representative_count == FINAL_REPRESENTATIVES:
                write_tsv(
                    landmark_dir / f"{system}_landmarks.tsv",
                    [
                        {
                            "system": system,
                            "landmark_rank": rank + 1,
                            "residue_index_zero_based": int(index),
                            "chain": str(chains[index]),
                            "confidence": float(final_landmarks.combined_score[index]),
                            "local_deformation_index": float(final_landmarks.local_deformation_index[index]),
                            "representative_residual_A": float(final_landmarks.representative_residual[index]),
                        }
                        for rank, index in enumerate(final_landmarks.indices)
                    ],
                )
        cache.close()


def _final_landmarks(coords, chains) -> LandmarkSet:
    evidence = build_landmark_evidence(
        coords,
        chain_ids=chains,
        representative_count=FINAL_REPRESENTATIVES,
    )
    return select_sparse_landmarks(
        np.asarray(coords)[0],
        evidence,
        chain_ids=chains,
        landmark_count=min(FINAL_LANDMARKS, coords.shape[1]),
        method=FINAL_METHOD,
    )


def _stream_quality_row(
    system: str,
    coords: np.ndarray,
    cache,
    result,
    *,
    keyframe_fraction: float,
    refinement_rounds: int,
    landmark_overhead_seconds: float,
) -> dict[str, object]:
    audit, profile = _sample_indices(coords.shape[0])
    quality_start = time.perf_counter()
    quality = _quality(
        coords,
        result.rotations,
        result.translations,
        np.asarray(cache["exact_weights"]),
        np.asarray(cache["exact_rmsd"]),
        np.asarray(cache["exact_rmsf"]),
        audit_indices=audit,
        profile_indices=profile,
    )
    quality_seconds = time.perf_counter() - quality_start
    runtime = landmark_overhead_seconds + result.runtime_seconds
    frozen = frozen_gpu_runtime(system)
    catastrophic = bool(
        not quality["finite_outputs"]
        or not quality["valid_transforms"]
        or quality["P95_coordinate_displacement_A"] > 10.0
    )
    return {
        "system": system,
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "mode": result.mode,
        "representative_frames": result.landmark_set.representative_indices.size,
        "landmark_count": result.landmark_set.indices.size,
        "landmark_method": result.landmark_set.method,
        "target_keyframe_fraction": keyframe_fraction,
        "actual_keyframe_fraction": result.keyframe_indices.size / coords.shape[0],
        "keyframe_count": result.keyframe_indices.size,
        "maximum_keyframe_gap": result.metadata["maximum_keyframe_gap"],
        "maximum_refinement_rounds": refinement_rounds,
        "refined_interval_count": result.refined_intervals.shape[0],
        "unresolved_interval_count": result.unresolved_intervals.shape[0],
        "refinement_rate": result.refined_intervals.shape[0] / max(1, result.keyframe_indices.size - 1),
        "landmark_estimation_seconds": landmark_overhead_seconds,
        "stream_execution_seconds": result.runtime_seconds,
        "algorithm_runtime_seconds": runtime,
        "sampled_quality_evaluation_seconds_excluded": quality_seconds,
        "frozen_GPU_throughput_seconds": frozen,
        "historical_runtime_ratio_frozen_GPU_over_stream": frozen / runtime if frozen else "",
        "speedup_comparison_status": "same_input_hardware_but_approximate_profile_contract",
        "peak_RSS_mb": result.peak_ram_mb,
        "peak_VRAM_bytes": result.peak_vram_bytes,
        "registered_catastrophic_mismatch": catastrophic,
        **quality,
    }


def run_keyframe_and_refinement_ablation() -> None:
    key_rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    atlas_rows: list[dict[str, object]] = []
    for system in ATLAS_SYSTEMS:
        coords, chains, cache, _ = load_atlas(system)
        landmarks = _final_landmarks(coords, chains)
        overhead = landmarks.timing["total_landmark_estimation_seconds"]
        for fraction in KEYFRAME_FRACTIONS:
            result = run_stream_contract(
                coords,
                chain_ids=chains,
                mode="stream-align",
                keyframe_fraction=fraction,
                maximum_refinement_rounds=0,
                backend=NEW_BACKEND,
                landmark_set=landmarks,
            )
            key_rows.append(
                _stream_quality_row(
                    system,
                    coords,
                    cache,
                    result,
                    keyframe_fraction=fraction,
                    refinement_rounds=0,
                    landmark_overhead_seconds=overhead,
                )
            )
            write_tsv(OUT / "keyframe_ablation.tsv", key_rows)
        for rounds in REFINEMENT_ROUNDS:
            result = run_stream_contract(
                coords,
                chain_ids=chains,
                mode="stream-align",
                keyframe_fraction=FINAL_KEYFRAME_FRACTION,
                maximum_refinement_rounds=rounds,
                backend=NEW_BACKEND,
                landmark_set=landmarks,
            )
            row = _stream_quality_row(
                system,
                coords,
                cache,
                result,
                keyframe_fraction=FINAL_KEYFRAME_FRACTION,
                refinement_rounds=rounds,
                landmark_overhead_seconds=overhead,
            )
            refinement_rows.append(row)
            if rounds == FINAL_REFINEMENT_ROUNDS:
                atlas_rows.append(row)
                interval_dir = OUT / "interval_audits"
                interval_dir.mkdir(exist_ok=True)
                interval_rows = [
                    {
                        "system": system,
                        "start_frame": int(values[0]),
                        "end_frame": int(values[1]),
                        "max_rotation_error_degrees_vs_direct_sparse_audit": float(values[2]),
                        "max_translation_error_A_vs_direct_sparse_audit": float(values[3]),
                        "max_coordinate_error_A_vs_direct_sparse_audit": float(values[4]),
                        "risk_score": float(values[5]),
                        "unresolved": bool(values[5] > 1.0),
                    }
                    for values in result.interval_error_estimates
                ]
                if interval_rows:
                    write_tsv(interval_dir / f"{system}_interval_audit.tsv", interval_rows)
            write_tsv(OUT / "refinement_ablation.tsv", refinement_rows)
        cache.close()
    write_tsv(OUT / "atlas_quality_speed.tsv", atlas_rows)


def _historical_generic_gpu(n_frames: int, n_residues: int) -> tuple[float | None, str]:
    rows = read_tsv(ROOT / "results/throughput_gpu_v1/t6_scaling/cpu_gpu_scaling_matrix.tsv")
    candidates = [
        row
        for row in rows
        if row["method"] == "GPU_throughput"
        and row["status"] == "success"
        and int(row["n_frames"]) == n_frames
    ]
    if not candidates:
        return None, "unavailable"
    row = min(candidates, key=lambda value: abs(int(value["n_residues"]) - n_residues))
    return float(row["warm_loaded_seconds"]), (
        f"historical_different_synthetic_input_{row['n_frames']}x{row['n_residues']}_not_a_matched_speedup"
    )


def run_synthetic_scaling() -> None:
    rows: list[dict[str, object]] = []
    paths = sorted((SYNTHETIC / "runs").glob("*/100000/transition_aligned_coords.npz"))
    for full_path in paths:
        domain = full_path.parents[1].name
        for frames in (10_000, 50_000, 100_000):
            path = full_path.parents[1] / str(frames) / "transition_aligned_coords.npz"
            started = time.perf_counter()
            with np.load(path, allow_pickle=True) as archive:
                coords = np.asarray(archive["coords"])
                metadata = archive["residue_metadata"]
            input_seconds = time.perf_counter() - started
            chains = np.asarray(
                [
                    str(item.get("chain", "A")) if isinstance(item, dict) else "A"
                    for item in metadata
                ]
            )
            result = run_stream_contract(
                coords,
                chain_ids=chains,
                mode="stream-align",
                representative_count=FINAL_REPRESENTATIVES,
                landmark_count=min(FINAL_LANDMARKS, coords.shape[1]),
                landmark_method=FINAL_METHOD,
                keyframe_fraction=FINAL_KEYFRAME_FRACTION,
                maximum_refinement_rounds=FINAL_REFINEMENT_ROUNDS,
                backend=NEW_BACKEND,
            )
            historical, status = _historical_generic_gpu(frames, coords.shape[1])
            rows.append(
                {
                    "panel": "transition_aligned_synthetic_long_not_biological_continuity",
                    "domain": domain,
                    "n_frames": frames,
                    "n_residues": coords.shape[1],
                    "representative_frames": result.landmark_set.representative_indices.size,
                    "landmark_count": result.landmark_set.indices.size,
                    "keyframe_count": result.keyframe_indices.size,
                    "actual_keyframe_fraction": result.keyframe_indices.size / frames,
                    "refined_interval_count": result.refined_intervals.shape[0],
                    "unresolved_interval_count": result.unresolved_intervals.shape[0],
                    "input_decode_seconds_excluded": input_seconds,
                    "algorithm_runtime_seconds": result.runtime_seconds,
                    "frames_per_second": frames / result.runtime_seconds,
                    "peak_RSS_mb": result.peak_ram_mb,
                    "peak_VRAM_bytes": result.peak_vram_bytes,
                    "finite_outputs": bool(
                        np.isfinite(result.rotations).all()
                        and np.isfinite(result.translations).all()
                        and np.isfinite(result.approximate_rmsd).all()
                    ),
                    "frozen_generic_GPU_seconds": historical if historical is not None else "",
                    "historical_runtime_ratio": historical / result.runtime_seconds if historical else "",
                    "comparison_status": status,
                }
            )
            write_tsv(OUT / "long_scaling_results.tsv", rows)


def run_tail_panel() -> None:
    panel = read_tsv(ROOT / "results/throughput_gpu_v1/t3_balanced_lite/locked_audit_panel.tsv")
    tails = [row for row in panel if row["predefined_tail_failure"] == "1"]
    rows: list[dict[str, object]] = []
    for source in tails:
        started = time.perf_counter()
        try:
            coords, _, metadata_rows, metadata = extract_ca_trajectory(
                Path(source["h5_path"]),
                coordinate_path=source["coordinate_path"],
                min_ca_atoms=3,
                min_frames=20,
            )
            input_seconds = time.perf_counter() - started
            chains = np.asarray([str(row.get("chain", "A")) for row in metadata_rows])
            result = run_stream_contract(
                coords,
                chain_ids=chains,
                mode="stream-align",
                representative_count=min(FINAL_REPRESENTATIVES, coords.shape[0]),
                landmark_count=min(FINAL_LANDMARKS, coords.shape[1]),
                landmark_method=FINAL_METHOD,
                keyframe_fraction=FINAL_KEYFRAME_FRACTION,
                maximum_refinement_rounds=FINAL_REFINEMENT_ROUNDS,
                backend=NEW_BACKEND,
            )
            profile_rows = read_tsv(Path(source["exact_profile_path"]))
            exact_weights = np.asarray(
                [float(row["soft_stability_weight"]) for row in profile_rows]
            )
            exact_rmsf = np.asarray(
                [float(row["robust_RMSF"]) for row in profile_rows]
            )
            exact_core = np.asarray(
                [row["core_label"] in {"1", "True", "true"} for row in profile_rows]
            )
            audit, profile = _sample_indices(coords.shape[0])
            sampled_reference = _reference_audit(coords, exact_weights, audit)
            exact_rmsd = np.zeros(coords.shape[0], dtype=np.float64)
            exact_rmsd[audit] = sampled_reference.rmsd
            quality = _quality(
                coords,
                result.rotations,
                result.translations,
                exact_weights,
                exact_rmsd,
                exact_rmsf,
                audit_indices=audit,
                profile_indices=profile,
            )
            landmark_mask = np.zeros(coords.shape[1], dtype=bool)
            landmark_mask[result.landmark_set.indices] = True
            rows.append(
                {
                    "trajectory_id": source["trajectory_id"],
                    "domain_id": source["domain_id"],
                    "split": source["split"],
                    "predefined_old_throughput_tail": True,
                    "status": "success",
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "unit_status": metadata["unit_status"],
                    "input_decode_seconds_excluded": input_seconds,
                    "algorithm_runtime_seconds": result.runtime_seconds,
                    "landmark_count": result.landmark_set.indices.size,
                    "keyframe_count": result.keyframe_indices.size,
                    "refined_interval_count": result.refined_intervals.shape[0],
                    "unresolved_interval_count": result.unresolved_intervals.shape[0],
                    "finite_outputs": bool(np.isfinite(result.approximate_rmsd).all()),
                    "valid_transforms": bool(np.max(np.abs(np.linalg.det(result.rotations) - 1.0)) < 5e-3),
                    "landmark_precision_within_frozen_exact_core": float(
                        np.count_nonzero(landmark_mask & exact_core)
                        / max(1, np.count_nonzero(landmark_mask))
                    ),
                    "frozen_exact_core_coverage_by_landmarks": float(
                        np.count_nonzero(landmark_mask & exact_core)
                        / max(1, np.count_nonzero(exact_core))
                    ),
                    **quality,
                    "warning": "unresolved_interval" if result.unresolved_intervals.size else "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "trajectory_id": source["trajectory_id"],
                    "domain_id": source["domain_id"],
                    "split": source["split"],
                    "predefined_old_throughput_tail": True,
                    "status": "failure",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        write_tsv(OUT / "tail_panel_results.tsv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("landmarks", "keyframes", "synthetic", "tails", "all"),
    )
    args = parser.parse_args()
    if args.phase in {"landmarks", "all"}:
        run_landmark_ablation()
    if args.phase in {"keyframes", "all"}:
        run_keyframe_and_refinement_ablation()
    if args.phase in {"synthetic", "all"}:
        run_synthetic_scaling()
    if args.phase in {"tails", "all"}:
        run_tail_panel()
    write_json(
        OUT / "s6_execution_status.json",
        {
            "requested_phase": args.phase,
            "only_new_algorithms_executed": True,
            "old_baseline_executions": 0,
            "external_tool_executions": 0,
            "new_downloads": 0,
            "backend": NEW_BACKEND,
        },
    )


if __name__ == "__main__":
    main()
