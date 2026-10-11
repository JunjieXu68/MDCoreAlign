"""Shared compact evaluation primitives for the chronology-calibrated v3 study."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .all_frame_engine import fit_all_frames_fixed_weights
from .chronology_adaptive import AdaptiveRiskRule, estimate_adaptive_chronology
from .fast_kabsch import apply_transforms_batch
from .parallel_weights import (
    ParallelWeightResult,
    estimate_batch_then_sequential,
    estimate_blockwise_warm_start,
    estimate_shared_weight,
)
from .sequential_weights import estimate_representative_weights_sequential_mean
from .stage_timing import StageTiming, all_frame_seconds
from .window_refine import temporally_stratified_indices


ROBUST_PARAMETERS = {
    "rule": "huber",
    "delta": 1.0,
    "max_iterations": 20,
    "tolerance": 1e-6,
    "objective_tolerance": 1e-5,
    "objective_patience": 2,
    "sequence_continuity_strength": 0.05,
    "contact_consistency_strength": 0.0,
    "record_history": False,
}


@dataclass(slots=True)
class CompactStudyResult:
    """Matched compact outputs used for all v3 quality comparisons."""

    method: str
    weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    robust_rmsf: np.ndarray
    mean_residual: np.ndarray
    runtime_seconds: float
    stage_timing: StageTiming
    metadata: dict[str, object]


def representative_indices(n_frames: int, count: int = 2000) -> np.ndarray:
    """Return the canonical deterministic temporal representatives."""

    return temporally_stratified_indices(int(n_frames), min(int(count), int(n_frames)))


def _fit(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], float, float]:
    result = fit_all_frames_fixed_weights(
        coords,
        reference,
        weights,
        chunk_size=chunk_size,
        return_aligned=False,
        compute_ordinary_rmsf=False,
    )
    if any(value is None for value in (result.rmsd, result.weighted_rmsd, result.robust_rmsf, result.mean_residual)):
        raise RuntimeError("compact all-frame fit omitted a required output")
    total, rmsf = all_frame_seconds(result.timing)
    return {
        "rotations": result.rotations,
        "translations": result.translations,
        "rmsd": result.rmsd,
        "weighted_rmsd": result.weighted_rmsd,
        "robust_rmsf": result.robust_rmsf,
        "mean_residual": result.mean_residual,
    }, total, rmsf


def _stage_timing(
    *,
    total: float,
    weight_total: float,
    all_frame_total: float,
    stable_rmsf: float,
    exact: bool,
    children: dict[str, float] | None = None,
) -> StageTiming:
    children = children or {}
    result_construction = max(0.0, total - weight_total - all_frame_total)
    timing = StageTiming(
        end_to_end_total_seconds=total,
        weight_estimation_total_seconds=weight_total,
        candidate_initialisation_seconds=float(children.get("candidate_initialisation", 0.0)),
        chronological_correction_seconds=float(children.get("chronological_correction", 0.0)),
        blockwise_propagation_seconds=float(children.get("blockwise_propagation", 0.0)),
        risk_audit_seconds=float(children.get("risk_audit", 0.0)),
        refinement_seconds=float(children.get("refinement", 0.0)),
        all_frame_total_seconds=all_frame_total,
        stable_rmsf_seconds=stable_rmsf,
        result_construction_seconds=result_construction,
    )
    timing.validate(exact_sequential=exact)
    return timing


def run_exact(
    coords: np.ndarray,
    chain_ids: np.ndarray,
    *,
    selected_indices: np.ndarray | None = None,
    reference_index: int = 0,
    chunk_size: int = 2000,
) -> CompactStudyResult:
    """Run the unchanged chronological float64 exact estimator."""

    total_start = time.perf_counter()
    selected = representative_indices(coords.shape[0]) if selected_indices is None else np.asarray(selected_indices, dtype=int)
    reference = np.asarray(coords[int(reference_index)], dtype=np.float64)
    weight_start = time.perf_counter()
    estimate = estimate_representative_weights_sequential_mean(
        np.asarray(coords[selected], dtype=np.float64),
        reference,
        sequence_chain_ids=chain_ids,
        collect_timing=True,
        **ROBUST_PARAMETERS,
    )
    weight_total = time.perf_counter() - weight_start
    fit, all_frame_total, stable_rmsf = _fit(
        coords,
        reference,
        estimate.mean_weights,
        chunk_size=chunk_size,
    )
    total = time.perf_counter() - total_start
    stages = _stage_timing(
        total=total,
        weight_total=weight_total,
        all_frame_total=all_frame_total,
        stable_rmsf=stable_rmsf,
        exact=True,
    )
    return CompactStudyResult(
        method="exact_sequential_f64_compact",
        weights=np.asarray(estimate.mean_weights, dtype=np.float64),
        runtime_seconds=total,
        stage_timing=stages,
        metadata={
            "representative_count": int(selected.size),
            "representative_indices": selected.tolist(),
            "reference_index": int(reference_index),
            "chunk_size": int(chunk_size),
            "average_iterations": estimate.average_iterations_per_frame,
            "converged_fraction": estimate.converged_fraction,
            "warnings": list(estimate.warnings),
        },
        **fit,
    )


def _candidate_children(method: str, estimate: ParallelWeightResult) -> dict[str, float]:
    if method == "throughput_compact_f64":
        internal = sum(
            float(value)
            for key, value in estimate.timing.items()
            if key.startswith("representative_")
        )
        return {"candidate_initialisation": min(internal, estimate.total_runtime_seconds)}
    if method in {"C_correction_1_calibration", "C_correction_2_v3"}:
        return {
            "candidate_initialisation": float(estimate.timing.get("batched_initialisation", 0.0)),
            "chronological_correction": float(estimate.timing.get("chronological_correction", 0.0)),
        }
    if method == "D_block_64_v3":
        internal = sum(
            float(value)
            for key, value in estimate.timing.items()
            if key != "blockwise_total"
        )
        internal = min(internal, estimate.total_runtime_seconds)
        return {
            "candidate_initialisation": internal,
            "blockwise_propagation": max(0.0, estimate.total_runtime_seconds - internal),
        }
    if method == "adaptive_C1_C2_D64_v3":
        return {
            "candidate_initialisation": float(estimate.timing.get("candidate_initialisation", 0.0)),
            "chronological_correction": float(estimate.timing.get("chronological_correction", 0.0)),
            "risk_audit": float(estimate.timing.get("risk_audit", 0.0)),
            "refinement": float(estimate.timing.get("refinement", 0.0)),
        }
    raise ValueError(f"unsupported v3 candidate: {method}")


def run_candidate(
    coords: np.ndarray,
    chain_ids: np.ndarray,
    method: str,
    *,
    adaptive_rule: AdaptiveRiskRule | None = None,
    selected_indices: np.ndarray | None = None,
    reference_index: int = 0,
    chunk_size: int = 2000,
) -> CompactStudyResult:
    """Run one frozen v3 candidate under the matched compact contract."""

    total_start = time.perf_counter()
    selected = representative_indices(coords.shape[0]) if selected_indices is None else np.asarray(selected_indices, dtype=int)
    reference = np.asarray(coords[int(reference_index)], dtype=np.float64)
    representatives = np.asarray(coords[selected], dtype=np.float64)
    if method == "throughput_compact_f64":
        estimate = estimate_shared_weight(representatives, reference, chain_ids=chain_ids)
    elif method in {"C_correction_1_calibration", "C_correction_2_v3"}:
        estimate = estimate_batch_then_sequential(
            representatives,
            reference,
            chain_ids=chain_ids,
            correction_iterations=1 if method == "C_correction_1_calibration" else 2,
        )
    elif method == "D_block_64_v3":
        estimate = estimate_blockwise_warm_start(
            representatives,
            reference,
            chain_ids=chain_ids,
            block_size=64,
        )
    elif method == "adaptive_C1_C2_D64_v3":
        if adaptive_rule is None:
            raise ValueError("adaptive_rule is required for adaptive candidate")
        estimate = estimate_adaptive_chronology(
            representatives,
            reference,
            chain_ids=chain_ids,
            rule=adaptive_rule,
        )
    else:
        raise ValueError(f"unsupported v3 candidate: {method}")
    weight_total = estimate.total_runtime_seconds
    fit, all_frame_total, stable_rmsf = _fit(
        coords,
        reference,
        estimate.mean_weights,
        chunk_size=chunk_size,
    )
    total = time.perf_counter() - total_start
    children = _candidate_children(method, estimate)
    child_sum = sum(children.values())
    if child_sum > weight_total:
        scale = weight_total / child_sum if child_sum > 0.0 else 0.0
        children = {key: value * scale for key, value in children.items()}
    stages = _stage_timing(
        total=total,
        weight_total=weight_total,
        all_frame_total=all_frame_total,
        stable_rmsf=stable_rmsf,
        exact=False,
        children=children,
    )
    return CompactStudyResult(
        method=method,
        weights=np.asarray(estimate.mean_weights, dtype=np.float64),
        runtime_seconds=total,
        stage_timing=stages,
        metadata={
            **estimate.metadata,
            "representative_count": int(selected.size),
            "reference_index": int(reference_index),
            "chunk_size": int(chunk_size),
            "warnings": list(estimate.warnings),
        },
        **fit,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).ravel()
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    first = 0
    while first < values.size:
        last = first + 1
        while last < values.size and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + last - 1)
        first = last
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float(np.array_equal(left_rank, right_rank))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def core_mask(weights: np.ndarray) -> np.ndarray:
    count = max(1, int(np.floor(0.60 * weights.size)))
    order = np.lexsort((np.arange(weights.size), -np.asarray(weights)))
    mask = np.zeros(weights.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def compare_results(
    candidate: CompactStudyResult,
    reference: CompactStudyResult,
    coords: np.ndarray,
) -> dict[str, object]:
    """Compare two results using the single registered v3 metric implementation."""

    candidate_core = core_mask(candidate.weights)
    reference_core = core_mask(reference.weights)
    union = np.count_nonzero(candidate_core | reference_core)
    jaccard = 1.0 if union == 0 else float(np.count_nonzero(candidate_core & reference_core) / union)
    rmsd_difference = np.abs(candidate.rmsd - reference.rmsd)
    relative = np.einsum("fij,fkj->fik", candidate.rotations, reference.rotations)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_angles = np.degrees(np.arccos(cosine))
    sampled = representative_indices(coords.shape[0], min(32, coords.shape[0]))
    candidate_aligned = apply_transforms_batch(
        np.asarray(coords[sampled], dtype=np.float64),
        candidate.rotations[sampled],
        candidate.translations[sampled],
    )
    reference_aligned = apply_transforms_batch(
        np.asarray(coords[sampled], dtype=np.float64),
        reference.rotations[sampled],
        reference.translations[sampled],
    )
    sampled_displacement = np.linalg.norm(candidate_aligned - reference_aligned, axis=2)
    finite = bool(
        np.isfinite(candidate.weights).all()
        and np.isfinite(candidate.rotations).all()
        and np.isfinite(candidate.translations).all()
        and np.isfinite(candidate.rmsd).all()
        and np.isfinite(candidate.robust_rmsf).all()
    )
    p95 = float(np.quantile(rmsd_difference, 0.95))
    return {
        "stable_weight_max_difference": float(np.max(np.abs(candidate.weights - reference.weights))),
        "stable_weight_mean_difference": float(np.mean(np.abs(candidate.weights - reference.weights))),
        "weight_spearman": spearman(candidate.weights, reference.weights),
        "core_jaccard": jaccard,
        "rotation_geodesic_median_degrees": float(np.median(rotation_angles)),
        "rotation_geodesic_p95_degrees": float(np.quantile(rotation_angles, 0.95)),
        "sampled_aligned_coordinate_difference_median_A": float(np.median(sampled_displacement)),
        "sampled_aligned_coordinate_difference_p95_A": float(np.quantile(sampled_displacement, 0.95)),
        "rmsd_absolute_difference_median_A": float(np.median(rmsd_difference)),
        "rmsd_absolute_difference_p95_A": p95,
        "rmsd_absolute_difference_max_A": float(np.max(rmsd_difference)),
        "stable_core_rmsf_spearman": spearman(candidate.robust_rmsf, reference.robust_rmsf),
        "stable_core_rmsf_mae_A": float(np.mean(np.abs(candidate.robust_rmsf - reference.robust_rmsf))),
        "non_finite_transform": not finite,
        "catastrophic_mismatch": bool((not finite) or jaccard < 0.5 or p95 > 1.0),
        "candidate_runtime_seconds": candidate.runtime_seconds,
        "reference_runtime_seconds": reference.runtime_seconds,
        "speedup_vs_exact": reference.runtime_seconds / candidate.runtime_seconds,
    }
