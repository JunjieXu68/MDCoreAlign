"""Internal deterministic chronology cascade for the v3 CPU study."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fast_kabsch import apply_transforms_batch, weighted_kabsch_transform_batch
from .parallel_weights import (
    ParallelWeightResult,
    _bounded_chronological_correction,
    estimate_blockwise_warm_start,
    estimate_shared_weight,
)


@dataclass(frozen=True, slots=True)
class AdaptiveRiskRule:
    """Prospectively locked monotonic thresholds for C1/C2 acceptance."""

    c1_core_jaccard_min: float
    c1_weight_spearman_min: float
    c1_cutoff_margin_min: float
    c1_residual_tail_max: float
    c2_core_jaccard_min: float
    c2_weight_spearman_min: float
    c2_cutoff_margin_min: float
    c2_residual_tail_max: float


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _ranks(np.asarray(left, dtype=np.float64))
    right_ranks = _ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return float(np.array_equal(left_ranks, right_ranks))
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _core_mask(weights: np.ndarray) -> np.ndarray:
    count = max(1, int(np.floor(0.60 * weights.size)))
    order = np.lexsort((np.arange(weights.size), -np.asarray(weights)))
    mask = np.zeros(weights.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _core_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = _core_mask(left)
    right_mask = _core_mask(right)
    union = np.count_nonzero(left_mask | right_mask)
    return 1.0 if union == 0 else float(np.count_nonzero(left_mask & right_mask) / union)


def _cutoff_margin(weights: np.ndarray) -> float:
    count = max(1, int(np.floor(0.60 * weights.size)))
    ordered = np.sort(np.asarray(weights, dtype=np.float64))[::-1]
    if count >= ordered.size:
        return float("inf")
    return float(ordered[count - 1] - ordered[count])


def _residual_tail(
    representatives: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    transform = weighted_kabsch_transform_batch(
        representatives,
        reference,
        weights,
        validate_inputs=False,
    )
    aligned = apply_transforms_batch(
        representatives,
        transform.rotations,
        transform.translations,
    )
    residuals = np.linalg.norm(aligned - reference[None, :, :], axis=2)
    frame_tail = np.quantile(residuals, 0.95, axis=1)
    return float(np.quantile(frame_tail, 0.95))


def adaptive_risk_features(
    representatives: np.ndarray,
    reference: np.ndarray,
    initial_weights: np.ndarray,
    corrected_weights: np.ndarray,
) -> dict[str, float]:
    """Calculate deterministic features available without exact outputs."""

    return {
        "core_jaccard_vs_initial": _core_jaccard(corrected_weights, initial_weights),
        "weight_spearman_vs_initial": _spearman(corrected_weights, initial_weights),
        "cutoff_margin": _cutoff_margin(corrected_weights),
        "maximum_weight_change_vs_initial": float(
            np.max(np.abs(np.asarray(corrected_weights) - np.asarray(initial_weights)))
        ),
        "residual_tail_p95": _residual_tail(
            representatives,
            reference,
            corrected_weights,
        ),
    }


def _high_risk(features: dict[str, float], rule: AdaptiveRiskRule, level: str) -> bool:
    prefix = "c1" if level == "C1" else "c2"
    return bool(
        features["core_jaccard_vs_initial"] < getattr(rule, f"{prefix}_core_jaccard_min")
        or features["weight_spearman_vs_initial"] < getattr(rule, f"{prefix}_weight_spearman_min")
        or features["cutoff_margin"] < getattr(rule, f"{prefix}_cutoff_margin_min")
        or features["residual_tail_p95"] > getattr(rule, f"{prefix}_residual_tail_max")
    )


def estimate_adaptive_chronology(
    representatives: np.ndarray,
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None,
    rule: AdaptiveRiskRule,
) -> ParallelWeightResult:
    """Run the locked C1 to C2 to D64 cascade without exact fallback."""

    total_start = time.perf_counter()
    initial = estimate_shared_weight(representatives, reference, chain_ids=chain_ids)
    c1_weights, c1_seconds, _ = _bounded_chronological_correction(
        np.asarray(representatives, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        initial.mean_weights,
        correction_iterations=1,
        chain_ids=chain_ids,
    )
    risk_start = time.perf_counter()
    c1_features = adaptive_risk_features(
        representatives,
        reference,
        initial.mean_weights,
        c1_weights,
    )
    risk_seconds = time.perf_counter() - risk_start
    accepted_level = "C1"
    final_weights = c1_weights
    refinement_seconds = 0.0
    c2_features: dict[str, float] = {}
    if _high_risk(c1_features, rule, "C1"):
        c2_weights, c2_seconds, _ = _bounded_chronological_correction(
            np.asarray(representatives, dtype=np.float64),
            np.asarray(reference, dtype=np.float64),
            initial.mean_weights,
            correction_iterations=2,
            chain_ids=chain_ids,
        )
        refinement_seconds += c2_seconds
        risk_start = time.perf_counter()
        c2_features = adaptive_risk_features(
            representatives,
            reference,
            initial.mean_weights,
            c2_weights,
        )
        risk_seconds += time.perf_counter() - risk_start
        accepted_level = "C2"
        final_weights = c2_weights
        if _high_risk(c2_features, rule, "C2"):
            d64 = estimate_blockwise_warm_start(
                representatives,
                reference,
                chain_ids=chain_ids,
                block_size=64,
            )
            refinement_seconds += d64.total_runtime_seconds
            accepted_level = "D64"
            final_weights = d64.mean_weights
    total = time.perf_counter() - total_start
    return ParallelWeightResult(
        candidate="adaptive_C1_C2_D64_v3",
        mean_weights=np.asarray(final_weights, dtype=np.float64),
        total_runtime_seconds=total,
        timing={
            "candidate_initialisation": initial.total_runtime_seconds,
            "chronological_correction": c1_seconds,
            "risk_audit": risk_seconds,
            "refinement": refinement_seconds,
        },
        refinement_fraction=0.0 if accepted_level == "C1" else 1.0,
        metadata={
            "accepted_level": accepted_level,
            "c1_features": c1_features,
            "c2_features": c2_features,
            "representative_count": int(representatives.shape[0]),
        },
    )
