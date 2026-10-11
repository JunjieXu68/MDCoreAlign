"""Independent metrics for the untimed MDCoreAlign-MDLovoFit agreement audit."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from scipy.stats import kendalltau, linregress, pearsonr, rankdata, spearmanr

from mdsuperpose.fast_kabsch import weighted_kabsch_batch


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Select exactly k highest scores with deterministic index tie breaking."""
    values = np.asarray(scores, dtype=np.float64)
    count = int(k)
    if not 1 <= count <= values.size:
        raise ValueError("k must be between one and score count")
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def safe_kendall(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(kendalltau(left, right).statistic)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def mask_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    left_count = int(np.count_nonzero(a))
    right_count = int(np.count_nonzero(b))
    return {
        "intersection_count": intersection,
        "union_count": union,
        "Jaccard": intersection / union if union else 1.0,
        "Dice": 2.0 * intersection / (left_count + right_count) if left_count + right_count else 1.0,
        "precision": intersection / right_count if right_count else float("nan"),
        "recall": intersection / left_count if left_count else float("nan"),
    }


def _segments(mask: np.ndarray, chains: np.ndarray) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []
    start: int | None = None
    for index in range(mask.size):
        if index > 0 and chains[index] != chains[index - 1] and start is not None:
            output.append((str(chains[index - 1]), start, index - 1))
            start = None
        if mask[index] and start is None:
            start = index
        if start is not None and (not mask[index] or index == mask.size - 1):
            stop = index if mask[index] else index - 1
            output.append((str(chains[stop]), start, stop))
            start = None
    return output


def segment_metrics(left: np.ndarray, right: np.ndarray, chains: np.ndarray) -> dict[str, object]:
    left_segments = _segments(left, chains)
    right_segments = _segments(right, chains)
    left_boundaries = {(chain, start) for chain, start, _ in left_segments} | {(chain, stop) for chain, _, stop in left_segments}
    right_boundaries = {(chain, start) for chain, start, _ in right_segments} | {(chain, stop) for chain, _, stop in right_segments}
    distances: list[int] = []
    for chain, position in left_boundaries:
        candidates = [abs(position - other) for other_chain, other in right_boundaries if other_chain == chain]
        if candidates:
            distances.append(min(candidates))
    disagreement = np.asarray(left, dtype=bool) != np.asarray(right, dtype=bool)
    ranges = _segments(disagreement, chains)
    return {
        "left_segment_count": len(left_segments),
        "right_segment_count": len(right_segments),
        "segment_overlap_Jaccard": mask_metrics(left, right)["Jaccard"],
        "mean_boundary_distance_residues": float(np.mean(distances)) if distances else float("nan"),
        "maximum_boundary_distance_residues": max(distances, default=0),
        "disagreement_ranges": ";".join(f"{chain}:{start + 1}-{stop + 1}" for chain, start, stop in ranges),
    }


def chainwise_jaccard(left: np.ndarray, right: np.ndarray, chains: np.ndarray) -> tuple[float, str]:
    values = []
    labels = []
    for chain in dict.fromkeys(str(item) for item in chains):
        selection = np.asarray([str(item) == chain for item in chains])
        value = float(mask_metrics(left[selection], right[selection])["Jaccard"])
        values.append(value)
        labels.append(f"{chain}:{value:.6f}")
    return float(np.mean(values)), ";".join(labels)


def rank_biased_overlap(left_scores: np.ndarray, right_scores: np.ndarray, *, p: float = 0.9) -> float:
    left = np.lexsort((np.arange(left_scores.size), -left_scores))
    right = np.lexsort((np.arange(right_scores.size), -right_scores))
    left_seen: set[int] = set()
    right_seen: set[int] = set()
    value = 0.0
    for depth, (left_index, right_index) in enumerate(zip(left, right), start=1):
        left_seen.add(int(left_index))
        right_seen.add(int(right_index))
        value += (1.0 - p) * (p ** (depth - 1)) * len(left_seen & right_seen) / depth
    return float(value + (p ** left.size) * len(left_seen & right_seen) / left.size)


def fit_with_weights(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.empty((coords.shape[0], 3, 3), dtype=np.float64)
    translations = np.empty((coords.shape[0], 3), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    for start in range(0, coords.shape[0], chunk_size):
        stop = min(start + chunk_size, coords.shape[0])
        result = weighted_kabsch_batch(
            coords[start:stop],
            coords[0],
            weights,
            validate_inputs=False,
            compute_metrics=False,
        )
        rotations[start:stop] = result.rotations
        translations[start:stop] = result.translations
    return rotations, translations


def fit_with_mask(
    coords: np.ndarray,
    mask: np.ndarray,
    *,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    return fit_with_weights(coords, np.asarray(mask, dtype=np.float64), chunk_size=chunk_size)


def _apply(coords: np.ndarray, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    return np.einsum("tni,tji->tnj", coords, rotations) + translations[:, None, :]


def _rmsf_from_moments(total: np.ndarray, squared_total: np.ndarray, n_frames: int) -> np.ndarray:
    mean = total / n_frames
    variance_sum = squared_total / n_frames - mean * mean
    return np.sqrt(np.maximum(np.sum(variance_sum, axis=1), 0.0))


def alignment_pair_metrics(
    coords: np.ndarray,
    left_rotations: np.ndarray,
    left_translations: np.ndarray,
    right_rotations: np.ndarray,
    right_translations: np.ndarray,
    left_core: np.ndarray,
    right_core: np.ndarray,
    *,
    chunk_size: int = 500,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    n_frames, n_residues, _ = coords.shape
    reference = coords[0]
    left_sum = np.zeros((n_residues, 3), dtype=np.float64)
    right_sum = np.zeros((n_residues, 3), dtype=np.float64)
    left_squared_sum = np.zeros((n_residues, 3), dtype=np.float64)
    right_squared_sum = np.zeros((n_residues, 3), dtype=np.float64)
    left_rmsd = np.empty(n_frames, dtype=np.float64)
    right_rmsd = np.empty(n_frames, dtype=np.float64)
    coordinate_full = np.empty(n_frames, dtype=np.float64)
    coordinate_left_core = np.empty(n_frames, dtype=np.float64)
    coordinate_right_core = np.empty(n_frames, dtype=np.float64)
    coordinate_intersection = np.empty(n_frames, dtype=np.float64)
    coordinate_union = np.empty(n_frames, dtype=np.float64)
    coordinate_mobile = np.empty(n_frames, dtype=np.float64)
    intersection = left_core & right_core
    union = left_core | right_core
    mobile = ~(left_core | right_core)
    masks = [left_core, right_core, intersection, union, mobile]
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        left = _apply(coords[start:stop], left_rotations[start:stop], left_translations[start:stop])
        right = _apply(coords[start:stop], right_rotations[start:stop], right_translations[start:stop])
        left_sum += np.sum(left, axis=0)
        right_sum += np.sum(right, axis=0)
        left_squared_sum += np.sum(left * left, axis=0)
        right_squared_sum += np.sum(right * right, axis=0)
        left_delta = left - reference[None, :, :]
        right_delta = right - reference[None, :, :]
        left_rmsd[start:stop] = np.sqrt(np.mean(np.sum(left_delta * left_delta, axis=2), axis=1))
        right_rmsd[start:stop] = np.sqrt(np.mean(np.sum(right_delta * right_delta, axis=2), axis=1))
        difference = np.sum((left - right) ** 2, axis=2)
        targets = [coordinate_left_core, coordinate_right_core, coordinate_intersection, coordinate_union, coordinate_mobile]
        coordinate_full[start:stop] = np.sqrt(np.mean(difference, axis=1))
        for mask, target in zip(masks, targets):
            target[start:stop] = np.sqrt(np.mean(difference[:, mask], axis=1)) if np.any(mask) else np.nan
    left_rmsf = _rmsf_from_moments(left_sum, left_squared_sum, n_frames)
    right_rmsf = _rmsf_from_moments(right_sum, right_squared_sum, n_frames)
    rmsd_difference = np.abs(left_rmsd - right_rmsd)
    relative_rotations = np.einsum("fij,fkj->fik", left_rotations, right_rotations)
    cosines = np.clip((np.trace(relative_rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    translations = np.linalg.norm(left_translations - right_translations, axis=1)
    regression = linregress(right_rmsd, left_rmsd) if np.ptp(right_rmsd) > 0 else None
    metrics = {
        "median_aligned_coordinate_difference_A": float(np.median(coordinate_full)),
        "P95_aligned_coordinate_difference_A": float(np.quantile(coordinate_full, 0.95)),
        "median_left_core_coordinate_difference_A": float(np.nanmedian(coordinate_left_core)),
        "median_right_core_coordinate_difference_A": float(np.nanmedian(coordinate_right_core)),
        "median_intersection_core_coordinate_difference_A": float(np.nanmedian(coordinate_intersection)),
        "median_union_core_coordinate_difference_A": float(np.nanmedian(coordinate_union)),
        "median_mobile_coordinate_difference_A": float(np.nanmedian(coordinate_mobile)),
        "median_rotation_geodesic_degrees": float(np.median(angles)),
        "P95_rotation_geodesic_degrees": float(np.quantile(angles, 0.95)),
        "median_translation_difference_A": float(np.median(translations)),
        "RMSD_Pearson": safe_pearson(left_rmsd, right_rmsd),
        "RMSD_Spearman": safe_spearman(left_rmsd, right_rmsd),
        "RMSD_MAE_A": float(np.mean(rmsd_difference)),
        "RMSD_median_absolute_difference_A": float(np.median(rmsd_difference)),
        "RMSD_P95_absolute_difference_A": float(np.quantile(rmsd_difference, 0.95)),
        "RMSD_calibration_slope": float(regression.slope) if regression else float("nan"),
        "RMSD_calibration_intercept_A": float(regression.intercept) if regression else float("nan"),
        "RMSF_Spearman": safe_spearman(left_rmsf, right_rmsf),
        "RMSF_MAE_A": float(np.mean(np.abs(left_rmsf - right_rmsf))),
        "mean_left_RMSF_A": float(np.mean(left_rmsf)),
        "mean_right_RMSF_A": float(np.mean(right_rmsf)),
    }
    high_count = max(1, int(math.floor(0.20 * n_residues)))
    metrics["high_RMSF_Jaccard"] = float(mask_metrics(top_k_mask(left_rmsf, high_count), top_k_mask(right_rmsf, high_count))["Jaccard"])
    return metrics, left_rmsf, right_rmsf


def heldout_geometry(
    coords: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    max_pairs: int = 512,
    max_frames: int = 128,
) -> dict[str, float | int]:
    indices = np.flatnonzero(mask)
    all_pairs = list(combinations(indices.tolist(), 2))
    rng = np.random.default_rng(seed)
    if len(all_pairs) > max_pairs:
        chosen = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = np.asarray([all_pairs[index] for index in sorted(chosen)], dtype=int)
    else:
        pairs = np.asarray(all_pairs, dtype=int)
    if pairs.size == 0:
        return {"heldout_pair_count": 0, "heldout_pairwise_distance_variance_A2": float("nan"), "heldout_contact_persistence": float("nan")}
    frames = np.linspace(0, coords.shape[0] - 1, min(max_frames, coords.shape[0]), dtype=int)
    distances = np.linalg.norm(coords[frames][:, pairs[:, 0]] - coords[frames][:, pairs[:, 1]], axis=2)
    reference_contacts = distances[0] <= 8.0
    contact_persistence = float(np.mean(distances[:, reference_contacts] <= 8.0)) if np.any(reference_contacts) else float("nan")
    return {
        "heldout_pair_count": int(pairs.shape[0]),
        "heldout_pairwise_distance_variance_A2": float(np.mean(np.var(distances, axis=0))),
        "heldout_contact_persistence": contact_persistence,
    }


def descriptive_category(core_jaccard: float, score_spearman: float, coordinate_difference: float, rmsf_spearman: float) -> str:
    if core_jaccard >= 0.80 and score_spearman >= 0.80 and coordinate_difference <= 0.10 and rmsf_spearman >= 0.98:
        return "high_practical_agreement"
    if core_jaccard >= 0.60 and score_spearman >= 0.60 and coordinate_difference <= 0.25 and rmsf_spearman >= 0.95:
        return "broad_moderate_practical_agreement"
    return "method_dependent_result"
