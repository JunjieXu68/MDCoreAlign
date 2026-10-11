"""Shared utilities for the U0-U6 matched GPU validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import resource
import statistics
import time
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from mdsuperpose.flexibility import detect_flexible_segments


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "gpu_matched_validation_v1"
PANEL = RESULTS / "u2_real_panel" / "real_continuous_panel.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_panel() -> list[dict[str, str]]:
    return [row for row in read_tsv(PANEL) if row["status"] == "ready"]


def load_coords(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / row["cache_path"]
    with np.load(path, allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"], dtype=np.float64)
        chains = np.asarray(archive["chain_ids"])
    return coords, chains


def core_mask(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=bool)
    right_values = np.asarray(right, dtype=bool)
    union = np.count_nonzero(left_values | right_values)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_values & right_values) / union)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def flexible_mask(weights: np.ndarray, chains: np.ndarray) -> np.ndarray:
    segments = detect_flexible_segments(
        np.clip(1.0 - np.asarray(weights, dtype=np.float64), 0.0, 1.0),
        percentile_cutoff=0.8,
        min_segment_length=3,
        merge_gap=0,
        chain=chains,
    )
    return segments > 0


def aligned_from_result(
    coords: np.ndarray,
    result,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.arange(coords.shape[0]) if indices is None else indices
    row_rotations = np.swapaxes(result.rotations[selected], 1, 2)
    return (
        np.matmul(coords[selected], row_rotations)
        + result.translations[selected, None, :]
    )


def heldout_indices(n_frames: int, count: int = 64) -> np.ndarray:
    return np.unique(
        np.linspace(0, n_frames - 1, min(count, n_frames), dtype=np.int64)
    )


def _sample_pairs(
    reference: np.ndarray,
    *,
    residue_mask: np.ndarray | None = None,
    max_pairs: int = 512,
    seed: int = 20260731,
) -> tuple[np.ndarray, np.ndarray]:
    if residue_mask is None:
        residue_indices = np.arange(reference.shape[0])
    else:
        residue_indices = np.flatnonzero(np.asarray(residue_mask, dtype=bool))
    left_local, right_local = np.triu_indices(residue_indices.size, k=1)
    left = residue_indices[left_local]
    right = residue_indices[right_local]
    if left.size > max_pairs:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(left.size, size=max_pairs, replace=False)
        )
        left = left[selected]
        right = right[selected]
    return left, right


def heldout_pairwise_variance(
    aligned: np.ndarray,
    reference: np.ndarray,
    residue_mask: np.ndarray | None = None,
) -> float:
    left, right = _sample_pairs(reference, residue_mask=residue_mask)
    if left.size == 0:
        return float("nan")
    distances = np.linalg.norm(
        aligned[:, left, :] - aligned[:, right, :],
        axis=2,
    )
    return float(np.mean(np.var(distances, axis=0)))


def contact_persistence(
    aligned: np.ndarray,
    reference: np.ndarray,
    residue_mask: np.ndarray | None = None,
    cutoff: float = 8.0,
) -> np.ndarray:
    left, right = _sample_pairs(reference, residue_mask=residue_mask)
    if left.size == 0:
        return np.full(4, np.nan)
    reference_distances = np.linalg.norm(
        reference[left] - reference[right],
        axis=1,
    )
    selected = reference_distances <= cutoff
    if not np.any(selected):
        selected = np.argsort(reference_distances)[: min(64, left.size)]
        left = left[selected]
        right = right[selected]
    else:
        left = left[selected]
        right = right[selected]
    distances = np.linalg.norm(
        aligned[:, left, :] - aligned[:, right, :],
        axis=2,
    )
    persistence = np.mean(distances <= cutoff, axis=0)
    return np.asarray(
        [
            np.mean(persistence),
            np.quantile(persistence, 0.1),
            np.quantile(persistence, 0.5),
            np.quantile(persistence, 0.9),
        ],
        dtype=np.float64,
    )


def quality_against_exact(
    coords: np.ndarray,
    chains: np.ndarray,
    candidate,
    exact,
) -> dict[str, object]:
    rmsd_delta = np.abs(candidate.rmsd - exact.rmsd)
    indices = heldout_indices(coords.shape[0])
    exact_aligned = aligned_from_result(coords, exact, indices)
    candidate_aligned = aligned_from_result(coords, candidate, indices)
    exact_core = core_mask(exact.frozen_weights)
    candidate_core = core_mask(candidate.frozen_weights)
    exact_variance = heldout_pairwise_variance(
        exact_aligned,
        coords[0],
        exact_core,
    )
    candidate_variance = heldout_pairwise_variance(
        candidate_aligned,
        coords[0],
        candidate_core,
    )
    denominator = max(abs(exact_variance), 1e-12)
    exact_contacts = contact_persistence(
        exact_aligned,
        coords[0],
        exact_core,
    )
    candidate_contacts = contact_persistence(
        candidate_aligned,
        coords[0],
        candidate_core,
    )
    return {
        "core_jaccard": jaccard(
            candidate_core,
            exact_core,
        ),
        "weight_spearman": spearman(
            candidate.frozen_weights,
            exact.frozen_weights,
        ),
        "rmsf_spearman": spearman(
            candidate.robust_rmsf,
            exact.robust_rmsf,
        ),
        "rmsf_mae_A": float(
            np.mean(np.abs(candidate.robust_rmsf - exact.robust_rmsf))
        ),
        "flexible_label_agreement": float(
            np.mean(
                flexible_mask(candidate.frozen_weights, chains)
                == flexible_mask(exact.frozen_weights, chains)
            )
        ),
        "flexible_label_jaccard": jaccard(
            flexible_mask(candidate.frozen_weights, chains),
            flexible_mask(exact.frozen_weights, chains),
        ),
        "median_abs_rmsd_difference_A": float(np.median(rmsd_delta)),
        "p95_abs_rmsd_difference_A": float(
            np.quantile(rmsd_delta, 0.95)
        ),
        "heldout_pairwise_variance_exact": exact_variance,
        "heldout_pairwise_variance_candidate": candidate_variance,
        "heldout_pairwise_variance_relative_difference": float(
            abs(candidate_variance - exact_variance) / denominator
        ),
        "contact_persistence_mae": float(
            np.mean(np.abs(candidate_contacts - exact_contacts))
        ),
        "contact_persistence_spearman": spearman(
            candidate_contacts,
            exact_contacts,
        ),
        "finite_outputs": bool(
            all(
                np.isfinite(values).all()
                for values in (
                    candidate.frozen_weights,
                    candidate.rotations,
                    candidate.translations,
                    candidate.rmsd,
                    candidate.robust_rmsf,
                )
            )
        ),
    }


def repeat_call(
    function: Callable[[], object],
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> tuple[object, list[float]]:
    for _ in range(warmups):
        function()
    result = None
    runtimes: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        runtimes.append(time.perf_counter() - start)
    assert result is not None
    return result, runtimes


def timing_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(array)),
        "q1_seconds": float(np.quantile(array, 0.25)),
        "q3_seconds": float(np.quantile(array, 0.75)),
        "iqr_seconds": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "min_seconds": float(np.min(array)),
        "max_seconds": float(np.max(array)),
        "mean_seconds": float(np.mean(array)),
        "sd_seconds": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
