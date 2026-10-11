"""Quality reporting for stable-core analysis."""

from __future__ import annotations

import numpy as np

from .types import QualityReport


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Jaccard similarity between boolean masks."""

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mean_rmsd(coords: np.ndarray, reference: np.ndarray, atom_mask: np.ndarray | None = None) -> float:
    """Compute mean frame RMSD to a reference."""

    if atom_mask is not None:
        coords = coords[:, atom_mask, :]
        reference = reference[atom_mask]
    squared = ((coords - reference[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(np.sqrt(np.mean(squared, axis=1))))


def score_correlation(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Compute a safe Pearson-like correlation."""

    if np.allclose(scores_a.std(), 0.0) or np.allclose(scores_b.std(), 0.0):
        return 1.0
    return float(np.corrcoef(scores_a, scores_b)[0, 1])


def build_quality_report(
    coarse_mask: np.ndarray,
    refined_mask: np.ndarray,
    aligned_coords: np.ndarray,
    reference: np.ndarray,
    coarse_scores: np.ndarray,
    refined_scores: np.ndarray,
    confidence: np.ndarray,
    core_atom_mask: np.ndarray | None = None,
    selected_quantile: float | None = None,
) -> QualityReport:
    """Assemble top-level quality metrics."""

    notes = []
    if core_atom_mask is None or not np.any(core_atom_mask):
        notes.append("Core atom mask was empty; core RMSD reused all-atom RMSD.")

    all_atom_rmsd = mean_rmsd(aligned_coords, reference)
    core_rmsd = mean_rmsd(aligned_coords, reference, atom_mask=core_atom_mask) if core_atom_mask is not None and np.any(core_atom_mask) else all_atom_rmsd

    return QualityReport(
        coarse_refined_core_jaccard=jaccard_similarity(coarse_mask, refined_mask),
        all_atom_rmsd_mean=all_atom_rmsd,
        core_rmsd_mean=core_rmsd,
        subsampling_score_correlation=score_correlation(coarse_scores, refined_scores),
        confidence_mean=float(np.mean(confidence)),
        selected_quantile=selected_quantile,
        core_fraction=float(np.mean(refined_mask)),
        notes=notes,
    )
