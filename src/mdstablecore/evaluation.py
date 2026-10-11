"""Evaluation utilities for robustness and sensitivity analysis."""

from __future__ import annotations

import numpy as np

from .alignment import align_trajectory
from .core_detection import detect_final_core
from .pipeline import run_stable_core_pipeline
from .quality_metrics import jaccard_similarity, score_correlation
from .scoring import score_residues
from .trajectory_io import residue_centroids
from .types import PipelineConfig, TrajectoryData


def subset_trajectory(data: TrajectoryData, frame_indices: np.ndarray) -> TrajectoryData:
    """Build a trajectory view on a subset of frames."""

    frame_indices = np.asarray(frame_indices, dtype=int)
    return TrajectoryData(
        coords=data.coords[frame_indices],
        atom_to_residue=data.atom_to_residue,
        residue_ids=data.residue_ids,
        atom_names=data.atom_names,
        frame_ids=frame_indices,
    )


def reference_sensitivity(
    data: TrajectoryData,
    config: PipelineConfig,
    n_references: int = 5,
) -> dict[str, float]:
    """Estimate sensitivity of scoring to the initial reference frame."""

    candidate_refs = np.unique(np.linspace(0, data.n_frames - 1, num=min(n_references, data.n_frames), dtype=int))
    all_scores = []
    all_masks = []

    for reference_index in candidate_refs:
        aligned = align_trajectory(data.coords, data.coords[reference_index])
        aligned_residues = residue_centroids(
            TrajectoryData(
                coords=aligned,
                atom_to_residue=data.atom_to_residue,
                residue_ids=data.residue_ids,
                atom_names=data.atom_names,
                frame_ids=data.frame_ids,
            )
        )
        scores = score_residues(
            aligned_residues,
            local_window_radius=config.local_window_radius,
            contact_cutoff=config.contact_cutoff,
            temporal_blocks=config.temporal_blocks,
        ).combined
        all_scores.append(scores)
        all_masks.append(detect_final_core(scores, quantile=config.final_core_quantile))

    correlations = []
    jaccards = []
    for i in range(len(all_scores)):
        for j in range(i + 1, len(all_scores)):
            correlations.append(score_correlation(all_scores[i], all_scores[j]))
            jaccards.append(jaccard_similarity(all_masks[i], all_masks[j]))

    return {
        "reference_pairwise_score_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "reference_pairwise_core_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
    }


def subsampling_sensitivity(
    data: TrajectoryData,
    config: PipelineConfig,
    fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
    repeats: int = 3,
) -> dict[str, float]:
    """Measure robustness of the final core to frame subsampling."""

    full_result = run_stable_core_pipeline(data, config=config)
    rng = np.random.default_rng(config.random_seed)
    summary: dict[str, float] = {}

    for fraction in fractions:
        subset_size = max(config.min_coarse_frames, int(round(data.n_frames * fraction)))
        subset_size = min(subset_size, data.n_frames)
        core_jaccards = []
        score_correlations = []

        for _ in range(repeats):
            frame_indices = np.sort(rng.choice(data.n_frames, size=subset_size, replace=False))
            subset_result = run_stable_core_pipeline(subset_trajectory(data, frame_indices), config=config)
            core_jaccards.append(jaccard_similarity(full_result.core_mask, subset_result.core_mask))
            score_correlations.append(
                score_correlation(full_result.residue_scores.combined, subset_result.residue_scores.combined)
            )

        label = f"{fraction:.2f}"
        summary[f"subsampling_{label}_core_jaccard"] = float(np.mean(core_jaccards))
        summary[f"subsampling_{label}_score_correlation"] = float(np.mean(score_correlations))

    return summary


def bootstrap_reproducibility_summary(confidence: np.ndarray) -> dict[str, float]:
    """Summarize bootstrap reproducibility from residue confidence values."""

    return {
        "confidence_mean": float(np.mean(confidence)),
        "high_confidence_fraction": float(np.mean(confidence >= 0.8)),
        "low_confidence_fraction": float(np.mean(confidence <= 0.2)),
    }


def coarse_refined_agreement(candidate_core_mask: np.ndarray, final_core_mask: np.ndarray) -> dict[str, float]:
    """Summarize agreement between coarse and refined stable-core masks."""

    changed = np.logical_xor(candidate_core_mask, final_core_mask)
    return {
        "coarse_refined_core_jaccard": jaccard_similarity(candidate_core_mask, final_core_mask),
        "coarse_refined_changed_fraction": float(np.mean(changed)),
    }


def evaluate_pipeline(
    data: TrajectoryData,
    config: PipelineConfig | None = None,
) -> dict[str, float]:
    """Run the pipeline and collect a compact evaluation summary."""

    if config is None:
        config = PipelineConfig()

    result = run_stable_core_pipeline(data, config=config)
    summary: dict[str, float] = {
        "core_size": float(result.core_mask.sum()),
        "uncertain_size": float(result.uncertain_mask.sum()),
        "all_atom_rmsd_mean": result.quality.all_atom_rmsd_mean,
        "core_rmsd_mean": result.quality.core_rmsd_mean,
    }
    summary.update(bootstrap_reproducibility_summary(result.confidence))
    summary.update(coarse_refined_agreement(result.candidate_core_mask, result.core_mask))
    summary.update(reference_sensitivity(data, config=config))
    summary.update(subsampling_sensitivity(data, config=config))
    return summary
