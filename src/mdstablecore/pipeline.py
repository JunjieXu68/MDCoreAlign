"""Top-level coarse-to-fine stable-core pipeline."""

from __future__ import annotations

import numpy as np

from .alignment import align_trajectory, medoid_frame_index
from .competitive import run_competitive_pipeline
from .confidence import bootstrap_confidence
from .core_detection import detect_candidate_core, detect_final_core, uncertain_residues
from .fastcore import run_fastcore_pipeline
from .quality_metrics import build_quality_report
from .sampling import adaptive_sample_frames, choose_coarse_frame_count
from .scoring import score_residues
from .trajectory_io import residue_centroids
from .types import CoreResult, PipelineConfig, QualityReport, TrajectoryData


def _core_atom_mask(atom_to_residue: np.ndarray, residue_mask: np.ndarray) -> np.ndarray:
    """Expand a residue mask to an atom mask."""

    return residue_mask[atom_to_residue]


def _quality_objective(quality: QualityReport) -> float:
    """Internal objective used to select the final stable-core quantile."""

    core_fraction = quality.core_fraction or 0.0
    core_size_score = max(0.0, 1.0 - abs(core_fraction - 0.5) / 0.5)
    core_rmsd_score = 1.0 / (1.0 + quality.core_rmsd_mean)
    all_rmsd_score = 1.0 / (1.0 + quality.all_atom_rmsd_mean)
    return (
        0.22 * quality.confidence_mean
        + 0.33 * core_size_score
        + 0.20 * core_rmsd_score
        + 0.15 * all_rmsd_score
        + 0.05 * quality.coarse_refined_core_jaccard
        + 0.05 * quality.subsampling_score_correlation
    )


def _evaluate_final_core_candidates(
    data: TrajectoryData,
    config: PipelineConfig,
    candidate_core: np.ndarray,
    coarse_scores: np.ndarray,
    refined_reference: np.ndarray,
    aligned_full: np.ndarray,
    refined_residue_coords: np.ndarray,
):
    """Scan several final-core quantiles and keep the best internal solution."""

    refined_scores = score_residues(
        refined_residue_coords,
        local_window_radius=config.local_window_radius,
        contact_cutoff=config.contact_cutoff,
        temporal_blocks=config.temporal_blocks,
    )

    best_result = None
    for quantile in config.final_core_quantiles:
        confidence = bootstrap_confidence(
            refined_residue_coords,
            quantile=quantile,
            rounds=config.bootstrap_rounds,
            fraction=config.bootstrap_fraction,
            block_fraction=config.bootstrap_block_fraction,
            local_window_radius=config.local_window_radius,
            contact_cutoff=config.contact_cutoff,
            temporal_blocks=config.temporal_blocks,
            min_run_length=config.min_core_run_length,
            confidence_threshold=config.confidence_threshold,
            confidence_weight=config.confidence_weight,
            terminal_exclusion=config.terminal_exclusion,
            random_seed=config.random_seed,
        )
        final_core = detect_final_core(
            refined_scores.combined,
            quantile=quantile,
            confidence=confidence,
            min_run_length=config.min_core_run_length,
            confidence_threshold=config.confidence_threshold,
            confidence_weight=config.confidence_weight,
            terminal_exclusion=config.terminal_exclusion,
        )
        uncertain_mask = uncertain_residues(
            refined_scores.combined,
            quantile=quantile,
            margin=config.uncertainty_margin,
        ) | ((confidence > 0.2) & (confidence < 0.8))
        quality = build_quality_report(
            coarse_mask=candidate_core,
            refined_mask=final_core,
            aligned_coords=aligned_full,
            reference=refined_reference,
            coarse_scores=coarse_scores,
            refined_scores=refined_scores.combined,
            confidence=confidence,
            core_atom_mask=_core_atom_mask(data.atom_to_residue, final_core),
            selected_quantile=quantile,
        )
        objective = _quality_objective(quality)
        current = (objective, final_core, confidence, uncertain_mask, quality, refined_scores)
        if best_result is None or objective > best_result[0]:
            best_result = current

    if best_result is None:
        raise RuntimeError("No final-core candidate could be evaluated")

    _, final_core, confidence, uncertain_mask, quality, refined_scores = best_result
    return final_core, confidence, uncertain_mask, quality, refined_scores


def _run_coarse_to_fine_pipeline(
    data: TrajectoryData,
    config: PipelineConfig | None = None,
) -> CoreResult:
    """Run a coarse-to-fine stable-core analysis with quantile scanning."""

    if config is None:
        config = PipelineConfig()

    coarse_count = choose_coarse_frame_count(
        data.n_frames,
        coarse_fraction=config.coarse_fraction,
        min_frames=config.min_coarse_frames,
        max_frames=config.max_coarse_frames,
    )

    raw_residue_coords = residue_centroids(data)
    coarse_indices = adaptive_sample_frames(raw_residue_coords, coarse_count, random_seed=config.random_seed)
    coarse_coords = data.coords[coarse_indices]
    coarse_residue_coords = raw_residue_coords[coarse_indices]

    coarse_reference_index = medoid_frame_index(coarse_residue_coords)
    coarse_reference = coarse_coords[coarse_reference_index]
    aligned_coarse = align_trajectory(coarse_coords, coarse_reference)
    aligned_coarse_residues = residue_centroids(
        TrajectoryData(
            coords=aligned_coarse,
            atom_to_residue=data.atom_to_residue,
            residue_ids=data.residue_ids,
            atom_names=data.atom_names,
            frame_ids=coarse_indices,
        )
    )
    coarse_scores = score_residues(
        aligned_coarse_residues,
        local_window_radius=config.local_window_radius,
        contact_cutoff=config.contact_cutoff,
        temporal_blocks=config.temporal_blocks,
    )
    candidate_core = detect_candidate_core(coarse_scores.combined, quantile=config.candidate_core_quantile)
    if not np.any(candidate_core):
        candidate_core = detect_candidate_core(coarse_scores.combined, quantile=0.5)
    candidate_atom_mask = _core_atom_mask(data.atom_to_residue, candidate_core)

    fit_selection = candidate_core if np.any(candidate_core) else slice(None)
    refined_reference_index = medoid_frame_index(aligned_coarse_residues[:, fit_selection, :])
    refined_reference = data.coords[coarse_indices[refined_reference_index]]
    aligned_full = align_trajectory(
        data.coords,
        refined_reference,
        atom_mask=candidate_atom_mask if np.any(candidate_atom_mask) else None,
    )
    refined_residue_coords = residue_centroids(
        TrajectoryData(
            coords=aligned_full,
            atom_to_residue=data.atom_to_residue,
            residue_ids=data.residue_ids,
            atom_names=data.atom_names,
            frame_ids=data.frame_ids,
        )
    )

    final_core, confidence, uncertain_mask, quality, refined_scores = _evaluate_final_core_candidates(
        data=data,
        config=config,
        candidate_core=candidate_core,
        coarse_scores=coarse_scores.combined,
        refined_reference=refined_reference,
        aligned_full=aligned_full,
        refined_residue_coords=refined_residue_coords,
    )

    return CoreResult(
        residue_scores=refined_scores,
        confidence=confidence,
        core_mask=final_core,
        uncertain_mask=uncertain_mask,
        candidate_core_mask=candidate_core,
        coarse_frame_indices=coarse_indices,
        refined_reference_index=int(coarse_indices[refined_reference_index]),
        quality=quality,
    )


def run_stable_core_pipeline(
    data: TrajectoryData,
    config: PipelineConfig | None = None,
) -> CoreResult:
    """Run the configured stable-core analysis pipeline."""

    if config is None:
        config = PipelineConfig()

    if config.algorithm == "coarse_to_fine":
        return _run_coarse_to_fine_pipeline(data, config=config)
    if config.algorithm == "fastcore":
        return run_fastcore_pipeline(data, config=config)
    if config.algorithm == "competitive":
        return run_competitive_pipeline(data, config=config)
    raise ValueError(f"Unknown pipeline algorithm: {config.algorithm}")
