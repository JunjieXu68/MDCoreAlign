"""Fast competitive stable-core pipeline."""

from __future__ import annotations

import numpy as np

from .alignment import align_trajectory, medoid_frame_index
from .baseline import run_mdlovofit_style
from .core_detection import detect_final_core, uncertain_residues
from .quality_metrics import build_quality_report, mean_rmsd
from .trajectory_io import residue_centroids
from .types import CoreResult, PipelineConfig, ResidueScores, TrajectoryData


def _robust_inverse_scale(values: np.ndarray) -> np.ndarray:
    """Convert a dispersion-like signal into a stability score."""

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-8
    z_like = (values - median) / mad
    return 1.0 / (1.0 + np.clip(z_like, 0.0, None))


def _residue_rmsd_to_reference(residue_coords: np.ndarray, reference_residues: np.ndarray) -> np.ndarray:
    """Compute per-residue RMSD relative to a reference frame."""

    squared = np.sum((residue_coords - reference_residues[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=0))


def _residue_rmsf(residue_coords: np.ndarray) -> np.ndarray:
    """Compute per-residue RMSF around the mean position."""

    mean_positions = residue_coords.mean(axis=0)
    squared = np.sum((residue_coords - mean_positions[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=0))


def _local_neighbor_dispersion(residue_coords: np.ndarray, radius: int) -> np.ndarray:
    """Measure how stable local sequence-neighbor distances remain."""

    n_residues = residue_coords.shape[1]
    if n_residues <= 1 or radius <= 0:
        return np.zeros(n_residues, dtype=float)

    local_dispersion = np.zeros(n_residues, dtype=float)
    counts = np.zeros(n_residues, dtype=float)
    for delta in range(1, radius + 1):
        if delta >= n_residues:
            break
        distances = np.linalg.norm(residue_coords[:, delta:, :] - residue_coords[:, :-delta, :], axis=2)
        dispersion = np.std(distances, axis=0)
        local_dispersion[:-delta] += dispersion
        local_dispersion[delta:] += dispersion
        counts[:-delta] += 1.0
        counts[delta:] += 1.0

    counts[counts == 0.0] = 1.0
    return local_dispersion / counts


def _temporal_block_dispersion(residue_coords: np.ndarray, n_blocks: int) -> np.ndarray:
    """Measure how reproducible per-residue RMSF is across time blocks."""

    blocks = [block for block in np.array_split(residue_coords, n_blocks, axis=0) if block.shape[0] > 1]
    if not blocks:
        return np.zeros(residue_coords.shape[1], dtype=float)

    block_rmsf = np.vstack([_residue_rmsf(block) for block in blocks])
    return np.std(block_rmsf, axis=0)


def _competitive_scores(
    residue_coords: np.ndarray,
    reference_residues: np.ndarray,
    config: PipelineConfig,
) -> ResidueScores:
    """Build a fast multi-signal residue score bundle."""

    positional = _robust_inverse_scale(_residue_rmsd_to_reference(residue_coords, reference_residues))
    rmsf_score = _robust_inverse_scale(_residue_rmsf(residue_coords))
    local_geometry = _robust_inverse_scale(
        _local_neighbor_dispersion(residue_coords, radius=config.local_window_radius)
    )
    temporal_consistency = _robust_inverse_scale(
        _temporal_block_dispersion(residue_coords, n_blocks=max(3, config.temporal_blocks + 1))
    )
    combined = (
        0.35 * positional
        + 0.35 * rmsf_score
        + 0.20 * local_geometry
        + 0.10 * temporal_consistency
    )
    return ResidueScores(
        positional=positional,
        local_geometry=local_geometry,
        contact_persistence=rmsf_score,
        temporal_consistency=temporal_consistency,
        combined=combined,
    )


def _candidate_objective(
    mask: np.ndarray,
    scores: np.ndarray,
    aligned_coords: np.ndarray,
    reference: np.ndarray,
    atom_to_residue: np.ndarray,
    target_fraction: float,
) -> float:
    """Score a candidate core using compactness, separation, and size balance."""

    atom_mask = mask[atom_to_residue]
    core_rmsd = mean_rmsd(aligned_coords, reference, atom_mask=atom_mask) if np.any(atom_mask) else np.inf
    all_rmsd = mean_rmsd(aligned_coords, reference)
    if np.any(mask) and np.any(~mask):
        separation = float(scores[mask].mean() - scores[~mask].mean())
    else:
        separation = 0.0
    core_fraction = float(np.mean(mask))
    size_score = max(0.0, 1.0 - abs(core_fraction - target_fraction) / max(target_fraction, 1e-8))
    return (
        0.45 * (1.0 / (1.0 + core_rmsd))
        + 0.20 * (1.0 / (1.0 + all_rmsd))
        + 0.25 * separation
        + 0.10 * size_score
    )


def _confidence_from_candidates(candidates: list[dict[str, object]], n_residues: int) -> np.ndarray:
    """Convert nearby candidate solutions into a soft residue confidence profile."""

    if not candidates:
        return np.zeros(n_residues, dtype=float)

    objectives = np.array([float(candidate["objective"]) for candidate in candidates], dtype=float)
    scale = max(1e-3, float(np.std(objectives)) * 0.5)
    weights = np.exp((objectives - objectives.max()) / scale)
    weights /= weights.sum()

    confidence = np.zeros(n_residues, dtype=float)
    for weight, candidate in zip(weights, candidates):
        confidence += weight * np.asarray(candidate["mask"], dtype=float)
    return confidence


def _append_fraction_candidates(
    candidates: list[dict[str, object]],
    data: TrajectoryData,
    config: PipelineConfig,
    fraction: float,
    reference: np.ndarray,
    reference_index: int,
    fit_residue_mask: np.ndarray,
    coarse_mask: np.ndarray,
    coarse_scores: np.ndarray,
    stage: str,
) -> None:
    """Score one aligned fraction setting and append all valid quantile candidates."""

    fit_atom_mask = fit_residue_mask[data.atom_to_residue]
    aligned_coords = align_trajectory(
        data.coords,
        reference,
        atom_mask=fit_atom_mask if np.any(fit_atom_mask) else None,
    )
    aligned_data = TrajectoryData(
        coords=aligned_coords,
        atom_to_residue=data.atom_to_residue,
        residue_ids=data.residue_ids,
        atom_names=data.atom_names,
        frame_ids=data.frame_ids,
    )
    aligned_residue_coords = residue_centroids(aligned_data)
    scores = _competitive_scores(
        aligned_residue_coords,
        reference_residues=aligned_residue_coords[reference_index],
        config=config,
    )

    for quantile in config.competitive_quantiles:
        mask = detect_final_core(
            scores.combined,
            quantile=quantile,
            confidence=None,
            min_run_length=config.min_core_run_length,
            terminal_exclusion=config.terminal_exclusion,
        )
        if not np.any(mask):
            continue
        objective = _candidate_objective(
            mask=mask,
            scores=scores.combined,
            aligned_coords=aligned_coords,
            reference=reference,
            atom_to_residue=data.atom_to_residue,
            target_fraction=config.competitive_target_core_fraction,
        )
        candidates.append(
            {
                "fraction": fraction,
                "quantile": quantile,
                "mask": mask,
                "scores": scores,
                "coarse_mask": coarse_mask.copy(),
                "coarse_scores": coarse_scores.copy(),
                "aligned_coords": aligned_coords,
                "objective": objective,
                "reference": reference.copy(),
                "reference_index": reference_index,
                "stage": stage,
            }
        )


def run_competitive_pipeline(
    data: TrajectoryData,
    config: PipelineConfig,
) -> CoreResult:
    """Run a fast MDLovoFit-competitive stable-core pipeline."""

    raw_residue_coords = residue_centroids(data)
    reference_index = medoid_frame_index(raw_residue_coords)
    reference = data.coords[reference_index]
    candidates: list[dict[str, object]] = []
    for fraction in config.competitive_fractions:
        baseline_result = run_mdlovofit_style(
            data,
            fraction=fraction,
            reference_frame=reference_index,
            iterations=config.competitive_iterations,
        )
        coarse_scores = _robust_inverse_scale(baseline_result.residue_rmsf)
        _append_fraction_candidates(
            candidates=candidates,
            data=data,
            config=config,
            fraction=fraction,
            reference=reference,
            reference_index=reference_index,
            fit_residue_mask=baseline_result.selected_residue_mask,
            coarse_mask=baseline_result.selected_residue_mask,
            coarse_scores=coarse_scores,
            stage="seed",
        )

    if not candidates:
        raise RuntimeError("Competitive pipeline could not generate a valid core candidate")

    if config.competitive_refinement_rounds > 0:
        for refinement_round in range(config.competitive_refinement_rounds):
            best_seed = max(candidates, key=lambda candidate: float(candidate["objective"]))
            best_aligned_coords = np.asarray(best_seed["aligned_coords"], dtype=float)
            best_mask = np.asarray(best_seed["mask"], dtype=bool)
            refined_residue_coords = residue_centroids(
                TrajectoryData(
                    coords=best_aligned_coords,
                    atom_to_residue=data.atom_to_residue,
                    residue_ids=data.residue_ids,
                    atom_names=data.atom_names,
                    frame_ids=data.frame_ids,
                )
            )
            refined_reference_index = medoid_frame_index(refined_residue_coords[:, best_mask, :])
            refined_reference = best_aligned_coords[refined_reference_index]
            _append_fraction_candidates(
                candidates=candidates,
                data=data,
                config=config,
                fraction=float(best_seed["fraction"]),
                reference=refined_reference,
                reference_index=refined_reference_index,
                fit_residue_mask=best_mask,
                coarse_mask=np.asarray(best_seed["coarse_mask"], dtype=bool),
                coarse_scores=np.asarray(best_seed["coarse_scores"], dtype=float),
                stage=f"refine_{refinement_round + 1}",
            )

    best = max(candidates, key=lambda candidate: float(candidate["objective"]))
    confidence = _confidence_from_candidates(candidates, n_residues=data.n_residues)
    best_scores = best["scores"]
    best_mask = np.asarray(best["mask"], dtype=bool)
    best_coarse_mask = np.asarray(best["coarse_mask"], dtype=bool)
    best_quantile = float(best["quantile"])
    best_aligned_coords = np.asarray(best["aligned_coords"], dtype=float)
    best_reference = np.asarray(best["reference"], dtype=float)
    best_reference_index = int(best["reference_index"])
    core_atom_mask = best_mask[data.atom_to_residue]
    quality = build_quality_report(
        coarse_mask=best_coarse_mask,
        refined_mask=best_mask,
        aligned_coords=best_aligned_coords,
        reference=best_reference,
        coarse_scores=np.asarray(best["coarse_scores"], dtype=float),
        refined_scores=best_scores.combined,
        confidence=confidence,
        core_atom_mask=core_atom_mask,
        selected_quantile=best_quantile,
    )
    quality.notes.append(
        "competitive pipeline selected "
        f"fraction={float(best['fraction']):.2f} quantile={best_quantile:.2f} stage={best['stage']}"
    )
    uncertain_mask = uncertain_residues(best_scores.combined, quantile=best_quantile, margin=config.uncertainty_margin)
    uncertain_mask |= (confidence > 0.2) & (confidence < 0.8)

    return CoreResult(
        residue_scores=best_scores,
        confidence=confidence,
        core_mask=best_mask,
        uncertain_mask=uncertain_mask,
        candidate_core_mask=best_coarse_mask,
        coarse_frame_indices=np.array([reference_index], dtype=int),
        refined_reference_index=best_reference_index,
        quality=quality,
    )
