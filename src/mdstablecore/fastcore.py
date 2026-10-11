"""Fast non-Lovo stable-core pipeline."""

from __future__ import annotations

import numpy as np

from .alignment import align_trajectory, medoid_frame_index
from .core_detection import detect_final_core, uncertain_residues
from .quality_metrics import build_quality_report
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


def _fast_scores(
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
        0.40 * positional
        + 0.30 * rmsf_score
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
    squared_deviations: np.ndarray,
    atom_mask: np.ndarray,
    all_rmsd: float,
    target_fraction: float,
) -> float:
    """Score a candidate core using compactness, separation, and size balance."""

    if np.any(atom_mask):
        core_rmsd = float(np.mean(np.sqrt(np.mean(squared_deviations[:, atom_mask], axis=1))))
    else:
        core_rmsd = np.inf
    if np.any(mask) and np.any(~mask):
        separation = float(scores[mask].mean() - scores[~mask].mean())
    else:
        separation = 0.0
    core_fraction = float(np.mean(mask))
    size_score = max(0.0, 1.0 - abs(core_fraction - target_fraction) / max(target_fraction, 1e-8))
    return (
        0.45 * (1.0 / (1.0 + core_rmsd))
        + 0.25 * (1.0 / (1.0 + all_rmsd))
        + 0.20 * separation
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


def run_fastcore_pipeline(
    data: TrajectoryData,
    config: PipelineConfig,
) -> CoreResult:
    """Run a fast stable-core pipeline without Lovo-style seeding."""

    raw_residue_coords = residue_centroids(data)
    initial_reference_index = medoid_frame_index(raw_residue_coords)
    initial_reference = data.coords[initial_reference_index]

    aligned_initial = align_trajectory(data.coords, initial_reference)
    aligned_initial_data = TrajectoryData(
        coords=aligned_initial,
        atom_to_residue=data.atom_to_residue,
        residue_ids=data.residue_ids,
        atom_names=data.atom_names,
        frame_ids=data.frame_ids,
    )
    initial_residue_coords = residue_centroids(aligned_initial_data)
    initial_scores = _fast_scores(
        initial_residue_coords,
        reference_residues=initial_residue_coords[initial_reference_index],
        config=config,
    )
    seed_mask = detect_final_core(
        initial_scores.combined,
        quantile=config.fastcore_seed_quantile,
        confidence=None,
        min_run_length=config.min_core_run_length,
        terminal_exclusion=config.terminal_exclusion,
    )
    if not np.any(seed_mask):
        seed_mask = detect_final_core(
            initial_scores.combined,
            quantile=0.5,
            confidence=None,
            min_run_length=config.min_core_run_length,
            terminal_exclusion=config.terminal_exclusion,
        )

    refined_reference_index = (
        medoid_frame_index(initial_residue_coords[:, seed_mask, :]) if np.any(seed_mask) else initial_reference_index
    )
    refined_reference = aligned_initial[refined_reference_index]

    aligned_final = align_trajectory(
        data.coords,
        refined_reference,
        atom_mask=seed_mask[data.atom_to_residue] if np.any(seed_mask) else None,
    )
    aligned_final_data = TrajectoryData(
        coords=aligned_final,
        atom_to_residue=data.atom_to_residue,
        residue_ids=data.residue_ids,
        atom_names=data.atom_names,
        frame_ids=data.frame_ids,
    )
    final_residue_coords = residue_centroids(aligned_final_data)
    final_scores = _fast_scores(
        final_residue_coords,
        reference_residues=final_residue_coords[refined_reference_index],
        config=config,
    )
    squared_deviations = np.sum((aligned_final - refined_reference[None, :, :]) ** 2, axis=2)
    all_rmsd = float(np.mean(np.sqrt(np.mean(squared_deviations, axis=1))))

    candidates: list[dict[str, object]] = []
    for quantile in config.fastcore_quantiles:
        mask = detect_final_core(
            final_scores.combined,
            quantile=quantile,
            confidence=None,
            min_run_length=config.min_core_run_length,
            terminal_exclusion=config.terminal_exclusion,
        )
        if not np.any(mask):
            continue
        atom_mask = mask[data.atom_to_residue]
        objective = _candidate_objective(
            mask=mask,
            scores=final_scores.combined,
            squared_deviations=squared_deviations,
            atom_mask=atom_mask,
            all_rmsd=all_rmsd,
            target_fraction=config.fastcore_target_core_fraction,
        )
        candidates.append(
            {
                "quantile": quantile,
                "mask": mask,
                "objective": objective,
            }
        )

    if not candidates:
        raise RuntimeError("Fastcore pipeline could not generate a valid core candidate")

    best = max(candidates, key=lambda candidate: float(candidate["objective"]))
    best_mask = np.asarray(best["mask"], dtype=bool)
    confidence = _confidence_from_candidates(candidates, n_residues=data.n_residues)
    uncertain_mask = uncertain_residues(
        final_scores.combined,
        quantile=float(best["quantile"]),
        margin=config.uncertainty_margin,
    )
    uncertain_mask |= (confidence > 0.2) & (confidence < 0.8)
    core_atom_mask = best_mask[data.atom_to_residue]
    quality = build_quality_report(
        coarse_mask=seed_mask,
        refined_mask=best_mask,
        aligned_coords=aligned_final,
        reference=refined_reference,
        coarse_scores=initial_scores.combined,
        refined_scores=final_scores.combined,
        confidence=confidence,
        core_atom_mask=core_atom_mask,
        selected_quantile=float(best["quantile"]),
    )
    quality.notes.append(f"fastcore pipeline selected quantile={float(best['quantile']):.2f}")

    return CoreResult(
        residue_scores=final_scores,
        confidence=confidence,
        core_mask=best_mask,
        uncertain_mask=uncertain_mask,
        candidate_core_mask=seed_mask,
        coarse_frame_indices=np.array([initial_reference_index], dtype=int),
        refined_reference_index=refined_reference_index,
        quality=quality,
    )
