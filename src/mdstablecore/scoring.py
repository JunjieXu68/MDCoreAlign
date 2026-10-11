"""Residue-level robust scoring functions."""

from __future__ import annotations

import numpy as np

from .types import ResidueScores


def _robust_scale_to_unit(values: np.ndarray) -> np.ndarray:
    """Convert non-negative dispersion values into stability-like scores."""

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-8
    z_like = (values - median) / mad
    return 1.0 / (1.0 + np.clip(z_like, 0.0, None))


def positional_stability(residue_coords: np.ndarray) -> np.ndarray:
    """Estimate positional stability with median absolute deviations."""

    median_positions = np.median(residue_coords, axis=0)
    radial_distances = np.linalg.norm(residue_coords - median_positions[None, :, :], axis=2)
    dispersion = np.median(np.abs(radial_distances - np.median(radial_distances, axis=0)), axis=0)
    return _robust_scale_to_unit(dispersion)


def local_geometry_stability(residue_coords: np.ndarray, radius: int = 2) -> np.ndarray:
    """Score stability of local sequence-neighbor geometry."""

    n_frames, n_residues, _ = residue_coords.shape
    local_dispersion = np.zeros(n_residues, dtype=float)

    for residue in range(n_residues):
        offsets = []
        for delta in range(-radius, radius + 1):
            if delta == 0:
                continue
            neighbor = residue + delta
            if 0 <= neighbor < n_residues:
                distances = np.linalg.norm(
                    residue_coords[:, residue, :] - residue_coords[:, neighbor, :],
                    axis=1,
                )
                offsets.append(np.median(np.abs(distances - np.median(distances))))
        if offsets:
            local_dispersion[residue] = float(np.mean(offsets))

    return _robust_scale_to_unit(local_dispersion)


def contact_persistence_score(residue_coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Estimate how persistently each residue forms non-local contacts."""

    n_frames, n_residues, _ = residue_coords.shape
    if n_residues < 4:
        return np.ones(n_residues, dtype=float)

    counts = np.zeros((n_frames, n_residues), dtype=float)
    persistent_fraction = np.zeros(n_residues, dtype=float)

    for frame_index in range(n_frames):
        diffs = residue_coords[frame_index][:, None, :] - residue_coords[frame_index][None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        contact_matrix = (distances < cutoff).astype(float)
        for residue in range(n_residues):
            lo = max(0, residue - 2)
            hi = min(n_residues, residue + 3)
            contact_matrix[residue, lo:hi] = 0.0
        counts[frame_index] = contact_matrix.sum(axis=1)

    residue_contact_mask = counts > 0.0
    persistence = residue_contact_mask.mean(axis=0)
    magnitude = np.clip(counts.mean(axis=0) / max(1.0, counts.max()), 0.0, 1.0)
    persistent_fraction[:] = 0.6 * persistence + 0.4 * magnitude
    return persistent_fraction


def temporal_consistency_score(residue_coords: np.ndarray, n_blocks: int = 4) -> np.ndarray:
    """Estimate how reproducible positional stability is across time blocks."""

    block_edges = np.linspace(0, residue_coords.shape[0], num=n_blocks + 1, dtype=int)
    block_scores = []
    for start, end in zip(block_edges[:-1], block_edges[1:]):
        if end - start < 2:
            continue
        block_scores.append(positional_stability(residue_coords[start:end]))

    if not block_scores:
        return np.ones(residue_coords.shape[1], dtype=float)

    score_stack = np.vstack(block_scores)
    dispersion = np.median(np.abs(score_stack - np.median(score_stack, axis=0)), axis=0)
    return _robust_scale_to_unit(dispersion)


def combine_scores(
    positional: np.ndarray,
    local_geometry: np.ndarray,
    contact_persistence: np.ndarray,
    temporal_consistency: np.ndarray,
) -> ResidueScores:
    """Combine residue-level scores into a weighted summary."""

    combined = (
        0.4 * positional
        + 0.25 * local_geometry
        + 0.2 * contact_persistence
        + 0.15 * temporal_consistency
    )
    return ResidueScores(
        positional=positional,
        local_geometry=local_geometry,
        contact_persistence=contact_persistence,
        temporal_consistency=temporal_consistency,
        combined=combined,
    )


def score_residues(
    residue_coords: np.ndarray,
    local_window_radius: int = 2,
    contact_cutoff: float = 8.0,
    temporal_blocks: int = 4,
) -> ResidueScores:
    """Compute all residue-level scoring components."""

    positional = positional_stability(residue_coords)
    local_geometry = local_geometry_stability(residue_coords, radius=local_window_radius)
    contacts = contact_persistence_score(residue_coords, cutoff=contact_cutoff)
    temporal = temporal_consistency_score(residue_coords, n_blocks=temporal_blocks)
    return combine_scores(positional, local_geometry, contacts, temporal)
