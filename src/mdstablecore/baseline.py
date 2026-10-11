"""Simple baseline methods for comparison."""

from __future__ import annotations

import numpy as np

from .alignment import align_trajectory, medoid_frame_index
from .core_detection import detect_final_core
from .quality_metrics import mean_rmsd
from .trajectory_io import residue_centroids
from .types import MDLovoFitStyleResult, TrajectoryData


def variance_only_residue_score(residue_coords: np.ndarray) -> np.ndarray:
    """Baseline residue score using inverse positional variance only."""

    variances = np.var(residue_coords, axis=0).sum(axis=1)
    median = float(np.median(variances))
    mad = float(np.median(np.abs(variances - median))) + 1e-8
    return 1.0 / (1.0 + np.clip((variances - median) / mad, 0.0, None))


def run_variance_baseline(
    data: TrajectoryData,
    core_quantile: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a simple variance-only baseline after all-atom alignment."""

    raw_residues = residue_centroids(data)
    reference_index = medoid_frame_index(raw_residues)
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
    scores = variance_only_residue_score(aligned_residues)
    core_mask = detect_final_core(scores, quantile=core_quantile)
    return scores, core_mask


def _residue_rmsd_to_reference(residue_coords: np.ndarray, reference_residues: np.ndarray) -> np.ndarray:
    """Compute per-residue RMSD relative to the reference frame."""

    squared = np.sum((residue_coords - reference_residues[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=0))


def _residue_rmsf(residue_coords: np.ndarray) -> np.ndarray:
    """Compute per-residue RMSF around the time-averaged position."""

    mean_positions = residue_coords.mean(axis=0)
    squared = np.sum((residue_coords - mean_positions[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=0))


def run_mdlovofit_style(
    data: TrajectoryData,
    fraction: float = 0.7,
    reference_frame: int = 0,
    iterations: int = 4,
) -> MDLovoFitStyleResult:
    """Approximate a MDLovoFit-style fraction-based robust alignment.

    This implementation follows the documented high-level idea:
    align the trajectory using only the fraction of residues with the lowest RMSD
    to a reference frame, updating the selected fraction iteratively.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")

    n_residues = data.n_residues
    n_selected = max(1, int(round(fraction * n_residues)))
    selected_residues = np.ones(n_residues, dtype=bool)
    reference = data.coords[reference_frame]
    reference_residue_coords = residue_centroids(
        TrajectoryData(
            coords=data.coords[[reference_frame]],
            atom_to_residue=data.atom_to_residue,
            residue_ids=data.residue_ids,
            atom_names=data.atom_names,
            frame_ids=np.array([reference_frame], dtype=int),
        )
    )[0]

    aligned = data.coords.copy()
    residue_coords = residue_centroids(data)

    for _ in range(iterations):
        atom_mask = selected_residues[data.atom_to_residue]
        aligned = align_trajectory(data.coords, reference, atom_mask=atom_mask if np.any(atom_mask) else None)
        residue_coords = residue_centroids(
            TrajectoryData(
                coords=aligned,
                atom_to_residue=data.atom_to_residue,
                residue_ids=data.residue_ids,
                atom_names=data.atom_names,
                frame_ids=data.frame_ids,
            )
        )
        residue_rmsd = _residue_rmsd_to_reference(residue_coords, reference_residue_coords)
        selected_indices = np.argsort(residue_rmsd)[:n_selected]
        selected_residues = np.zeros(n_residues, dtype=bool)
        selected_residues[selected_indices] = True

    residue_rmsd = _residue_rmsd_to_reference(residue_coords, reference_residue_coords)
    residue_rmsf = _residue_rmsf(residue_coords)
    low_atom_mask = selected_residues[data.atom_to_residue]
    high_atom_mask = ~low_atom_mask

    rmsd_low = mean_rmsd(aligned, reference, atom_mask=low_atom_mask)
    rmsd_high = mean_rmsd(aligned, reference, atom_mask=high_atom_mask) if np.any(high_atom_mask) else 0.0
    rmsd_all = mean_rmsd(aligned, reference)

    return MDLovoFitStyleResult(
        fraction=fraction,
        selected_residue_mask=selected_residues,
        residue_rmsd=residue_rmsd,
        residue_rmsf=residue_rmsf,
        rmsd_low_mean=rmsd_low,
        rmsd_high_mean=rmsd_high,
        rmsd_all_mean=rmsd_all,
    )


def scan_mdlovofit_style_fractions(
    data: TrajectoryData,
    fractions: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    reference_frame: int = 0,
    iterations: int = 4,
) -> list[MDLovoFitStyleResult]:
    """Run the MDLovoFit-style approximation across a fraction sweep."""

    return [
        run_mdlovofit_style(
            data,
            fraction=fraction,
            reference_frame=reference_frame,
            iterations=iterations,
        )
        for fraction in fractions
    ]
