"""Simple evaluation metrics for alignment experiments."""

from __future__ import annotations

import numpy as np

from .utils import as_float_array


def mean_rmsd(coords: np.ndarray, reference: np.ndarray, atom_mask: np.ndarray | None = None) -> float:
    """Compute mean RMSD to a fixed reference."""

    coords = as_float_array(coords)
    reference = as_float_array(reference)
    if atom_mask is not None:
        coords = coords[:, atom_mask, :]
        reference = reference[atom_mask]
    squared = np.sum((coords - reference[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.sqrt(np.mean(squared, axis=1))))
