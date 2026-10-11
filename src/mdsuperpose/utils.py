"""Shared low-level helpers for the scaffold package."""

from __future__ import annotations

import numpy as np


def as_float_array(array: np.ndarray) -> np.ndarray:
    """Convert input to a float NumPy array."""

    return np.asarray(array, dtype=float)


def ensure_xyz_frames(coords: np.ndarray) -> np.ndarray:
    """Validate an array of shape (n_frames, n_atoms, 3)."""

    coords = as_float_array(coords)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    return coords


def ensure_atom_vector(values: np.ndarray, n_atoms: int, name: str) -> np.ndarray:
    """Validate a one-dimensional per-atom vector."""

    values = as_float_array(values)
    if values.shape != (n_atoms,):
        raise ValueError(f"{name} must have shape ({n_atoms},)")
    return values
