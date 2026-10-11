"""Trajectory input helpers for array-first workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import TrajectoryData


def make_trajectory_data(
    coords: np.ndarray,
    atom_to_residue: np.ndarray,
    residue_ids: list[str] | None = None,
    atom_names: list[str] | None = None,
) -> TrajectoryData:
    """Validate raw arrays and construct a trajectory container."""

    coords = np.asarray(coords, dtype=float)
    atom_to_residue = np.asarray(atom_to_residue, dtype=int)

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    if atom_to_residue.ndim != 1 or atom_to_residue.shape[0] != coords.shape[1]:
        raise ValueError("atom_to_residue must have shape (n_atoms,)")
    if atom_to_residue.min() < 0:
        raise ValueError("atom_to_residue cannot contain negative values")

    n_residues = int(atom_to_residue.max()) + 1
    if residue_ids is None:
        residue_ids = [f"RES{i + 1}" for i in range(n_residues)]
    if len(residue_ids) != n_residues:
        raise ValueError("residue_ids length does not match residue count")

    return TrajectoryData(
        coords=coords,
        atom_to_residue=atom_to_residue,
        residue_ids=residue_ids,
        atom_names=atom_names,
        frame_ids=np.arange(coords.shape[0], dtype=int),
    )


def residue_centroids(data: TrajectoryData, atom_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute residue centroids for every frame."""

    coords = data.coords if atom_mask is None else data.coords[:, atom_mask, :]
    mapping = data.atom_to_residue if atom_mask is None else data.atom_to_residue[atom_mask]
    n_residues = data.n_residues
    centroids = np.zeros((data.n_frames, n_residues, 3), dtype=float)

    for residue_index in range(n_residues):
        residue_atoms = np.where(mapping == residue_index)[0]
        if residue_atoms.size == 0:
            continue
        centroids[:, residue_index, :] = coords[:, residue_atoms, :].mean(axis=1)

    return centroids


def load_mdanalysis_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path,
    selection: str = "protein and name CA",
    stride: int = 1,
    max_frames: int | None = None,
) -> TrajectoryData:
    """Load a trajectory through MDAnalysis into the internal array format."""

    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path), str(trajectory_path))
    atoms = universe.select_atoms(selection)

    if atoms.n_atoms == 0:
        raise ValueError(f"Selection returned no atoms: {selection}")

    frames = []
    for frame_index, _ in enumerate(universe.trajectory[::stride]):
        frames.append(atoms.positions.astype(float).copy())
        if max_frames is not None and (frame_index + 1) >= max_frames:
            break

    coords = np.asarray(frames, dtype=float)
    residue_indices = atoms.resindices
    _, atom_to_residue = np.unique(residue_indices, return_inverse=True)
    residue_ids = [f"{res.resname}{res.resid}" for res in atoms.residues]
    atom_names = atoms.names.tolist()

    return make_trajectory_data(
        coords=coords,
        atom_to_residue=atom_to_residue,
        residue_ids=residue_ids,
        atom_names=atom_names,
    )
