"""Trajectory container and lightweight I/O helpers for array-first workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import as_float_array, ensure_xyz_frames


@dataclass(slots=True)
class Trajectory:
    """Minimal trajectory container used by the first prototype."""

    coords: np.ndarray

    def __post_init__(self) -> None:
        """Validate trajectory coordinates after construction."""

        self.coords = ensure_xyz_frames(self.coords)

    @property
    def n_frames(self) -> int:
        """Return the number of frames."""

        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms."""

        return int(self.coords.shape[1])


@dataclass(slots=True)
class ProteinCoordinates:
    """One protein-like coordinate set with explicit residue labels."""

    coords: np.ndarray
    residue_labels: tuple[str, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate one set of paired protein coordinates."""

        self.coords = _ensure_xyz_points(self.coords)
        self.residue_labels = tuple(str(label) for label in self.residue_labels)
        if len(self.residue_labels) != self.coords.shape[0]:
            raise ValueError("residue_labels must match the number of coordinate rows")

    @property
    def n_atoms(self) -> int:
        """Return the number of coordinate rows."""

        return int(self.coords.shape[0])


@dataclass(slots=True)
class PairedProteinCoordinates:
    """A validated pair of corresponding protein coordinate sets."""

    mobile_coords: np.ndarray
    reference_coords: np.ndarray
    residue_labels: tuple[str, ...]
    correspondence_mode: str
    mobile_source: str | None = None
    reference_source: str | None = None

    @property
    def n_atoms(self) -> int:
        """Return the number of paired coordinates."""

        return int(self.mobile_coords.shape[0])


def _ensure_xyz_points(coords: np.ndarray) -> np.ndarray:
    """Validate one coordinate set of shape `(n_atoms, 3)`."""

    coords = as_float_array(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3:
        raise ValueError("coords must contain at least three paired points")
    if not np.isfinite(coords).all():
        raise ValueError("coords must contain only finite values")
    return coords


def _default_residue_labels(n_atoms: int) -> tuple[str, ...]:
    """Build default order-based residue labels for array input."""

    return tuple(f"index:{index}" for index in range(n_atoms))


def load_protein_coords_from_array(
    coords: np.ndarray,
    *,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    source: str | None = None,
) -> ProteinCoordinates:
    """Load one protein-like coordinate set directly from a NumPy-style array.

    When `residue_labels` are omitted, correspondence is assumed purely by
    coordinate order.
    """

    coords = _ensure_xyz_points(coords)
    if residue_labels is None:
        residue_labels = _default_residue_labels(coords.shape[0])
    return ProteinCoordinates(
        coords=coords,
        residue_labels=tuple(str(label) for label in residue_labels),
        source=source or "array",
    )


def _format_correspondence_error(
    mobile_labels: tuple[str, ...],
    reference_labels: tuple[str, ...],
) -> str:
    """Format a clear label-mismatch error for paired protein inputs."""

    first_mismatch = next(
        (
            (index, mobile_label, reference_label)
            for index, (mobile_label, reference_label) in enumerate(zip(mobile_labels, reference_labels, strict=True))
            if mobile_label != reference_label
        ),
        None,
    )
    mobile_only = [label for label in mobile_labels if label not in set(reference_labels)][:3]
    reference_only = [label for label in reference_labels if label not in set(mobile_labels)][:3]
    parts = [
        "paired protein coordinates require consistent atom correspondence",
        "residue labels did not match exactly",
    ]
    if first_mismatch is not None:
        index, mobile_label, reference_label = first_mismatch
        parts.append(
            f"first mismatch at index {index}: mobile={mobile_label}, reference={reference_label}"
        )
    if mobile_only:
        parts.append(f"labels only in mobile: {mobile_only}")
    if reference_only:
        parts.append(f"labels only in reference: {reference_only}")
    parts.append("missing residues or inconsistent numbering must be fixed before alignment")
    return "; ".join(parts)


def make_paired_protein_coordinates(
    mobile: ProteinCoordinates,
    reference: ProteinCoordinates,
    *,
    correspondence_mode: str,
) -> PairedProteinCoordinates:
    """Validate and combine two protein coordinate sets into one paired object."""

    if mobile.coords.shape != reference.coords.shape:
        raise ValueError("mobile and reference coordinates must have the same shape")
    if mobile.residue_labels != reference.residue_labels:
        raise ValueError(_format_correspondence_error(mobile.residue_labels, reference.residue_labels))
    return PairedProteinCoordinates(
        mobile_coords=mobile.coords,
        reference_coords=reference.coords,
        residue_labels=mobile.residue_labels,
        correspondence_mode=correspondence_mode,
        mobile_source=mobile.source,
        reference_source=reference.source,
    )


def load_paired_protein_coords_from_arrays(
    mobile_coords: np.ndarray,
    reference_coords: np.ndarray,
    *,
    mobile_residue_labels: list[str] | tuple[str, ...] | None = None,
    reference_residue_labels: list[str] | tuple[str, ...] | None = None,
) -> PairedProteinCoordinates:
    """Load paired protein coordinates directly from arrays.

    If explicit residue labels are not provided, order-based correspondence is
    assumed and cannot detect missing residues beyond shape mismatches.
    """

    mobile = load_protein_coords_from_array(
        mobile_coords,
        residue_labels=mobile_residue_labels,
        source="array_mobile",
    )
    reference = load_protein_coords_from_array(
        reference_coords,
        residue_labels=reference_residue_labels,
        source="array_reference",
    )
    return make_paired_protein_coordinates(
        mobile,
        reference,
        correspondence_mode="order" if mobile_residue_labels is None and reference_residue_labels is None else "residue_labels",
    )


def load_ca_coords_from_pdb(path: str | Path, *, chain_id: str | None = None) -> ProteinCoordinates:
    """Read CA-only coordinates from a simple PDB file.

    This lightweight reader assumes that atom correspondence is determined by
    matching residue labels exactly. Only `ATOM` records with atom name `CA`
    are used. Alternate locations other than blank or `A` are ignored.
    """

    path = Path(path)
    coords: list[list[float]] = []
    residue_labels: list[str] = []
    seen: set[str] = set()

    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            chain = line[21].strip() or "_"
            if chain_id is not None and chain != chain_id:
                continue
            resname = line[17:20].strip() or "UNK"
            resseq = line[22:26].strip()
            insertion_code = line[26].strip()
            if not resseq:
                raise ValueError(f"missing residue sequence number in {path}")
            residue_label = f"{chain}:{resname}{resseq}{insertion_code}"
            if residue_label in seen:
                raise ValueError(
                    f"duplicate CA entry for residue {residue_label} in {path}; "
                    "please preprocess alternate locations or duplicates explicitly"
                )
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            residue_labels.append(residue_label)
            seen.add(residue_label)

    if not coords:
        if chain_id is None:
            raise ValueError(f"no CA atoms could be parsed from {path}")
        raise ValueError(f"no CA atoms could be parsed from {path} for chain {chain_id}")
    return ProteinCoordinates(
        coords=np.asarray(coords, dtype=float),
        residue_labels=tuple(residue_labels),
        source=str(path),
    )


def load_paired_ca_coords_from_pdb(
    mobile_path: str | Path,
    reference_path: str | Path,
    *,
    mobile_chain_id: str | None = None,
    reference_chain_id: str | None = None,
) -> PairedProteinCoordinates:
    """Load paired CA-only protein coordinates from two PDB files.

    Missing residues are rejected by exact residue-label comparison.
    """

    mobile = load_ca_coords_from_pdb(mobile_path, chain_id=mobile_chain_id)
    reference = load_ca_coords_from_pdb(reference_path, chain_id=reference_chain_id)
    return make_paired_protein_coordinates(
        mobile,
        reference,
        correspondence_mode="residue_labels",
    )


def load_trajectory_from_array_file(path: str | Path, key: str | None = None) -> Trajectory:
    """Load a trajectory from a `.npy` or `.npz` coordinate file."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        coords = np.load(path)
        return Trajectory(coords=coords)
    if suffix == ".npz":
        with np.load(path) as archive:
            if key is None:
                key = "coords" if "coords" in archive.files else archive.files[0]
            if key not in archive.files:
                raise KeyError(f"key '{key}' was not found in {path}")
            coords = archive[key]
        return Trajectory(coords=coords)
    raise ValueError("array trajectory files must use the .npy or .npz extension")


def load_trajectory_from_multimodel_pdb(path: str | Path) -> Trajectory:
    """Load a trajectory from a multi-model or repeated-model PDB file."""

    path = Path(path)
    frames: list[list[list[float]]] = []
    current: list[list[float]] = []

    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM"):
                current.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            elif line.startswith("ENDMDL"):
                if current:
                    frames.append(current)
                    current = []

    if current:
        frames.append(current)
    if not frames:
        raise ValueError(f"no coordinate frames could be parsed from {path}")

    n_atoms = len(frames[0])
    if n_atoms == 0:
        raise ValueError(f"no atoms were parsed from the first frame of {path}")
    if any(len(frame) != n_atoms for frame in frames):
        raise ValueError("all parsed PDB frames must contain the same number of atoms")
    return Trajectory(coords=np.asarray(frames, dtype=float))
