"""Chain-aware coordinate-scale integrity checks for protein trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import numpy as np


@dataclass(frozen=True, slots=True)
class CoordinateScaleAudit:
    """Summary of a read-only coordinate-scale audit."""

    status: str
    inferred_internal_unit: str
    n_frames: int
    n_atoms: int
    n_valid_adjacent_pairs: int
    n_reference_contacts: int
    median_adjacent_ca_distance: float
    p05_adjacent_ca_distance: float
    p95_adjacent_ca_distance: float
    minimum_frame_median_adjacent_ca_distance: float
    maximum_frame_median_adjacent_ca_distance: float
    median_nonlocal_contact_distance: float
    minimum_scale_ratio_to_first: float
    maximum_scale_ratio_to_first: float
    abrupt_scale_change_count: int
    factor_ten_frame_count: int
    factor_tenth_frame_count: int
    notes: str

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-ready summary."""

        return asdict(self)


def _residue_number(value: object) -> int | None:
    """Parse the integer portion of a residue identifier."""

    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def chain_aware_adjacent_pairs(
    n_atoms: int,
    chain_ids: np.ndarray | None = None,
    residue_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return adjacent indices excluding chain breaks and residue gaps."""

    if n_atoms < 2:
        return np.empty((0, 2), dtype=np.int64)
    left = np.arange(n_atoms - 1, dtype=np.int64)
    right = left + 1
    valid = np.ones(n_atoms - 1, dtype=bool)
    if chain_ids is not None:
        chains = np.asarray(chain_ids).reshape(-1)
        if chains.size != n_atoms:
            raise ValueError("chain_ids must match the coordinate atom count")
        valid &= chains[left] == chains[right]
    if residue_ids is not None:
        residues = np.asarray(residue_ids).reshape(-1)
        if residues.size != n_atoms:
            raise ValueError("residue_ids must match the coordinate atom count")
        numbers = np.asarray(
            [_residue_number(value) for value in residues], dtype=object
        )
        sequential = np.asarray(
            [
                numbers[j] is not None
                and numbers[i] is not None
                and int(numbers[j]) - int(numbers[i]) == 1
                for i, j in zip(left, right, strict=True)
            ],
            dtype=bool,
        )
        valid &= sequential
    return np.column_stack((left[valid], right[valid]))


def _reference_contact_pairs(
    reference: np.ndarray,
    chain_ids: np.ndarray | None,
    cutoff: float,
) -> np.ndarray:
    """Build a fixed sparse nonlocal reference-contact set."""

    n_atoms = reference.shape[0]
    left, right = np.triu_indices(n_atoms, k=1)
    if chain_ids is None:
        nonlocal_mask = right - left > 3
    else:
        chains = np.asarray(chain_ids).reshape(-1)
        nonlocal_mask = (chains[left] != chains[right]) | (right - left > 3)
    left = left[nonlocal_mask]
    right = right[nonlocal_mask]
    distances = np.linalg.norm(reference[left] - reference[right], axis=1)
    selected = np.isfinite(distances) & (distances <= cutoff)
    return np.column_stack((left[selected], right[selected]))


def audit_coordinate_scale(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_ids: np.ndarray | None = None,
    contact_cutoff: float = 8.0,
) -> tuple[CoordinateScaleAudit, list[dict[str, object]]]:
    """Audit coordinate scale without modifying the input coordinates."""

    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (frames, atoms, 3)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("coords must contain frames and at least two atoms")
    if not np.isfinite(values).all():
        raise ValueError("coords contain NaN or infinite values")

    pairs = chain_aware_adjacent_pairs(
        values.shape[1], chain_ids, residue_ids
    )
    if pairs.size == 0:
        raise ValueError("no valid chain-aware adjacent residue pairs")
    adjacent = np.linalg.norm(
        values[:, pairs[:, 0]] - values[:, pairs[:, 1]], axis=2
    )
    frame_medians = np.median(adjacent, axis=1)
    baseline = float(frame_medians[0])
    ratios = frame_medians / baseline if baseline > 0.0 else np.full(
        frame_medians.shape, np.nan
    )
    step_ratios = np.divide(
        frame_medians[1:],
        frame_medians[:-1],
        out=np.full(frame_medians.size - 1, np.nan),
        where=frame_medians[:-1] > 0.0,
    )
    abrupt = (step_ratios > 3.0) | (step_ratios < (1.0 / 3.0))
    factor_ten = (ratios >= 8.0) & (ratios <= 12.0)
    factor_tenth = (ratios >= 0.08) & (ratios <= 0.12)

    contact_pairs = _reference_contact_pairs(
        values[0], chain_ids, contact_cutoff
    )
    if contact_pairs.size:
        contact_distances = np.linalg.norm(
            values[:, contact_pairs[:, 0]]
            - values[:, contact_pairs[:, 1]],
            axis=2,
        )
        frame_contact_medians = np.median(contact_distances, axis=1)
    else:
        frame_contact_medians = np.full(values.shape[0], np.nan)

    overall_median = float(np.median(frame_medians))
    minimum = float(np.min(frame_medians))
    maximum = float(np.max(frame_medians))
    if minimum < 8.0 and maximum > 20.0:
        status = "mixed_scale_factor_ten"
        unit = "mixed"
        notes = "Both Angstrom-like and approximately tenfold CA geometry occur."
    elif 0.25 <= overall_median <= 0.55:
        status = "uniform_nanometer_suspect"
        unit = "nanometer_suspect"
        notes = "CA geometry is approximately one tenth of expected Angstrom scale."
    elif 2.5 <= overall_median <= 5.5 and not np.any(abrupt):
        status = "angstrom_consistent"
        unit = "angstrom"
        notes = "Chain-aware CA geometry is consistent with Angstrom coordinates."
    elif 25.0 <= overall_median <= 55.0:
        status = "uniform_factor_ten_suspect"
        unit = "tenfold_angstrom_suspect"
        notes = "CA geometry is approximately tenfold larger than expected."
    else:
        status = "undetermined"
        unit = "unknown"
        notes = "Coordinate scale could not be classified from CA geometry."

    frame_rows = [
        {
            "frame_index": int(index),
            "median_adjacent_ca_distance": float(frame_medians[index]),
            "p05_adjacent_ca_distance": float(
                np.quantile(adjacent[index], 0.05)
            ),
            "p95_adjacent_ca_distance": float(
                np.quantile(adjacent[index], 0.95)
            ),
            "median_nonlocal_contact_distance": float(
                frame_contact_medians[index]
            ),
            "scale_ratio_to_first_valid_frame": float(ratios[index]),
            "abrupt_scale_change_from_previous": bool(
                index > 0 and abrupt[index - 1]
            ),
            "factor_ten_cluster_vs_first": bool(factor_ten[index]),
            "factor_tenth_cluster_vs_first": bool(factor_tenth[index]),
        }
        for index in range(values.shape[0])
    ]
    summary = CoordinateScaleAudit(
        status=status,
        inferred_internal_unit=unit,
        n_frames=values.shape[0],
        n_atoms=values.shape[1],
        n_valid_adjacent_pairs=pairs.shape[0],
        n_reference_contacts=contact_pairs.shape[0],
        median_adjacent_ca_distance=overall_median,
        p05_adjacent_ca_distance=float(np.quantile(adjacent, 0.05)),
        p95_adjacent_ca_distance=float(np.quantile(adjacent, 0.95)),
        minimum_frame_median_adjacent_ca_distance=minimum,
        maximum_frame_median_adjacent_ca_distance=maximum,
        median_nonlocal_contact_distance=float(
            np.nanmedian(frame_contact_medians)
        )
        if contact_pairs.size
        else float("nan"),
        minimum_scale_ratio_to_first=float(np.nanmin(ratios)),
        maximum_scale_ratio_to_first=float(np.nanmax(ratios)),
        abrupt_scale_change_count=int(np.count_nonzero(abrupt)),
        factor_ten_frame_count=int(np.count_nonzero(factor_ten)),
        factor_tenth_frame_count=int(np.count_nonzero(factor_tenth)),
        notes=notes,
    )
    return summary, frame_rows
