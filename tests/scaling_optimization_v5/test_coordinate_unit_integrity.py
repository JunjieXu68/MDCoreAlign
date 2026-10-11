"""Regression tests for the Phase 7B coordinate-unit integrity guard."""

from __future__ import annotations

import numpy as np

from mdsuperpose.coordinate_scale import (
    audit_coordinate_scale,
    chain_aware_adjacent_pairs,
)
from scripts.mdcath_benchmark.build_transition_aligned_stitched_trajectory import (
    build_transition_aligned,
)
from scripts.mdcath_benchmark.mdcath_common import (
    maybe_convert_units_to_angstrom,
)


def _ca_trajectory(n_frames: int = 8, n_residues: int = 12) -> np.ndarray:
    reference = np.zeros((n_residues, 3), dtype=np.float64)
    reference[:, 0] = 3.8 * np.arange(n_residues)
    coords = np.repeat(reference[None], n_frames, axis=0)
    coords[:, 5:, 1] += np.linspace(0.0, 0.2, n_frames)[:, None]
    return coords


def test_explicit_angstrom_metadata_is_never_rescaled() -> None:
    coords = _ca_trajectory()
    observed, status = maybe_convert_units_to_angstrom(
        coords, {"unit": "Angstrom"}
    )
    np.testing.assert_array_equal(observed, coords)
    assert status == "verified_angstrom_metadata"


def test_explicit_nanometer_metadata_is_converted_to_angstrom() -> None:
    coords_nm = _ca_trajectory() / 10.0
    observed, status = maybe_convert_units_to_angstrom(
        coords_nm, {"unit": "nm"}
    )
    np.testing.assert_allclose(observed, _ca_trajectory())
    assert status == "converted_nm_to_angstrom"
    audit, _ = audit_coordinate_scale(observed)
    assert audit.status == "angstrom_consistent"


def test_mixed_angstrom_nanometer_trajectory_is_detected() -> None:
    coords = _ca_trajectory()
    coords[4:] *= 10.0
    audit, rows = audit_coordinate_scale(coords)
    assert audit.status == "mixed_scale_factor_ten"
    assert audit.abrupt_scale_change_count == 1
    assert sum(row["factor_ten_cluster_vs_first"] for row in rows) == 4


def test_chain_breaks_and_missing_residues_are_excluded() -> None:
    chains = np.asarray(["A", "A", "A", "B", "B", "B"])
    residues = np.asarray(["1", "2", "4", "1", "2", "4"])
    pairs = chain_aware_adjacent_pairs(6, chains, residues)
    np.testing.assert_array_equal(pairs, np.asarray([[0, 1], [3, 4]]))


def test_valid_angstrom_trajectory_passes_chain_aware_guard() -> None:
    coords = _ca_trajectory()
    audit, rows = audit_coordinate_scale(
        coords,
        chain_ids=np.asarray(["A"] * coords.shape[1]),
        residue_ids=np.arange(1, coords.shape[1] + 1),
    )
    assert audit.status == "angstrom_consistent"
    assert audit.inferred_internal_unit == "angstrom"
    assert len(rows) == coords.shape[0]


def test_transition_aligned_stitching_preserves_consistent_scale() -> None:
    coords = _ca_trajectory(n_frames=6)
    metadata = [
        {"chain": "A", "residue_id": str(i + 1), "resname": "ALA"}
        for i in range(coords.shape[1])
    ]
    labels = [f"A:{i + 1}:ALA" for i in range(coords.shape[1])]
    raw = {
        label: (coords + index, labels, metadata)
        for index, label in enumerate("ABCDE")
    }
    stitched, _, _, _ = build_transition_aligned(
        "toy",
        raw,
        ["A", "B", "A"],
        16,
        "canonical_cycle",
        "keep_boundary_first_frame",
    )
    audit, _ = audit_coordinate_scale(
        stitched,
        chain_ids=np.asarray(["A"] * coords.shape[1]),
        residue_ids=np.arange(1, coords.shape[1] + 1),
    )
    assert audit.status == "angstrom_consistent"
