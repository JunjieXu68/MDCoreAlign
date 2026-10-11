"""Focused parity tests for the internal no-materialisation benchmark path."""

from __future__ import annotations

import numpy as np

from mdcorealign.benchmark_minimal import (
    load_ca_multimodel_pdb_minimal,
    solve_minimal_cpu,
)
from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming


def _trajectory() -> np.ndarray:
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(24, 3))
    frames = np.repeat(reference[None, :, :], 12, axis=0)
    frames += rng.normal(scale=0.03, size=frames.shape)
    frames[:, -5:] += np.linspace(0.0, 1.2, frames.shape[0])[:, None, None]
    return frames


def test_minimal_pdb_reader_roundtrip(tmp_path):
    coords = _trajectory()
    path = tmp_path / "toy.pdb"
    write_multimodel_ca_pdb_streaming(coords, path)
    observed, chains = load_ca_multimodel_pdb_minimal(
        path,
        expected_frames=coords.shape[0],
        expected_residues=coords.shape[1],
    )
    np.testing.assert_allclose(observed, coords, atol=5.1e-4)
    assert set(chains) == {"A"}


def test_exact_minimal_matches_standard_without_rmsf():
    coords = _trajectory()
    standard = solve_exact_contract(
        coords,
        representative_count=coords.shape[0],
        chunk_size=8,
        store_aligned=False,
    )
    minimal = solve_minimal_cpu(
        coords,
        mode="exact",
        representative_count=coords.shape[0],
        chunk_size=8,
    )
    np.testing.assert_allclose(minimal.frozen_weights, standard.frozen_weights)
    np.testing.assert_allclose(minimal.rotations, standard.rotations)
    np.testing.assert_allclose(minimal.translations, standard.translations)
    np.testing.assert_allclose(minimal.rmsd, standard.rmsd)
    np.testing.assert_allclose(minimal.weighted_rmsd, standard.weighted_rmsd)


def test_throughput_minimal_matches_standard_without_rmsf():
    coords = _trajectory()
    standard = solve_fast_global_alignment(
        coords,
        representative_count=coords.shape[0],
        update_rounds=4,
        chunk_size=8,
        store_aligned=False,
        compute_confidence=False,
    )
    minimal = solve_minimal_cpu(
        coords,
        mode="throughput",
        representative_count=coords.shape[0],
        update_rounds=4,
        chunk_size=8,
    )
    np.testing.assert_allclose(minimal.frozen_weights, standard.frozen_weights)
    np.testing.assert_allclose(minimal.rotations, standard.rotations)
    np.testing.assert_allclose(minimal.translations, standard.translations)
    np.testing.assert_allclose(minimal.rmsd, standard.rmsd)
    np.testing.assert_allclose(minimal.weighted_rmsd, standard.weighted_rmsd)
    assert minimal.metadata["aligned_coordinates_materialised"] is False
    assert minimal.metadata["RMSF_computed"] is False
