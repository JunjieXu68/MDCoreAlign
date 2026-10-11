"""Tests for the D0-D1 database-scale benchmark support code."""

from __future__ import annotations

import h5py
import numpy as np

from scripts.database_scale_gpu_v1.common import (
    core_mask,
    cpu_gpu_agreement,
    jaccard,
    mapping_digest,
    quality_against_exact,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)


class Result:
    """Minimal alignment-result fixture."""

    def __init__(self, weights: np.ndarray) -> None:
        self.frozen_weights = weights
        self.rotations = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
        self.translations = np.zeros((4, 3))
        self.robust_rmsf = np.linspace(0.1, 0.4, weights.size)
        self.rmsd = np.linspace(0.2, 0.5, 4)
        self.weighted_rmsd = self.rmsd * 0.8


def test_mapping_digest_is_order_sensitive() -> None:
    mapping = [("A", "1", "ALA"), ("A", "2", "GLY")]
    assert mapping_digest(mapping) == mapping_digest(list(mapping))
    assert mapping_digest(mapping) != mapping_digest(list(reversed(mapping)))


def test_core_mask_is_deterministic_for_ties() -> None:
    weights = np.ones(10)
    first = core_mask(weights)
    second = core_mask(weights)
    assert np.array_equal(first, second)
    assert np.flatnonzero(first).tolist() == [0, 1, 2, 3, 4, 5]


def test_quality_and_cpu_gpu_schema() -> None:
    weights = np.linspace(0.1, 1.0, 10)
    exact = Result(weights)
    candidate = Result(weights.copy())
    chains = np.asarray(["A"] * 10)
    exact_core = core_mask(weights)
    metrics = quality_against_exact(
        candidate,
        exact_weights=weights,
        exact_rmsd=exact.rmsd,
        exact_rmsf=exact.robust_rmsf,
        exact_core=exact_core,
        exact_flexible=np.zeros(10, dtype=bool),
        chains=chains,
    )
    agreement = cpu_gpu_agreement(exact, candidate, chains=chains)
    assert metrics["finite_outputs"]
    assert metrics["core_jaccard_vs_exact"] == 1.0
    assert metrics["median_absolute_rmsd_difference_vs_exact_A"] == 0.0
    assert agreement["cpu_gpu_core_jaccard"] == 1.0
    assert np.isclose(agreement["cpu_gpu_weight_spearman"], 1.0)
    assert jaccard(exact_core, exact_core) == 1.0


def test_contiguous_hdf5_read_matches_fancy_ca_read(tmp_path) -> None:
    h5_path = tmp_path / "mdcath_dataset_toyA00.h5"
    pdb = "".join(
        f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
        f"{index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        for index in range(1, 31)
    )
    coords = np.arange(20 * 30 * 3, dtype=np.float32).reshape(20, 30, 3)
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("toyA00")
        group.create_dataset("pdb", data=np.bytes_(pdb))
        dataset = handle.create_dataset(
            "toyA00/320/0/coords",
            data=coords,
        )
        dataset.attrs["unit"] = "Angstrom"
    fancy, fancy_labels, _, fancy_meta = extract_ca_trajectory(
        h5_path,
        coordinate_path="/toyA00/320/0/coords",
    )
    contiguous, contiguous_labels, _, contiguous_meta = (
        extract_ca_trajectory(
            h5_path,
            coordinate_path="/toyA00/320/0/coords",
            contiguous_full_atom_read=True,
        )
    )
    direct, direct_labels, _, direct_meta = extract_ca_trajectory(
        h5_path,
        coordinate_path="/toyA00/320/0/coords",
        direct_contiguous_read=True,
    )
    assert np.array_equal(fancy, contiguous)
    assert np.array_equal(fancy, direct)
    assert fancy_labels == contiguous_labels
    assert fancy_labels == direct_labels
    assert fancy_meta["read_strategy"] == "HDF5_CA_fancy_index"
    assert (
        contiguous_meta["read_strategy"]
        == "contiguous_full_atom_then_CA"
    )
    assert direct_meta["read_strategy"] == "direct_contiguous_bytes_then_CA"
