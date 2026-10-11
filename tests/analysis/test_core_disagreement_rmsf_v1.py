"""Necessary numerical and provenance checks for the new RMSF diagnostic."""

import csv

import numpy as np
import pytest

from mdsuperpose.fast_kabsch import apply_transforms_batch, weighted_kabsch_transform_batch
from scripts.analysis.core_disagreement_rmsf_v1 import (
    binary_fit_rmsf,
    index_ranges,
    neighborhood_mask,
    source_sets,
)


def test_streamed_binary_rmsf_matches_dense_temporal_variance():
    rng = np.random.default_rng(20260915)
    reference = rng.normal(size=(12, 3)) * 5
    coords = reference + rng.normal(size=(19, 12, 3)) * 0.2
    coords += rng.normal(size=(19, 1, 3)) * 30
    mask = np.arange(12) < 8
    transform = weighted_kabsch_transform_batch(coords, reference, mask.astype(float))
    aligned = apply_transforms_batch(coords, transform.rotations, transform.translations)
    expected = np.sqrt(np.var(aligned, axis=0, ddof=0).sum(axis=1))
    for block_size in [1, 6, 40]:
        actual, audit = binary_fit_rmsf(coords, reference, mask, chunk_size=block_size)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert audit["evaluated_frames"] == 19
        assert not audit["aligned_trajectory_retained"]


def test_collinear_reference_and_mobile_frames_are_unevaluable():
    reference = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    with pytest.raises(ValueError, match="collinear reference"):
        binary_fit_rmsf(np.repeat(reference[None], 3, axis=0), reference, np.ones(4, bool))
    reference[2, 1] = 1
    coords = np.repeat(reference[None], 3, axis=0)
    coords[1, :, 1] = 0
    with pytest.raises(ValueError, match="cross covariance.*frame 1"):
        binary_fit_rmsf(coords, reference, np.ones(4, bool), chunk_size=1)


def test_planar_noncollinear_points_are_evaluable():
    reference = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    coords = np.repeat(reference[None], 3, axis=0)
    coords[1] += [8, 9, -3]
    profile, _ = binary_fit_rmsf(coords, reference, np.ones(4, bool))
    np.testing.assert_allclose(profile, 0, atol=1e-12)


def test_neighborhood_is_sequence_based_and_does_not_cross_chains():
    residues = [
        dict(chain="A", sequence_position=i + 1, source_resid=100 * (i + 1)) for i in range(7)
    ]
    residues[6]["chain"] = "B"
    np.testing.assert_array_equal(neighborhood_mask(residues, 3), [True] * 6 + [False])
    assert index_ranges([0, 1, 2, 7, 9, 10]) == "[[0, 3], [7, 8], [9, 11]]"


def test_saved_binary_columns_and_full_training_scopes_remain_distinct(tmp_path):
    residues = [dict(chain="A", source_resid=i + 20, icode="", resname="ALA") for i in range(3)]
    rows = []
    for scope in ["full", "train"]:
        for i, residue in enumerate(residues):
            rows.append(
                {
                    **residue,
                    "atom_index": i,
                    "scope": scope,
                    "MDCoreAlign_matched_k": int(i < 2),
                    "MDLovoFit_core": int(i != 1 if scope == "full" else i < 2),
                }
            )
    with (tmp_path / "residue_results.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    case = dict(source_case_directory=str(tmp_path), residues=residues)
    assert source_sets(case, "full")["C"].sum() == 1
    assert source_sets(case, "train")["C"].sum() == 2
