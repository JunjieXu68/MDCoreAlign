"""Scientific invariants for the bounded residue-disagreement analysis."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.core_disagreement_sensitivity_v1 import (
    classify,
    decompose,
    relative_geometry,
    stable_ranks,
    variance_matrix,
)
from scripts.atlas_strict_single_thread_cpu_v1.extract_cores import overlap, top_k


def residues(n=6):
    return [{"chain": "A", "sequence_position": i + 1} for i in range(n)]


def pairs():
    return {
        "left": np.array([0, 0, 1]),
        "right": np.array([4, 5, 5]),
        "variance_A2": np.array([1.0, 4.0, 9.0]),
    }


def test_partition_and_matched_dice_integer_identity():
    a = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    b = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
    labels = classify(a, b)
    assert labels.tolist() == ["MCA_only", "shared", "shared", "MLF_only", "neither", "neither"]
    assert sum(
        np.count_nonzero(labels == g) for g in ["shared", "MCA_only", "MLF_only", "neither"]
    ) == len(a)
    assert overlap(a, b)["dice"] == np.count_nonzero(a & b) / a.sum()
    assert overlap(a, b)["jaccard"] == 2 / 4


def test_exact_ties_use_input_index_not_sort_instability():
    weights = np.array([0.7, 0.9, 0.7, 0.9, 0.4])
    assert stable_ranks(weights).tolist() == [3, 1, 4, 2, 5]
    assert np.array_equal(stable_ranks(weights) <= 3, top_k(weights, 3))


def test_nearly_equal_weights_are_not_exact_ties():
    weights = np.array([0.5, np.nextafter(0.5, 1), 0.2])
    assert stable_ranks(weights).tolist() == [2, 1, 3]


def test_geometry_symmetric_self_and_neighbour_exclusions():
    matrix = variance_matrix(pairs(), residues())
    np.testing.assert_equal(matrix, matrix.T)
    assert np.isnan(np.diag(matrix)).all()
    assert np.isnan(matrix[0, 3]) and matrix[0, 4] == 1
    reference = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    h, n = relative_geometry(matrix, reference)
    assert n[5] == 2 and h[5] == pytest.approx(np.sqrt(6.5))
    assert n[2] == 0 and np.isnan(h[2])


def test_cross_chain_close_indices_remain_valid():
    rs = residues(2)
    rs[1]["chain"] = "B"
    matrix = variance_matrix(
        {"left": np.array([0]), "right": np.array([1]), "variance_A2": np.array([2.0])}, rs
    )
    assert matrix[0, 1] == 2


@pytest.mark.parametrize("left,right", [(0, 0), (0, 3), (4, 0)])
def test_invalid_cached_pair_is_rejected(left, right):
    with pytest.raises(AssertionError):
        variance_matrix(
            {"left": np.array([left]), "right": np.array([right]), "variance_A2": np.array([1.0])},
            residues(),
        )


def test_decomposition_rebuilds_g_squared_with_method_specific_denominator():
    common = np.array([1, 0, 0, 0, 1, 0], dtype=bool)
    ua = np.array([0, 1, 0, 0, 0, 1], dtype=bool)
    ub = np.array([0, 0, 1, 0, 0, 1], dtype=bool)
    a = decompose(pairs(), common, ua)
    b = decompose(pairs(), common, ub)
    assert [r["pair_count"] for r in a] == [1, 1, 1, 3]
    assert [r["variance_sum_A2"] for r in a] == [1, 4, 9, 14]
    assert sum(r["G2_contribution_A2"] for r in a[:3]) == pytest.approx(
        a[-1]["reconstructed_G_A"] ** 2
    )
    assert a[-1]["reconstructed_G_A"] == pytest.approx(np.sqrt(14 / 3))
    assert b[-1]["pair_count"] == 2 and b[-1]["reconstructed_G_A"] == pytest.approx(np.sqrt(5 / 2))


def test_empty_pair_group_is_na_not_zero_variance():
    common = np.array([1, 0, 0, 0, 1, 0], dtype=bool)
    data = decompose(pairs(), common, np.zeros(6, dtype=bool))
    assert data[2]["pair_count"] == 0 and data[2]["mean_variance_A2"] is None


def test_frozen_case_native_rounding_splits_and_frameworks():
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "results/biological_core_validation_v1/analysis_config.json").read_text()
    )
    selected = {c["case_id"]: c for c in config["cases"] if c["case_id"] in ["ADK", "3nci_A"]}
    adk = selected["ADK"]
    rb = selected["3nci_A"]
    assert not set(adk["train_indices"]) & set(adk["test_indices"])
    assert adk["train_indices"] == list(range(2093))
    assert adk["test_indices"] == list(range(2093, 4187))
    assert Path(rb["coords_path"]).resolve() != Path(rb["test_coords_path"]).resolve()
    mask = np.load(
        root
        / "results/biological_core_validation_v1/cases/3nci_A/analysis/mdlovofit_full/membership.npz"
    )["core"]
    assert int(mask.sum()) == 541
    assert int(np.floor(0.6 * len(rb["residues"]) + 0.5)) == 542
    a = np.asarray(rb["regions"]["framework_A"])
    b = np.asarray(rb["regions"]["framework_B"])
    assert [int(x.sum()) for x in [a & ~b, b & ~a, a & b, ~(a | b)]] == [202, 289, 29, 383]
