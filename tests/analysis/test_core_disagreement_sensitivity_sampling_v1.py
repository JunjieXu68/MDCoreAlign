"""Targeted tests for the analysis adapter, not production-algorithm changes."""

import json

import numpy as np
import pytest

from mdcorealign.benchmark_minimal import solve_minimal_cpu
from mdsuperpose import window_refine
from scripts.analysis.core_disagreement_sensitivity_sampling_v1 import (
    SEEDS,
    estimate_with_indices,
    prepare_selection_cache,
    stable_ranks,
    stratified_sample,
)


def test_stratified_draws_keep_budget_endpoints_order_and_seed_identity():
    draws = [stratified_sample(2093, 256, seed) for seed in SEEDS]
    for seed, indices in zip(SEEDS, draws, strict=True):
        assert len(indices) == 256
        assert indices[0] == 0 and indices[-1] == 2092
        assert np.all(np.diff(indices) > 0)
        np.testing.assert_array_equal(indices, stratified_sample(2093, 256, seed))
    assert len({tuple(indices) for indices in draws}) == 5
    np.testing.assert_array_equal(stratified_sample(7, 7, SEEDS[0]), np.arange(7))
    with pytest.raises(ValueError):
        stratified_sample(10, 11, SEEDS[0])


def test_adapter_replay_matches_real_throughput_and_trace_is_observational():
    rng = np.random.default_rng(517)
    reference = rng.normal(size=(12, 3)) * 3
    coords = reference[None] + rng.normal(size=(41, 12, 3)) * np.linspace(.1, 1.5, 12)[None, :, None]
    chains = np.asarray(["A"] * 6 + ["B"] * 6)
    old = coords.copy()
    indices = window_refine.temporally_stratified_indices(len(coords), 19)
    production = solve_minimal_cpu(coords, mode="throughput", representative_count=19, chain_ids=chains)
    original_function = window_refine.apply_sequence_continuity_regularization_batch
    captured, trace, rounds = estimate_with_indices(coords, coords[0], indices, chains, diagnostics=True)
    plain, _, _ = estimate_with_indices(coords, coords[0], indices, chains)
    np.testing.assert_array_equal(captured, production.frozen_weights)
    np.testing.assert_array_equal(captured, plain)
    np.testing.assert_array_equal(coords, old)
    assert window_refine.apply_sequence_continuity_regularization_batch is original_function
    assert [r["round"] for r in rounds] == [1, 2, 3, 4]
    for index, record in enumerate(rounds):
        np.testing.assert_array_equal(record["weights_before_update"], trace["weight_history"][index])
        np.testing.assert_array_equal(record["normalized_weights_after_update"], trace["weight_history"][index + 1])
        assert record["reference_distances_A"].shape == (19, 12)
    # The recorded final-round distances were generated with the preceding
    # weights, not a newly fitted residual under the final frozen weights.
    assert np.any(rounds[-1]["weights_before_update"] != captured)


def test_adapter_keeps_reference_separate_from_representatives_and_restores_on_error(monkeypatch):
    rng = np.random.default_rng(721)
    coords = rng.normal(size=(12, 7, 3))
    chains = np.repeat("A", 7)
    indices = np.asarray([2, 4, 7, 10])
    expected, *_ = window_refine.bounded_global_weight_refinement(
        coords[indices], coords[0], np.ones(7), update_rounds=4, sequence_chain_ids=chains,
    )
    actual, _, _ = estimate_with_indices(coords, coords[0], indices, chains, diagnostics=True)
    np.testing.assert_array_equal(actual, expected)

    def failure(*args, **kwargs):
        raise RuntimeError("simulated continuity failure")

    monkeypatch.setattr(window_refine, "apply_sequence_continuity_regularization_batch", failure)
    with pytest.raises(RuntimeError, match="simulated continuity failure"):
        estimate_with_indices(coords, coords[0], indices, chains, diagnostics=True)
    assert window_refine.apply_sequence_continuity_regularization_batch is failure


def test_stable_ranking_resolves_equal_weights_by_original_input_index():
    np.testing.assert_array_equal(stable_ranks(np.asarray([.3, .8, .8, .2, .3])), [3, 1, 2, 5, 4])


def test_selection_cache_uses_existing_pdb_precision_and_rejects_corruption(tmp_path):
    from scripts.analysis.biological_core_validation_v1 import write_ca

    residues = [{"resname": "ALA", "chain": "A", "source_resid": i + 1} for i in range(5)]
    values = np.random.default_rng(88).normal(size=(4, 5, 3))
    source = tmp_path / "common_input.pdb"
    reference = tmp_path / "reference.pdb"
    write_ca(values, source, residues)
    write_ca(values[:1], reference, residues)
    (tmp_path / "case_summary.json").write_text(json.dumps({"n_frames": 4}))
    case = {
        "case_id": "TEST", "source_case_directory": str(tmp_path),
        "selection_input_pdb": str(source), "selection_coords_cache": str(tmp_path / "selection_coords.npy"),
        "reference_source": str(reference), "residues": residues,
    }
    first, _, _ = prepare_selection_cache(case, {"run_id": "TEST"})
    second, _, _ = prepare_selection_cache(case, {"run_id": "TEST"})
    np.testing.assert_array_equal(first, second)
    # No further centring or higher-precision raw-coordinate substitution.
    assert np.max(np.abs(first - values)) <= .0005
    assert not np.array_equal(first, values)
    np.save(case["selection_coords_cache"], np.zeros_like(first))
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_selection_cache(case, {"run_id": "TEST"})
