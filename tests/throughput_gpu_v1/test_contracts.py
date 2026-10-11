"""Tests for the explicit practical execution contracts."""

from __future__ import annotations

import numpy as np

from mdcorealign.contracts import (
    run_fixed_cost_sampled_audit,
    solve_contract,
)


def _trajectory(n_frames: int = 80, n_residues: int = 36) -> np.ndarray:
    rng = np.random.default_rng(708)
    reference = np.cumsum(rng.normal(size=(n_residues, 3)), axis=0)
    coords = np.repeat(reference[None, :, :], n_frames, axis=0)
    coords += rng.normal(scale=0.01, size=coords.shape)
    coords[:, 14:21] += rng.normal(
        scale=0.14,
        size=(n_frames, 7, 3),
    )
    return coords


def test_throughput_contract_never_runs_exact_fallback() -> None:
    result = solve_contract(
        _trajectory(),
        contract="throughput",
        representative_count=32,
        update_rounds=2,
        chunk_size=80,
    )

    assert result.contract == "throughput"
    assert result.backend_used == "cpu"
    assert result.metadata["automatic_exact_fallback"] is False
    assert result.alignment is not None
    assert result.alignment.rotations.shape == (80, 3, 3)


def test_balanced_lite_audit_has_fixed_frame_budget() -> None:
    short = _trajectory(100)
    long = np.tile(short, (10, 1, 1))
    short_fast = solve_contract(
        short,
        contract="throughput",
        representative_count=32,
        update_rounds=2,
    )
    long_fast = solve_contract(
        long,
        contract="throughput",
        representative_count=32,
        update_rounds=2,
    )

    short_audit = run_fixed_cost_sampled_audit(
        short,
        short_fast.alignment.frozen_weights,
        subset_count=16,
        heldout_count=24,
    )
    long_audit = run_fixed_cost_sampled_audit(
        long,
        long_fast.alignment.frozen_weights,
        subset_count=16,
        heldout_count=24,
    )

    assert short_audit.subset_a_indices.size <= 16
    assert long_audit.subset_a_indices.size <= 16
    assert short_audit.heldout_indices.size <= 24
    assert long_audit.heldout_indices.size <= 24
    assert short_audit.metadata["audit_cost_scales_with_all_frames"] is False


def test_balanced_lite_performs_at_most_one_refinement_and_no_exact() -> None:
    result = solve_contract(
        _trajectory(),
        contract="balanced-lite",
        representative_count=24,
        update_rounds=2,
        audit_subset_count=12,
        audit_heldout_count=16,
        refinement_representative_count=40,
    )

    assert result.metadata["maximum_refinement_stages"] == 1
    assert result.metadata["automatic_exact_fallback"] is False
    assert result.timing["full_exact_fallback"] == 0.0
    assert result.sampled_audit is not None


def test_local_deformation_contract_does_not_claim_rmsd_or_domains() -> None:
    result = solve_contract(
        _trajectory(),
        contract="local-deformation",
        evaluate_blocks=False,
    )

    assert result.alignment is None
    assert result.local_deformation is not None
    assert result.metadata["alignment_free"] is True
    assert result.metadata["hard_motion_domains_mature"] is False


def test_exact_auto_contract_remains_on_cpu() -> None:
    result = solve_contract(
        _trajectory(24),
        contract="exact",
        backend="auto",
        representative_count=12,
        chunk_size=24,
    )

    assert result.backend_used == "cpu"
    assert result.alignment.metadata["weight_estimator"] == "sequential_lean"
