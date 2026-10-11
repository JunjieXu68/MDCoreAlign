"""End-to-end contract tests for the frozen v3 internal candidates."""

from __future__ import annotations

import numpy as np

from mdsuperpose.chronology_adaptive import AdaptiveRiskRule
from mdsuperpose.chronology_study import compare_results, run_candidate, run_exact


def trajectory() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260821)
    reference = rng.normal(size=(24, 3))
    frames = []
    for index in range(18):
        frame = reference + rng.normal(scale=0.02, size=reference.shape)
        frame[18:] += np.array([0.08 * index, 0.0, 0.0])
        frames.append(frame)
    return np.asarray(frames), np.asarray(["A"] * 12 + ["B"] * 12)


def permissive_rule() -> AdaptiveRiskRule:
    return AdaptiveRiskRule(
        c1_core_jaccard_min=0.0,
        c1_weight_spearman_min=-1.0,
        c1_cutoff_margin_min=-1.0,
        c1_residual_tail_max=float("inf"),
        c2_core_jaccard_min=0.0,
        c2_weight_spearman_min=-1.0,
        c2_cutoff_margin_min=-1.0,
        c2_residual_tail_max=float("inf"),
    )


def test_all_frozen_candidates_share_compact_output_contract() -> None:
    coords, chains = trajectory()
    exact = run_exact(coords, chains, chunk_size=7)
    assert exact.stage_timing.chronological_correction_seconds == 0.0
    for method in (
        "throughput_compact_f64",
        "D_block_64_v3",
        "C_correction_2_v3",
        "adaptive_C1_C2_D64_v3",
    ):
        result = run_candidate(
            coords,
            chains,
            method,
            adaptive_rule=permissive_rule() if method.startswith("adaptive") else None,
            chunk_size=7,
        )
        assert result.weights.shape == (coords.shape[1],)
        assert result.rotations.shape == (coords.shape[0], 3, 3)
        assert result.translations.shape == (coords.shape[0], 3)
        assert result.rmsd.shape == (coords.shape[0],)
        assert result.robust_rmsf.shape == (coords.shape[1],)
        assert np.isfinite(result.weights).all()
        assert np.isfinite(result.rotations).all()
        assert np.isfinite(result.rmsd).all()
        result.stage_timing.validate()
        metrics = compare_results(result, exact, coords)
        assert metrics["non_finite_transform"] is False


def test_adaptive_route_is_recorded_without_public_mode_change() -> None:
    coords, chains = trajectory()
    result = run_candidate(
        coords,
        chains,
        "adaptive_C1_C2_D64_v3",
        adaptive_rule=permissive_rule(),
    )
    assert result.metadata["accepted_level"] == "C1"
