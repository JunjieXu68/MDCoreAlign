"""Focused regression tests for the CPU fast-path study."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdcorealign.cli import run  # noqa: E402
from mdsuperpose.parallel_weights import (  # noqa: E402
    estimate_batch_then_sequential,
    estimate_blockwise_warm_start,
    estimate_independent_batch,
    estimate_risk_triggered_refinement,
    estimate_shared_weight,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(412)
    reference = rng.normal(size=(24, 3))
    coords = reference[None, :, :] + rng.normal(scale=0.08, size=(18, 24, 3))
    coords[:, 17:] += np.linspace(0.0, 1.4, coords.shape[0])[:, None, None]
    chains = np.asarray(["A"] * 12 + ["B"] * 12)
    return coords, reference, chains


def test_internal_parallel_candidates_return_finite_deterministic_weights() -> None:
    coords, reference, chains = _trajectory()
    candidates = (
        estimate_independent_batch(coords, reference, chain_ids=chains),
        estimate_shared_weight(coords, reference, chain_ids=chains),
        estimate_batch_then_sequential(
            coords, reference, chain_ids=chains, correction_iterations=1
        ),
        estimate_blockwise_warm_start(
            coords, reference, chain_ids=chains, block_size=8
        ),
        estimate_risk_triggered_refinement(
            coords,
            reference,
            chain_ids=chains,
            risk_quantile=0.8,
            interval_padding=1,
            correction_iterations=1,
        ),
    )
    for result in candidates:
        assert result.mean_weights.shape == (coords.shape[1],)
        assert np.isfinite(result.mean_weights).all()
        assert np.all((0.0 <= result.mean_weights) & (result.mean_weights <= 1.0))
        assert result.total_runtime_seconds >= 0.0

    repeated = estimate_shared_weight(coords, reference, chain_ids=chains)
    np.testing.assert_array_equal(candidates[1].mean_weights, repeated.mean_weights)


def test_risk_triggered_refinement_reports_actual_bounded_fraction() -> None:
    coords, reference, chains = _trajectory()
    result = estimate_risk_triggered_refinement(
        coords,
        reference,
        chain_ids=chains,
        risk_quantile=0.9,
        interval_padding=0,
        correction_iterations=2,
    )
    assert 0.0 < result.refinement_fraction < 1.0
    assert result.metadata["flagged_representatives"] > 0
    assert result.timing["chronological_refinement"] >= 0.0


def test_cli_default_omits_ordinary_rmsf_and_opt_in_restores_it(tmp_path) -> None:
    toy = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    default_output = tmp_path / "default"
    common = [
        "align",
        "--input",
        str(toy),
        "--input-format",
        "ca-pdb",
        "--mode",
        "throughput",
        "--backend",
        "cpu",
        "--fast-representative-count",
        "5",
        "--fast-update-rounds",
        "2",
    ]
    assert run([*common, "--output-dir", str(default_output)]) == 0
    with (default_output / "residue_profile.tsv").open(newline="") as handle:
        default_fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    default_diagnostics = json.loads(
        (default_output / "diagnostics" / "diagnostics.json").read_text()
    )
    assert "ordinary_RMSF" not in default_fields
    assert default_diagnostics["ordinary_rmsf_computed"] is False

    opt_in_output = tmp_path / "opt_in"
    assert (
        run(
            [
                *common,
                "--ordinary-rmsf",
                "--output-dir",
                str(opt_in_output),
            ]
        )
        == 0
    )
    with (opt_in_output / "residue_profile.tsv").open(newline="") as handle:
        opt_in_fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert "ordinary_RMSF" in opt_in_fields
