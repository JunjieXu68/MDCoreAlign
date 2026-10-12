"""Output-contract and parity tests for the shared all-frame CPU engine."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdcorealign.cli import run  # noqa: E402
from mdsuperpose.all_frame_engine import (  # noqa: E402
    CPUAlignmentWorkspace,
    fit_all_frames_fixed_weights,
)
from mdsuperpose.fast_kabsch import weighted_kabsch_batch  # noqa: E402
from mdsuperpose.window_refine import exact_all_frame_fit  # noqa: E402


def _coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    reference = rng.normal(size=(27, 3))
    coords = reference[None, :, :] + rng.normal(scale=0.12, size=(13, 27, 3))
    weights = rng.uniform(0.1, 1.0, size=27)
    return coords, reference, weights


def test_historical_wrapper_matches_framewise_compatibility_results() -> None:
    coords, reference, weights = _coords()
    result = exact_all_frame_fit(coords, reference, weights, chunk_size=4)
    legacy = weighted_kabsch_batch(coords, reference, weights)

    np.testing.assert_allclose(result.rotations, legacy.rotations, atol=1e-12)
    np.testing.assert_allclose(result.translations, legacy.translations, atol=1e-12)
    np.testing.assert_allclose(result.rmsd, legacy.rmsd, atol=1e-12)
    np.testing.assert_allclose(result.weighted_rmsd, legacy.weighted_rmsd, atol=1e-12)
    assert result.robust_rmsf is not None
    assert result.ordinary_rmsf is not None
    assert result.mean_residual is not None
    assert result.aligned_coords is None


def test_transform_only_contract_does_not_retain_coordinate_or_metric_arrays() -> None:
    coords, reference, weights = _coords()
    result = fit_all_frames_fixed_weights(
        coords,
        reference,
        weights,
        chunk_size=5,
        return_aligned=False,
        compute_rmsd=False,
        compute_weighted_rmsd=False,
        compute_mean_residual=False,
        compute_robust_rmsf=False,
        compute_ordinary_rmsf=False,
    )

    assert result.rotations.shape == (coords.shape[0], 3, 3)
    assert result.translations.shape == (coords.shape[0], 3)
    assert result.aligned_coords is None
    assert result.rmsd is None
    assert result.weighted_rmsd is None
    assert result.robust_rmsf is None
    assert result.ordinary_rmsf is None
    assert result.mean_residual is None


def test_ordinary_rmsf_is_independently_controllable_without_changing_robust_outputs() -> None:
    coords, reference, weights = _coords()
    full = exact_all_frame_fit(coords, reference, weights, chunk_size=6)
    without_ordinary = exact_all_frame_fit(
        coords,
        reference,
        weights,
        chunk_size=6,
        compute_ordinary_rmsf=False,
    )

    assert without_ordinary.ordinary_rmsf is None
    np.testing.assert_allclose(without_ordinary.rotations, full.rotations, atol=1e-12)
    np.testing.assert_allclose(without_ordinary.translations, full.translations, atol=1e-12)
    np.testing.assert_allclose(without_ordinary.rmsd, full.rmsd, atol=1e-12)
    np.testing.assert_allclose(without_ordinary.robust_rmsf, full.robust_rmsf, atol=1e-12)


def test_workspace_reuse_preserves_results_and_rejects_shape_mismatch() -> None:
    coords, reference, weights = _coords()
    workspace = CPUAlignmentWorkspace(coords.shape[1])
    first = fit_all_frames_fixed_weights(coords, reference, weights, workspace=workspace)
    second = fit_all_frames_fixed_weights(coords, reference, weights, workspace=workspace)
    np.testing.assert_array_equal(first.rotations, second.rotations)
    np.testing.assert_array_equal(first.ordinary_rmsf, second.ordinary_rmsf)

    bad = CPUAlignmentWorkspace(coords.shape[1] + 1)
    try:
        fit_all_frames_fixed_weights(coords, reference, weights, workspace=bad)
    except ValueError as error:
        assert "workspace" in str(error)
    else:
        raise AssertionError("incompatible workspace was accepted")


def test_cli_marks_ordinary_rmsf_unavailable_without_zero_filling(tmp_path) -> None:
    output = tmp_path / "compact"
    toy = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    status = run(
        [
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
            "--no-ordinary-rmsf",
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0
    with (output / "residue_profile.tsv").open(newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    diagnostics = json.loads(
        (output / "diagnostics" / "diagnostics.json").read_text()
    )
    assert "ordinary_RMSF" not in fields
    assert diagnostics["ordinary_rmsf_computed"] is False
