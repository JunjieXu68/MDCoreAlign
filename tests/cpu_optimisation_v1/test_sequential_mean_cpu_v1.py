"""Parity tests for the mean-only chronological exact estimator."""

from __future__ import annotations

import os
import sys
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose.sequential_weights import (  # noqa: E402
    SequentialMeanWeightEstimationResult,
    SequentialWeightEstimationResult,
    estimate_representative_weights_sequential_lean,
    estimate_representative_weights_sequential_mean,
)


def _input() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    reference = rng.normal(size=(48, 3))
    coords = reference[None, :, :] + rng.normal(scale=0.12, size=(24, 48, 3))
    coords[:, 35:, 1] += np.linspace(-0.8, 0.8, coords.shape[0])[:, None]
    return coords, reference


def _kwargs() -> dict[str, object]:
    return {
        "rule": "huber",
        "delta": 0.4,
        "max_iterations": 12,
        "tolerance": 1e-7,
        "objective_tolerance": 1e-6,
        "objective_patience": 2,
        "sequence_continuity_strength": 0.05,
        "record_history": False,
    }


def test_mean_only_preserves_chronological_exact_weights_and_metadata() -> None:
    coords, reference = _input()
    detailed = estimate_representative_weights_sequential_lean(
        coords,
        reference,
        **_kwargs(),
    )
    mean_only = estimate_representative_weights_sequential_mean(
        coords,
        reference,
        **_kwargs(),
    )
    assert isinstance(detailed, SequentialWeightEstimationResult)
    assert isinstance(mean_only, SequentialMeanWeightEstimationResult)

    np.testing.assert_allclose(mean_only.mean_weights, detailed.mean_weights, atol=1e-14)
    np.testing.assert_allclose(mean_only.mean_residual, detailed.mean_residual, atol=1e-14)
    assert mean_only.average_iterations_per_frame == detailed.average_iterations_per_frame
    assert mean_only.converged_fraction == detailed.converged_fraction
    assert mean_only.max_iterations_reached_count == int(np.sum(~detailed.converged))
    assert mean_only.call_counts["retained_per_representative_detail_arrays"] == 0


def test_mean_only_has_lower_python_traced_peak_memory() -> None:
    coords, reference = _input()

    tracemalloc.start()
    estimate_representative_weights_sequential_lean(coords, reference, **_kwargs())
    _, detailed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    estimate_representative_weights_sequential_mean(coords, reference, **_kwargs())
    _, mean_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert mean_peak < detailed_peak
