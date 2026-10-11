"""Regression tests for the U0-U6 GPU matched-validation support code."""

from __future__ import annotations

import numpy as np

from scripts.gpu_matched_validation_v1.common import (
    core_mask,
    jaccard,
    timing_summary,
)
from scripts.gpu_matched_validation_v1.run_u3_scaling import _trajectory


def test_core_mask_is_deterministic_at_fixed_fraction() -> None:
    weights = np.asarray([0.9, 0.9, 0.4, 0.2, 0.1])
    selected = core_mask(weights, fraction=0.4)

    np.testing.assert_array_equal(
        selected,
        np.asarray([True, True, False, False, False]),
    )
    assert jaccard(selected, selected) == 1.0


def test_timing_summary_reports_required_repeat_statistics() -> None:
    summary = timing_summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["median_seconds"] == 3.0
    assert summary["q1_seconds"] == 2.0
    assert summary["q3_seconds"] == 4.0
    assert summary["iqr_seconds"] == 2.0
    assert summary["min_seconds"] == 1.0
    assert summary["max_seconds"] == 5.0


def test_scaling_trajectory_is_finite_and_has_requested_shape() -> None:
    coords = _trajectory(24, 30)

    assert coords.shape == (24, 30, 3)
    assert np.isfinite(coords).all()
    assert np.max(np.std(coords, axis=0)) > 0.0
