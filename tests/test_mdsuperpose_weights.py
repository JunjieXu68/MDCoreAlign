"""Unit tests for robust distance-to-weight rules."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    apply_sequence_continuity_regularization,
    gaussian_weights,
    huber_weights,
    normalize_weights,
    rescale_weights,
    tukey_weights,
    update_irls_weights,
)


class WeightRuleTest(unittest.TestCase):
    """Validate basic behavior of the robust weighting rules."""

    def test_gaussian_weights_are_monotone_and_unit_at_zero(self) -> None:
        """Gaussian weights should decrease smoothly from one."""

        distances = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
        weights = gaussian_weights(distances, sigma=1.0)

        self.assertAlmostEqual(weights[0], 1.0, places=12)
        self.assertTrue(np.all(np.diff(weights) <= 0.0))
        self.assertGreater(weights[-1], 0.0)

    def test_huber_weights_have_expected_piecewise_behavior(self) -> None:
        """Huber weights should stay flat then decay as 1 / d."""

        distances = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
        weights = huber_weights(distances, delta=1.0)

        expected = np.array([1.0, 1.0, 1.0, 0.5, 0.25], dtype=float)
        np.testing.assert_allclose(weights, expected, atol=1e-12, rtol=1e-12)

    def test_tukey_weights_are_bounded_and_zero_after_cutoff(self) -> None:
        """Tukey weights should drop to zero beyond the cutoff."""

        distances = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
        weights = tukey_weights(distances, cutoff=1.5)

        self.assertAlmostEqual(weights[0], 1.0, places=12)
        self.assertTrue(np.all(np.diff(weights[:4]) <= 0.0))
        self.assertEqual(weights[-1], 0.0)
        self.assertTrue(np.all((weights >= 0.0) & (weights <= 1.0)))

    def test_dispatcher_matches_direct_rule_calls(self) -> None:
        """The generic dispatcher should reproduce the direct rule helpers."""

        distances = np.array([0.0, 0.2, 0.7, 1.4], dtype=float)

        np.testing.assert_allclose(
            update_irls_weights(distances, rule="gaussian", sigma=0.8),
            gaussian_weights(distances, sigma=0.8),
        )
        np.testing.assert_allclose(
            update_irls_weights(distances, rule="huber", delta=0.8),
            huber_weights(distances, delta=0.8),
        )
        np.testing.assert_allclose(
            update_irls_weights(distances, rule="tukey", cutoff=1.2),
            tukey_weights(distances, cutoff=1.2),
        )

    def test_normalize_and_rescale_helpers_work(self) -> None:
        """Weight post-processing helpers should preserve valid ranges."""

        weights = np.array([2.0, 4.0, 8.0], dtype=float)

        np.testing.assert_allclose(normalize_weights(weights, mode="sum").sum(), 1.0, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(float(np.max(normalize_weights(weights, mode="max"))), 1.0, places=12)

        rescaled = rescale_weights(weights, min_value=0.2, max_value=0.8)
        self.assertAlmostEqual(float(np.min(rescaled)), 0.2, places=12)
        self.assertAlmostEqual(float(np.max(rescaled)), 0.8, places=12)

    def test_sequence_continuity_regularization_reduces_fragmentation(self) -> None:
        """Sequence smoothing should reduce alternating weight variation."""

        weights = np.array([1.0, 0.1, 1.0, 0.1, 1.0, 0.1], dtype=float)
        smoothed = apply_sequence_continuity_regularization(weights, strength=0.6, passes=2)

        original_total_variation = float(np.sum(np.abs(np.diff(weights))))
        smoothed_total_variation = float(np.sum(np.abs(np.diff(smoothed))))
        self.assertLess(smoothed_total_variation, original_total_variation)
        self.assertTrue(np.all(smoothed >= 0.0))

    def test_sequence_continuity_regularization_zero_strength_is_identity(self) -> None:
        """Zero sequence-regularization strength should leave weights unchanged."""

        weights = np.array([0.2, 0.8, 0.3, 0.7], dtype=float)
        smoothed = apply_sequence_continuity_regularization(weights, strength=0.0, passes=3)
        np.testing.assert_allclose(smoothed, weights)

    def test_invalid_inputs_raise(self) -> None:
        """Negative distances and invalid parameters should be rejected."""

        distances = np.array([0.0, 1.0, 2.0], dtype=float)

        with self.assertRaises(ValueError):
            gaussian_weights(np.array([0.0, -1.0, 2.0], dtype=float), sigma=1.0)

        with self.assertRaises(ValueError):
            gaussian_weights(distances, sigma=0.0)

        with self.assertRaises(ValueError):
            huber_weights(distances, delta=0.0)

        with self.assertRaises(ValueError):
            tukey_weights(distances, cutoff=0.0)

        with self.assertRaises(ValueError):
            update_irls_weights(distances, rule="unknown")

        with self.assertRaises(ValueError):
            apply_sequence_continuity_regularization(distances, strength=-0.1)

        with self.assertRaises(ValueError):
            apply_sequence_continuity_regularization(distances, strength=0.5, passes=-1)


if __name__ == "__main__":
    unittest.main()
