"""Tests for hard core extraction post-processing."""

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
    aggregate_weights_by_group,
    format_selected_atoms,
    select_core_by_target_fraction,
    select_core_by_threshold,
    select_core_by_top_k,
    summarize_core_selection,
)


class CoreSelectionTest(unittest.TestCase):
    """Check deterministic hard core extraction behavior."""

    def test_threshold_selection_returns_expected_indices(self) -> None:
        """Threshold selection should keep atoms at or above the cutoff."""

        weights = np.array([0.90, 0.70, 0.70, 0.20], dtype=float)
        result = select_core_by_threshold(weights, threshold=0.70)

        np.testing.assert_array_equal(result.core_indices, np.array([0, 1, 2]))
        np.testing.assert_array_equal(result.non_core_indices, np.array([3]))
        self.assertEqual(result.core_size, 3)
        self.assertAlmostEqual(result.core_fraction, 0.75)

    def test_top_k_uses_deterministic_tie_breaking(self) -> None:
        """Top-k selection should prefer lower atom indices for equal weights."""

        weights = np.array([0.90, 0.80, 0.80, 0.70], dtype=float)
        result_a = select_core_by_top_k(weights, k=2)
        result_b = select_core_by_top_k(weights, k=2)

        np.testing.assert_array_equal(result_a.core_indices, np.array([0, 1]))
        np.testing.assert_array_equal(result_a.core_indices, result_b.core_indices)

    def test_target_fraction_rounds_to_deterministic_core_size(self) -> None:
        """Target-fraction selection should map to a deterministic size."""

        weights = np.array([0.90, 0.80, 0.30, 0.10, 0.05], dtype=float)
        result = select_core_by_target_fraction(weights, fraction=0.50)

        np.testing.assert_array_equal(result.core_indices, np.array([0, 1, 2]))
        self.assertEqual(result.core_size, 3)
        self.assertAlmostEqual(result.core_fraction, 3.0 / 5.0)

    def test_group_aggregation_supports_future_residue_level_use(self) -> None:
        """Grouped aggregation should summarize atoms by a later group label."""

        weights = np.array([1.0, 0.5, 0.2, 0.8], dtype=float)
        group_ids = np.array(["A", "A", "B", "B"], dtype=object)
        aggregated = aggregate_weights_by_group(weights, group_ids, reduction="mean")

        np.testing.assert_array_equal(aggregated.group_ids, np.array(["A", "B"], dtype=object))
        np.testing.assert_allclose(aggregated.group_weights, np.array([0.75, 0.50]))
        np.testing.assert_array_equal(aggregated.group_sizes, np.array([2, 2]))

    def test_summary_utility_reports_selected_atoms(self) -> None:
        """Summary helpers should expose readable selected-atom information."""

        weights = np.array([0.9, 0.8, 0.1], dtype=float)
        result = select_core_by_top_k(weights, k=2)
        labels = ["GLY1-CA", "ALA2-CA", "LYS3-CA"]

        formatted = format_selected_atoms(result.core_indices, labels=labels)
        summary = summarize_core_selection(result, labels=labels, max_items=1)

        self.assertEqual(formatted, ["GLY1-CA", "ALA2-CA"])
        self.assertEqual(summary["core_size"], 2)
        self.assertEqual(summary["core_preview"], ["GLY1-CA"])
        self.assertEqual(summary["non_core_preview"], ["LYS3-CA"])

    def test_invalid_inputs_are_rejected(self) -> None:
        """Selection helpers should reject malformed inputs."""

        with self.assertRaises(ValueError):
            select_core_by_top_k(np.array([0.5, -0.1]), k=1)
        with self.assertRaises(ValueError):
            select_core_by_target_fraction(np.array([0.5, 0.4]), fraction=0.0)
        with self.assertRaises(ValueError):
            aggregate_weights_by_group(np.array([0.5, 0.4]), np.array([1]), reduction="mean")
        with self.assertRaises(ValueError):
            summarize_core_selection(
                select_core_by_threshold(np.array([0.7, 0.2]), threshold=0.5),
                labels=["only-one-label"],
            )


if __name__ == "__main__":
    unittest.main()
