"""Unit tests for the weighted rigid alignment solver."""

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

from mdsuperpose import solve_rigid_alignment, solve_weighted_rigid_alignment


class WeightedRigidAlignmentTest(unittest.TestCase):
    """Validate the weighted Kabsch implementation."""

    def test_known_transform_is_recovered_with_non_normalized_weights(self) -> None:
        """The weighted solver should recover a clean rigid transform."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.2, 0.3, 0.7],
            ],
            dtype=float,
        )
        theta = 0.4
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([-1.5, 0.8, 2.2], dtype=float)
        y = x @ rotation.T + translation
        weights = np.array([3.0, 1.0, 7.5, 2.0, 0.5], dtype=float)

        result = solve_weighted_rigid_alignment(x, y, weights)

        np.testing.assert_allclose(result.rotation, rotation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.translation, translation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.aligned_coords, y, atol=1e-7, rtol=1e-7)
        self.assertLess(result.rmsd, 1e-10)
        self.assertLess(result.weighted_rmsd, 1e-10)

    def test_equal_weights_match_unweighted_solver(self) -> None:
        """Uniform weights should reproduce the unweighted solution."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.3, 0.8, 0.2],
                [0.1, 0.4, 1.1],
                [1.4, 0.3, 0.5],
                [0.7, 1.1, 0.9],
            ],
            dtype=float,
        )
        theta = -0.35
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.4, -0.2, 1.3], dtype=float)
        y = x @ rotation.T + translation
        weights = np.full(x.shape[0], 5.0, dtype=float)

        unweighted = solve_rigid_alignment(x, y)
        weighted = solve_weighted_rigid_alignment(x, y, weights)

        np.testing.assert_allclose(weighted.rotation, unweighted.rotation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(weighted.translation, unweighted.translation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(weighted.aligned_coords, unweighted.aligned_coords, atol=1e-7, rtol=1e-7)
        self.assertAlmostEqual(weighted.rmsd, unweighted.rmsd, places=12)
        self.assertAlmostEqual(weighted.weighted_rmsd, unweighted.weighted_rmsd, places=12)

    def test_unnormalized_weights_do_not_change_the_solution(self) -> None:
        """Scaling all weights by a constant should not change the transform."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [0.1, 0.3, 0.9],
            ],
            dtype=float,
        )
        theta = 0.2
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.5, 1.2, -0.7], dtype=float)
        y = x @ rotation.T + translation

        result_a = solve_weighted_rigid_alignment(x, y, np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
        result_b = solve_weighted_rigid_alignment(x, y, np.array([10.0, 20.0, 30.0, 40.0], dtype=float))

        np.testing.assert_allclose(result_a.rotation, result_b.rotation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result_a.translation, result_b.translation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result_a.aligned_coords, result_b.aligned_coords, atol=1e-7, rtol=1e-7)
        self.assertAlmostEqual(result_a.weighted_rmsd, result_b.weighted_rmsd, places=12)

    def test_invalid_weights_raise(self) -> None:
        """Negative, non-finite, or all-zero weights should be rejected."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        y = x.copy()

        with self.assertRaises(ValueError):
            solve_weighted_rigid_alignment(x, y, np.array([0.0, 0.0, 0.0], dtype=float))

        with self.assertRaises(ValueError):
            solve_weighted_rigid_alignment(x, y, np.array([1.0, -1.0, 2.0], dtype=float))

        with self.assertRaises(ValueError):
            solve_weighted_rigid_alignment(x, y, np.array([1.0, np.inf, 2.0], dtype=float))


if __name__ == "__main__":
    unittest.main()
