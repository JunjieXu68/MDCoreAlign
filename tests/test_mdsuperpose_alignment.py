"""Unit tests for the baseline rigid alignment solver."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import solve_rigid_alignment


class BaselineRigidAlignmentTest(unittest.TestCase):
    """Validate the clean baseline Kabsch implementation."""

    def test_identity_pair_has_zero_rmsd(self) -> None:
        """Identical coordinate sets should align with near-zero error."""

        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        result = solve_rigid_alignment(coords, coords)

        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.translation, np.zeros(3), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.aligned_coords, coords, atol=1e-7, rtol=1e-7)
        self.assertLess(result.rmsd, 1e-10)

    def test_known_rotation_and_translation_are_recovered(self) -> None:
        """A known rigid transform should be recovered from synthetic data."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.1],
                [0.0, 0.4, 1.1],
                [1.2, 0.3, 0.5],
            ],
            dtype=float,
        )
        theta = 0.6
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([2.0, -1.0, 0.7], dtype=float)
        y = x @ rotation.T + translation

        result = solve_rigid_alignment(x, y)

        np.testing.assert_allclose(result.rotation, rotation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.translation, translation, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(result.aligned_coords, y, atol=1e-7, rtol=1e-7)
        self.assertLess(result.rmsd, 1e-10)

    def test_invalid_shapes_raise(self) -> None:
        """Shape mismatches should fail fast with a clear error."""

        x = np.zeros((4, 3), dtype=float)
        y = np.zeros((5, 3), dtype=float)

        with self.assertRaises(ValueError):
            solve_rigid_alignment(x, y)

        with self.assertRaises(ValueError):
            solve_rigid_alignment(np.zeros((4, 2), dtype=float), np.zeros((4, 2), dtype=float))

        with self.assertRaises(ValueError):
            solve_rigid_alignment(np.zeros((2, 3), dtype=float), np.zeros((2, 3), dtype=float))

    def test_non_finite_inputs_raise(self) -> None:
        """NaN or Inf entries should be rejected."""

        x = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        y = np.zeros((3, 3), dtype=float)

        with self.assertRaises(ValueError):
            solve_rigid_alignment(x, y)


if __name__ == "__main__":
    unittest.main()
