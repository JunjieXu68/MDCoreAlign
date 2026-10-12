"""Checks for batched alignment utilities."""

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

from mdstablecore.alignment import align_frame, align_trajectory  # noqa: E402


class AlignmentConsistencyTest(unittest.TestCase):
    """Ensure batched and per-frame alignment agree."""

    def test_align_trajectory_matches_align_frame(self) -> None:
        """The vectorized trajectory aligner should match frame-wise alignment."""

        rng = np.random.default_rng(7)
        coords = rng.normal(size=(8, 12, 3))
        reference = coords[0].copy()
        atom_mask = np.array([True, False, True, True, False, True, True, False, True, True, True, False])

        expected = np.stack([align_frame(frame, reference, atom_mask=atom_mask) for frame in coords], axis=0)
        observed = align_trajectory(coords, reference, atom_mask=atom_mask)

        np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=1e-7)

    def test_rigid_transform_is_recovered(self) -> None:
        """A pure rigid-body transform should align back to the reference."""

        rng = np.random.default_rng(11)
        reference = rng.normal(size=(16, 3))
        theta = 0.7
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([3.0, -2.0, 1.5])
        frame = reference @ rotation.T + translation

        aligned = align_frame(frame, reference)

        np.testing.assert_allclose(aligned, reference, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
