"""Smoke tests for the minimal scaffold package."""

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
    Trajectory,
    apply_transform,
    identity_transform,
    mean_rmsd,
    top_fraction_core_mask,
    uniform_weights,
)


class MDSuperposeSmokeTest(unittest.TestCase):
    """Check that the scaffold package can be imported and exercised."""

    def test_identity_scaffold_run(self) -> None:
        """The minimal scaffold should support one tiny end-to-end call path."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]],
            ],
            dtype=float,
        )
        trajectory = Trajectory(coords=coords)
        transform = identity_transform()
        transformed = apply_transform(trajectory.coords, transform)
        weights = uniform_weights(trajectory.n_atoms)
        core_mask = top_fraction_core_mask(weights, fraction=1.0 / 3.0)

        self.assertEqual(trajectory.n_frames, 2)
        self.assertEqual(trajectory.n_atoms, 3)
        self.assertEqual(int(core_mask.sum()), 1)
        np.testing.assert_allclose(transformed, trajectory.coords)
        self.assertGreaterEqual(mean_rmsd(trajectory.coords, trajectory.coords[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
