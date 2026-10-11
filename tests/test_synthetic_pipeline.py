"""Basic end-to-end test for the synthetic stable-core pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdstablecore.pipeline import run_stable_core_pipeline  # noqa: E402
from mdstablecore.trajectory_io import make_trajectory_data  # noqa: E402
from mdstablecore.types import PipelineConfig  # noqa: E402
from scripts.demo_synthetic import build_synthetic_trajectory  # noqa: E402


class SyntheticPipelineTest(unittest.TestCase):
    """End-to-end checks for the MVP pipeline."""

    def test_pipeline_recovers_most_of_synthetic_core(self) -> None:
        """The MVP should recover the majority of synthetic core residues."""

        coords, atom_to_residue, expected_core = build_synthetic_trajectory(n_frames=120)
        data = make_trajectory_data(coords, atom_to_residue)
        result = run_stable_core_pipeline(data, config=PipelineConfig(bootstrap_rounds=8))

        overlap = np.logical_and(result.core_mask, expected_core).sum()
        expected_size = int(expected_core.sum())

        self.assertGreaterEqual(overlap, int(0.9 * expected_size))
        self.assertEqual(result.confidence.shape[0], expected_core.shape[0])


if __name__ == "__main__":
    unittest.main()
