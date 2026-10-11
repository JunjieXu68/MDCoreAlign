"""Tests for the soft-weighted superposition prototype."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdstablecore.quality_metrics import jaccard_similarity  # noqa: E402
from mdstablecore.soft_superposition import (  # noqa: E402
    soft_superpose_pair,
    soft_superpose_trajectory,
)
from scripts.demo_synthetic import build_synthetic_trajectory  # noqa: E402


class SoftSuperpositionTest(unittest.TestCase):
    """Checks for the stage-1 soft superposition prototype."""

    def test_weighted_pair_alignment_recovers_rigid_transform(self) -> None:
        """A rigid transform with uniform weights should align exactly."""

        rng = np.random.default_rng(5)
        reference = rng.normal(size=(18, 3))
        theta = 0.45
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([1.2, -0.4, 2.5])
        mobile = reference @ rotation.T + translation

        result = soft_superpose_pair(mobile, reference, init_weights=np.ones(reference.shape[0]), max_iterations=2)

        np.testing.assert_allclose(result.aligned_coords, reference, atol=1e-7, rtol=1e-7)

    def test_soft_weights_recover_synthetic_core(self) -> None:
        """Stable residues should receive higher mean soft weights."""

        coords, _, expected_core = build_synthetic_trajectory(n_frames=120, atoms_per_residue=1, random_seed=9)
        result = soft_superpose_trajectory(
            coords,
            reference=coords[0],
            warm_start=True,
            core_fraction=float(np.mean(expected_core)),
            max_iterations=8,
        )

        mean_core_weight = float(result.mean_weights[expected_core].mean())
        mean_flexible_weight = float(result.mean_weights[~expected_core].mean())
        overlap = jaccard_similarity(result.hard_core_mask, expected_core)

        self.assertGreater(mean_core_weight, mean_flexible_weight + 0.15)
        self.assertGreaterEqual(overlap, 0.65)

    def test_warm_start_preserves_solution_quality(self) -> None:
        """Warm-started weights should stay close to the cold-start solution."""

        coords, _, expected_core = build_synthetic_trajectory(n_frames=80, atoms_per_residue=1, random_seed=13)
        warm = soft_superpose_trajectory(
            coords,
            reference=coords[0],
            warm_start=True,
            core_fraction=float(np.mean(expected_core)),
            max_iterations=8,
        )
        cold = soft_superpose_trajectory(
            coords,
            reference=coords[0],
            warm_start=False,
            core_fraction=float(np.mean(expected_core)),
            max_iterations=8,
        )

        np.testing.assert_allclose(warm.mean_weights, cold.mean_weights, atol=0.08, rtol=0.12)
        self.assertGreaterEqual(jaccard_similarity(warm.hard_core_mask, cold.hard_core_mask), 0.8)


if __name__ == "__main__":
    unittest.main()
