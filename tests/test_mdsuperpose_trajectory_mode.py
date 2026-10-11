"""Tests for trajectory-mode robust alignment and warm start."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    Trajectory,
    benchmark_trajectory_robust_alignment,
    compare_trajectory_warm_start,
    solve_trajectory_robust_alignment,
    summarize_trajectory_solution_difference,
)


def _build_smooth_toy_trajectory() -> Trajectory:
    """Build a smooth synthetic trajectory with a flexible tail region."""

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
            [1.5, 2.0, 0.0],
            [2.5, 2.0, 0.0],
            [3.5, 2.0, 0.0],
        ],
        dtype=float,
    )
    frames = []
    for time_index in range(18):
        theta = 0.02 * time_index
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.03 * time_index, -0.015 * time_index, 0.0], dtype=float)
        frame = base @ rotation.T + translation
        frame[[8, 9, 10, 11]] += np.array([0.0, 0.15 * np.sin(0.35 * time_index), 0.0], dtype=float)
        frame[[10, 11]] += np.array([0.0, 0.18 * np.cos(0.25 * time_index), 0.0], dtype=float)
        frames.append(frame)
    return Trajectory(coords=np.asarray(frames, dtype=float))


class TrajectoryModeTest(unittest.TestCase):
    """Check trajectory-mode warm start behavior on a smooth synthetic series."""

    def test_first_frame_warm_start_reduces_iterations_and_preserves_solution(self) -> None:
        """Warm start should help when all frames align to the same reference frame."""

        trajectory = _build_smooth_toy_trajectory()
        cold = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode="first_frame",
            warm_start=False,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )
        warm = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode="first_frame",
            warm_start=True,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )
        difference = summarize_trajectory_solution_difference(cold, warm)

        self.assertLessEqual(warm.average_iterations_per_frame, cold.average_iterations_per_frame)
        self.assertAlmostEqual(warm.mean_rmsd, cold.mean_rmsd, places=6)
        self.assertLess(difference["max_aligned_coordinate_difference"], 1e-5)
        self.assertLess(difference["max_weight_linf_difference"], 1e-4)

    def test_previous_frame_mode_runs_and_remains_consistent(self) -> None:
        """Frame-to-frame mode should run with or without warm start."""

        trajectory = _build_smooth_toy_trajectory()
        cold = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode="previous_frame",
            warm_start=False,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )
        warm = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode="previous_frame",
            warm_start=True,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )

        self.assertEqual(cold.frame_results[0].reference_frame_index, 0)
        self.assertEqual(cold.frame_results[1].reference_frame_index, 0)
        self.assertEqual(cold.frame_results[2].reference_frame_index, 1)
        self.assertTrue(any(frame.used_warm_start for frame in warm.frame_results[1:]))
        self.assertAlmostEqual(warm.mean_rmsd, cold.mean_rmsd, places=8)
        self.assertAlmostEqual(warm.mean_weighted_rmsd, cold.mean_weighted_rmsd, places=8)

    def test_trajectory_benchmark_reports_warm_start_summary(self) -> None:
        """Trajectory benchmark helpers should expose warm-start comparison metrics."""

        trajectory = _build_smooth_toy_trajectory()
        result = benchmark_trajectory_robust_alignment(
            trajectory,
            reference_mode="first_frame",
            warm_start=True,
            core_fraction=0.5,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )
        results, difference = compare_trajectory_warm_start(
            trajectory,
            reference_mode="first_frame",
            core_fraction=0.5,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )

        self.assertEqual(result.extra_metadata["reference_mode"], "first_frame")
        self.assertTrue(result.extra_metadata["warm_start"])
        self.assertEqual(len(result.extra_metadata["per_frame_iterations"]), trajectory.n_frames)
        self.assertEqual(len(results), 2)
        self.assertIn("mean_aligned_coordinate_difference", difference)

    def test_collect_timing_writes_optional_timing_json(self) -> None:
        """Timing collection should be opt-in and write a compact JSON file."""

        trajectory = _build_smooth_toy_trajectory()
        timing_path = ROOT / "_tmp_trajectory_timing.json"
        if timing_path.exists():
            timing_path.unlink()
        result = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode="first_frame",
            warm_start=True,
            rule="huber",
            delta=0.16,
            max_iterations=5,
            tolerance=1e-6,
            collect_timing=True,
            timing_output_path=timing_path,
        )

        self.assertIsNotNone(result.timing)
        self.assertTrue(timing_path.exists())
        self.assertIn("weighted_fit_sec", result.timing)
        self.assertGreaterEqual(result.timing["total_runtime_sec"], result.total_runtime_seconds)
        timing_path.unlink()


if __name__ == "__main__":
    unittest.main()
