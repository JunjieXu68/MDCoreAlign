"""Tests for the first iterative robust alignment prototype."""

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
    build_contact_graph,
    contact_graph_disagreement,
    select_core_by_target_fraction,
    solve_iterative_hard_trim_alignment,
    solve_iterative_robust_alignment,
    solve_rigid_alignment,
)


class RobustAlignmentPrototypeTest(unittest.TestCase):
    """Validate the minimal iterative robust alignment loop."""

    def test_history_is_recorded_and_weights_stay_non_negative(self) -> None:
        """The prototype should return a non-empty optimization history."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.5],
            ],
            dtype=float,
        )
        theta = 0.25
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.4, -0.2, 1.0], dtype=float)
        y = x @ rotation.T + translation

        result = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.5,
            max_iterations=10,
            tolerance=1e-8,
        )

        self.assertGreaterEqual(len(result.history), 1)
        self.assertEqual(result.weights.shape, (x.shape[0],))
        self.assertTrue(np.all(result.weights >= 0.0))
        self.assertGreaterEqual(result.weighted_rmsd, 0.0)
        self.assertLess(result.rmsd, 1e-10)

    def test_record_history_can_be_disabled_without_losing_iteration_count(self) -> None:
        """The solver should support a lighter benchmark path without full history."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        y = x + np.array([0.2, -0.1, 0.3], dtype=float)

        result = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.3,
            max_iterations=12,
            tolerance=1e-8,
            record_history=False,
        )

        self.assertEqual(result.history, [])
        self.assertGreaterEqual(result.iteration_count, 1)
        self.assertIsNotNone(result.final_weight_change)
        self.assertIsNotNone(result.final_transform_change)
        self.assertLess(result.rmsd, 1e-10)

    def test_robust_alignment_downweights_outliers_and_improves_inlier_fit(self) -> None:
        """The robust loop should improve inlier fit on a toy outlier problem."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        theta = 0.45
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([1.5, -0.8, 0.4], dtype=float)
        y_clean = x @ rotation.T + translation
        y_noisy = y_clean.copy()
        y_noisy[6] += np.array([7.0, -6.0, 5.0], dtype=float)
        y_noisy[7] += np.array([-8.0, 5.0, -4.0], dtype=float)
        inlier_mask = np.array([True, True, True, True, True, True, False, False], dtype=bool)

        unweighted = solve_rigid_alignment(x, y_noisy)
        robust = solve_iterative_robust_alignment(
            x,
            y_noisy,
            rule="tukey",
            cutoff=3.0,
            max_iterations=25,
            tolerance=1e-6,
        )

        unweighted_inlier_rmsd = float(
            np.sqrt(np.mean(np.sum((unweighted.aligned_coords[inlier_mask] - y_clean[inlier_mask]) ** 2, axis=1)))
        )
        robust_inlier_rmsd = float(
            np.sqrt(np.mean(np.sum((robust.aligned_coords[inlier_mask] - y_clean[inlier_mask]) ** 2, axis=1)))
        )

        self.assertLess(robust_inlier_rmsd, unweighted_inlier_rmsd)
        self.assertLess(np.mean(robust.weights[~inlier_mask]), np.mean(robust.weights[inlier_mask]))

    def test_invalid_configuration_raises(self) -> None:
        """Basic invalid loop settings should be rejected."""

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
            solve_iterative_robust_alignment(x, y, max_iterations=0)

        with self.assertRaises(ValueError):
            solve_iterative_robust_alignment(x, y, tolerance=-1.0)

        with self.assertRaises(ValueError):
            solve_iterative_robust_alignment(x, y, initial_weights=np.array([0.0, 0.0, 0.0], dtype=float))

    def test_sequence_regularization_reduces_core_fragmentation(self) -> None:
        """Sequence regularization should produce a smoother hard core on a toy example."""

        x = np.column_stack(
            [
                np.linspace(0.0, 11.0, 12),
                np.sin(np.linspace(0.0, 2.2, 12)),
                np.cos(np.linspace(0.0, 1.8, 12)),
            ]
        )
        theta = 0.12
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.3, -0.2, 0.1], dtype=float)
        y = x @ rotation.T + translation
        y[[4, 7]] += np.array([0.0, 0.9, 0.0], dtype=float)
        y[[0, 11]] += np.array([0.0, 1.3, 0.0], dtype=float)

        no_reg = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.35,
            max_iterations=20,
            tolerance=1e-6,
        )
        with_reg = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.35,
            max_iterations=20,
            tolerance=1e-6,
            sequence_continuity_strength=0.6,
            sequence_continuity_passes=2,
        )

        no_reg_core = select_core_by_target_fraction(no_reg.weights, fraction=0.5).core_mask.astype(int)
        with_reg_core = select_core_by_target_fraction(with_reg.weights, fraction=0.5).core_mask.astype(int)

        no_reg_transitions = int(np.sum(np.abs(np.diff(no_reg_core))))
        with_reg_transitions = int(np.sum(np.abs(np.diff(with_reg_core))))
        self.assertLessEqual(with_reg_transitions, no_reg_transitions)
        self.assertLess(np.sum(np.abs(np.diff(with_reg.weights))), np.sum(np.abs(np.diff(no_reg.weights))))

    def test_contact_regularization_reduces_contact_disagreement(self) -> None:
        """Contact regularization should align weights across sequence-distant contacts."""

        strand_a = np.column_stack([np.array([0.0, 2.0, 4.0, 6.0]), np.zeros(4), np.zeros(4)])
        strand_b = np.column_stack([np.array([0.0, 2.0, 4.0, 6.0]), np.full(4, 2.2), np.zeros(4)])
        x = np.vstack([strand_a, strand_b])
        theta = 0.18
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.4, -0.3, 0.2], dtype=float)
        y = x @ rotation.T + translation
        y[[5, 6]] += np.array([0.0, 0.8, 0.0], dtype=float)

        graph = build_contact_graph(
            x,
            cutoff=2.5,
            min_sequence_separation=4,
            max_neighbors=1,
        )
        no_reg = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.30,
            max_iterations=20,
            tolerance=1e-6,
        )
        with_reg = solve_iterative_robust_alignment(
            x,
            y,
            rule="huber",
            delta=0.30,
            max_iterations=20,
            tolerance=1e-6,
            contact_graph=graph,
            contact_consistency_strength=0.5,
            contact_consistency_passes=2,
        )

        self.assertLess(contact_graph_disagreement(with_reg.weights, graph), contact_graph_disagreement(no_reg.weights, graph))
        self.assertLess(abs(with_reg.weights[1] - with_reg.weights[5]), abs(no_reg.weights[1] - no_reg.weights[5]))
        self.assertLess(abs(with_reg.weights[2] - with_reg.weights[6]), abs(no_reg.weights[2] - no_reg.weights[6]))

    def test_hard_trim_control_returns_binary_weights_with_expected_fraction(self) -> None:
        """The hard-trim control should keep a deterministic binary subset."""

        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
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
        translation = np.array([0.5, -0.3, 0.2], dtype=float)
        y = x @ rotation.T + translation
        y[5] += np.array([3.0, -2.0, 2.0], dtype=float)

        result = solve_iterative_hard_trim_alignment(
            x,
            y,
            core_fraction=2.0 / 3.0,
            max_iterations=20,
            tolerance=1e-8,
        )

        self.assertTrue(np.all(np.logical_or(np.isclose(result.weights, 0.0), np.isclose(result.weights, 1.0))))
        self.assertEqual(int(np.sum(result.weights > 0.5)), 4)
        self.assertGreaterEqual(result.iteration_count, 1)
        self.assertGreaterEqual(result.weighted_rmsd, 0.0)


if __name__ == "__main__":
    unittest.main()
