"""Tests for sparse contact-graph regularization helpers."""

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
    apply_contact_consistency_regularization,
    build_contact_graph,
    contact_graph_disagreement,
)


class ContactGraphTest(unittest.TestCase):
    """Validate sparse contact-graph construction and smoothing behavior."""

    def test_contact_graph_is_sparse_and_sequence_aware(self) -> None:
        """Graph construction should drop nearby sequence neighbors when requested."""

        reference = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        graph = build_contact_graph(
            reference,
            cutoff=1.5,
            min_sequence_separation=2,
            max_neighbors=1,
        )

        self.assertEqual(graph.edge_count, 3)
        self.assertEqual(graph.neighbors[0].tolist(), [2, 4])
        self.assertEqual(graph.neighbors[1].tolist(), [4])
        self.assertEqual(graph.neighbors[2].tolist(), [0])
        self.assertEqual(graph.neighbors[3].tolist(), [])
        self.assertEqual(graph.neighbors[4].tolist(), [0, 1])
        for node_index, neighbors in enumerate(graph.neighbors):
            for neighbor_index in neighbors:
                self.assertGreaterEqual(abs(neighbor_index - node_index), 2)
                self.assertNotEqual(neighbor_index, node_index)

    def test_contact_regularization_reduces_edge_disagreement(self) -> None:
        """Graph smoothing should reduce weight disagreement across contacts."""

        reference = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.2, 0.0],
                [1.0, 2.2, 0.0],
            ],
            dtype=float,
        )
        graph = build_contact_graph(
            reference,
            cutoff=2.5,
            min_sequence_separation=2,
            max_neighbors=1,
        )
        weights = np.array([1.0, 0.9, 0.2, 0.1], dtype=float)

        smoothed = apply_contact_consistency_regularization(
            weights,
            graph,
            strength=0.6,
            passes=2,
        )

        self.assertLess(
            contact_graph_disagreement(smoothed, graph),
            contact_graph_disagreement(weights, graph),
        )
        self.assertTrue(np.all(smoothed >= 0.0))

    def test_invalid_contact_configuration_raises(self) -> None:
        """Invalid graph-construction and smoothing settings should be rejected."""

        reference = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        graph = build_contact_graph(reference, cutoff=1.5)

        with self.assertRaises(ValueError):
            build_contact_graph(reference, cutoff=0.0)

        with self.assertRaises(ValueError):
            apply_contact_consistency_regularization(np.array([[1.0, 0.5]]), graph, strength=0.5)

        with self.assertRaises(ValueError):
            apply_contact_consistency_regularization(np.array([1.0, 0.5, 0.2]), graph, strength=-0.1)


if __name__ == "__main__":
    unittest.main()
