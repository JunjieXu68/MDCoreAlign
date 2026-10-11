"""Chain, residue-gap, and descriptor parity tests for Phase 6A."""

from __future__ import annotations

import numpy as np

from mdsuperpose.window_descriptors import (
    build_local_window_edges,
    edge_squared_distances,
    scalar_edge_squared_distances,
)
from mdsuperpose.window_scan import scan_window_local


def _coords() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(10, 12, 3))


def test_no_edge_crosses_chain_boundary() -> None:
    chain = np.asarray(["A"] * 6 + ["B"] * 6)
    edges = build_local_window_edges(12, chain_ids=chain)
    assert not np.any((edges.source < 6) & (edges.target >= 6))
    assert edges.continuity_breaks[5]


def test_no_edge_crosses_missing_residue_gap() -> None:
    positions = np.asarray([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16])
    edges = build_local_window_edges(12, residue_positions=positions)
    assert not np.any((edges.source < 5) & (edges.target >= 5))
    assert edges.continuity_breaks[4]


def test_signal_does_not_cross_chain_boundary() -> None:
    coords = np.repeat(_coords()[0:1], 40, axis=0)
    rng = np.random.default_rng(8)
    coords[:, 4:6] += rng.normal(scale=1.0, size=(40, 2, 3))
    chain = np.asarray(["A"] * 6 + ["B"] * 6)
    result = scan_window_local(coords, chain_ids=chain, feature_set="full")
    np.testing.assert_allclose(result.movement_score[6:], 0.0, atol=1e-15)


def test_vectorised_descriptor_matches_scalar_reference() -> None:
    coords = _coords()
    edges = build_local_window_edges(coords.shape[1], feature_set="full")
    observed = edge_squared_distances(coords, edges, chunk_size=3)
    expected = scalar_edge_squared_distances(coords, edges)
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)


def test_terminal_support_is_reported_as_more_uncertain() -> None:
    result = scan_window_local(_coords(), feature_set="full")
    assert result.support_count[0] < result.support_count[5]
    assert result.support_uncertainty[0] > result.support_uncertainty[5]
