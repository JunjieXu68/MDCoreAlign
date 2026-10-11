"""Sparse graph determinism, bounds, and hinge-risk tests."""

from __future__ import annotations

import numpy as np

from mdsuperpose.stability_graph import (
    build_global_anchor_edges,
    build_reference_contact_edges,
    scan_window_graph,
    select_global_anchors,
)
from mdsuperpose.window_scan import scan_window_local


def _compact_reference(n_residues: int = 40) -> np.ndarray:
    index = np.arange(n_residues, dtype=float)
    return np.column_stack(
        (
            3.8 * np.cos(index * 0.6),
            3.8 * np.sin(index * 0.6),
            1.2 * index,
        )
    )


def _hinge(equal_size: bool = True) -> np.ndarray:
    rng = np.random.default_rng(42)
    first_size = 20 if equal_size else 28
    second_size = 20 if equal_size else 12
    first = rng.normal(scale=2.0, size=(first_size, 3))
    second = rng.normal(scale=2.0, size=(second_size, 3)) + np.asarray((8.0, 0.0, 0.0))
    reference = np.vstack((first, second))
    frames = []
    pivot = np.asarray((4.0, 0.0, 0.0))
    for angle in np.linspace(0.0, 1.0, 80):
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        frame = reference.copy()
        frame[first_size:] = (frame[first_size:] - pivot) @ rotation.T + pivot
        frames.append(frame)
    return np.asarray(frames)


def test_anchor_selection_is_deterministic() -> None:
    reference = _compact_reference()
    first = select_global_anchors(reference, n_anchors=10)
    second = select_global_anchors(reference, n_anchors=10)
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 10


def test_sparse_graph_edge_counts_are_linear_bounds() -> None:
    reference = _compact_reference(80)
    contacts = build_reference_contact_edges(
        reference,
        max_contacts_per_residue=4,
    )
    anchors, selected = build_global_anchor_edges(
        reference,
        n_anchors=10,
        anchors_per_residue=2,
    )
    assert contacts.n_edges <= 8 * reference.shape[0]
    assert anchors.n_edges <= 2 * reference.shape[0]
    assert selected.size == 10


def test_graph_detects_nonlocal_hinge_signal_missed_by_local_window() -> None:
    coords = _hinge(equal_size=False)
    local = scan_window_local(coords, statistic="winsorised_variance")
    graph = scan_window_graph(coords, variant="local_anchor")
    boundary = 28
    far_from_hinge = np.r_[0 : boundary - 4, boundary + 4 : coords.shape[1]]
    assert float(np.mean(local.movement_score[far_from_hinge])) < 1e-6
    assert float(np.mean(graph.scan.movement_score[far_from_hinge])) > 0.1
    assert graph.diagnostics.high_risk


def test_equal_rigid_domains_produce_ambiguity_signal() -> None:
    result = scan_window_graph(_hinge(equal_size=True), variant="full")
    assert result.diagnostics.high_risk
    assert (
        result.diagnostics.ambiguous
        or "local_graph_disagreement" in result.diagnostics.risk_reasons
    )
