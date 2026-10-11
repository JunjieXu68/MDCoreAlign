"""Sparse contact-graph helpers for optional core-membership regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import as_float_array


@dataclass(slots=True, frozen=True)
class ContactGraph:
    """Sparse undirected contact graph built from one coordinate frame."""

    neighbors: tuple[np.ndarray, ...]
    degrees: np.ndarray
    directed_sources: np.ndarray
    directed_targets: np.ndarray
    cutoff: float
    min_sequence_separation: int
    max_neighbors: int | None
    edge_count: int
    mean_degree: float


def _validate_reference_coords(reference: np.ndarray) -> np.ndarray:
    """Validate one coordinate frame used for contact-graph construction."""

    reference = as_float_array(reference)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference must have shape (n_atoms, 3)")
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one atom")
    if not np.isfinite(reference).all():
        raise ValueError("reference must contain only finite values")
    return reference


def validate_contact_graph(contact_graph: ContactGraph, n_nodes: int) -> ContactGraph:
    """Validate a contact graph against an expected node count."""

    if len(contact_graph.neighbors) != n_nodes:
        raise ValueError(f"contact_graph must contain {n_nodes} nodes")
    degrees = np.asarray(contact_graph.degrees, dtype=int)
    if degrees.shape != (n_nodes,):
        raise ValueError("contact_graph.degrees must have shape (n_nodes,)")
    if np.any(degrees < 0):
        raise ValueError("contact_graph.degrees must be non-negative")
    directed_sources = np.asarray(contact_graph.directed_sources, dtype=int)
    directed_targets = np.asarray(contact_graph.directed_targets, dtype=int)
    if directed_sources.ndim != 1 or directed_targets.ndim != 1:
        raise ValueError("contact graph edge arrays must be one-dimensional")
    if directed_sources.shape != directed_targets.shape:
        raise ValueError("contact graph edge arrays must have the same shape")
    if directed_sources.shape[0] != int(np.sum(degrees)):
        raise ValueError("contact graph edge arrays must match the node degrees")
    for node_index, neighbors in enumerate(contact_graph.neighbors):
        neighbors = np.asarray(neighbors, dtype=int)
        if neighbors.ndim != 1:
            raise ValueError("each contact-graph neighbor list must be one-dimensional")
        if np.any(neighbors < 0) or np.any(neighbors >= n_nodes):
            raise ValueError("contact-graph neighbors must be valid node indices")
        if np.any(neighbors == node_index):
            raise ValueError("contact graph must not contain self-edges")
        if neighbors.shape[0] != degrees[node_index]:
            raise ValueError("contact graph degrees must match the neighbor lists")
    return contact_graph


def build_contact_graph(
    reference: np.ndarray,
    *,
    cutoff: float = 8.0,
    min_sequence_separation: int = 2,
    max_neighbors: int | None = 8,
) -> ContactGraph:
    """Build a sparse undirected contact graph from one coordinate frame.

    Contacts are defined by Euclidean distance within `cutoff` and filtered by
    a minimum sequence separation to avoid duplicating the sequence regularizer.
    The graph is sparsified further by keeping at most `max_neighbors` nearest
    contacts per node before symmetrization.
    """

    reference = _validate_reference_coords(reference)
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be non-negative")
    if max_neighbors is not None and max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive when provided")

    n_nodes = reference.shape[0]
    deltas = reference[:, None, :] - reference[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2))

    directed_neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    for node_index in range(n_nodes):
        candidates = np.where(
            (distances[node_index] <= cutoff)
            & (distances[node_index] > 0.0)
            & (np.abs(np.arange(n_nodes) - node_index) >= min_sequence_separation)
        )[0]
        if candidates.size == 0:
            continue
        order = np.argsort(distances[node_index, candidates], kind="stable")
        if max_neighbors is not None:
            order = order[:max_neighbors]
        directed_neighbors[node_index] = candidates[order].tolist()

    undirected = [set(neighbors) for neighbors in directed_neighbors]
    for node_index, neighbors in enumerate(directed_neighbors):
        for neighbor_index in neighbors:
            undirected[neighbor_index].add(node_index)

    neighbor_arrays = tuple(
        np.asarray(sorted(neighbors), dtype=int)
        for neighbors in undirected
    )
    degrees = np.asarray([len(neighbors) for neighbors in neighbor_arrays], dtype=int)
    if np.any(degrees > 0):
        directed_sources = np.repeat(np.arange(n_nodes, dtype=int), degrees)
        directed_targets = np.concatenate(neighbor_arrays)
    else:
        directed_sources = np.empty(0, dtype=int)
        directed_targets = np.empty(0, dtype=int)
    edge_count = int(np.sum(degrees) // 2)
    mean_degree = float(np.mean(degrees))
    return ContactGraph(
        neighbors=neighbor_arrays,
        degrees=degrees,
        directed_sources=directed_sources,
        directed_targets=directed_targets,
        cutoff=float(cutoff),
        min_sequence_separation=int(min_sequence_separation),
        max_neighbors=None if max_neighbors is None else int(max_neighbors),
        edge_count=edge_count,
        mean_degree=mean_degree,
    )


def apply_contact_consistency_regularization(
    weights: np.ndarray,
    contact_graph: ContactGraph,
    *,
    strength: float = 0.0,
    passes: int = 1,
    validate: bool = True,
) -> np.ndarray:
    """Smooth weights over a sparse contact graph with Jacobi-style averaging."""

    weights = as_float_array(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if passes < 0:
        raise ValueError("passes must be non-negative")

    if validate:
        contact_graph = validate_contact_graph(contact_graph, weights.shape[0])
    if strength == 0.0 or passes == 0 or contact_graph.edge_count == 0:
        return weights.copy()

    smoothed = weights.copy()
    updated = weights.copy()
    neighbor_sums = np.zeros_like(weights)
    active = contact_graph.degrees > 0
    degrees = contact_graph.degrees[active]
    directed_sources = contact_graph.directed_sources
    directed_targets = contact_graph.directed_targets
    for _ in range(passes):
        updated[:] = smoothed
        neighbor_sums.fill(0.0)
        np.add.at(neighbor_sums, directed_sources, smoothed[directed_targets])
        updated[active] = (
            (1.0 - strength) * smoothed[active]
            + strength * (neighbor_sums[active] / degrees)
        )
        smoothed, updated = updated, smoothed
    return smoothed


def contact_graph_disagreement(weights: np.ndarray, contact_graph: ContactGraph) -> float:
    """Measure mean absolute disagreement across contact edges."""

    weights = as_float_array(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    contact_graph = validate_contact_graph(contact_graph, weights.shape[0])
    if contact_graph.edge_count == 0:
        return 0.0

    total = 0.0
    counted = 0
    for node_index, neighbors in enumerate(contact_graph.neighbors):
        for neighbor_index in neighbors:
            if neighbor_index <= node_index:
                continue
            total += abs(float(weights[node_index] - weights[neighbor_index]))
            counted += 1
    return total / counted if counted > 0 else 0.0
