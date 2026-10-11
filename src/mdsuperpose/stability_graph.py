"""Sparse non-local stability graph for experimental window scanning."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .window_descriptors import (
    SparseEdgeSet,
    build_local_window_edges,
    combine_edge_sets,
    continuity_breaks,
    segment_ids_from_breaks,
    validate_coordinates,
)
from .window_scan import (
    WindowScanResult,
    aggregate_edge_movement,
    map_movement_to_stability,
    percentile_ranks,
    scan_sparse_edges,
    smooth_weights_chain_aware,
    stable_labels_from_weights,
)


@dataclass(frozen=True, slots=True)
class StabilityGraphDiagnostics:
    """Connected-component and method-disagreement risk diagnostics."""

    component_labels: np.ndarray
    component_sizes: np.ndarray
    ambiguity_score: float
    ambiguous: bool
    local_graph_weight_spearman: float
    high_risk: bool
    risk_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StabilityGraphResult:
    """Graph-derived alignment-free scan and diagnostics."""

    scan: WindowScanResult
    diagnostics: StabilityGraphDiagnostics
    anchor_indices: np.ndarray
    edge_family_counts: dict[str, int]
    graph_build_seconds: float


def _topology_segments(
    n_residues: int,
    chain_ids: np.ndarray | None,
    residue_positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    breaks = continuity_breaks(
        n_residues,
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    return breaks, segment_ids_from_breaks(n_residues, breaks)


def _coordinate_scale(
    reference: np.ndarray,
    segments: np.ndarray,
) -> float:
    """Estimate coordinate units from valid adjacent C-alpha distances."""

    valid = segments[1:] == segments[:-1]
    if not np.any(valid):
        return 1.0
    distances = np.linalg.norm(reference[1:][valid] - reference[:-1][valid], axis=1)
    median = float(np.median(distances))
    return max(median / 3.8, np.finfo(np.float64).eps)


def build_reference_contact_edges(
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    cutoff_angstrom: float = 9.0,
    max_contacts_per_residue: int = 6,
    minimum_sequence_separation: int = 5,
) -> SparseEdgeSet:
    """Build bounded non-local reference contacts without a dense matrix."""

    coords = np.asarray(reference, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("reference must have shape (n_residues, 3)")
    if max_contacts_per_residue < 1:
        raise ValueError("max_contacts_per_residue must be positive")
    breaks, segments = _topology_segments(
        coords.shape[0],
        chain_ids,
        residue_positions,
    )
    cutoff = float(cutoff_angstrom) * _coordinate_scale(coords, segments)
    selected: set[tuple[int, int]] = set()
    indices = np.arange(coords.shape[0])
    for source in range(coords.shape[0]):
        distances = np.linalg.norm(coords - coords[source], axis=1)
        nonlocal_mask = (segments != segments[source]) | (
            np.abs(indices - source) >= int(minimum_sequence_separation)
        )
        candidates = np.flatnonzero(
            nonlocal_mask & (indices != source) & (distances <= cutoff)
        )
        order = np.lexsort((candidates, distances[candidates]))
        for target in candidates[order[: int(max_contacts_per_residue)]]:
            selected.add(tuple(sorted((source, int(target)))))
    rows = sorted(selected)
    source = np.asarray([row[0] for row in rows], dtype=np.int64)
    target = np.asarray([row[1] for row in rows], dtype=np.int64)
    return SparseEdgeSet(
        source=source,
        target=target,
        family=np.full(source.size, "contact", dtype="U24"),
        n_residues=coords.shape[0],
        continuity_breaks=breaks,
    )


def select_global_anchors(
    reference: np.ndarray,
    *,
    n_anchors: int = 12,
) -> np.ndarray:
    """Select deterministic spatial anchors by farthest-point sampling."""

    coords = np.asarray(reference, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("reference must have shape (n_residues, 3)")
    n_anchors = min(max(1, int(n_anchors)), coords.shape[0])
    centroid = np.mean(coords, axis=0)
    centroid_distances = np.einsum(
        "ni,ni->n",
        coords - centroid,
        coords - centroid,
    )
    first = int(np.argmax(centroid_distances))
    anchors = [first]
    minimum_squared = np.einsum(
        "ni,ni->n",
        coords - coords[first],
        coords - coords[first],
    )
    while len(anchors) < n_anchors:
        minimum_squared[np.asarray(anchors, dtype=int)] = -1.0
        next_anchor = int(np.argmax(minimum_squared))
        anchors.append(next_anchor)
        squared = np.einsum(
            "ni,ni->n",
            coords - coords[next_anchor],
            coords - coords[next_anchor],
        )
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.asarray(anchors, dtype=np.int64)


def build_global_anchor_edges(
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    n_anchors: int = 12,
    anchors_per_residue: int = 2,
) -> tuple[SparseEdgeSet, np.ndarray]:
    """Connect residues to spatially near and far deterministic anchors."""

    coords = np.asarray(reference, dtype=np.float64)
    breaks = continuity_breaks(
        coords.shape[0],
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    anchors = select_global_anchors(coords, n_anchors=n_anchors)
    count = min(max(1, int(anchors_per_residue)), anchors.size)
    selected: set[tuple[int, int]] = set()
    for residue in range(coords.shape[0]):
        distances = np.linalg.norm(coords[anchors] - coords[residue], axis=1)
        available = anchors[anchors != residue]
        available_distances = distances[anchors != residue]
        if available.size == 0:
            continue
        nearest_order = np.lexsort((available, available_distances))
        chosen: list[int] = []
        if count >= 1:
            chosen.append(int(available[nearest_order[0]]))
        if count >= 2:
            farthest_order = np.lexsort((available, -available_distances))
            chosen.append(int(available[farthest_order[0]]))
        if count > 2:
            for target in available[nearest_order]:
                if int(target) not in chosen:
                    chosen.append(int(target))
                if len(chosen) == count:
                    break
        for target in chosen:
            selected.add(tuple(sorted((residue, target))))
    rows = sorted(selected)
    source = np.asarray([row[0] for row in rows], dtype=np.int64)
    target = np.asarray([row[1] for row in rows], dtype=np.int64)
    edges = SparseEdgeSet(
        source=source,
        target=target,
        family=np.full(source.size, "anchor", dtype="U24"),
        n_residues=coords.shape[0],
        continuity_breaks=breaks,
    )
    return edges, anchors


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = percentile_ranks(np.asarray(left, dtype=np.float64))
    right_ranks = percentile_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return 1.0 if np.allclose(left_ranks, right_ranks) else 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _family_balanced_node_movement(
    edge_movement: np.ndarray,
    edges: SparseEdgeSet,
) -> np.ndarray:
    """Average per-family movement percentiles so local edges do not dominate."""

    family_scores: list[np.ndarray] = []
    for family in np.unique(edges.family):
        mask = edges.family == family
        subset = SparseEdgeSet(
            source=edges.source[mask],
            target=edges.target[mask],
            family=edges.family[mask],
            n_residues=edges.n_residues,
            continuity_breaks=edges.continuity_breaks,
        )
        node_movement, support = aggregate_edge_movement(edge_movement[mask], subset)
        percentiles = percentile_ranks(node_movement)
        percentiles[~support.astype(bool)] = np.nan
        family_scores.append(percentiles)
    stacked = np.asarray(family_scores)
    return np.nanmean(stacked, axis=0)


def stable_components(
    edges: SparseEdgeSet,
    edge_movement: np.ndarray,
    *,
    stable_fraction: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Find components connected by the lowest-movement sparse edges."""

    edge_percentile = percentile_ranks(edge_movement)
    stable = edge_percentile <= float(stable_fraction)
    adjacency: list[list[int]] = [[] for _ in range(edges.n_residues)]
    for source, target in zip(
        edges.source[stable],
        edges.target[stable],
        strict=True,
    ):
        adjacency[int(source)].append(int(target))
        adjacency[int(target)].append(int(source))
    labels = np.full(edges.n_residues, -1, dtype=np.int64)
    sizes: list[int] = []
    component = 0
    for start in range(edges.n_residues):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in adjacency[node]:
                if labels[neighbour] < 0:
                    labels[neighbour] = component
                    stack.append(neighbour)
        sizes.append(size)
        component += 1
    return labels, np.asarray(sizes, dtype=np.int64)


def scan_window_graph(
    coords: np.ndarray,
    *,
    variant: str = "full",
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    statistic: str = "winsorised_variance",
    mapping: str = "rank",
    n_anchors: int = 12,
    max_contacts_per_residue: int = 6,
) -> StabilityGraphResult:
    """Run local, contact and/or anchor sparse stability scanning."""

    values = validate_coordinates(coords)
    if variant not in {"local_contact", "local_anchor", "full"}:
        raise ValueError("variant must be 'local_contact', 'local_anchor', or 'full'")
    build_start = time.perf_counter()
    local = build_local_window_edges(
        values.shape[1],
        window_size=5,
        feature_set="minimal",
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    edge_sets = [local]
    anchors = np.asarray([], dtype=np.int64)
    if variant in {"local_contact", "full"}:
        contact = build_reference_contact_edges(
            values[0],
            chain_ids=chain_ids,
            residue_positions=residue_positions,
            max_contacts_per_residue=max_contacts_per_residue,
        )
        if contact.n_edges:
            edge_sets.append(contact)
    if variant in {"local_anchor", "full"}:
        anchor_edges, anchors = build_global_anchor_edges(
            values[0],
            chain_ids=chain_ids,
            residue_positions=residue_positions,
            n_anchors=n_anchors,
        )
        if anchor_edges.n_edges:
            edge_sets.append(anchor_edges)
    edges = combine_edge_sets(*edge_sets)
    build_seconds = time.perf_counter() - build_start
    base = scan_sparse_edges(
        values,
        edges,
        statistic=statistic,
        mapping=mapping,
    )
    balanced_movement = _family_balanced_node_movement(
        base.edge_movement,
        edges,
    )
    weights, percentiles = map_movement_to_stability(
        balanced_movement,
        mapping=mapping,
    )
    weights = smooth_weights_chain_aware(
        weights,
        edges.continuity_breaks,
        strength=0.05,
    )
    labels = stable_labels_from_weights(weights, core_fraction=0.6)
    scan = WindowScanResult(
        movement_score=balanced_movement,
        movement_percentile=percentiles,
        stability_weights=weights,
        candidate_stable_label=labels,
        support_count=base.support_count,
        support_uncertainty=base.support_uncertainty,
        edge_movement=base.edge_movement,
        edge_centre=base.edge_centre,
        edge_scale=base.edge_scale,
        edges=edges,
        statistic=base.statistic,
        mapping=base.mapping,
        timing={
            **base.timing,
            "graph_build": build_seconds,
            "total_scan_only": base.timing["total_scan_only"] + build_seconds,
        },
        metadata={
            **base.metadata,
            "graph_variant": variant,
            "n_anchors": int(anchors.size),
            "max_contacts_per_residue": int(max_contacts_per_residue),
        },
    )
    component_labels, component_sizes = stable_components(
        edges,
        base.edge_movement,
    )
    ordered_sizes = np.sort(component_sizes)[::-1]
    ambiguity = (
        float(ordered_sizes[1] / ordered_sizes[0])
        if ordered_sizes.size > 1 and ordered_sizes[0] > 0
        else 0.0
    )
    local_scan = scan_sparse_edges(values, local, statistic=statistic, mapping=mapping)
    agreement = _spearman(local_scan.stability_weights, weights)
    reasons: list[str] = []
    if ambiguity >= 0.8 and ordered_sizes[0] < 0.8 * values.shape[1]:
        reasons.append("similar_stable_components")
    if agreement < 0.8:
        reasons.append("local_graph_disagreement")
    effective_core = float(np.sum(weights) / max(float(np.max(weights)), 1e-12))
    if effective_core < 0.3 * values.shape[1]:
        reasons.append("small_effective_core")
    diagnostics = StabilityGraphDiagnostics(
        component_labels=component_labels,
        component_sizes=component_sizes,
        ambiguity_score=ambiguity,
        ambiguous="similar_stable_components" in reasons,
        local_graph_weight_spearman=agreement,
        high_risk=bool(reasons),
        risk_reasons=tuple(reasons),
    )
    counts = {
        str(family): int(np.count_nonzero(edges.family == family))
        for family in np.unique(edges.family)
    }
    return StabilityGraphResult(
        scan=scan,
        diagnostics=diagnostics,
        anchor_indices=anchors,
        edge_family_counts=counts,
        graph_build_seconds=build_seconds,
    )
