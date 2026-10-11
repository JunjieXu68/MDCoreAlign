"""Experimental alignment-free local kinematics and motion-domain analysis.

This module is intentionally not part of the public default path. Pairwise
distance changes are used to describe local deformation and coherent domains.
Rigid transforms, RMSD and RMSF are produced only by the optional explicit
domain-anchor alignment step.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fast_kabsch import weighted_kabsch_batch
from .flexibility import detect_flexible_segments
from .stability_graph import (
    build_global_anchor_edges,
    build_reference_contact_edges,
)
from .window_descriptors import (
    SparseEdgeSet,
    build_local_window_edges,
    combine_edge_sets,
    validate_coordinates,
)
from .window_scan import (
    WindowScanResult,
    percentile_ranks,
    scan_sparse_edges,
    scan_window_local,
)


@dataclass(frozen=True, slots=True)
class LocalDeformationResult:
    """Alignment-free local-deformation output.

    ``local_deformation_index`` is dimensionless pair-distance instability. It
    is not RMSD and does not describe coherent global domain displacement.
    """

    local_deformation_index: np.ndarray
    local_rigidity_score: np.ndarray
    locally_deforming_segments: np.ndarray
    confidence: np.ndarray
    temporal_block_reproducibility: float
    temporal_block_segment_jaccard: float
    block_count: int
    scan: WindowScanResult
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class MotionDomainResult:
    """Sparse-graph coherent motion-domain and hinge diagnostics."""

    local: LocalDeformationResult
    motion_domain_id: np.ndarray
    domain_confidence: np.ndarray
    hinge_boundary_score: np.ndarray
    inter_domain_edge_instability: np.ndarray
    edge_temporal_instability: np.ndarray
    edges: SparseEdgeSet
    temporal_block_boundary_jaccard: float
    edge_family_counts: dict[str, int]
    timing: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class DomainAlignmentResult:
    """Downstream exact fit after an explicit domain-anchor choice."""

    anchor_domain_id: int
    anchor_selection: str
    anchor_mask: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    rmsf: np.ndarray
    aligned_coords: np.ndarray
    timing: dict[str, float]
    metadata: dict[str, object]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute deterministic rank correlation with flat-vector handling."""

    left_rank = percentile_ranks(np.asarray(left, dtype=np.float64))
    right_rank = percentile_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return 1.0 if np.allclose(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return Boolean-set Jaccard with an empty-set convention of one."""

    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    union = np.count_nonzero(first | second)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first & second) / union)


def _temporal_blocks(n_frames: int, block_count: int) -> list[np.ndarray]:
    """Return non-empty contiguous temporal blocks."""

    count = min(max(2, int(block_count)), int(n_frames))
    return [
        indices
        for indices in np.array_split(np.arange(n_frames), count)
        if indices.size >= 2
    ]


def _mean_pairwise(values: list[np.ndarray], metric) -> float:
    """Average a pair metric over all block pairs."""

    scores = [
        metric(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return float(np.mean(scores)) if scores else 1.0


def scan_local_kinematics(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    block_count: int = 4,
    evaluate_blocks: bool = True,
    deforming_percentile: float = 0.8,
    minimum_segment_length: int = 3,
    merge_gap: int = 1,
    descriptor_chunk_size: int = 2000,
) -> LocalDeformationResult:
    """Run the fixed chain-aware W5 local-deformation analysis.

    The descriptor uses robust relative MAD of squared pair distances from
    unique W5 edges with sequence separations 2-4. This makes it invariant to
    translation and rotation, normalised under global coordinate scaling, and
    robust to isolated outlier frames.
    """

    total_start = time.perf_counter()
    values = validate_coordinates(coords)
    chains = (
        np.asarray([""] * values.shape[1], dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    full_start = time.perf_counter()
    full = scan_window_local(
        values,
        window_size=5,
        feature_set="minimal",
        chain_ids=chains,
        residue_positions=residue_positions,
        statistic="relative_mad",
        mapping="rank",
        descriptor_chunk_size=descriptor_chunk_size,
    )
    full_seconds = time.perf_counter() - full_start
    segment_ids = detect_flexible_segments(
        full.movement_score,
        percentile_cutoff=deforming_percentile,
        min_segment_length=minimum_segment_length,
        merge_gap=merge_gap,
        chain=chains,
    )

    block_start = time.perf_counter()
    blocks = (
        _temporal_blocks(values.shape[0], block_count)
        if evaluate_blocks
        else []
    )
    block_indices: list[np.ndarray] = []
    block_segments: list[np.ndarray] = []
    for indices in blocks:
        scan = scan_window_local(
            values[indices],
            window_size=5,
            feature_set="minimal",
            chain_ids=chains,
            residue_positions=residue_positions,
            statistic="relative_mad",
            mapping="rank",
            descriptor_chunk_size=descriptor_chunk_size,
        )
        block_indices.append(scan.movement_score)
        block_segments.append(
            detect_flexible_segments(
                scan.movement_score,
                percentile_cutoff=deforming_percentile,
                min_segment_length=minimum_segment_length,
                merge_gap=merge_gap,
                chain=chains,
            )
            > 0
        )
    reproducibility = _mean_pairwise(block_indices, _spearman)
    segment_reproducibility = _mean_pairwise(block_segments, _jaccard)
    if block_indices:
        block_percentiles = np.asarray(
            [percentile_ranks(index) for index in block_indices]
        )
        rank_dispersion = np.std(block_percentiles, axis=0)
    else:
        rank_dispersion = np.zeros(values.shape[1], dtype=np.float64)
    support_confidence = 1.0 - full.support_uncertainty
    confidence = support_confidence * np.clip(
        1.0 - rank_dispersion / 0.5, 0.0, 1.0
    )
    block_seconds = time.perf_counter() - block_start
    total_seconds = time.perf_counter() - total_start
    return LocalDeformationResult(
        local_deformation_index=np.asarray(
            full.movement_score, dtype=np.float64
        ),
        local_rigidity_score=np.asarray(
            full.stability_weights, dtype=np.float64
        ),
        locally_deforming_segments=segment_ids,
        confidence=confidence,
        temporal_block_reproducibility=reproducibility,
        temporal_block_segment_jaccard=segment_reproducibility,
        block_count=len(blocks),
        scan=full,
        timing={
            "full_local_scan_seconds": full_seconds,
            "temporal_block_reproducibility_seconds": block_seconds,
            "total_seconds": total_seconds,
        },
        metadata={
            "status": "experimental",
            "output_contract": "alignment_free_local_kinematics",
            "descriptor": "W5_relative_MAD_squared_pair_distance",
            "translation_invariant": True,
            "rotation_invariant": True,
            "scale_normalised": True,
            "outlier_robust": True,
            "score_called_rmsd": False,
            "score_called_global_mobility": False,
            "deforming_percentile": float(deforming_percentile),
            "minimum_segment_length": int(minimum_segment_length),
            "merge_gap": int(merge_gap),
            "temporal_blocks_evaluated": bool(evaluate_blocks),
            "descriptor_chunk_size": int(descriptor_chunk_size),
        },
    )


def build_motion_domain_edges(
    reference: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    n_anchors: int = 12,
    anchors_per_residue: int = 2,
    max_contacts_per_residue: int = 6,
) -> tuple[SparseEdgeSet, np.ndarray]:
    """Build the fixed local/contact/anchor sparse motion graph."""

    local = build_local_window_edges(
        reference.shape[0],
        window_size=5,
        feature_set="minimal",
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    contacts = build_reference_contact_edges(
        reference,
        chain_ids=chain_ids,
        residue_positions=residue_positions,
        max_contacts_per_residue=max_contacts_per_residue,
    )
    anchors, anchor_indices = build_global_anchor_edges(
        reference,
        chain_ids=chain_ids,
        residue_positions=residue_positions,
        n_anchors=n_anchors,
        anchors_per_residue=anchors_per_residue,
    )
    sets = [local]
    if contacts.n_edges:
        sets.append(contacts)
    if anchors.n_edges:
        sets.append(anchors)
    return combine_edge_sets(*sets), anchor_indices


def _edge_subset(edges: SparseEdgeSet, mask: np.ndarray) -> SparseEdgeSet:
    """Return a topology-preserving subset of sparse graph edges."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (edges.n_edges,):
        raise ValueError("edge mask has the wrong shape")
    return SparseEdgeSet(
        source=edges.source[selected],
        target=edges.target[selected],
        family=edges.family[selected],
        n_residues=edges.n_residues,
        continuity_breaks=edges.continuity_breaks,
    )


def _motion_edge_instability(
    coords: np.ndarray,
    edges: SparseEdgeSet,
    local_scan: WindowScanResult,
    *,
    nonlocal_sample_count: int,
    descriptor_chunk_size: int,
) -> tuple[np.ndarray, int]:
    """Combine exact W5 statistics with stratified non-local edge statistics."""

    edge_movement = np.empty(edges.n_edges, dtype=np.float64)
    local_lookup = {
        (int(source), int(target)): float(value)
        for source, target, value in zip(
            local_scan.edges.source,
            local_scan.edges.target,
            local_scan.edge_movement,
            strict=True,
        )
    }
    local_mask = edges.family == "local_w5"
    for edge_index in np.flatnonzero(local_mask):
        key = (int(edges.source[edge_index]), int(edges.target[edge_index]))
        edge_movement[edge_index] = local_lookup[key]

    nonlocal_mask = ~local_mask
    if not np.any(nonlocal_mask):
        return edge_movement, 0
    sample_count = min(
        max(2, int(nonlocal_sample_count)),
        int(coords.shape[0]),
    )
    sample_indices = np.linspace(
        0, coords.shape[0] - 1, sample_count, dtype=np.int64
    )
    nonlocal_edges = _edge_subset(edges, nonlocal_mask)
    nonlocal_scan = scan_sparse_edges(
        coords[sample_indices],
        nonlocal_edges,
        statistic="relative_mad",
        mapping="rank",
        descriptor_chunk_size=descriptor_chunk_size,
    )
    edge_movement[nonlocal_mask] = nonlocal_scan.edge_movement
    return edge_movement, sample_count


def _hinge_boundary_scores(
    edges: SparseEdgeSet,
    edge_instability: np.ndarray,
    local_deformation: np.ndarray,
    chain_ids: np.ndarray,
) -> np.ndarray:
    """Aggregate short/intermediate crossing-edge instability at each cut."""

    n_residues = edges.n_residues
    raw = np.zeros(n_residues, dtype=np.float64)
    edge_values = np.asarray(edge_instability, dtype=np.float64).copy()
    local_values = np.asarray(local_deformation, dtype=np.float64).copy()
    maximum_signal = max(
        float(np.max(edge_values)), float(np.max(local_values))
    )
    if maximum_signal <= 1e-10:
        return raw
    numerical_floor = max(1e-12, maximum_signal * 1e-10)
    edge_values[edge_values < numerical_floor] = 0.0
    local_values[local_values < numerical_floor] = 0.0
    edge_percentile = percentile_ranks(edge_values)
    local_percentile = percentile_ranks(local_values)
    span_limit = max(12, int(np.ceil(0.15 * n_residues)))
    for boundary in range(n_residues - 1):
        if chain_ids[boundary] != chain_ids[boundary + 1]:
            continue
        same_chain = chain_ids[edges.source] == chain_ids[boundary]
        crossing = (
            same_chain
            & (edges.source <= boundary)
            & (edges.target > boundary)
            & ((edges.target - edges.source) <= span_limit)
        )
        if np.any(crossing):
            crossing_score = float(
                np.quantile(edge_percentile[crossing], 0.75)
            )
        else:
            crossing_score = 0.0
        neighbourhood = local_percentile[
            max(0, boundary - 1) : min(n_residues, boundary + 3)
        ]
        raw[boundary] = 0.8 * crossing_score + 0.2 * float(
            np.mean(neighbourhood)
        )
    raw[-1] = 0.0
    if np.allclose(raw, 0.0, atol=1e-15, rtol=0.0):
        return raw
    return percentile_ranks(raw)


def _choose_boundaries(
    scores: np.ndarray,
    chain_ids: np.ndarray,
    *,
    threshold: float,
    minimum_domain_size: int,
) -> np.ndarray:
    """Select separated local maxima without crossing chain boundaries."""

    n_residues = scores.size
    forced = set(
        (np.flatnonzero(chain_ids[1:] != chain_ids[:-1]) + 1).tolist()
    )
    candidates: list[int] = []
    for boundary in range(n_residues - 1):
        if chain_ids[boundary] != chain_ids[boundary + 1]:
            continue
        left = scores[boundary - 1] if boundary else -np.inf
        right = scores[boundary + 1] if boundary + 1 < n_residues else -np.inf
        if scores[boundary] >= threshold and scores[boundary] >= left and scores[
            boundary
        ] >= right:
            candidates.append(boundary + 1)
    accepted = sorted(forced)
    for boundary in sorted(
        candidates, key=lambda item: (-scores[item - 1], item)
    ):
        chain_indices = np.flatnonzero(chain_ids == chain_ids[boundary])
        chain_start = int(chain_indices[0])
        chain_stop = int(chain_indices[-1]) + 1
        existing = sorted(
            [
                value
                for value in accepted
                if chain_start < value < chain_stop
            ]
        )
        cuts = [chain_start, *existing, chain_stop]
        segment_start = max(value for value in cuts if value <= boundary)
        segment_stop = min(value for value in cuts if value >= boundary)
        if (
            boundary - segment_start >= minimum_domain_size
            and segment_stop - boundary >= minimum_domain_size
        ):
            accepted.append(boundary)
    return np.asarray(sorted(set(accepted)), dtype=np.int64)


def _labels_from_boundaries(
    n_residues: int, boundaries: np.ndarray
) -> np.ndarray:
    """Convert residue-start boundaries to one-based domain labels."""

    labels = np.ones(n_residues, dtype=np.int64)
    for boundary in np.asarray(boundaries, dtype=np.int64):
        labels[boundary:] += 1
    return labels


def _domain_confidence(
    labels: np.ndarray,
    edges: SparseEdgeSet,
    edge_instability: np.ndarray,
) -> np.ndarray:
    """Assign each residue its domain's internal edge-stability confidence."""

    edge_percentile = percentile_ranks(edge_instability)
    output = np.zeros(labels.size, dtype=np.float64)
    for domain_id in np.unique(labels):
        residues = labels == domain_id
        internal = residues[edges.source] & residues[edges.target]
        confidence = (
            1.0 - float(np.median(edge_percentile[internal]))
            if np.any(internal)
            else 0.0
        )
        output[residues] = confidence
    return output


def _inter_domain_instability(
    labels: np.ndarray,
    edges: SparseEdgeSet,
    edge_instability: np.ndarray,
) -> np.ndarray:
    """Aggregate incident between-domain edge instability per residue."""

    edge_percentile = percentile_ranks(edge_instability)
    incident: list[list[float]] = [[] for _ in range(labels.size)]
    between = labels[edges.source] != labels[edges.target]
    for source, target, score in zip(
        edges.source[between],
        edges.target[between],
        edge_percentile[between],
        strict=True,
    ):
        incident[int(source)].append(float(score))
        incident[int(target)].append(float(score))
    return np.asarray(
        [
            float(np.median(values)) if values else 0.0
            for values in incident
        ],
        dtype=np.float64,
    )


def _expanded_boundary_mask(
    domain_ids: np.ndarray, radius: int = 2
) -> np.ndarray:
    """Return a tolerance-expanded residue mask around domain boundaries."""

    mask = np.zeros(domain_ids.size, dtype=bool)
    boundaries = np.flatnonzero(domain_ids[1:] != domain_ids[:-1]) + 1
    for boundary in boundaries:
        mask[
            max(0, boundary - radius) : min(
                domain_ids.size, boundary + radius + 1
            )
        ] = True
    return mask


def scan_motion_domains(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    block_count: int = 4,
    boundary_threshold: float = 0.90,
    minimum_domain_size: int = 8,
    n_anchors: int = 12,
    anchors_per_residue: int = 2,
    max_contacts_per_residue: int = 6,
    nonlocal_sample_count: int = 512,
    evaluate_blocks: bool = True,
    descriptor_chunk_size: int = 2000,
    deforming_percentile: float = 0.8,
    minimum_segment_length: int = 3,
    merge_gap: int = 1,
) -> MotionDomainResult:
    """Decompose local deformation and coherent sequence motion domains."""

    total_start = time.perf_counter()
    values = validate_coordinates(coords)
    chains = (
        np.asarray([""] * values.shape[1], dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    if chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must have shape (n_residues,)")
    local = scan_local_kinematics(
        values,
        chain_ids=chains,
        residue_positions=residue_positions,
        block_count=block_count,
        evaluate_blocks=evaluate_blocks,
        descriptor_chunk_size=descriptor_chunk_size,
        deforming_percentile=deforming_percentile,
        minimum_segment_length=minimum_segment_length,
        merge_gap=merge_gap,
    )
    graph_start = time.perf_counter()
    edges, anchors = build_motion_domain_edges(
        values[0],
        chain_ids=chains,
        residue_positions=residue_positions,
        n_anchors=n_anchors,
        anchors_per_residue=anchors_per_residue,
        max_contacts_per_residue=max_contacts_per_residue,
    )
    graph_build_seconds = time.perf_counter() - graph_start
    scan_start = time.perf_counter()
    edge_movement, sampled_nonlocal_frames = _motion_edge_instability(
        values,
        edges,
        local.scan,
        nonlocal_sample_count=nonlocal_sample_count,
        descriptor_chunk_size=descriptor_chunk_size,
    )
    scan_seconds = time.perf_counter() - scan_start
    boundary_scores = _hinge_boundary_scores(
        edges,
        edge_movement,
        local.local_deformation_index,
        chains,
    )
    boundaries = _choose_boundaries(
        boundary_scores,
        chains,
        threshold=boundary_threshold,
        minimum_domain_size=minimum_domain_size,
    )
    domain_ids = _labels_from_boundaries(values.shape[1], boundaries)
    domain_confidence = _domain_confidence(
        domain_ids, edges, edge_movement
    )
    inter_domain = _inter_domain_instability(
        domain_ids, edges, edge_movement
    )

    block_start = time.perf_counter()
    block_masks: list[np.ndarray] = []
    if evaluate_blocks:
        for indices in _temporal_blocks(values.shape[0], block_count):
            block_local_scan = scan_window_local(
                values[indices],
                chain_ids=chains,
                residue_positions=residue_positions,
                statistic="relative_mad",
                mapping="rank",
                descriptor_chunk_size=descriptor_chunk_size,
            )
            block_edge_movement, _ = _motion_edge_instability(
                values[indices],
                edges,
                block_local_scan,
                nonlocal_sample_count=nonlocal_sample_count,
                descriptor_chunk_size=descriptor_chunk_size,
            )
            block_scores = _hinge_boundary_scores(
                edges,
                block_edge_movement,
                block_local_scan.movement_score,
                chains,
            )
            block_boundaries = _choose_boundaries(
                block_scores,
                chains,
                threshold=boundary_threshold,
                minimum_domain_size=minimum_domain_size,
            )
            block_domains = _labels_from_boundaries(
                values.shape[1], block_boundaries
            )
            block_masks.append(_expanded_boundary_mask(block_domains))
    boundary_reproducibility = _mean_pairwise(block_masks, _jaccard)
    block_seconds = time.perf_counter() - block_start
    total_seconds = time.perf_counter() - total_start
    return MotionDomainResult(
        local=local,
        motion_domain_id=domain_ids,
        domain_confidence=domain_confidence,
        hinge_boundary_score=boundary_scores,
        inter_domain_edge_instability=inter_domain,
        edge_temporal_instability=np.asarray(
            edge_movement, dtype=np.float64
        ),
        edges=edges,
        temporal_block_boundary_jaccard=boundary_reproducibility,
        edge_family_counts={
            str(family): int(np.count_nonzero(edges.family == family))
            for family in np.unique(edges.family)
        },
        timing={
            "local_kinematics_seconds": local.timing["total_seconds"],
            "graph_build_seconds": graph_build_seconds,
            "graph_scan_seconds": scan_seconds,
            "temporal_block_domain_seconds": block_seconds,
            "total_seconds": total_seconds,
        },
        metadata={
            "status": "experimental",
            "output_contract": "sparse_motion_domain_graph",
            "single_global_core_required": False,
            "boundary_threshold": float(boundary_threshold),
            "minimum_domain_size": int(minimum_domain_size),
            "n_anchors": int(anchors.size),
            "anchors_per_residue": int(anchors_per_residue),
            "max_contacts_per_residue": int(max_contacts_per_residue),
            "local_edge_sampling": "all_frames_exact_W5_reuse",
            "nonlocal_edge_sampling": "temporally_stratified",
            "nonlocal_edge_sample_count": sampled_nonlocal_frames,
            "descriptor_chunk_size": int(descriptor_chunk_size),
            "score_called_rmsd": False,
        },
    )


def select_anchor_domain(
    result: MotionDomainResult,
    *,
    selection: str = "largest_stable",
    domain_id: int | None = None,
) -> int:
    """Select one motion domain for optional downstream alignment."""

    labels = result.motion_domain_id
    domains = np.unique(labels)
    if selection == "user_specified":
        if domain_id is None or int(domain_id) not in domains:
            raise ValueError("domain_id must identify an available domain")
        return int(domain_id)
    if selection == "largest_stable":
        return int(
            max(
                domains,
                key=lambda current: (
                    int(np.count_nonzero(labels == current)),
                    float(np.mean(result.domain_confidence[labels == current])),
                    -int(current),
                ),
            )
        )
    if selection == "least_deforming":
        return int(
            min(
                domains,
                key=lambda current: (
                    float(
                        np.mean(
                            result.local.local_deformation_index[
                                labels == current
                            ]
                        )
                    ),
                    -int(np.count_nonzero(labels == current)),
                    int(current),
                ),
            )
        )
    raise ValueError(
        "selection must be 'largest_stable', 'least_deforming', or "
        "'user_specified'"
    )


def align_to_motion_domain(
    coords: np.ndarray,
    domains: MotionDomainResult,
    *,
    selection: str = "largest_stable",
    domain_id: int | None = None,
) -> DomainAlignmentResult:
    """Perform exact global transforms after explicit anchor selection."""

    total_start = time.perf_counter()
    values = validate_coordinates(coords)
    selected_domain = select_anchor_domain(
        domains, selection=selection, domain_id=domain_id
    )
    mask = domains.motion_domain_id == selected_domain
    if np.count_nonzero(mask) < 3:
        raise ValueError("the selected anchor domain has fewer than 3 residues")
    weights = mask.astype(np.float64)
    fit_start = time.perf_counter()
    fit = weighted_kabsch_batch(
        values,
        values[0],
        weights,
        validate_inputs=False,
        compute_metrics=False,
    )
    fit_seconds = time.perf_counter() - fit_start
    metric_start = time.perf_counter()
    squared = np.einsum(
        "tni,tni->tn",
        fit.aligned_coords - values[0][None, :, :],
        fit.aligned_coords - values[0][None, :, :],
    )
    rmsd = np.sqrt(np.mean(squared, axis=1))
    mean = np.mean(fit.aligned_coords, axis=0)
    rmsf = np.sqrt(
        np.mean(
            np.einsum(
                "tni,tni->tn",
                fit.aligned_coords - mean[None, :, :],
                fit.aligned_coords - mean[None, :, :],
            ),
            axis=0,
        )
    )
    metric_seconds = time.perf_counter() - metric_start
    return DomainAlignmentResult(
        anchor_domain_id=selected_domain,
        anchor_selection=selection,
        anchor_mask=mask,
        rotations=fit.rotations,
        translations=fit.translations,
        rmsd=rmsd,
        rmsf=rmsf,
        aligned_coords=fit.aligned_coords,
        timing={
            "exact_domain_anchor_fit_seconds": fit_seconds,
            "downstream_rmsd_rmsf_seconds": metric_seconds,
            "total_seconds": time.perf_counter() - total_start,
        },
        metadata={
            "status": "experimental",
            "output_contract": "optional_domain_anchor_alignment",
            "rmsd_role": "downstream_geometric_audit",
            "primary_window_validation_metric": False,
            "anchor_residue_count": int(np.count_nonzero(mask)),
        },
    )


def local_profile_rows(
    result: MotionDomainResult,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Build one transparent residue row per experimental output."""

    n_residues = result.motion_domain_id.size
    chains = (
        np.asarray([""] * n_residues, dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    positions = (
        np.arange(1, n_residues + 1)
        if residue_positions is None
        else np.asarray(residue_positions)
    )
    return [
        {
            "atom_index": index,
            "chain": chains[index],
            "residue_position": positions[index],
            "local_deformation_index": result.local.local_deformation_index[
                index
            ],
            "local_rigidity_score": result.local.local_rigidity_score[index],
            "locally_deforming_segment_id": (
                result.local.locally_deforming_segments[index]
            ),
            "local_confidence": result.local.confidence[index],
            "motion_domain_id": result.motion_domain_id[index],
            "domain_confidence": result.domain_confidence[index],
            "hinge_boundary_score_after_residue": (
                result.hinge_boundary_score[index]
            ),
            "inter_domain_edge_instability": (
                result.inter_domain_edge_instability[index]
            ),
        }
        for index in range(n_residues)
    ]
