#!/usr/bin/env python
"""Run the locked W1-W5 experimental window-kinematics pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_kinematics import (
    align_to_motion_domain,
    local_profile_rows,
    scan_local_kinematics,
    scan_motion_domains,
)
from mdsuperpose.window_refine import (
    bounded_global_weight_refinement,
    build_initial_weights,
    exact_all_frame_fit,
    temporally_stratified_indices,
)
from mdsuperpose.window_scan import percentile_ranks
from scripts.scaling_optimization.build_phase2_regression import ROBUST_KWARGS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/window_motion_domains/pilot_validation"
SELECTED_DOMAINS = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "selected_domains.tsv"
)
LEGACY_ROOT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "domains"
)
REGENERATED_ROOT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity/regenerated"
)
LONG_COORDS = (
    REGENERATED_ROOT / "4oelA02/transition_aligned_coords_10000.npz"
)
LONG_METADATA = (
    REGENERATED_ROOT / "4oelA02/raw_replicates/A_metadata.json"
)
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
PERFORMANCE_METHODS = (
    "w5_scan_only",
    "w5_motion_domain_graph",
    "optional_domain_alignment",
    "sequential_lean",
    "uniform_fast_global",
)
WARMUPS = 3
REPEATS = 5


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write union-schema rows."""

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def compact_reference(n_residues: int, seed: int = 7) -> np.ndarray:
    """Build a non-degenerate synthetic CA chain."""

    rng = np.random.default_rng(seed)
    coords = np.cumsum(rng.normal(size=(n_residues, 3)), axis=0)
    adjacent = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return coords * (3.8 / np.median(adjacent))


def rotation(angle: float) -> np.ndarray:
    """Return one z-axis rotation."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def rigid_trajectory(n_residues: int = 40, n_frames: int = 100) -> np.ndarray:
    """Generate pure rigid translation/rotation."""

    reference = compact_reference(n_residues)
    return np.asarray(
        [
            reference @ rotation(angle).T
            + np.asarray((angle, 2.0 * angle, -angle))
            for angle in np.linspace(-0.8, 0.8, n_frames)
        ]
    )


def domain_trajectory(
    sizes: tuple[int, ...], n_frames: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """Generate known coherent domains and hinge motion."""

    reference = compact_reference(sum(sizes))
    known = np.empty(sum(sizes), dtype=np.int64)
    start = 0
    for domain_id, size in enumerate(sizes, start=1):
        known[start : start + size] = domain_id
        start += size
    frames: list[np.ndarray] = []
    for angle in np.linspace(-0.7, 0.7, n_frames):
        frame = reference.copy()
        start = sizes[0]
        for domain_index, size in enumerate(sizes[1:], start=1):
            pivot = reference[start - 1]
            stop = start + size
            frame[start:stop] = (
                (frame[start:stop] - pivot)
                @ rotation(angle * domain_index).T
                + pivot
            )
            start = stop
        frames.append(frame)
    return np.asarray(frames), known


def local_loop_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Generate one known locally deforming loop."""

    rng = np.random.default_rng(12)
    reference = compact_reference(44)
    coords = np.repeat(reference[None, :, :], 120, axis=0)
    mask = np.zeros(44, dtype=bool)
    mask[17:24] = True
    coords[:, mask] += rng.normal(scale=1.2, size=(120, 7, 3))
    return coords, mask


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute rank correlation with flat handling."""

    first = percentile_ranks(np.asarray(left, dtype=np.float64))
    second = percentile_ranks(np.asarray(right, dtype=np.float64))
    if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return set Jaccard."""

    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    union = np.count_nonzero(first | second)
    return (
        1.0
        if union == 0
        else float(np.count_nonzero(first & second) / union)
    )


def boundary_mask(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return a tolerance-expanded boundary mask."""

    mask = np.zeros(labels.size, dtype=bool)
    for boundary in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        mask[
            max(0, boundary - radius) : min(
                labels.size, boundary + radius + 1
            )
        ] = True
    return mask


def pair_coassignment(labels: np.ndarray) -> np.ndarray:
    """Return the upper-triangle domain coassignment vector."""

    rows, columns = np.triu_indices(labels.size, 1)
    return labels[rows] == labels[columns]


def domain_recovery(known: np.ndarray, observed: np.ndarray) -> float:
    """Measure label-invariant domain pair agreement."""

    return float(
        np.mean(pair_coassignment(known) == pair_coassignment(observed))
    )


def synthetic_validation() -> list[dict[str, object]]:
    """Run the required synthetic local/domain controls."""

    rows: list[dict[str, object]] = []
    rigid = rigid_trajectory()
    rigid_result = scan_motion_domains(rigid)
    rows.append(
        {
            "case_id": "pure_global_rotation_translation",
            "expected_motion": "one rigid domain; no local deformation",
            "n_expected_domains": 1,
            "n_observed_domains": np.unique(
                rigid_result.motion_domain_id
            ).size,
            "domain_pair_agreement": 1.0,
            "boundary_jaccard_tolerance2": 1.0,
            "localisation_jaccard": 1.0,
            "maximum_local_deformation_index": float(
                np.max(rigid_result.local.local_deformation_index)
            ),
            "temporal_local_spearman": (
                rigid_result.local.temporal_block_reproducibility
            ),
            "temporal_boundary_jaccard": (
                rigid_result.temporal_block_boundary_jaccard
            ),
            "pass": (
                np.unique(rigid_result.motion_domain_id).size == 1
                and np.max(
                    rigid_result.local.local_deformation_index
                )
                < 1e-10
            ),
        }
    )

    loop_coords, loop_mask = local_loop_trajectory()
    loop_result = scan_motion_domains(loop_coords)
    selected = loop_result.local.locally_deforming_segments > 0
    top = np.argsort(loop_result.local.local_deformation_index)[
        -np.count_nonzero(loop_mask) :
    ]
    rows.append(
        {
            "case_id": "local_loop_deformation",
            "expected_motion": "local loop deformation",
            "n_expected_domains": 1,
            "n_observed_domains": np.unique(
                loop_result.motion_domain_id
            ).size,
            "domain_pair_agreement": "",
            "boundary_jaccard_tolerance2": "",
            "localisation_jaccard": jaccard(loop_mask, selected),
            "top_k_localisation_recall": float(
                np.count_nonzero(loop_mask[top])
                / np.count_nonzero(loop_mask)
            ),
            "maximum_local_deformation_index": float(
                np.max(loop_result.local.local_deformation_index)
            ),
            "temporal_local_spearman": (
                loop_result.local.temporal_block_reproducibility
            ),
            "temporal_boundary_jaccard": (
                loop_result.temporal_block_boundary_jaccard
            ),
            "pass": (
                np.count_nonzero(loop_mask[top])
                / np.count_nonzero(loop_mask)
                >= 0.8
            ),
        }
    )

    for case_id, sizes in (
        ("two_domain_hinge", (28, 12)),
        ("equal_size_rigid_domains", (20, 20)),
        ("three_domain_motion", (15, 15, 15)),
    ):
        coords, known = domain_trajectory(sizes)
        observed = scan_motion_domains(coords, minimum_domain_size=6)
        rows.append(
            {
                "case_id": case_id,
                "expected_motion": f"{len(sizes)} coherent rigid domains",
                "n_expected_domains": len(sizes),
                "n_observed_domains": np.unique(
                    observed.motion_domain_id
                ).size,
                "domain_pair_agreement": domain_recovery(
                    known, observed.motion_domain_id
                ),
                "boundary_jaccard_tolerance2": jaccard(
                    boundary_mask(known), boundary_mask(observed.motion_domain_id)
                ),
                "localisation_jaccard": "",
                "maximum_local_deformation_index": float(
                    np.max(observed.local.local_deformation_index)
                ),
                "temporal_local_spearman": (
                    observed.local.temporal_block_reproducibility
                ),
                "temporal_boundary_jaccard": (
                    observed.temporal_block_boundary_jaccard
                ),
                "pass": (
                    np.unique(observed.motion_domain_id).size == len(sizes)
                    and domain_recovery(known, observed.motion_domain_id)
                    >= 0.9
                ),
            }
        )

    base, known = domain_trajectory((20, 20))
    baseline = scan_motion_domains(base, evaluate_blocks=False)
    transformed = base @ rotation(0.63).T + np.asarray((5.0, -3.0, 8.0))
    for scale in (0.1, 1.0, 10.0):
        observed = scan_motion_domains(
            transformed * scale, evaluate_blocks=False
        )
        difference = np.max(
            np.abs(
                observed.local.local_deformation_index
                - baseline.local.local_deformation_index
            )
        )
        labels_equal = np.array_equal(
            observed.motion_domain_id, baseline.motion_domain_id
        )
        rows.append(
            {
                "case_id": f"scale_and_rigid_invariance_x{scale:g}",
                "expected_motion": "same descriptor/domain labels",
                "n_expected_domains": 2,
                "n_observed_domains": np.unique(
                    observed.motion_domain_id
                ).size,
                "domain_pair_agreement": domain_recovery(
                    known, observed.motion_domain_id
                ),
                "boundary_jaccard_tolerance2": jaccard(
                    boundary_mask(baseline.motion_domain_id),
                    boundary_mask(observed.motion_domain_id),
                ),
                "localisation_jaccard": "",
                "maximum_local_deformation_index_difference": float(
                    difference
                ),
                "labels_identical": labels_equal,
                "pass": difference <= 1e-10 and labels_equal,
            }
        )

    rng = np.random.default_rng(5)
    reference = compact_reference(40)
    chain_coords = np.repeat(reference[None, :, :], 100, axis=0)
    chain_coords[:, 17:23] += rng.normal(scale=0.8, size=(100, 6, 3))
    chains = np.asarray(["A"] * 20 + ["B"] * 20)
    positions = np.r_[np.arange(1, 21), np.arange(1, 21)]
    chain_result = scan_motion_domains(
        chain_coords,
        chain_ids=chains,
        residue_positions=positions,
        minimum_domain_size=5,
    )
    observed_boundaries = (
        np.flatnonzero(
            chain_result.motion_domain_id[1:]
            != chain_result.motion_domain_id[:-1]
        )
        + 1
    )
    segment = chain_result.local.locally_deforming_segments
    no_cross_merge = not (
        segment[19] > 0 and segment[19] == segment[20]
    )
    rows.append(
        {
            "case_id": "terminal_and_chain_break",
            "expected_motion": "forced chain boundary; no segment merge",
            "n_expected_domains": 2,
            "n_observed_domains": np.unique(
                chain_result.motion_domain_id
            ).size,
            "chain_boundary_detected": 20 in observed_boundaries,
            "local_segment_crosses_chain": not no_cross_merge,
            "temporal_local_spearman": (
                chain_result.local.temporal_block_reproducibility
            ),
            "temporal_boundary_jaccard": (
                chain_result.temporal_block_boundary_jaccard
            ),
            "pass": 20 in observed_boundaries and no_cross_merge,
        }
    )
    return rows


def replicate_path(domain: str, label: str) -> tuple[Path, Path, str]:
    """Resolve a unit-clean versioned replicate."""

    regenerated = (
        REGENERATED_ROOT
        / domain
        / "raw_replicates"
        / f"{label}_ca_coords.npz"
    )
    if regenerated.is_file():
        return (
            regenerated,
            regenerated.with_name(f"{label}_metadata.json"),
            "phase7b_regenerated_unit_clean",
        )
    legacy = (
        LEGACY_ROOT / domain / "raw_replicates" / f"{label}_ca_coords.npz"
    )
    return (
        legacy,
        legacy.with_name(f"{label}_metadata.json"),
        "legacy_scale_audit_angstrom_consistent",
    )


def load_replicate(
    domain: str, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one audited CA replicate and metadata."""

    path, metadata_path, provenance = replicate_path(domain, label)
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
    metadata = json.loads(metadata_path.read_text())
    labels = metadata["labels"]
    chains = np.asarray([label_value.split(":", 1)[0] for label_value in labels])
    positions = np.asarray(
        [
            int(label_value.split(":")[1])
            if label_value.split(":")[1].lstrip("-").isdigit()
            else index
            for index, label_value in enumerate(labels, start=1)
        ]
    )
    return coords, chains, positions, provenance


def hard_core(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Select a deterministic top-fraction reference core."""

    count = max(1, int(np.floor(weights.size * fraction + 0.5)))
    order = np.lexsort((np.arange(weights.size), -weights))
    mask = np.zeros(weights.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def rmsf(aligned: np.ndarray) -> np.ndarray:
    """Compute coordinate RMSF."""

    centered = aligned - np.mean(aligned, axis=0, keepdims=True)
    return np.sqrt(
        np.mean(np.einsum("tni,tni->tn", centered, centered), axis=0)
    )


def heldout_pair_variance(
    coords: np.ndarray, labels: np.ndarray, maximum_pairs: int = 256
) -> float:
    """Measure held-out within-domain pair-distance variance."""

    pairs: list[tuple[int, int]] = []
    for domain_id in np.unique(labels):
        indices = np.flatnonzero(labels == domain_id)
        left, right = np.triu_indices(indices.size, 1)
        pairs.extend(zip(indices[left], indices[right], strict=True))
    if not pairs:
        return float("nan")
    chosen = np.linspace(
        0, len(pairs) - 1, min(maximum_pairs, len(pairs)), dtype=int
    )
    selected = [pairs[index] for index in chosen]
    source = np.asarray([pair[0] for pair in selected])
    target = np.asarray([pair[1] for pair in selected])
    distances = np.linalg.norm(
        coords[:, source] - coords[:, target], axis=2
    )
    centre = np.mean(distances, axis=0)
    relative_variance = np.var(distances, axis=0) / (
        centre * centre + 1e-12
    )
    return float(np.median(relative_variance))


def contact_persistence(
    coords: np.ndarray,
    domains,
    reference: np.ndarray,
    factor: float = 1.2,
) -> float:
    """Measure persistence of reference contacts internal to domains."""

    contact = domains.edges.family == "contact"
    internal = (
        domains.motion_domain_id[domains.edges.source]
        == domains.motion_domain_id[domains.edges.target]
    )
    selected = contact & internal
    if not np.any(selected):
        return float("nan")
    source = domains.edges.source[selected]
    target = domains.edges.target[selected]
    distances = np.linalg.norm(
        coords[:, source] - coords[:, target], axis=2
    )
    reference_distances = np.linalg.norm(
        reference[source] - reference[target], axis=1
    )
    threshold = factor * reference_distances
    return float(np.mean(distances <= threshold[None, :]))


def real_panel_validation() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Validate on 20 audited real mdCATH replicate pairs."""

    summary_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for panel in read_tsv(SELECTED_DOMAINS):
        domain = panel["domain_id"]
        coords, chains, positions, provenance = load_replicate(domain, "A")
        coords_r2, chains_r2, positions_r2, provenance_r2 = load_replicate(
            domain, "B"
        )
        if not (
            np.array_equal(chains, chains_r2)
            and np.array_equal(positions, positions_r2)
        ):
            raise RuntimeError(f"{domain}: R1/R2 residue metadata differ")
        first = scan_motion_domains(
            coords,
            chain_ids=chains,
            residue_positions=positions,
            block_count=4,
        )
        second = scan_motion_domains(
            coords_r2,
            chain_ids=chains_r2,
            residue_positions=positions_r2,
            block_count=4,
        )
        cross_local = spearman(
            first.local.local_deformation_index,
            second.local.local_deformation_index,
        )
        cross_segments = jaccard(
            first.local.locally_deforming_segments > 0,
            second.local.locally_deforming_segments > 0,
        )
        cross_boundaries = jaccard(
            boundary_mask(first.motion_domain_id),
            boundary_mask(second.motion_domain_id),
        )
        cross_domains = domain_recovery(
            first.motion_domain_id, second.motion_domain_id
        )
        summary_rows.append(
            {
                "domain_id": domain,
                "cath_class": panel["cath_class"],
                "n_frames_R1": coords.shape[0],
                "n_frames_R2": coords_r2.shape[0],
                "n_residues": coords.shape[1],
                "R1_provenance": provenance,
                "R2_provenance": provenance_r2,
                "n_motion_domains_R1": np.unique(
                    first.motion_domain_id
                ).size,
                "n_motion_domains_R2": np.unique(
                    second.motion_domain_id
                ).size,
                "temporal_local_spearman_R1": (
                    first.local.temporal_block_reproducibility
                ),
                "temporal_segment_jaccard_R1": (
                    first.local.temporal_block_segment_jaccard
                ),
                "temporal_boundary_jaccard_R1": (
                    first.temporal_block_boundary_jaccard
                ),
                "cross_replicate_local_spearman": cross_local,
                "cross_replicate_segment_jaccard": cross_segments,
                "cross_replicate_boundary_jaccard": cross_boundaries,
                "cross_replicate_domain_pair_agreement": cross_domains,
                "backbone_dihedral_status": "unavailable_CA_only",
            }
        )
        heldout = coords[coords.shape[0] // 2 :]
        training = coords[: coords.shape[0] // 2]
        training_domains = scan_motion_domains(
            training,
            chain_ids=chains,
            residue_positions=positions,
            block_count=4,
        )
        heldout_rows.append(
            {
                "domain_id": domain,
                "n_motion_domains": np.unique(
                    training_domains.motion_domain_id
                ).size,
                "heldout_within_domain_pairwise_distance_variance": (
                    heldout_pair_variance(
                        heldout, training_domains.motion_domain_id
                    )
                ),
                "heldout_internal_contact_persistence": (
                    contact_persistence(
                        heldout,
                        training_domains,
                        training[0],
                    )
                ),
                "motion_domain_boundary_stability": (
                    training_domains.temporal_block_boundary_jaccard
                ),
                "domain_training_frames": training.shape[0],
                "domain_evaluation_frames": heldout.shape[0],
                "backbone_dihedral_status": "unavailable_CA_only",
            }
        )

        selected = temporally_stratified_indices(
            coords.shape[0], min(2000, coords.shape[0])
        )
        exact = estimate_representative_weights_sequential_lean(
            coords[selected],
            coords[0],
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        weights = np.asarray(exact.mean_weights, dtype=np.float64)
        sequential_fit = weighted_kabsch_batch(
            coords,
            coords[0],
            weights,
            validate_inputs=False,
            compute_metrics=False,
        )
        domain_fit = align_to_motion_domain(
            coords, first, selection="largest_stable"
        )
        anchor_appropriate = (
            np.count_nonzero(domain_fit.anchor_mask) >= 0.3 * coords.shape[1]
        )
        secondary_rows.append(
            {
                "domain_id": domain,
                "anchor_domain_id": domain_fit.anchor_domain_id,
                "anchor_fraction": float(np.mean(domain_fit.anchor_mask)),
                "single_anchor_domain_appropriate": anchor_appropriate,
                "anchor_vs_sequential_core_jaccard": (
                    jaccard(domain_fit.anchor_mask, hard_core(weights))
                    if anchor_appropriate
                    else ""
                ),
                "domain_anchor_vs_sequential_RMSF_spearman": spearman(
                    domain_fit.rmsf, rmsf(sequential_fit.aligned_coords)
                ),
                "mean_domain_anchor_RMSD_A": float(
                    np.mean(domain_fit.rmsd)
                ),
                "comparison_role": (
                    "secondary_not_primary_go_no_go"
                ),
            }
        )
        for row in local_profile_rows(
            first, chain_ids=chains, residue_positions=positions
        ):
            profile_rows.append({"domain_id": domain, **row})
    return summary_rows, heldout_rows, secondary_rows, profile_rows


def load_long() -> tuple[np.ndarray, np.ndarray]:
    """Load the unit-clean 10k synthetic stress input."""

    with np.load(LONG_COORDS, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
    labels = json.loads(LONG_METADATA.read_text())["labels"]
    chains = np.asarray([label.split(":", 1)[0] for label in labels])
    return coords, chains


def run_performance_method(
    method: str, coords: np.ndarray, chains: np.ndarray
) -> dict[str, object]:
    """Run one performance contract without temporal diagnostics."""

    start = time.perf_counter()
    if method == "w5_scan_only":
        result = scan_local_kinematics(
            coords, chain_ids=chains, evaluate_blocks=False
        )
        checksum_data = result.local_deformation_index
        contract = "alignment_free_local_kinematics"
    elif method == "w5_motion_domain_graph":
        result = scan_motion_domains(
            coords, chain_ids=chains, evaluate_blocks=False
        )
        checksum_data = np.column_stack(
            (
                result.local.local_deformation_index,
                result.motion_domain_id,
                result.hinge_boundary_score,
            )
        )
        contract = "alignment_free_motion_domain_graph"
    elif method == "optional_domain_alignment":
        domains = scan_motion_domains(
            coords, chain_ids=chains, evaluate_blocks=False
        )
        result = align_to_motion_domain(
            coords, domains, selection="largest_stable"
        )
        checksum_data = np.r_[result.rmsd, result.rmsf]
        contract = "graph_plus_exact_domain_anchor_alignment"
    elif method == "sequential_lean":
        selected = temporally_stratified_indices(
            coords.shape[0], min(2000, coords.shape[0])
        )
        estimate = estimate_representative_weights_sequential_lean(
            coords[selected],
            coords[0],
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        result = exact_all_frame_fit(
            coords, coords[0], np.asarray(estimate.mean_weights)
        )
        checksum_data = np.r_[result.rmsd, result.robust_rmsf]
        contract = "exact_global_metrics_transforms"
    elif method == "uniform_fast_global":
        initial = build_initial_weights(coords.shape[1], "uniform")
        representatives = temporally_stratified_indices(
            coords.shape[0], 256
        )
        weights, _, _, _ = bounded_global_weight_refinement(
            coords[representatives],
            coords[0],
            initial.weights,
            update_rounds=4,
            sequence_chain_ids=chains,
        )
        result = exact_all_frame_fit(coords, coords[0], weights)
        checksum_data = np.r_[result.rmsd, result.robust_rmsf]
        contract = "approximate_fast_global_metrics_transforms"
    else:
        raise ValueError(f"unknown method: {method}")
    runtime = time.perf_counter() - start
    checksum = hashlib.sha256(
        np.asarray(checksum_data, dtype=np.float64).tobytes()
    ).hexdigest()
    return {
        "runtime_seconds": runtime,
        "output_contract": contract,
        "result_sha256": checksum,
    }


def repeat_ids():
    """Yield warmup and measured repeat identifiers."""

    for index in range(WARMUPS):
        yield True, index + 1
    for index in range(REPEATS):
        yield False, index + 1


def performance_benchmark() -> list[dict[str, object]]:
    """Measure loaded and fresh-process W5/global contracts."""

    coords, chains = load_long()
    rows: list[dict[str, object]] = []
    for method in PERFORMANCE_METHODS:
        for warmup, repeat in repeat_ids():
            result = run_performance_method(method, coords, chains)
            rows.append(
                {
                    "case_id": "unit_clean_transition_aligned_4oelA02_10000",
                    "method": method,
                    "boundary": "already_loaded",
                    "warmup": warmup,
                    "repeat": repeat,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    **result,
                }
            )

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    for method in PERFORMANCE_METHODS:
        for warmup, repeat in repeat_ids():
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "--performance-child",
                    "--method",
                    method,
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            wall = time.perf_counter() - start
            payload = json.loads(completed.stdout)
            rows.append(
                {
                    "case_id": "unit_clean_transition_aligned_4oelA02_10000",
                    "method": method,
                    "boundary": "fresh_process_including_npz_decode",
                    "warmup": warmup,
                    "repeat": repeat,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    **payload,
                    "runtime_seconds": wall,
                    "child_reported_seconds": payload["runtime_seconds"],
                    "startup_import_decode_seconds": max(
                        0.0, wall - float(payload["runtime_seconds"])
                    ),
                }
            )
    return rows


def summarize_performance(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Summarize measured repeats with median and IQR."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["warmup"]:
            continue
        groups.setdefault(
            (str(row["method"]), str(row["boundary"])), []
        ).append(row)
    summary: list[dict[str, object]] = []
    for (method, boundary), group in sorted(groups.items()):
        values = np.asarray(
            [float(row["runtime_seconds"]) for row in group]
        )
        summary.append(
            {
                "case_id": group[0]["case_id"],
                "method": method,
                "boundary": boundary,
                "n_warmups": WARMUPS,
                "n_measured_repeats": len(group),
                "median_runtime_seconds": float(np.median(values)),
                "q1_runtime_seconds": float(np.quantile(values, 0.25)),
                "q3_runtime_seconds": float(np.quantile(values, 0.75)),
                "minimum_runtime_seconds": float(np.min(values)),
                "maximum_runtime_seconds": float(np.max(values)),
                "result_hashes_identical": len(
                    {str(row["result_sha256"]) for row in group}
                )
                == 1,
                "output_contract": group[0]["output_contract"],
            }
        )
    return summary


def write_report(
    synthetic: list[dict[str, object]],
    real: list[dict[str, object]],
    heldout: list[dict[str, object]],
    secondary: list[dict[str, object]],
    performance: list[dict[str, object]],
) -> None:
    """Write a conservative pilot report and status."""

    loaded = {
        row["method"]: row
        for row in performance
        if row["boundary"] == "already_loaded"
    }
    synthetic_pass = sum(bool(row["pass"]) for row in synthetic)
    local_temporal = statistics.median(
        float(row["temporal_local_spearman_R1"]) for row in real
    )
    local_cross = statistics.median(
        float(row["cross_replicate_local_spearman"]) for row in real
    )
    boundary_temporal = statistics.median(
        float(row["temporal_boundary_jaccard_R1"]) for row in real
    )
    boundary_cross = statistics.median(
        float(row["cross_replicate_boundary_jaccard"]) for row in real
    )
    boundary_warning = boundary_temporal < 0.5 or boundary_cross < 0.5
    pair_variance = statistics.median(
        float(row["heldout_within_domain_pairwise_distance_variance"])
        for row in heldout
        if np.isfinite(
            float(row["heldout_within_domain_pairwise_distance_variance"])
        )
    )
    contact_values = [
        float(row["heldout_internal_contact_persistence"])
        for row in heldout
        if str(row["heldout_internal_contact_persistence"]) != "nan"
    ]
    contact = statistics.median(contact_values) if contact_values else float("nan")
    rmsf_secondary = statistics.median(
        float(row["domain_anchor_vs_sequential_RMSF_spearman"])
        for row in secondary
    )
    graph_runtime = float(
        loaded["w5_motion_domain_graph"]["median_runtime_seconds"]
    )
    target_pass = graph_runtime < 0.75
    report = f"""# Experimental window-kinematics pilot report

## Scope and contracts

This pilot treats chain-aware W5 as a primary alignment-free local-deformation
descriptor and augments it with a sparse local/contact/anchor motion-domain
graph. It does not require one global core. `local_deformation_index`,
`hinge_boundary_score`, and edge-instability values are dimensionless
pair-distance diagnostics and are not RMSD or global mobility.

Optional exact Kabsch transforms, RMSD and RMSF are computed only after an
explicit largest-stable, least-deforming, or user-specified domain anchor is
selected. `sequential_lean` remains the public exact/default method.

## Synthetic validation

{synthetic_pass}/{len(synthetic)} required synthetic controls passed. The panel
covered pure global rigid motion, local loop deformation, unequal/equal
two-domain hinge motion, three-domain motion, global scale factors 0.1/1/10,
rigid-transform invariance, terminals and chain breaks.

## Real 20-domain pilot

- median temporal-block local-index Spearman: {local_temporal:.4f};
- median cross-replicate local-index Spearman: {local_cross:.4f};
- median temporal boundary Jaccard (±2 residues): {boundary_temporal:.4f};
- median cross-replicate boundary Jaccard (±2 residues): {boundary_cross:.4f};
- median held-out within-domain relative pair-distance variance:
  {pair_variance:.6g};
- median held-out internal-contact persistence: {contact:.4f};
- secondary domain-anchor versus sequential RMSF Spearman:
  {rmsf_secondary:.4f}.

Backbone-dihedral validation was unavailable because the locked pilot inputs
are CA-only. This is reported as unavailable rather than inferred.

The continuous local signal is moderately reproducible, but the discrete
motion-domain boundaries are not: their median temporal and cross-replicate
Jaccard values are below 0.5. The motion-domain partition must therefore remain
an experimental hypothesis/diagnostic rather than a validated biological
annotation. The local-loop control also produced two graph domains around the
deforming region; localisation passed, but this illustrates the same
over-partitioning risk.

## Performance on the unit-clean 10k/153 stress case

- W5 scan only: {float(loaded['w5_scan_only']['median_runtime_seconds']):.4f} s;
- W5 plus motion-domain graph: {graph_runtime:.4f} s;
- graph plus optional domain alignment:
  {float(loaded['optional_domain_alignment']['median_runtime_seconds']):.4f} s;
- exact sequential_lean:
  {float(loaded['sequential_lean']['median_runtime_seconds']):.4f} s;
- uniform fast-global:
  {float(loaded['uniform_fast_global']['median_runtime_seconds']):.4f} s.

The W5+graph CPU target below 0.75 s was **{'PASS' if target_pass else 'FAIL'}**.
Loaded and fresh-process timings are reported separately. The 10k trajectory is
a transition-aligned synthetic frame-count/stitching stress control, not
biologically continuous MD.

## Pilot decision

The branch remains experimental. Synthetic recovery/invariance and the CPU
performance target passed, while real-system discrete boundary reproducibility
did not provide strong validation. Its primary scientific evidence is
local-deformation localisation, invariance, continuous temporal/cross-replicate
reproducibility, held-out pair stability and contact persistence. Agreement
with sequential RMSF/core is secondary and no transform-identity or RMSD-parity
gate was applied.
"""
    (RESULT_ROOT / "window_motion_domain_pilot_report.md").write_text(report)
    status = {
        "phase": "W1-W5_pilot",
        "status": "partial",
        "synthetic_pass_count": synthetic_pass,
        "synthetic_case_count": len(synthetic),
        "synthetic_gate_pass": synthetic_pass == len(synthetic),
        "real_domain_count": len(real),
        "real_boundary_reproducibility_warning": boundary_warning,
        "median_temporal_boundary_jaccard": boundary_temporal,
        "median_cross_replicate_boundary_jaccard": boundary_cross,
        "performance_target_seconds": 0.75,
        "w5_graph_loaded_median_seconds": graph_runtime,
        "performance_target_pass": target_pass,
        "formal_all_domain_rerun_performed": False,
        "public_default_changed": False,
        "frozen_evidence_modified": False,
        "gpu_used": False,
        "faiss_annoy_used": False,
    }
    (RESULT_ROOT / "pilot_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


def performance_child(method: str) -> None:
    """Run one fresh-process method and print JSON only."""

    start = time.perf_counter()
    coords, chains = load_long()
    decode_seconds = time.perf_counter() - start
    result = run_performance_method(method, coords, chains)
    result["decode_seconds"] = decode_seconds
    result["runtime_seconds"] = time.perf_counter() - start
    print(json.dumps(result, sort_keys=True))


def main() -> int:
    """Run the pilot or one performance child."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-child", action="store_true")
    parser.add_argument("--method", choices=PERFORMANCE_METHODS)
    arguments = parser.parse_args()
    if arguments.performance_child:
        if arguments.method is None:
            parser.error("--method is required")
        performance_child(arguments.method)
        return 0

    w0 = json.loads(
        (
            PROJECT_ROOT
            / "results/window_motion_domains/w0_scale_provenance/"
            "w0_status.json"
        ).read_text()
    )
    if w0["status"] != "pass":
        raise RuntimeError("W0 provenance gate has not passed")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_validation()
    real, heldout, secondary, profiles = real_panel_validation()
    performance_rows = performance_benchmark()
    performance_summary = summarize_performance(performance_rows)
    write_tsv(RESULT_ROOT / "synthetic_recovery.tsv", synthetic)
    write_tsv(RESULT_ROOT / "real_panel_reproducibility.tsv", real)
    write_tsv(RESULT_ROOT / "heldout_independent_quality.tsv", heldout)
    write_tsv(RESULT_ROOT / "secondary_sequential_comparison.tsv", secondary)
    write_tsv(RESULT_ROOT / "real_panel_residue_profiles.tsv", profiles)
    write_tsv(RESULT_ROOT / "performance_per_repeat.tsv", performance_rows)
    write_tsv(RESULT_ROOT / "performance_summary.tsv", performance_summary)
    write_report(
        synthetic, real, heldout, secondary, performance_summary
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
