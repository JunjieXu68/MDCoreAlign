#!/usr/bin/env python
"""Run Phase 6B window and sparse-stability-graph quality evaluation."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.flexibility import compute_rmsf, detect_flexible_segments
from mdsuperpose.representative_weights import estimate_representative_weights
from mdsuperpose.stability_graph import (
    StabilityGraphResult,
    build_reference_contact_edges,
    scan_window_graph,
)
from mdsuperpose.window_descriptors import edge_squared_distances
from mdsuperpose.window_scan import (
    WindowScanResult,
    scan_window_local,
    scan_window_multiscale,
)
from scripts.scaling_optimization.build_phase2_regression import (
    ROBUST_KWARGS,
    build_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v4/window_quality"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v4"
GOLDEN_MANIFEST = (
    PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
)
CORE_FRACTION = 0.6
SEED = 42


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a union-schema TSV."""

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


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a finite Spearman value with explicit constant handling."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0 if np.allclose(a, b, rtol=1e-10, atol=1e-12) else 0.0
    return float(spearmanr(a, b).statistic)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a finite Pearson value with explicit constant handling."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0 if np.allclose(a, b, rtol=1e-10, atol=1e-12) else 0.0
    return float(pearsonr(a, b).statistic)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jaccard overlap for boolean labels."""

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def binary_f1(left: np.ndarray, right: np.ndarray) -> float:
    """Return symmetric binary F1 overlap."""

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    denominator = np.count_nonzero(a) + np.count_nonzero(b)
    return (
        1.0
        if denominator == 0
        else float(2 * np.count_nonzero(a & b) / denominator)
    )


def exact_fit_metrics(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform an explicit global fit and return exact RMSD and RMSF."""

    aligned = np.empty_like(coords, dtype=np.float64)
    rmsd = np.empty(coords.shape[0], dtype=np.float64)
    for start in range(0, coords.shape[0], chunk_size):
        stop = min(start + chunk_size, coords.shape[0])
        fit = weighted_kabsch_batch(
            coords[start:stop],
            coords[0],
            weights,
            validate_inputs=False,
            compute_metrics=True,
        )
        aligned[start:stop] = fit.aligned_coords
        rmsd[start:stop] = fit.rmsd
    return rmsd, compute_rmsf(aligned)


def flexible_labels(weights: np.ndarray, chain: np.ndarray | None) -> np.ndarray:
    """Apply the existing flexible-segment contract to soft mobility."""

    chain_values = (
        np.asarray(chain)
        if chain is not None
        else np.asarray(["A"] * weights.size)
    )
    return (
        detect_flexible_segments(
            1.0 - np.clip(weights, 0.0, 1.0),
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=chain_values,
        )
        > 0
    )


def terminal_mask(n_residues: int, chain: np.ndarray | None) -> np.ndarray:
    """Mark two residues at each contiguous chain end."""

    chains = (
        np.asarray(chain)
        if chain is not None
        else np.asarray(["A"] * n_residues)
    )
    mask = np.zeros(n_residues, dtype=bool)
    start = 0
    while start < n_residues:
        stop = start + 1
        while stop < n_residues and chains[stop] == chains[start]:
            stop += 1
        mask[start : min(start + 2, stop)] = True
        mask[max(start, stop - 2) : stop] = True
        start = stop
    return mask


def pair_distance_stability(
    coords: np.ndarray,
    core: np.ndarray,
    *,
    maximum_pairs: int = 512,
) -> float:
    """Return held-out median relative distance MAD inside a candidate core."""

    indices = np.flatnonzero(core)
    if indices.size < 2:
        return float("nan")
    left, right = np.triu_indices(indices.size, k=1)
    left = indices[left]
    right = indices[right]
    if left.size > maximum_pairs:
        selected = np.linspace(0, left.size - 1, maximum_pairs, dtype=int)
        left = left[selected]
        right = right[selected]
    delta = coords[:, left] - coords[:, right]
    distances = np.sqrt(np.einsum("tpi,tpi->tp", delta, delta))
    centre = np.median(distances, axis=0)
    scale = 1.4826 * np.median(np.abs(distances - centre[None]), axis=0)
    return float(np.median(scale / (np.abs(centre) + 1e-12)))


def contact_persistence(
    coords: np.ndarray,
    core: np.ndarray,
    chain: np.ndarray | None,
) -> float:
    """Return persistence of reference contacts whose endpoints are in core."""

    edges = build_reference_contact_edges(
        coords[0],
        chain_ids=chain,
        max_contacts_per_residue=6,
    )
    if edges.n_edges == 0:
        return float("nan")
    keep = core[edges.source] & core[edges.target]
    if not np.any(keep):
        return float("nan")
    reference_d2 = np.sum(
        (coords[0, edges.source[keep]] - coords[0, edges.target[keep]]) ** 2,
        axis=1,
    )
    distances_d2 = edge_squared_distances(
        coords,
        type(edges)(
            source=edges.source[keep],
            target=edges.target[keep],
            family=edges.family[keep],
            n_residues=edges.n_residues,
            continuity_breaks=edges.continuity_breaks,
        ),
    )
    return float(np.mean(distances_d2 <= 1.2**2 * reference_d2[None]))


def load_golden_manifest() -> dict[str, dict[str, str]]:
    """Load frozen reference locations by case identifier."""

    with GOLDEN_MANIFEST.open(newline="") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["status"] == "success"
        }


def scan_method(
    method: str,
    coords: np.ndarray,
    chain: np.ndarray | None,
) -> tuple[WindowScanResult, dict[str, object]]:
    """Dispatch one experimental scan variant."""

    if method == "window_local_w5_minimal":
        result = scan_window_local(
            coords,
            feature_set="minimal",
            chain_ids=chain,
            statistic="winsorised_variance",
        )
        return result, {"risk_flag": False, "risk_reasons": ""}
    if method == "window_local_w5_full":
        result = scan_window_local(
            coords,
            feature_set="full",
            chain_ids=chain,
            statistic="winsorised_variance",
        )
        return result, {"risk_flag": False, "risk_reasons": ""}
    if method == "window_local_multiscale":
        result = scan_window_multiscale(
            coords,
            chain_ids=chain,
            statistic="winsorised_variance",
        )
        return result, {"risk_flag": False, "risk_reasons": ""}
    variants = {
        "window_graph_local_contact": "local_contact",
        "window_graph_local_anchor": "local_anchor",
        "window_graph_full": "full",
    }
    graph: StabilityGraphResult = scan_window_graph(
        coords,
        variant=variants[method],
        chain_ids=chain,
        statistic="winsorised_variance",
    )
    return graph.scan, {
        "risk_flag": graph.diagnostics.high_risk,
        "risk_reasons": ";".join(graph.diagnostics.risk_reasons),
        "ambiguity_score": graph.diagnostics.ambiguity_score,
        "component_sizes": ",".join(
            str(value) for value in np.sort(graph.diagnostics.component_sizes)[::-1]
        ),
        "local_graph_weight_spearman": (
            graph.diagnostics.local_graph_weight_spearman
        ),
        "n_anchors": graph.anchor_indices.size,
        "edge_family_counts": json.dumps(
            graph.edge_family_counts,
            sort_keys=True,
        ),
    }


WINDOW_METHODS = (
    "window_local_w5_minimal",
    "window_local_w5_full",
    "window_local_multiscale",
    "window_graph_local_contact",
    "window_graph_local_anchor",
    "window_graph_full",
)


def batched_reference(
    coords: np.ndarray,
    chain: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the Phase 3 independently initialised exact-fit comparison."""

    selected = np.unique(
        np.linspace(0, coords.shape[0] - 1, min(2000, coords.shape[0]), dtype=int)
    )
    estimate = estimate_representative_weights(
        coords[selected],
        coords[0],
        batch_size=min(512, selected.size),
        sequence_chain_ids=chain,
        **ROBUST_KWARGS,
    )
    weights = np.clip(estimate.mean_weights, 0.0, 1.0)
    rmsd, rmsf = exact_fit_metrics(coords, weights)
    core = select_core_by_target_fraction(weights, CORE_FRACTION).core_mask
    return weights, core, rmsd, rmsf


def quality_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate the complete 34-case panel without dropping failures."""

    golden_manifest = load_golden_manifest()
    rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    for case in build_cases():
        print(f"[phase6b] {case.case_id}", flush=True)
        case_start = time.perf_counter()
        try:
            coords, chain = case.loader()
            chain = np.asarray(chain) if chain is not None else None
            with np.load(
                PROJECT_ROOT / golden_manifest[case.case_id]["reference_npz"],
                allow_pickle=False,
            ) as golden:
                sequential_weights = np.asarray(
                    golden["legacy_long_frozen_weights"],
                    dtype=np.float64,
                )
                sequential_core = np.asarray(
                    golden["legacy_long_core_labels"],
                    dtype=bool,
                )
                sequential_rmsd = np.asarray(
                    golden["legacy_long_rmsd"],
                    dtype=np.float64,
                )
                sequential_rmsf = np.asarray(
                    golden["legacy_long_robust_rmsf"],
                    dtype=np.float64,
                )
                sequential_flexible = np.asarray(
                    golden["legacy_long_flexible_labels"],
                    dtype=bool,
                )
            batched_weights, batched_core, batched_rmsd, batched_rmsf = (
                batched_reference(coords, chain)
            )
            batched_flexible = flexible_labels(batched_weights, chain)
            rows.extend(
                [
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "method": "sequential_lean",
                        "status": "success",
                        "output_contract": "exact_reference",
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "core_jaccard_vs_sequential": 1.0,
                        "core_jaccard_vs_batched": jaccard(
                            sequential_core,
                            batched_core,
                        ),
                        "weight_spearman_vs_sequential": 1.0,
                        "weight_spearman_vs_batched": safe_spearman(
                            sequential_weights,
                            batched_weights,
                        ),
                        "weight_pearson_vs_sequential": 1.0,
                        "rmsf_spearman_vs_sequential": 1.0,
                        "rmsf_spearman_vs_batched": safe_spearman(
                            sequential_rmsf,
                            batched_rmsf,
                        ),
                        "flexible_jaccard_vs_sequential": 1.0,
                        "flexible_jaccard_vs_batched": jaccard(
                            sequential_flexible,
                            batched_flexible,
                        ),
                        "flexible_f1_vs_sequential": 1.0,
                        "risk_flag": False,
                        "cross_replicate_status": (
                            "unavailable_no_matching_local_replicate"
                        ),
                    },
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "method": "batched_independent",
                        "status": "success",
                        "output_contract": "exact_independent_reference",
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "core_jaccard_vs_sequential": jaccard(
                            batched_core,
                            sequential_core,
                        ),
                        "core_jaccard_vs_batched": 1.0,
                        "weight_spearman_vs_sequential": safe_spearman(
                            batched_weights,
                            sequential_weights,
                        ),
                        "weight_spearman_vs_batched": 1.0,
                        "weight_pearson_vs_sequential": safe_pearson(
                            batched_weights,
                            sequential_weights,
                        ),
                        "rmsf_spearman_vs_sequential": safe_spearman(
                            batched_rmsf,
                            sequential_rmsf,
                        ),
                        "rmsf_spearman_vs_batched": 1.0,
                        "flexible_jaccard_vs_sequential": jaccard(
                            batched_flexible,
                            sequential_flexible,
                        ),
                        "flexible_jaccard_vs_batched": 1.0,
                        "flexible_f1_vs_sequential": binary_f1(
                            batched_flexible,
                            sequential_flexible,
                        ),
                        "risk_flag": False,
                        "cross_replicate_status": (
                            "unavailable_no_matching_local_replicate"
                        ),
                    },
                ]
            )
            for method in WINDOW_METHODS:
                scan_start = time.perf_counter()
                scan, diagnostics = scan_method(method, coords, chain)
                scan_seconds = time.perf_counter() - scan_start
                weights = scan.stability_weights
                core = scan.candidate_stable_label
                fit_start = time.perf_counter()
                rmsd, rmsf = exact_fit_metrics(coords, weights)
                validation_fit_seconds = time.perf_counter() - fit_start
                flexible = flexible_labels(weights, chain)
                terminal = terminal_mask(coords.shape[1], chain)
                half = max(2, coords.shape[0] // 2)
                first_scan, _ = scan_method(method, coords[:half], chain)
                second_scan, _ = scan_method(method, coords[-half:], chain)
                block_core_jaccard = jaccard(
                    first_scan.candidate_stable_label,
                    second_scan.candidate_stable_label,
                )
                block_weight_spearman = safe_spearman(
                    first_scan.stability_weights,
                    second_scan.stability_weights,
                )
                risk_reasons = [
                    value
                    for value in str(diagnostics.get("risk_reasons", "")).split(";")
                    if value
                ]
                if block_core_jaccard < 0.65:
                    risk_reasons.append("temporal_block_instability")
                risk_flag = bool(diagnostics.get("risk_flag")) or bool(risk_reasons)
                row = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "source": case.source,
                    "method": method,
                    "status": "success",
                    "output_contract": (
                        "scan_only_plus_explicit_exact_validation_fit"
                    ),
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "scan_seconds": scan_seconds,
                    "validation_fit_seconds": validation_fit_seconds,
                    "n_edges": scan.edges.n_edges,
                    "core_jaccard_vs_sequential": jaccard(
                        core,
                        sequential_core,
                    ),
                    "core_jaccard_vs_batched": jaccard(core, batched_core),
                    "weight_spearman_vs_sequential": safe_spearman(
                        weights,
                        sequential_weights,
                    ),
                    "weight_spearman_vs_batched": safe_spearman(
                        weights,
                        batched_weights,
                    ),
                    "weight_pearson_vs_sequential": safe_pearson(
                        weights,
                        sequential_weights,
                    ),
                    "rmsf_spearman_vs_sequential": safe_spearman(
                        rmsf,
                        sequential_rmsf,
                    ),
                    "rmsf_spearman_vs_batched": safe_spearman(
                        rmsf,
                        batched_rmsf,
                    ),
                    "movement_vs_sequential_rmsf_spearman": safe_spearman(
                        scan.movement_score,
                        sequential_rmsf,
                    ),
                    "flexible_jaccard_vs_sequential": jaccard(
                        flexible,
                        sequential_flexible,
                    ),
                    "flexible_jaccard_vs_batched": jaccard(
                        flexible,
                        batched_flexible,
                    ),
                    "flexible_f1_vs_sequential": binary_f1(
                        flexible,
                        sequential_flexible,
                    ),
                    "flexible_segment_interval_overlap": jaccard(
                        flexible,
                        sequential_flexible,
                    ),
                    "median_rmsd_abs_difference_vs_sequential": float(
                        np.median(np.abs(rmsd - sequential_rmsd))
                    ),
                    "terminal_weight_mae_vs_sequential": float(
                        np.mean(np.abs(weights[terminal] - sequential_weights[terminal]))
                    ),
                    "nonterminal_weight_mae_vs_sequential": float(
                        np.mean(
                            np.abs(
                                weights[~terminal] - sequential_weights[~terminal]
                            )
                        )
                    ),
                    "heldout_pair_distance_variance": pair_distance_stability(
                        coords[coords.shape[0] // 2 :],
                        core,
                    ),
                    "contact_persistence": contact_persistence(
                        coords[coords.shape[0] // 2 :],
                        core,
                        chain,
                    ),
                    "cross_block_core_jaccard": block_core_jaccard,
                    "cross_block_weight_spearman": block_weight_spearman,
                    "cross_replicate_status": (
                        "unavailable_no_matching_local_replicate"
                    ),
                    "risk_flag": risk_flag,
                    "risk_reasons": ";".join(sorted(set(risk_reasons))),
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if key not in {"risk_flag", "risk_reasons"}
                    },
                }
                rows.append(row)
                ambiguity_rows.append(
                    {
                        "case_id": case.case_id,
                        "method": method,
                        "risk_flag": risk_flag,
                        "risk_reasons": row["risk_reasons"],
                        "ambiguity_score": diagnostics.get(
                            "ambiguity_score",
                            "",
                        ),
                        "component_sizes": diagnostics.get(
                            "component_sizes",
                            "",
                        ),
                        "cross_block_core_jaccard": block_core_jaccard,
                        "cross_block_weight_spearman": block_weight_spearman,
                        "core_jaccard_vs_sequential": row[
                            "core_jaccard_vs_sequential"
                        ],
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "method": "case_load_or_evaluation",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - case_start,
                }
            )
    return rows, ambiguity_rows


def bootstrap_median_ci(values: list[float]) -> tuple[float, float]:
    """Return deterministic 95% bootstrap CI for a median."""

    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    medians = np.empty(2000, dtype=np.float64)
    for index in range(medians.size):
        medians[index] = np.median(rng.choice(data, size=data.size, replace=True))
    lower, upper = np.quantile(medians, (0.025, 0.975))
    return float(lower), float(upper)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize all methods with distributions and continuation gates."""

    output: list[dict[str, object]] = []
    for method in ("sequential_lean", "batched_independent", *WINDOW_METHODS):
        selected = [
            row
            for row in rows
            if row.get("method") == method and row.get("status") == "success"
        ]
        if not selected:
            continue
        metric_fields = (
            "core_jaccard_vs_sequential",
            "core_jaccard_vs_batched",
            "weight_spearman_vs_sequential",
            "weight_spearman_vs_batched",
            "rmsf_spearman_vs_sequential",
            "rmsf_spearman_vs_batched",
            "flexible_jaccard_vs_sequential",
            "flexible_jaccard_vs_batched",
        )
        row: dict[str, object] = {
            "method": method,
            "n_cases": len(selected),
            "n_failures": sum(
                candidate.get("method") == method
                and candidate.get("status") != "success"
                for candidate in rows
            ),
        }
        for field in metric_fields:
            values = [float(candidate[field]) for candidate in selected]
            lower, upper = bootstrap_median_ci(values)
            row[f"median_{field}"] = statistics.median(values)
            row[f"mean_{field}"] = statistics.mean(values)
            row[f"min_{field}"] = min(values)
            row[f"bootstrap95_median_low_{field}"] = lower
            row[f"bootstrap95_median_high_{field}"] = upper
        poor_unflagged = [
            candidate
            for candidate in selected
            if float(candidate["core_jaccard_vs_sequential"]) < 0.65
            and not bool(candidate.get("risk_flag"))
        ]
        row["poor_core_unflagged_fraction"] = len(poor_unflagged) / len(selected)
        if method.startswith("window_"):
            gate = (
                min(
                    float(row["median_core_jaccard_vs_sequential"]),
                    float(row["median_core_jaccard_vs_batched"]),
                )
                >= 0.85
                and min(
                    float(row["median_weight_spearman_vs_sequential"]),
                    float(row["median_weight_spearman_vs_batched"]),
                )
                >= 0.90
                and min(
                    float(row["median_rmsf_spearman_vs_sequential"]),
                    float(row["median_rmsf_spearman_vs_batched"]),
                )
                >= 0.90
                and min(
                    float(row["median_flexible_jaccard_vs_sequential"]),
                    float(row["median_flexible_jaccard_vs_batched"]),
                )
                >= 0.80
                and float(row["poor_core_unflagged_fraction"]) <= 0.10
            )
            row["phase6b_quality_gate_without_control_checks"] = gate
        output.append(row)
    return output


def hinge_controls() -> list[dict[str, object]]:
    """Evaluate local-loop, hinge, equal-domain, and seam controls."""

    rng = np.random.default_rng(SEED)

    def domains(first_size: int, second_size: int) -> np.ndarray:
        first = rng.normal(scale=2.0, size=(first_size, 3))
        second = (
            rng.normal(scale=2.0, size=(second_size, 3))
            + np.asarray((8.0, 0.0, 0.0))
        )
        reference = np.vstack((first, second))
        frames = []
        for angle in np.linspace(0.0, 1.0, 100):
            rotation = np.asarray(
                (
                    (np.cos(angle), -np.sin(angle), 0.0),
                    (np.sin(angle), np.cos(angle), 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
            frame = reference.copy()
            frame[first_size:] = (
                frame[first_size:] - np.asarray((4.0, 0.0, 0.0))
            ) @ rotation.T + np.asarray((4.0, 0.0, 0.0))
            frames.append(frame)
        return np.asarray(frames)

    loop_reference = rng.normal(size=(40, 3))
    loop = np.repeat(loop_reference[None], 100, axis=0)
    loop[:, 17:23] += rng.normal(scale=0.8, size=(100, 6, 3))
    seam = np.concatenate((loop[:50], loop[:50]), axis=0)
    cases = {
        "local_loop": loop,
        "hinge_unequal_domains": domains(28, 12),
        "hinge_equal_domains": domains(20, 20),
        "transition_aligned_seam_control": seam,
    }
    rows: list[dict[str, object]] = []
    for case_id, coords in cases.items():
        for method in (
            "window_local_w5_minimal",
            "window_graph_local_contact",
            "window_graph_local_anchor",
            "window_graph_full",
        ):
            scan, diagnostics = scan_method(method, coords, None)
            boundary = 28 if "unequal" in case_id else 20
            outside = (
                np.r_[0 : max(0, boundary - 4), boundary + 4 : coords.shape[1]]
                if "hinge" in case_id
                else np.arange(coords.shape[1])
            )
            distributed_signal = float(np.mean(scan.movement_percentile[outside]))
            expected_limitation = (
                "hinge" in case_id and method == "window_local_w5_minimal"
            )
            detected = (
                bool(diagnostics.get("risk_flag"))
                if "hinge" in case_id and method.startswith("window_graph")
                else distributed_signal > 0.2
            )
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "distributed_movement_signal": distributed_signal,
                    "risk_flag": diagnostics.get("risk_flag", False),
                    "risk_reasons": diagnostics.get("risk_reasons", ""),
                    "ambiguity_score": diagnostics.get("ambiguity_score", ""),
                    "expected_local_window_limitation": expected_limitation,
                    "domain_or_local_motion_detected": detected,
                    "status": (
                        "expected_limitation"
                        if expected_limitation
                        else ("pass" if detected else "fail")
                    ),
                }
            )
    return rows


def write_reports(
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    controls: list[dict[str, object]],
) -> bool:
    """Write Phase 6B quality and sparse-graph interpretation reports."""

    candidates = [
        row
        for row in summary
        if row["method"].startswith("window_")
        and row.get("phase6b_quality_gate_without_control_checks")
    ]
    graph_hinge = [
        row
        for row in controls
        if "hinge" in row["case_id"]
        and row["method"] == "window_graph_full"
    ]
    hinge_pass = bool(graph_hinge) and all(
        bool(row["domain_or_local_motion_detected"]) for row in graph_hinge
    )
    timing = json.loads(
        (
            PROJECT_ROOT
            / "results/scaling_optimization_v4/window_local/phase6a_summary.json"
        ).read_text()
    )
    speed_pass = timing["winsorised_variance_10k_median_seconds"] < 1.5
    three_pio = [
        row
        for row in rows
        if row.get("case_id") == "mdcath_3pio100"
        and row.get("method", "").startswith("window_")
    ]
    three_pio_handled = any(
        float(row["core_jaccard_vs_sequential"]) >= 0.65 or bool(row["risk_flag"])
        for row in three_pio
    )
    continue_gate = bool(candidates) and hinge_pass and speed_pass and three_pio_handled
    best = max(
        [row for row in summary if row["method"].startswith("window_")],
        key=lambda row: (
            min(
                float(row["median_core_jaccard_vs_sequential"]),
                float(row["median_core_jaccard_vs_batched"]),
            ),
            min(
                float(row["median_weight_spearman_vs_sequential"]),
                float(row["median_weight_spearman_vs_batched"]),
            ),
        ),
    )
    quality_report = f"""# Phase 6B window quality report

## Panel

All 34 frozen cases were evaluated: four formal MD systems, 20 mdCATH
development domains including class 4/6 and slow/difficult examples, five real
500/501-frame replicates, order-sensitive/repeated-segment cases, and the 10k
transition-aligned synthetic scaling case. No failed row was excluded.
Cross-replicate reproducibility is unavailable because the local panel contains
only R1 for each real replicate; this limitation is explicit in the table.

## Best observed alignment-free candidate

- Method: `{best["method"]}`.
- Median core Jaccard versus sequential reference:
  {float(best["median_core_jaccard_vs_sequential"]):.4f}.
- Median core Jaccard versus batched-independent reference:
  {float(best["median_core_jaccard_vs_batched"]):.4f}.
- Median weight Spearman versus sequential:
  {float(best["median_weight_spearman_vs_sequential"]):.4f}.
- Median exact-fit RMSF Spearman versus sequential:
  {float(best["median_rmsf_spearman_vs_sequential"]):.4f}.
- Median flexible-label Jaccard versus sequential:
  {float(best["median_flexible_jaccard_vs_sequential"]):.4f}.

RMSF values in `case_metrics.tsv` were calculated only after an explicit exact
global fit with the scan-derived weights. The raw movement index is not RMSD.
The table also reports means, minima and deterministic bootstrap 95% intervals,
not only medians.

## Continuation decision

- Candidate passing numerical quality thresholds: {bool(candidates)}.
- Graph detects synthetic hinge/domain motion: {hinge_pass}.
- 10k scan-only remains below 1.5 s: {speed_pass}.
- `mdcath_3pio100` acceptable or risk-flagged: {three_pio_handled}.
- Phase 6B decision: **{"continue to Phase 6C" if continue_gate else "stop before Phase 6C"}**.
"""
    graph_report = f"""# Phase 6B sparse stability graph report

The direct W5 scan detects internal neighbourhood changes but is intentionally
blind to internally rigid domain motion away from a hinge. The synthetic hinge
controls retain this result as an expected theoretical limitation.

The sparse graph adds bounded non-local reference contacts and 12 deterministic
farthest-point anchors. Edge count remains O(kN); no `(T, N, N)` allocation is
used. Non-local edges detect relative-domain change, but pair distances are
symmetric and cannot always decide which equal-sized domain should define the
alignment core. Equal-component cases therefore produce an ambiguity/risk flag
instead of an arbitrary scientific claim.

The graph continuation check was {"passed" if hinge_pass else "not passed"}.
Detailed component sizes, local/graph disagreement, temporal-block agreement
and risk reasons are in `ambiguity_diagnostics.tsv`.
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "01_window_quality_report.md").write_text(quality_report)
    (DOC_ROOT / "02_sparse_stability_graph_report.md").write_text(graph_report)
    pio_rows = [
        row for row in three_pio if row.get("status") == "success"
    ]
    pio_report = "# mdcath_3pio100 window report\n\n"
    for row in pio_rows:
        pio_report += (
            f"- `{row['method']}`: core Jaccard "
            f"{float(row['core_jaccard_vs_sequential']):.4f}, weight Spearman "
            f"{float(row['weight_spearman_vs_sequential']):.4f}, risk "
            f"`{row['risk_flag']}` ({row.get('risk_reasons', '')}).\n"
        )
    (DOC_ROOT / "cases").mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "cases/mdcath_3pio100_window_report.md").write_text(pio_report)
    (RESULT_ROOT / "phase6b_summary.json").write_text(
        json.dumps(
            {
                "continue_to_phase6c": continue_gate,
                "candidate_methods_passing": [
                    row["method"] for row in candidates
                ],
                "hinge_gate": hinge_pass,
                "speed_gate": speed_pass,
                "mdcath_3pio100_gate": three_pio_handled,
                "best_alignment_free_method": best["method"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return continue_gate


def main() -> int:
    """Run Phase 6B in the canonical environment."""

    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use {PYTHON}; got {sys.executable}")
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "2":
            raise RuntimeError(f"{variable}=2 is required")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, ambiguity = quality_rows()
    write_tsv(RESULT_ROOT / "case_metrics.tsv", rows)
    write_tsv(RESULT_ROOT / "ambiguity_diagnostics.tsv", ambiguity)
    summary = summarize(rows)
    write_tsv(RESULT_ROOT / "summary_metrics.tsv", summary)
    controls = hinge_controls()
    write_tsv(RESULT_ROOT / "hinge_controls.tsv", controls)
    continue_gate = write_reports(rows, summary, controls)
    return 0 if continue_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
