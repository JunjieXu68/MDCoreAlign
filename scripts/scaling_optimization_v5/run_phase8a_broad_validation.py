#!/usr/bin/env python
"""Run the locked Phase 8A 512-domain unit-clean validation gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from mdsuperpose.coordinate_scale import audit_coordinate_scale
from mdsuperpose.flexibility import detect_flexible_segments
from mdsuperpose.representative_weights import (
    estimate_representative_weights,
)
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import (
    compute_frozen_window_signals,
    temporally_stratified_indices,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization.build_phase2_regression import ROBUST_KWARGS
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes
from scripts.scaling_optimization_v4.run_phase6b import (
    contact_persistence,
    pair_distance_stability,
)
from scripts.scaling_optimization_v4.run_phase6c_r import make_initial
from scripts.scaling_optimization_v4.run_phase7a_rmsd_audit import (
    MethodTransforms,
    apply_transforms_independent,
    fit_transforms,
    hard_core,
    independent_evaluation,
    jaccard,
    locked_candidates,
    normalized_weights,
    regression,
    rotation_angles,
    run_locked_candidate,
    safe_spearman,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT / "results/scaling_optimization_v5/phase8a_broad_validation"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/all_domain_canonical/aggregate/"
    "all_domain_canonical_per_domain_summary.tsv"
)
PHASE7B_AUDIT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity/"
    "all_trajectory_scale_audit.tsv"
)
DEV_PANEL = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "selected_domains.tsv"
)
PANEL_SIZE = 512
CLASS_QUOTAS = {
    "1": 120,
    "2": 105,
    "3": 190,
    "4": 45,
    "6": 32,
    "NA": 20,
}
METHOD_ORDER = (
    "sequential_lean_exact",
    "uniform_low_budget",
    "w5_preconditioned",
    "graph_preconditioned",
    "w5_scan_only",
    "batched_independent",
)
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stable_hash(value: str) -> str:
    """Return a deterministic selection key."""

    return hashlib.sha256(value.encode()).hexdigest()


def h5_path_from_notes(notes: str) -> Path:
    """Extract the retained local HDF5 path."""

    match = re.search(r"h5_file=([^;]+)", str(notes))
    if not match:
        raise ValueError("missing h5_file in all-domain notes")
    return Path(match.group(1))


def length_bin(n_residues: int) -> str:
    """Return a fixed residue-count stratum."""

    if n_residues <= 75:
        return "short_30_75"
    if n_residues <= 150:
        return "medium_76_150"
    return "long_gt150"


def mobility_bin(value: float) -> str:
    """Return a fixed historical clean-panel mobility stratum."""

    if value < 2.27:
        return "low"
    if value < 3.78:
        return "medium"
    return "high"


def select_locked_panel() -> list[dict[str, object]]:
    """Select a deterministic class/length/mobility/speed panel."""

    frozen_bad = {
        row["trajectory_id"].split("__")[0]
        for row in read_tsv(PHASE7B_AUDIT)
        if row["scope"] == "frozen_all_domain"
        and row["status"] != "angstrom_consistent"
    }
    development = {row["domain_id"] for row in read_tsv(DEV_PANEL)}
    candidates: list[dict[str, object]] = []
    for row in read_tsv(SUMMARY_PATH):
        if row["domain_id"] in frozen_bad | development:
            continue
        if row["mdcorealign_status"] != "success":
            continue
        n_residues = int(row["n_residues"])
        mobility = float(row["noncore_core_RMSF_ratio"])
        runtime_ratio = float(
            row["runtime_ratio_mdcorealign_over_mdlovofit"]
        )
        cath_class = row["cath_class"]
        if cath_class not in CLASS_QUOTAS:
            cath_class = "NA"
        h5_path = h5_path_from_notes(row["notes"])
        candidates.append(
            {
                "domain_id": row["domain_id"],
                "trajectory_id": f"{row['domain_id']}__T320__R1",
                "cath_class": cath_class,
                "cath_architecture": row["cath_architecture"],
                "cath_topology": row["cath_topology"],
                "cath_homologous_superfamily": row[
                    "cath_homologous_superfamily"
                ],
                "n_frames_expected": int(row["n_frames"]),
                "n_residues_expected": n_residues,
                "length_bin": length_bin(n_residues),
                "mobility_bin": mobility_bin(mobility),
                "speed_bin": "mdca_faster"
                if runtime_ratio < 1.0
                else "mdca_slower",
                "historical_noncore_core_rmsf_ratio": mobility,
                "historical_runtime_ratio": runtime_ratio,
                "h5_path": str(h5_path),
                "coordinate_path": f"/{row['domain_id']}/320/0/coords",
                "selection_key": stable_hash(row["domain_id"]),
                "excluded_from_development_panel": True,
                "historical_scale_audit_status": "angstrom_consistent",
            }
        )

    selected: list[dict[str, object]] = []
    for cath_class, quota in CLASS_QUOTAS.items():
        class_rows = [
            row for row in candidates if row["cath_class"] == cath_class
        ]
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in class_rows:
            key = (
                str(row["length_bin"]),
                str(row["mobility_bin"]),
                str(row["speed_bin"]),
            )
            groups.setdefault(key, []).append(row)
        for group in groups.values():
            group.sort(key=lambda row: str(row["selection_key"]))
        group_keys = sorted(groups)
        class_selected: list[dict[str, object]] = []
        while len(class_selected) < quota and group_keys:
            next_keys: list[tuple[str, str, str]] = []
            for key in group_keys:
                if groups[key] and len(class_selected) < quota:
                    row = groups[key].pop(0)
                    row["selection_stratum"] = "|".join(key)
                    class_selected.append(row)
                if groups[key]:
                    next_keys.append(key)
            group_keys = next_keys
        if len(class_selected) != quota:
            raise RuntimeError(
                f"class {cath_class} supplied {len(class_selected)}/{quota}"
            )
        selected.extend(class_selected)
    if len(selected) != PANEL_SIZE:
        raise RuntimeError(
            f"locked panel has {len(selected)} instead of {PANEL_SIZE}"
        )
    for panel_index, row in enumerate(selected, start=1):
        row["panel_index"] = panel_index
        row["cross_replicate_selected"] = panel_index % 8 == 0
    return selected


def lock_manifest() -> list[dict[str, object]]:
    """Create or validate the pre-analysis panel lock."""

    manifest = RESULT_ROOT / "broad_panel_manifest.tsv"
    lock_path = RESULT_ROOT / "broad_panel_lock.json"
    if manifest.is_file() and lock_path.is_file():
        lock = json.loads(lock_path.read_text())
        if sha256(manifest) != lock["manifest_sha256"]:
            raise RuntimeError("broad panel manifest hash changed")
        return read_tsv(manifest)
    rows = select_locked_panel()
    write_tsv(manifest, rows)
    lock = {
        "status": "locked_before_validation",
        "panel_size": len(rows),
        "manifest_sha256": sha256(manifest),
        "class_quotas": CLASS_QUOTAS,
        "excluded_development_domains": 20,
        "excluded_historical_unit_failures": 163,
        "temperature_K": 320,
        "replicate": "R1",
        "algorithms": METHOD_ORDER,
        "candidate_lock": (
            "results/scaling_optimization_v4/window_refine/"
            "locked_candidates.json"
        ),
        "post_validation_tuning_allowed": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return rows


def labels_to_arrays(
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return chain and residue-ID arrays."""

    chains: list[str] = []
    residues: list[str] = []
    for index, label in enumerate(labels, start=1):
        parts = label.split(":")
        chains.append(parts[0] if len(parts) > 1 else "_")
        residues.append(parts[1] if len(parts) > 2 else str(index))
    return np.asarray(chains), np.asarray(residues)


def flexible_labels(weights: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """Apply the frozen mobility-to-segment contract."""

    return (
        detect_flexible_segments(
            np.clip(1.0 - normalized_weights(weights), 0.0, 1.0),
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=chain,
        )
        > 0
    )


def exact_method(
    coords: np.ndarray, chain: np.ndarray
) -> tuple[MethodTransforms, float]:
    """Run sequential_lean and exact all-frame fitting."""

    start = time.perf_counter()
    selected = temporally_stratified_indices(
        coords.shape[0], min(2000, coords.shape[0])
    )
    estimate = estimate_representative_weights_sequential_lean(
        coords[selected],
        coords[0],
        sequence_chain_ids=chain,
        **ROBUST_KWARGS,
    )
    weights = np.asarray(estimate.mean_weights, dtype=np.float64)
    rotations, translations = fit_transforms(coords, coords[0], weights)
    return (
        MethodTransforms(
            method="sequential_lean_exact",
            weights=weights,
            rotations=rotations,
            translations=translations,
            source="phase8a_fresh_unit_clean",
        ),
        time.perf_counter() - start,
    )


def batched_method(
    coords: np.ndarray, chain: np.ndarray
) -> tuple[MethodTransforms, float]:
    """Run the independently initialized batched reference."""

    start = time.perf_counter()
    selected = temporally_stratified_indices(
        coords.shape[0], min(2000, coords.shape[0])
    )
    estimate = estimate_representative_weights(
        coords[selected],
        coords[0],
        batch_size=min(512, selected.size),
        sequence_chain_ids=chain,
        **ROBUST_KWARGS,
    )
    weights = np.asarray(estimate.mean_weights, dtype=np.float64)
    rotations, translations = fit_transforms(coords, coords[0], weights)
    return (
        MethodTransforms(
            method="batched_independent",
            weights=weights,
            rotations=rotations,
            translations=translations,
            source="phase8a_fresh_unit_clean",
        ),
        time.perf_counter() - start,
    )


def method_weight_on_block(
    method_name: str,
    coords: np.ndarray,
    chain: np.ndarray,
    candidate_by_method: dict[str, object],
) -> np.ndarray:
    """Estimate block-specific weights for temporal reproducibility."""

    if method_name == "sequential_lean_exact":
        selected = temporally_stratified_indices(
            coords.shape[0], min(2000, coords.shape[0])
        )
        result = estimate_representative_weights_sequential_lean(
            coords[selected],
            coords[0],
            sequence_chain_ids=chain,
            **ROBUST_KWARGS,
        )
        return np.asarray(result.mean_weights, dtype=np.float64)
    signals = compute_frozen_window_signals(coords, chain_ids=chain)
    result = run_locked_candidate(
        coords, chain, candidate_by_method[method_name], signals
    )
    return result.weights


def temporal_reproducibility(
    method_name: str,
    coords: np.ndarray,
    chain: np.ndarray,
    candidate_by_method: dict[str, object],
) -> tuple[float, float]:
    """Compare independently estimated first/second temporal halves."""

    middle = max(2, coords.shape[0] // 2)
    first = method_weight_on_block(
        method_name, coords[:middle], chain, candidate_by_method
    )
    second = method_weight_on_block(
        method_name, coords[middle:], chain, candidate_by_method
    )
    return (
        safe_spearman(first, second),
        jaccard(hard_core(first), hard_core(second)),
    )


def evaluate_case(
    panel_row: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract, guard and evaluate one locked domain."""

    extract_start = time.perf_counter()
    coords, labels, residue_metadata, metadata = extract_ca_trajectory(
        Path(panel_row["h5_path"]),
        coordinate_path=panel_row["coordinate_path"],
        min_ca_atoms=30,
        min_frames=20,
    )
    extraction_seconds = time.perf_counter() - extract_start
    chain, residue_ids = labels_to_arrays(labels)
    scale_start = time.perf_counter()
    scale, _ = audit_coordinate_scale(
        coords, chain_ids=chain, residue_ids=residue_ids
    )
    scale_seconds = time.perf_counter() - scale_start
    if (
        metadata["unit_status"] != "verified_angstrom_metadata"
        or scale.status != "angstrom_consistent"
    ):
        raise RuntimeError(
            f"unit guard failed: {metadata['unit_status']}/{scale.status}"
        )

    exact, exact_runtime = exact_method(coords, chain)
    exact_core = hard_core(exact.weights)
    exact_eval = independent_evaluation(
        coords,
        coords[0],
        exact,
        exact_core,
        exact_core,
        None,
    )
    exact_flexible = flexible_labels(exact.weights, chain)
    candidates = locked_candidates()
    candidate_by_method = {}
    for candidate in candidates:
        from scripts.scaling_optimization_v4.run_phase7a_rmsd_audit import (
            candidate_method_name,
        )

        candidate_by_method[candidate_method_name(candidate)] = candidate

    signal_start = time.perf_counter()
    signals = compute_frozen_window_signals(coords, chain_ids=chain)
    signal_seconds = time.perf_counter() - signal_start
    methods: list[tuple[MethodTransforms, float, float]] = [
        (exact, exact_runtime, 0.0)
    ]
    for method_name in (
        "uniform_low_budget",
        "w5_preconditioned",
        "graph_preconditioned",
    ):
        start = time.perf_counter()
        method = run_locked_candidate(
            coords, chain, candidate_by_method[method_name], signals
        )
        solver_seconds = time.perf_counter() - start
        standalone_scan = (
            0.0 if method_name == "uniform_low_budget" else signal_seconds
        )
        methods.append(
            (method, solver_seconds + standalone_scan, standalone_scan)
        )
    batched, batched_runtime = batched_method(coords, chain)
    methods.append((batched, batched_runtime, 0.0))

    scan_candidate = candidate_by_method["w5_preconditioned"]
    scan_start = time.perf_counter()
    scan_initial = make_initial(
        scan_candidate, coords.shape[1], signals
    )
    scan_build_seconds = time.perf_counter() - scan_start
    scan_weights = np.asarray(scan_initial.weights, dtype=np.float64)

    temporal: dict[str, tuple[float, float]] = {}
    for method_name in (
        "sequential_lean_exact",
        "uniform_low_budget",
        "w5_preconditioned",
    ):
        temporal[method_name] = temporal_reproducibility(
            method_name, coords, chain, candidate_by_method
        )

    cross_replicate: dict[str, tuple[float, float]] = {}
    if str(panel_row["cross_replicate_selected"]).lower() == "true":
        coords_r2, labels_r2, _, metadata_r2 = extract_ca_trajectory(
            Path(panel_row["h5_path"]),
            coordinate_path=(
                f"/{panel_row['domain_id']}/320/1/coords"
            ),
            min_ca_atoms=30,
            min_frames=20,
        )
        if labels_r2 != labels:
            raise RuntimeError("R1/R2 residue labels differ")
        scale_r2, _ = audit_coordinate_scale(
            coords_r2, chain_ids=chain, residue_ids=residue_ids
        )
        if (
            metadata_r2["unit_status"] != "verified_angstrom_metadata"
            or scale_r2.status != "angstrom_consistent"
        ):
            raise RuntimeError("R2 unit guard failed")
        r2_exact, _ = exact_method(coords_r2, chain)
        r2_signals = compute_frozen_window_signals(
            coords_r2, chain_ids=chain
        )
        r2_uniform = run_locked_candidate(
            coords_r2,
            chain,
            candidate_by_method["uniform_low_budget"],
            r2_signals,
        )
        r2_w5 = run_locked_candidate(
            coords_r2,
            chain,
            candidate_by_method["w5_preconditioned"],
            r2_signals,
        )
        first_by_name = {
            method.method: method
            for method, _, _ in methods
            if method.method
            in {
                "sequential_lean_exact",
                "uniform_low_budget",
                "w5_preconditioned",
            }
        }
        second_by_name = {
            "sequential_lean_exact": r2_exact,
            "uniform_low_budget": r2_uniform,
            "w5_preconditioned": r2_w5,
        }
        for method_name in second_by_name:
            cross_replicate[method_name] = (
                safe_spearman(
                    first_by_name[method_name].weights,
                    second_by_name[method_name].weights,
                ),
                jaccard(
                    hard_core(first_by_name[method_name].weights),
                    hard_core(second_by_name[method_name].weights),
                ),
            )

    rows: list[dict[str, object]] = []
    independent_rows: list[dict[str, object]] = []
    heldout = coords[max(1, coords.shape[0] // 2) :]
    exact_pair = pair_distance_stability(heldout, exact_core)
    exact_contact = contact_persistence(heldout, exact_core, chain)
    for method, runtime, scan_runtime in methods:
        core = hard_core(method.weights)
        evaluation = (
            exact_eval
            if method.method == "sequential_lean_exact"
            else independent_evaluation(
                coords,
                coords[0],
                method,
                exact_core,
                core,
                None,
            )
        )
        flex = flexible_labels(method.weights, chain)
        rmsd_difference = np.abs(
            evaluation.full_rmsd - exact_eval.full_rmsd
        )
        rmsf_correlation, rmsf_slope, rmsf_intercept = regression(
            exact_eval.rmsf, evaluation.rmsf
        )
        angle = rotation_angles(method.rotations, exact.rotations)
        translation = np.linalg.norm(
            method.translations - exact.translations, axis=1
        )
        aligned_displacement = np.sqrt(
            np.mean(
                np.sum(
                    (evaluation.aligned - exact_eval.aligned) ** 2,
                    axis=2,
                ),
                axis=1,
            )
        )
        pair_value = pair_distance_stability(heldout, core)
        contact_value = contact_persistence(heldout, core, chain)
        temporal_value = temporal.get(method.method)
        replicate_value = cross_replicate.get(method.method)
        row = {
            **panel_row,
            "status": "success",
            "method": method.method,
            "source_unit_status": metadata["unit_status"],
            "coordinate_scale_status": scale.status,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "extraction_seconds": extraction_seconds,
            "scale_audit_seconds": scale_seconds,
            "algorithm_runtime_seconds": runtime,
            "scan_runtime_seconds": scan_runtime,
            "peak_input_coordinate_mb": coords.nbytes / (1024 * 1024),
            "core_jaccard_vs_exact": jaccard(core, exact_core),
            "flexible_label_jaccard_vs_exact": jaccard(
                flex, exact_flexible
            ),
            "weight_spearman_vs_exact": safe_spearman(
                normalized_weights(method.weights),
                normalized_weights(exact.weights),
            ),
            "weight_mae_vs_exact": float(
                np.mean(
                    np.abs(
                        normalized_weights(method.weights)
                        - normalized_weights(exact.weights)
                    )
                )
            ),
            "effective_core_size": float(
                np.sum(normalized_weights(method.weights))
            ),
            "effective_core_size_difference_vs_exact": float(
                np.sum(normalized_weights(method.weights))
                - np.sum(normalized_weights(exact.weights))
            ),
            "rmsf_spearman_vs_exact": safe_spearman(
                evaluation.rmsf, exact_eval.rmsf
            ),
            "rmsf_mae_vs_exact_A": float(
                np.mean(np.abs(evaluation.rmsf - exact_eval.rmsf))
            ),
            "rmsf_calibration_correlation": rmsf_correlation,
            "rmsf_calibration_slope": rmsf_slope,
            "rmsf_calibration_intercept_A": rmsf_intercept,
            "median_absolute_rmsd_difference_vs_exact_A": float(
                np.median(rmsd_difference)
            ),
            "p95_absolute_rmsd_difference_vs_exact_A": float(
                np.quantile(rmsd_difference, 0.95)
            ),
            "mean_unified_posthoc_rmsd_A": float(
                np.mean(evaluation.full_rmsd)
            ),
            "rotation_geodesic_median_degrees": float(np.median(angle)),
            "rotation_geodesic_p95_degrees": float(
                np.quantile(angle, 0.95)
            ),
            "translation_difference_median_A": float(
                np.median(translation)
            ),
            "transform_coordinate_displacement_median_A": float(
                np.median(aligned_displacement)
            ),
            "heldout_core_pairwise_distance_variance": pair_value,
            "heldout_pair_variance_delta_vs_exact": pair_value - exact_pair,
            "heldout_core_contact_persistence": contact_value,
            "heldout_contact_persistence_delta_vs_exact": (
                contact_value - exact_contact
            ),
            "temporal_block_weight_spearman": (
                temporal_value[0]
                if temporal_value is not None
                else float("nan")
            ),
            "temporal_block_core_jaccard": (
                temporal_value[1]
                if temporal_value is not None
                else float("nan")
            ),
            "cross_replicate_weight_spearman": (
                replicate_value[0]
                if replicate_value is not None
                else float("nan")
            ),
            "cross_replicate_core_jaccard": (
                replicate_value[1]
                if replicate_value is not None
                else float("nan")
            ),
            "backbone_dihedral_status": (
                "unavailable_CA_only_formal_validation"
            ),
            "hinge_annotation_status": (
                "unavailable_no_locked_domain_annotation"
            ),
        }
        rows.append(row)
        independent_rows.append(
            {
                "domain_id": panel_row["domain_id"],
                "method": method.method,
                "heldout_core_pairwise_distance_variance": pair_value,
                "heldout_pair_variance_delta_vs_exact": pair_value
                - exact_pair,
                "heldout_core_contact_persistence": contact_value,
                "heldout_contact_persistence_delta_vs_exact": contact_value
                - exact_contact,
                "temporal_block_weight_spearman": row[
                    "temporal_block_weight_spearman"
                ],
                "temporal_block_core_jaccard": row[
                    "temporal_block_core_jaccard"
                ],
                "cross_replicate_weight_spearman": row[
                    "cross_replicate_weight_spearman"
                ],
                "cross_replicate_core_jaccard": row[
                    "cross_replicate_core_jaccard"
                ],
                "backbone_dihedral_status": row[
                    "backbone_dihedral_status"
                ],
                "hinge_annotation_status": row[
                    "hinge_annotation_status"
                ],
            }
        )

    scan_core = hard_core(scan_weights)
    scan_flexible = flexible_labels(scan_weights, chain)
    scan_pair = pair_distance_stability(heldout, scan_core)
    scan_contact = contact_persistence(heldout, scan_core, chain)
    rows.append(
        {
            **panel_row,
            "status": "success_local_diagnostic_only",
            "method": "w5_scan_only",
            "source_unit_status": metadata["unit_status"],
            "coordinate_scale_status": scale.status,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "extraction_seconds": extraction_seconds,
            "scale_audit_seconds": scale_seconds,
            "algorithm_runtime_seconds": signal_seconds
            + scan_build_seconds,
            "scan_runtime_seconds": signal_seconds + scan_build_seconds,
            "peak_input_coordinate_mb": coords.nbytes / (1024 * 1024),
            "core_jaccard_vs_exact": jaccard(scan_core, exact_core),
            "flexible_label_jaccard_vs_exact": jaccard(
                scan_flexible, exact_flexible
            ),
            "weight_spearman_vs_exact": safe_spearman(
                normalized_weights(scan_weights),
                normalized_weights(exact.weights),
            ),
            "weight_mae_vs_exact": float(
                np.mean(
                    np.abs(
                        normalized_weights(scan_weights)
                        - normalized_weights(exact.weights)
                    )
                )
            ),
            "effective_core_size": float(
                np.sum(normalized_weights(scan_weights))
            ),
            "effective_core_size_difference_vs_exact": float(
                np.sum(normalized_weights(scan_weights))
                - np.sum(normalized_weights(exact.weights))
            ),
            "rmsf_spearman_vs_exact": float("nan"),
            "rmsf_mae_vs_exact_A": float("nan"),
            "median_absolute_rmsd_difference_vs_exact_A": float("nan"),
            "p95_absolute_rmsd_difference_vs_exact_A": float("nan"),
            "mean_unified_posthoc_rmsd_A": float("nan"),
            "rotation_geodesic_median_degrees": float("nan"),
            "translation_difference_median_A": float("nan"),
            "transform_coordinate_displacement_median_A": float("nan"),
            "heldout_core_pairwise_distance_variance": scan_pair,
            "heldout_pair_variance_delta_vs_exact": scan_pair - exact_pair,
            "heldout_core_contact_persistence": scan_contact,
            "heldout_contact_persistence_delta_vs_exact": scan_contact
            - exact_contact,
            "backbone_dihedral_status": (
                "unavailable_CA_only_formal_validation"
            ),
            "hinge_annotation_status": (
                "unavailable_no_locked_domain_annotation"
            ),
            "output_contract": (
                "alignment_free_local_deformation_not_RMSD"
            ),
        }
    )
    return rows, independent_rows


def aggregate(
    rows: list[dict[str, object]], method: str
) -> dict[str, object]:
    """Aggregate a global-fit method across systems."""

    selected = [
        row
        for row in rows
        if row["method"] == method and row["status"] == "success"
    ]

    def values(field: str) -> np.ndarray:
        return np.asarray(
            [float(row[field]) for row in selected], dtype=np.float64
        )

    core = values("core_jaccard_vs_exact")
    weight = values("weight_spearman_vs_exact")
    rmsf = values("rmsf_spearman_vs_exact")
    rmsd_median = values(
        "median_absolute_rmsd_difference_vs_exact_A"
    )
    rmsd_p95 = values("p95_absolute_rmsd_difference_vs_exact_A")
    pair_delta = values("heldout_pair_variance_delta_vs_exact")
    contact_delta = values("heldout_contact_persistence_delta_vs_exact")
    return {
        "method": method,
        "n_systems": len(selected),
        "failure_count": PANEL_SIZE - len(selected),
        "median_core_jaccard": float(np.median(core)),
        "p10_core_jaccard": float(np.quantile(core, 0.10)),
        "p05_core_jaccard": float(np.quantile(core, 0.05)),
        "minimum_core_jaccard": float(np.min(core)),
        "catastrophic_core_count_lt_0_50": int(np.count_nonzero(core < 0.5)),
        "median_weight_spearman": float(np.median(weight)),
        "p10_weight_spearman": float(np.quantile(weight, 0.10)),
        "median_rmsf_spearman": float(np.median(rmsf)),
        "p10_rmsf_spearman": float(np.quantile(rmsf, 0.10)),
        "median_system_median_abs_rmsd_difference_A": float(
            np.median(rmsd_median)
        ),
        "p95_system_median_abs_rmsd_difference_A": float(
            np.quantile(rmsd_median, 0.95)
        ),
        "median_system_p95_abs_rmsd_difference_A": float(
            np.median(rmsd_p95)
        ),
        "p95_system_p95_abs_rmsd_difference_A": float(
            np.quantile(rmsd_p95, 0.95)
        ),
        "median_algorithm_runtime_seconds": float(
            np.median(values("algorithm_runtime_seconds"))
        ),
        "p90_algorithm_runtime_seconds": float(
            np.quantile(values("algorithm_runtime_seconds"), 0.90)
        ),
        "median_heldout_pair_variance_delta": float(
            np.median(pair_delta)
        ),
        "median_heldout_contact_persistence_delta": float(
            np.median(contact_delta)
        ),
    }


def gate(summary: dict[str, object]) -> bool:
    """Apply the locked provisional fast-mode engineering gate."""

    return bool(
        summary["n_systems"] == PANEL_SIZE
        and summary["median_core_jaccard"] >= 0.95
        and summary["p10_core_jaccard"] >= 0.85
        and summary["median_weight_spearman"] >= 0.98
        and summary["median_rmsf_spearman"] >= 0.99
        and summary[
            "median_system_median_abs_rmsd_difference_A"
        ]
        <= 0.05
        and summary["p95_system_p95_abs_rmsd_difference_A"] <= 0.20
        and summary["catastrophic_core_count_lt_0_50"] == 0
        and summary["median_heldout_pair_variance_delta"] <= 0.0
        and summary["median_heldout_contact_persistence_delta"] >= -0.01
    )


def w5_incremental_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare W5 directly with equal-budget uniform per system."""

    index = {
        (str(row["domain_id"]), str(row["method"])): row for row in rows
    }
    output: list[dict[str, object]] = []
    domains = sorted(
        {
            str(row["domain_id"])
            for row in rows
            if row["method"] == "w5_preconditioned"
        }
    )
    for domain_id in domains:
        w5 = index[(domain_id, "w5_preconditioned")]
        uniform = index[(domain_id, "uniform_low_budget")]
        output.append(
            {
                "domain_id": domain_id,
                "cath_class": w5["cath_class"],
                "length_bin": w5["length_bin"],
                "mobility_bin": w5["mobility_bin"],
                "core_jaccard_improvement_w5_minus_uniform": float(
                    w5["core_jaccard_vs_exact"]
                )
                - float(uniform["core_jaccard_vs_exact"]),
                "weight_spearman_improvement_w5_minus_uniform": float(
                    w5["weight_spearman_vs_exact"]
                )
                - float(uniform["weight_spearman_vs_exact"]),
                "rmsf_spearman_improvement_w5_minus_uniform": float(
                    w5["rmsf_spearman_vs_exact"]
                )
                - float(uniform["rmsf_spearman_vs_exact"]),
                "rmsd_median_improvement_uniform_minus_w5_A": float(
                    uniform[
                        "median_absolute_rmsd_difference_vs_exact_A"
                    ]
                )
                - float(
                    w5[
                        "median_absolute_rmsd_difference_vs_exact_A"
                    ]
                ),
                "heldout_pair_variance_improvement_uniform_minus_w5": float(
                    uniform["heldout_pair_variance_delta_vs_exact"]
                )
                - float(w5["heldout_pair_variance_delta_vs_exact"]),
                "heldout_contact_improvement_w5_minus_uniform": float(
                    w5["heldout_contact_persistence_delta_vs_exact"]
                )
                - float(
                    uniform[
                        "heldout_contact_persistence_delta_vs_exact"
                    ]
                ),
                "runtime_penalty_w5_over_uniform": float(
                    w5["algorithm_runtime_seconds"]
                )
                / max(float(uniform["algorithm_runtime_seconds"]), 1e-15),
                "uniform_failure_rescued_by_w5": int(
                    float(uniform["core_jaccard_vs_exact"]) < 0.85
                    and float(w5["core_jaccard_vs_exact"]) >= 0.85
                ),
            }
        )
    return output


def write_failure_catalogue(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    """Write worst systems and explicit processing failures."""

    approximate = [
        row
        for row in rows
        if row["method"]
        in {
            "uniform_low_budget",
            "w5_preconditioned",
            "graph_preconditioned",
        }
    ]
    worst = sorted(
        approximate,
        key=lambda row: (
            float(row["core_jaccard_vs_exact"]),
            -float(row["p95_absolute_rmsd_difference_vs_exact_A"]),
        ),
    )[:10]
    lines = [
        "# Phase 8A failure and worst-case catalogue",
        "",
        f"Processing failures: {len(failures)}.",
        "",
    ]
    for failure in failures:
        lines.append(
            f"- `{failure['domain_id']}`: {failure['failure_reason']}"
        )
    lines.extend(["", "## Worst ten method-system rows", ""])
    for row in worst:
        lines.append(
            f"- `{row['domain_id']}` / `{row['method']}`: core Jaccard "
            f"{float(row['core_jaccard_vs_exact']):.4f}, system P95 RMSD "
            f"difference "
            f"{float(row['p95_absolute_rmsd_difference_vs_exact_A']):.4f} A."
        )
    (RESULT_ROOT / "failure_case_catalogue.md").write_text(
        "\n".join(lines) + "\n"
    )


def write_decision(
    summaries: list[dict[str, object]],
    incremental: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    """Write the locked Phase 8A contract decision."""

    by_method = {row["method"]: row for row in summaries}
    uniform_pass = gate(by_method["uniform_low_budget"])
    w5_pass = gate(by_method["w5_preconditioned"])
    improvements = np.asarray(
        [
            float(row["core_jaccard_improvement_w5_minus_uniform"])
            for row in incremental
        ]
    )
    penalties = np.asarray(
        [float(row["runtime_penalty_w5_over_uniform"]) for row in incremental]
    )
    rescues = sum(
        int(row["uniform_failure_rescued_by_w5"]) for row in incremental
    )
    w5_material_benefit = bool(
        rescues > 0 or np.quantile(improvements, 0.10) >= 0.01
    )
    stage2 = bool(uniform_pass and w5_pass and not failures)
    text = f"""# Phase 8A contract decision

The locked 512-domain gate used fresh 320 K R1 extraction from explicit
Angstrom mdCATH sources. No Phase 6C-R parameter was tuned after panel locking.

- Uniform fast-mode gate: **{'PASS' if uniform_pass else 'FAIL'}**.
- W5-preconditioned gate: **{'PASS' if w5_pass else 'FAIL'}**.
- Processing failures: {len(failures)}.
- Median W5-minus-uniform core-Jaccard change:
  {float(np.median(improvements)):.6f}.
- 10th-percentile W5-minus-uniform core-Jaccard change:
  {float(np.quantile(improvements, 0.10)):.6f}.
- Uniform failures rescued by W5: {rescues}.
- Median standalone W5 runtime penalty over uniform:
  {float(np.median(penalties)):.3f}x.

W5 shows {'a locked global-path benefit' if w5_material_benefit else 'no material locked global-path benefit'} over equal-budget uniform refinement under the predefined rule.
It remains available as an alignment-free local-deformation diagnostic.

Stage-2 all-domain expansion decision: **{'GO' if stage2 else 'NO-GO'}**.
The expansion is not run when the 512-domain gate fails or has unresolved
processing failures.
"""
    (RESULT_ROOT / "phase8a_contract_decision.md").write_text(text)
    (RESULT_ROOT / "phase8a_status.json").write_text(
        json.dumps(
            {
                "uniform_gate_pass": uniform_pass,
                "w5_gate_pass": w5_pass,
                "processing_failure_count": len(failures),
                "w5_material_global_benefit": w5_material_benefit,
                "stage2_all_domain_go": stage2,
                "manifest_sha256": sha256(
                    RESULT_ROOT / "broad_panel_manifest.tsv"
                ),
                "post_validation_tuning_performed": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    """Run the locked 512-domain Phase 8A gate."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = verify_hashes()
    if before["status"] != "pass":
        raise RuntimeError("frozen hash integrity failed before Phase 8A")
    panel = lock_manifest()
    result_path = RESULT_ROOT / "broad_validation_per_system.tsv"
    independent_path = RESULT_ROOT / "independent_scientific_quality.tsv"
    failure_path = RESULT_ROOT / "processing_failures.tsv"
    rows = read_tsv(result_path) if result_path.is_file() else []
    independent = (
        read_tsv(independent_path) if independent_path.is_file() else []
    )
    failures = read_tsv(failure_path) if failure_path.is_file() else []
    completed = {
        row["domain_id"]
        for row in rows
        if row["method"] == "w5_scan_only"
    } | {row["domain_id"] for row in failures}
    for position, panel_row in enumerate(panel, start=1):
        domain_id = panel_row["domain_id"]
        if domain_id in completed:
            continue
        print(
            f"[phase8a] {position}/{len(panel)} {domain_id}",
            flush=True,
        )
        try:
            case_rows, quality_rows = evaluate_case(panel_row)
            rows.extend(case_rows)
            independent.extend(quality_rows)
        except Exception as error:
            failures.append(
                {
                    "domain_id": domain_id,
                    "panel_index": panel_row["panel_index"],
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
        write_tsv(result_path, rows)
        write_tsv(independent_path, independent)
        write_tsv(failure_path, failures)

    summaries = [
        aggregate(rows, method)
        for method in (
            "sequential_lean_exact",
            "uniform_low_budget",
            "w5_preconditioned",
            "graph_preconditioned",
            "batched_independent",
        )
    ]
    for summary in summaries:
        summary["engineering_gate_pass"] = (
            True
            if summary["method"] == "sequential_lean_exact"
            else gate(summary)
        )
    write_tsv(RESULT_ROOT / "broad_validation_summary.tsv", summaries)
    incremental = w5_incremental_rows(rows)
    write_tsv(
        RESULT_ROOT / "w5_incremental_value_over_uniform.tsv",
        incremental,
    )
    write_failure_catalogue(rows, failures)
    write_decision(summaries, incremental, failures)
    after = verify_hashes()
    (RESULT_ROOT / "frozen_hash_integrity.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2) + "\n"
    )
    if after["status"] != "pass":
        raise RuntimeError("frozen hash integrity failed after Phase 8A")
    print((RESULT_ROOT / "phase8a_status.json").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.update(THREAD_ENVIRONMENT)
    raise SystemExit(main())
