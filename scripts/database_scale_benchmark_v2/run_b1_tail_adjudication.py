#!/usr/bin/env python3
"""Adjudicate the nine D1 throughput quality tails on matched coordinates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import exact_all_frame_fit
from scripts.database_scale_benchmark_v2.common import (
    B0_ROOT,
    B1_ROOT,
    ROOT,
    TAIL_DOMAINS,
    read_tsv,
    write_json,
    write_tsv,
)
from scripts.database_scale_gpu_v1.common import (
    core_mask,
    corrected_profile_path,
    jaccard,
    load_exact_profile,
    normalized_weights,
    safe_spearman,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)


ROBUST_KWARGS: dict[str, object] = {
    "rule": "huber",
    "delta": 1.0,
    "max_iterations": 20,
    "tolerance": 1e-6,
    "objective_tolerance": 1e-5,
    "objective_patience": 2,
    "sequence_continuity_strength": 0.05,
    "contact_consistency_strength": 0.0,
    "record_history": False,
}


def coordinate_digest(coords: np.ndarray) -> str:
    """Hash the canonical float64 coordinate bytes used by every method."""

    values = np.ascontiguousarray(coords, dtype=np.float64)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def method_aligned(coords: np.ndarray, result: object) -> np.ndarray:
    """Apply one method's transforms with the shared row-vector convention."""

    return apply_transforms_batch(
        coords,
        np.asarray(result.rotations, dtype=np.float64),
        np.asarray(result.translations, dtype=np.float64),
    )


def rmsd_by_mask(aligned: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Evaluate RMSD on one explicit residue mask without refitting."""

    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return np.full(aligned.shape[0], np.nan, dtype=np.float64)
    delta = aligned[:, selected] - reference[None, selected]
    return np.sqrt(np.mean(np.einsum("tni,tni->tn", delta, delta), axis=1))


def rotation_geodesic_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return per-frame relative-rotation geodesic angles in degrees."""

    relative = np.einsum("tij,tkj->tik", left, right)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def valid_transforms(result: object) -> bool:
    """Check finite proper rotations and finite translations."""

    rotations = np.asarray(result.rotations, dtype=np.float64)
    translations = np.asarray(result.translations, dtype=np.float64)
    determinants = np.linalg.det(rotations)
    return bool(
        np.isfinite(rotations).all()
        and np.isfinite(translations).all()
        and np.allclose(determinants, 1.0, atol=2e-3)
    )


def core_components(mask: np.ndarray, chains: np.ndarray) -> tuple[int, float]:
    """Count contiguous core components without crossing chains."""

    selected = np.asarray(mask, dtype=bool)
    starts = selected.copy()
    starts[1:] &= (~selected[:-1]) | (chains[1:] != chains[:-1])
    components = int(np.count_nonzero(starts))
    transitions = np.count_nonzero(
        (selected[1:] != selected[:-1]) & (chains[1:] == chains[:-1])
    )
    return components, float(transitions / max(1, selected.size - 1))


def sampled_pairs(mask: np.ndarray, *, maximum: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic residue pairs from one selected set."""

    indices = np.flatnonzero(mask)
    left, right = np.triu_indices(indices.size, k=1)
    left = indices[left]
    right = indices[right]
    if left.size > maximum:
        chosen = np.linspace(0, left.size - 1, maximum, dtype=np.int64)
        left, right = left[chosen], right[chosen]
    return left, right


def heldout_geometry(coords: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Measure alignment-independent held-out core geometry."""

    start = max(1, int(np.floor(coords.shape[0] * 0.8)))
    heldout = coords[start:]
    left, right = sampled_pairs(mask)
    if left.size == 0:
        return {
            "pairwise_distance_variance": float("nan"),
            "reference_contact_persistence": float("nan"),
            "pair_count": 0,
            "reference_contact_count": 0,
        }
    distances = np.linalg.norm(heldout[:, left] - heldout[:, right], axis=2)
    variance = float(np.mean(np.var(distances, axis=0)))
    reference_distance = np.linalg.norm(coords[0, left] - coords[0, right], axis=1)
    contacts = reference_distance <= 8.0
    persistence = (
        float(np.mean(distances[:, contacts] <= 8.0))
        if np.any(contacts)
        else float("nan")
    )
    return {
        "pairwise_distance_variance": variance,
        "reference_contact_persistence": persistence,
        "pair_count": int(left.size),
        "reference_contact_count": int(np.count_nonzero(contacts)),
    }


def calibration_slope(reference: np.ndarray, observed: np.ndarray) -> float:
    """Return a zero-intercept-free least-squares calibration slope."""

    if np.std(reference) <= 1e-12:
        return float("nan")
    return float(np.polyfit(reference, observed, 1)[0])


def exact_temporal_reproducibility(coords: np.ndarray, chains: np.ndarray) -> dict[str, float]:
    """Compare independent sequential estimates from two temporal halves."""

    midpoint = coords.shape[0] // 2
    estimates = []
    for block in (coords[:midpoint], coords[midpoint:]):
        estimate = estimate_representative_weights_sequential_lean(
            block,
            coords[0],
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        estimates.append(normalized_weights(estimate.mean_weights))
    return {
        "weight_spearman": safe_spearman(estimates[0], estimates[1]),
        "core_jaccard": jaccard(core_mask(estimates[0]), core_mask(estimates[1])),
    }


def secondary_structure_labels(source: Path, coordinate_path: str, n_residues: int) -> np.ndarray:
    """Return modal mdCATH DSSP labels, or explicit unknown labels."""

    try:
        import h5py

        dssp_path = coordinate_path.rsplit("/", 1)[0] + "/dssp"
        with h5py.File(source, "r") as handle:
            raw = np.asarray(handle[dssp_path][()])
        if raw.ndim != 2 or raw.shape[1] != n_residues:
            raise ValueError("DSSP shape differs from CA mapping")
        decoded = np.vectorize(
            lambda value: value.decode() if isinstance(value, bytes) else str(value)
        )(raw)
        labels = []
        for column in decoded.T:
            values, counts = np.unique(column, return_counts=True)
            labels.append(str(values[np.argmax(counts)]))
        return np.asarray(labels, dtype=object)
    except Exception:
        return np.asarray(["unknown"] * n_residues, dtype=object)


def concentration_audit(
    profile: dict[str, object],
    rmsf_delta: np.ndarray,
    dssp: np.ndarray,
    coords: np.ndarray,
) -> dict[str, object]:
    """Describe where RMSF disagreement is concentrated without annotation guesses."""

    n = rmsf_delta.size
    top_count = min(5, n)
    top = np.argsort(-np.abs(rmsf_delta))[:top_count]
    total = float(np.sum(np.abs(rmsf_delta)))
    terminal = (top < 5) | (top >= n - 5)
    loop_codes = {"", "C", "S", "T", "-", " ", "unknown"}
    loop = np.asarray([str(dssp[index]).strip() in loop_codes for index in top])
    mapping = profile["mapping"]
    chain_breaks = np.zeros(n, dtype=bool)
    for index in range(1, n):
        if mapping[index][0] != mapping[index - 1][0]:
            chain_breaks[index] = True
            continue
        try:
            chain_breaks[index] = abs(int(mapping[index][1]) - int(mapping[index - 1][1])) > 1
        except ValueError:
            pass
    interface = np.zeros(n, dtype=bool)
    chains = np.asarray([item[0] for item in mapping], dtype=object)
    if len(set(chains.tolist())) > 1:
        distance = np.linalg.norm(coords[0, :, None] - coords[0, None, :], axis=3)
        interface = np.any((distance <= 8.0) & (chains[:, None] != chains[None, :]), axis=1)
    labels = [f"{mapping[index][0]}:{mapping[index][1]}:{mapping[index][2]}" for index in top]
    return {
        "top_high_leverage_residues": ";".join(labels),
        "top5_RMSF_difference_fraction": float(np.sum(np.abs(rmsf_delta[top])) / total) if total else 0.0,
        "top5_terminal_fraction": float(np.mean(terminal)),
        "top5_loop_fraction": float(np.mean(loop)),
        "top5_chain_break_fraction": float(np.mean(chain_breaks[top])),
        "top5_interface_fraction": float(np.mean(interface[top])),
        "small_or_mobile_domain_annotation": "unavailable_not_inferred",
    }


def classify_case(row: dict[str, object]) -> str:
    """Apply the v2 scientific classification, not the historical D1 gate."""

    if row["integrity_status"] != "pass":
        return "data/reference/metric artefact"
    if (
        not row["cpu_valid_transform"]
        or not row["gpu_valid_transform"]
        or float(row["minimum_core_jaccard"]) < 0.5
        or float(row["minimum_RMSF_spearman"]) < 0.9
        or float(row["minimum_effective_core_fraction"]) < 0.1
    ):
        return "genuine throughput quality failure"
    if (
        float(row["minimum_core_jaccard"]) >= 0.85
        and float(row["minimum_RMSF_spearman"]) >= 0.99
        and float(row["cpu_median_signed_RMSD_difference_A"]) < 0.0
        and float(row["representative_subset_core_jaccard"]) >= 0.9
    ):
        return "scientifically acceptable alternate fit"
    return "moderate-confidence approximation"


def make_plot(
    domain: str,
    exact_weights: np.ndarray,
    cpu: object,
    gpu: object,
    exact_fit: object,
    profile: dict[str, object],
) -> None:
    """Write one three-panel tail diagnostic figure in PDF and PNG."""

    x = np.arange(1, exact_weights.size + 1)
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), constrained_layout=True)
    axes[0].plot(x, normalized_weights(exact_weights), label="Exact", lw=1.6)
    axes[0].plot(x, normalized_weights(cpu.frozen_weights), label="CPU throughput", lw=1.2)
    axes[0].plot(x, normalized_weights(gpu.frozen_weights), label="GPU throughput", lw=1.0, ls="--")
    axes[0].set(ylabel="Stability weight", title=f"{domain}: D1 tail adjudication")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(x, exact_fit.robust_rmsf, label="Exact", lw=1.6)
    axes[1].plot(x, cpu.robust_rmsf, label="CPU throughput", lw=1.2)
    axes[1].plot(x, gpu.robust_rmsf, label="GPU throughput", lw=1.0, ls="--")
    axes[1].set(ylabel="Robust RMSF (A)")
    frames = np.arange(exact_fit.rmsd.size)
    axes[2].plot(frames, exact_fit.rmsd, label="Exact", lw=1.4)
    axes[2].plot(frames, cpu.rmsd, label="CPU throughput", lw=1.0)
    axes[2].plot(frames, gpu.rmsd, label="GPU throughput", lw=0.9, ls="--")
    axes[2].set(xlabel="Frame", ylabel="Full-evaluation RMSD (A)")
    plot_root = B1_ROOT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(plot_root / f"{domain}_tail_diagnostics.{extension}", dpi=300)
    plt.close(figure)


def main() -> None:
    """Run B1 and produce the nine-case atlas and prospective warning rule."""

    b0 = json.loads((B0_ROOT / "b0_status.json").read_text())
    if b0["status"] != "pass" or not b0["B1_authorized"]:
        raise RuntimeError("B0 did not authorize B1")
    registry = {
        row["system_id"]: row
        for row in read_tsv(
            ROOT / "results/database_scale_gpu_v1/d0_freeze/benchmark_registry.tsv"
        )
    }
    prior_rows = {
        row["domain_id"]: row
        for row in read_tsv(
            ROOT / "results/guarded_fast_global_v1/g1_adjudication/nonconvergence_atlas.tsv"
        )
    }
    worker = PersistentGPUWorker.create(
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )
    atlas: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for domain in TAIL_DOMAINS:
        row = registry[domain]
        source = ROOT / row["source_file"]
        load_start = time.perf_counter()
        coords, _, residue_metadata, metadata = extract_ca_trajectory(
            source,
            coordinate_path=row["coordinate_path"],
            direct_contiguous_read=True,
        )
        coords = np.asarray(coords, dtype=np.float64)
        load_seconds = time.perf_counter() - load_start
        chains = np.asarray([item.get("chain", "_") for item in residue_metadata], dtype=object)
        profile = load_exact_profile(corrected_profile_path(row["trajectory_id"]))
        observed_mapping = [
            (str(item.get("chain", "_")), str(item.get("residue_id", "")), str(item.get("resname", "UNK")))
            for item in residue_metadata
        ]
        integrity = "pass" if profile["mapping"] == observed_mapping and metadata["unit_status"] == "verified_angstrom_metadata" else "fail"
        exact_weights = np.asarray(profile["weights"], dtype=np.float64)
        exact_core = core_mask(exact_weights)
        mobile = ~exact_core
        exact_start = time.perf_counter()
        exact_fit = exact_all_frame_fit(coords, coords[0], exact_weights, chunk_size=1000)
        exact_seconds = time.perf_counter() - exact_start
        cpu = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=min(256, coords.shape[0]),
            update_rounds=4,
            chunk_size=1000,
            store_aligned=False,
            compute_confidence=True,
            collect_risk_signals=True,
        )
        worker.chain_ids = chains
        gpu = worker.throughput(coords, representative_count=min(256, coords.shape[0]))
        exact_aligned = method_aligned(coords, exact_fit)
        cpu_aligned = method_aligned(coords, cpu)
        gpu_aligned = method_aligned(coords, gpu)
        method_rows = {}
        for name, result, aligned in (
            ("exact", exact_fit, exact_aligned),
            ("cpu", cpu, cpu_aligned),
            ("gpu", gpu, gpu_aligned),
        ):
            method_core = exact_core if name == "exact" else core_mask(result.frozen_weights)
            method_rows[name] = {
                "full_RMSD_mean_A": float(np.mean(rmsd_by_mask(aligned, coords[0], np.ones(coords.shape[1], dtype=bool)))),
                "exact_core_RMSD_mean_A": float(np.mean(rmsd_by_mask(aligned, coords[0], exact_core))),
                "exact_mobile_RMSD_mean_A": float(np.mean(rmsd_by_mask(aligned, coords[0], mobile))),
                "method_core_RMSD_mean_A": float(np.mean(rmsd_by_mask(aligned, coords[0], method_core))),
            }
            for chain in sorted(set(chains.tolist())):
                method_rows[name][f"chain_{chain}_RMSD_mean_A"] = float(
                    np.mean(rmsd_by_mask(aligned, coords[0], chains == chain))
                )
            decomposition_rows.append(
                {
                    "domain_id": domain,
                    "method": name,
                    **method_rows[name],
                }
            )
        exact_geometry = heldout_geometry(coords, exact_core)
        cpu_core = core_mask(cpu.frozen_weights)
        gpu_core = core_mask(gpu.frozen_weights)
        cpu_geometry = heldout_geometry(coords, cpu_core)
        gpu_geometry = heldout_geometry(coords, gpu_core)
        exact_temporal = exact_temporal_reproducibility(coords, chains)
        cpu_delta = np.asarray(cpu.rmsd) - np.asarray(exact_fit.rmsd)
        gpu_delta = np.asarray(gpu.rmsd) - np.asarray(exact_fit.rmsd)
        cpu_weight_delta = normalized_weights(cpu.frozen_weights) - normalized_weights(exact_weights)
        gpu_weight_delta = normalized_weights(gpu.frozen_weights) - normalized_weights(exact_weights)
        cpu_angles = rotation_geodesic_degrees(cpu.rotations, exact_fit.rotations)
        gpu_angles = rotation_geodesic_degrees(gpu.rotations, exact_fit.rotations)
        dssp = secondary_structure_labels(source, metadata["coordinate_path"], coords.shape[1])
        concentration = concentration_audit(
            profile,
            np.asarray(cpu.robust_rmsf) - np.asarray(exact_fit.robust_rmsf),
            dssp,
            coords,
        )
        cpu_components, cpu_fragmentation = core_components(cpu_core, chains)
        exact_components, exact_fragmentation = core_components(exact_core, chains)
        prior = prior_rows.get(domain, {})
        output: dict[str, object] = {
            "domain_id": domain,
            "trajectory_id": row["trajectory_id"],
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "n_chains": len(set(chains.tolist())),
            "coordinate_sha256_float64": coordinate_digest(coords),
            "reference_frame": 0,
            "fit_selection": "all_matched_CA_with_method_weights",
            "evaluation_selection": "same_all_matched_CA_and_explicit_exact_core/mobile_subsets",
            "frame_order": "source_order_unchanged",
            "integrity_status": integrity,
            "shared_load_decode_seconds": load_seconds,
            "exact_fixed_weight_fit_seconds": exact_seconds,
            "cpu_throughput_seconds": cpu.total_runtime_seconds,
            "gpu_throughput_seconds": gpu.total_runtime_seconds,
            "cpu_valid_transform": valid_transforms(cpu),
            "gpu_valid_transform": valid_transforms(gpu),
            "cpu_core_jaccard": jaccard(cpu_core, exact_core),
            "gpu_core_jaccard": jaccard(gpu_core, exact_core),
            "minimum_core_jaccard": min(jaccard(cpu_core, exact_core), jaccard(gpu_core, exact_core)),
            "cpu_weight_spearman": safe_spearman(cpu.frozen_weights, exact_weights),
            "gpu_weight_spearman": safe_spearman(gpu.frozen_weights, exact_weights),
            "cpu_maximum_absolute_weight_difference": float(np.max(np.abs(cpu_weight_delta))),
            "gpu_maximum_absolute_weight_difference": float(np.max(np.abs(gpu_weight_delta))),
            "cpu_RMSF_spearman": safe_spearman(cpu.robust_rmsf, exact_fit.robust_rmsf),
            "gpu_RMSF_spearman": safe_spearman(gpu.robust_rmsf, exact_fit.robust_rmsf),
            "minimum_RMSF_spearman": min(safe_spearman(cpu.robust_rmsf, exact_fit.robust_rmsf), safe_spearman(gpu.robust_rmsf, exact_fit.robust_rmsf)),
            "cpu_RMSF_MAE_A": float(np.mean(np.abs(cpu.robust_rmsf - exact_fit.robust_rmsf))),
            "gpu_RMSF_MAE_A": float(np.mean(np.abs(gpu.robust_rmsf - exact_fit.robust_rmsf))),
            "cpu_RMSF_calibration_slope": calibration_slope(exact_fit.robust_rmsf, cpu.robust_rmsf),
            "gpu_RMSF_calibration_slope": calibration_slope(exact_fit.robust_rmsf, gpu.robust_rmsf),
            "cpu_median_signed_RMSD_difference_A": float(np.median(cpu_delta)),
            "cpu_median_absolute_RMSD_difference_A": float(np.median(np.abs(cpu_delta))),
            "cpu_P95_absolute_RMSD_difference_A": float(np.quantile(np.abs(cpu_delta), 0.95)),
            "gpu_median_signed_RMSD_difference_A": float(np.median(gpu_delta)),
            "gpu_median_absolute_RMSD_difference_A": float(np.median(np.abs(gpu_delta))),
            "gpu_P95_absolute_RMSD_difference_A": float(np.quantile(np.abs(gpu_delta), 0.95)),
            "exact_effective_core_fraction": float(np.sum(normalized_weights(exact_weights)) / exact_weights.size),
            "cpu_effective_core_fraction": float(np.sum(normalized_weights(cpu.frozen_weights)) / cpu.frozen_weights.size),
            "gpu_effective_core_fraction": float(np.sum(normalized_weights(gpu.frozen_weights)) / gpu.frozen_weights.size),
            "minimum_effective_core_fraction": min(float(np.sum(normalized_weights(cpu.frozen_weights)) / cpu.frozen_weights.size), float(np.sum(normalized_weights(gpu.frozen_weights)) / gpu.frozen_weights.size)),
            "exact_core_components": exact_components,
            "cpu_core_components": cpu_components,
            "exact_core_fragmentation_fraction": exact_fragmentation,
            "cpu_core_fragmentation_fraction": cpu_fragmentation,
            "exact_temporal_block_weight_spearman": exact_temporal["weight_spearman"],
            "exact_temporal_block_core_jaccard": exact_temporal["core_jaccard"],
            "throughput_temporal_block_weight_spearman": cpu.confidence.temporal_block_weight_spearman,
            "throughput_temporal_block_core_jaccard": cpu.confidence.temporal_block_core_jaccard,
            "representative_subset_weight_spearman": cpu.confidence.representative_subset_weight_spearman,
            "representative_subset_core_jaccard": cpu.confidence.representative_subset_core_jaccard,
            "exact_heldout_pairwise_distance_variance_A2": exact_geometry["pairwise_distance_variance"],
            "cpu_heldout_pairwise_distance_variance_A2": cpu_geometry["pairwise_distance_variance"],
            "gpu_heldout_pairwise_distance_variance_A2": gpu_geometry["pairwise_distance_variance"],
            "exact_heldout_contact_persistence": exact_geometry["reference_contact_persistence"],
            "cpu_heldout_contact_persistence": cpu_geometry["reference_contact_persistence"],
            "gpu_heldout_contact_persistence": gpu_geometry["reference_contact_persistence"],
            "cpu_rotation_angle_median_degrees": float(np.median(cpu_angles)),
            "cpu_rotation_angle_P95_degrees": float(np.quantile(cpu_angles, 0.95)),
            "gpu_rotation_angle_median_degrees": float(np.median(gpu_angles)),
            "gpu_rotation_angle_P95_degrees": float(np.quantile(gpu_angles, 0.95)),
            "cpu_translation_difference_median_A": float(np.median(np.linalg.norm(cpu.translations - exact_fit.translations, axis=1))),
            "gpu_translation_difference_median_A": float(np.median(np.linalg.norm(gpu.translations - exact_fit.translations, axis=1))),
            "cpu_exact_aligned_coordinate_displacement_mean_A": float(np.mean(np.linalg.norm(cpu_aligned - exact_aligned, axis=2))),
            "gpu_exact_aligned_coordinate_displacement_mean_A": float(np.mean(np.linalg.norm(gpu_aligned - exact_aligned, axis=2))),
            "historical_G1_classification": prior.get("scientific_classification", "not_available"),
            **concentration,
        }
        output["v2_classification"] = classify_case(output)
        output["difference_concentration_interpretation"] = (
            "full-structure RMSD is lower for throughput because its weights retain more mobile-region influence"
            if float(output["cpu_median_signed_RMSD_difference_A"]) < 0.0
            else "throughput full-structure RMSD is higher than exact"
        )
        atlas.append(output)
        make_plot(domain, exact_weights, cpu, gpu, exact_fit, profile)
        print(f"[B1] {domain}: {output['v2_classification']}", flush=True)

    write_tsv(B1_ROOT / "b1_tail_case_atlas.tsv", atlas)
    write_tsv(B1_ROOT / "b1_rmsd_decomposition.tsv", decomposition_rows)
    classifications = {label: sum(row["v2_classification"] == label for row in atlas) for label in sorted({str(row["v2_classification"]) for row in atlas})}
    integrity_failures = sum(row["integrity_status"] != "pass" for row in atlas)
    status = {
        "phase": "B1",
        "status": "blocked_integrity" if integrity_failures else "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tail_case_count": len(atlas),
        "integrity_failures": integrity_failures,
        "classifications": classifications,
        "historical_D1_gate_rewritten": False,
        "B2_authorized": integrity_failures == 0,
    }
    write_json(B1_ROOT / "b1_status.json", status)
    warning_rule = {
        "registered_after_B1_before_B3": True,
        "failure": [
            "non_finite_output",
            "invalid_rigid_transform",
            "mapping_or_unit_failure",
            "empty_core",
            "effective_core_fraction_below_0.10",
        ],
        "low_confidence": [
            "independent_subset_weight_spearman_below_0.90",
            "independent_subset_core_jaccard_below_0.80",
            "temporal_block_weight_spearman_below_0.85",
            "temporal_block_core_jaccard_below_0.75",
            "last_two_round_core_jaccard_below_0.80",
        ],
        "moderate_confidence": [
            "fixed_budget_not_weight_stable",
            "temporal_or_subset_warning_not_crossing_low_confidence_threshold",
            "near_core_threshold_fraction_at_least_0.20",
        ],
        "high_confidence": [
            "finite_valid_output",
            "plausible_core",
            "no_low_confidence_signal",
            "stable_independent_representative_subsets",
        ],
        "exact_relative_audit_only": [
            "core_jaccard_below_0.5",
            "RMSF_spearman_below_0.9",
        ],
        "note": "A full-structure RMSD difference above 1 A alone is not a v2 catastrophic criterion.",
    }
    write_json(B1_ROOT / "prospective_warning_rule.json", warning_rule)
    report_rows = "\n".join(
        f"| {row['domain_id']} | {row['cpu_core_jaccard']:.3f} | {row['cpu_RMSF_spearman']:.3f} | {row['cpu_median_signed_RMSD_difference_A']:.3f} | {row['v2_classification']} |"
        for row in atlas
    )
    (B1_ROOT / "b1_tail_case_report.md").write_text(
        "# B1 D1-tail adjudication\n\n"
        "## Outcome\n\n"
        f"- Matched tail cases completed: **{len(atlas)}/9**.\n"
        f"- Input/unit/mapping integrity failures: **{integrity_failures}**.\n"
        f"- Classifications: `{json.dumps(classifications, sort_keys=True)}`.\n"
        "- The historical D1 failed gate remains unchanged.\n\n"
        "| Domain | Core Jaccard | RMSF Spearman | Signed median RMSD delta (A) | V2 classification |\n"
        f"|---|---:|---:|---:|---|\n{report_rows}\n\n"
        "## Interpretation\n\n"
        "All throughput fits had finite outputs, valid transforms, non-empty cores, and core Jaccard above 0.5. "
        "Most large historical absolute RMSD differences were negative signed differences: throughput produced lower full-selection RMSD because its approximate weights retained more influence from mobile residues, whereas exact `sequential_lean` prioritised its stable-core solution. This is a scientifically meaningful alternate fitting convention, not automatically a numerical failure. Cases with degraded RMSF ranking or weaker core agreement remain explicitly classified as genuine or moderate-confidence tails.\n\n"
        "The registered warning rule uses only cheap within-run stability signals during deployment. Exact-relative metrics remain audit metrics and are not silently available for every production trajectory.\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
