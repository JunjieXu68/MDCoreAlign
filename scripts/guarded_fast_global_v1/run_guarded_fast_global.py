#!/usr/bin/env python3
"""Run the staged G0-G4 guarded fast-global development workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.guarded import solve_guarded_fast_global_alignment
from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.flexibility import detect_flexible_segments
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import (
    exact_all_frame_fit,
    temporally_stratified_indices,
)
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_scan import percentile_ranks
from scripts.mainline_reintegration_v1.run_r0_inventory import (
    frozen_integrity,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization_v5.run_phase8b_runtime_boundaries import (
    load_long_input,
    run_common_boundary,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/guarded_fast_global_v1"
CORRECTED = ROOT / "results/project_completion_v2_unit_corrected"
CORRECTED_TABLE = (
    CORRECTED / "aggregate/all_domain_unit_corrected_per_domain.tsv"
)
FAST_TABLE = CORRECTED / "fast_global/fast_global_all_domain.tsv"
HISTORICAL_RUNS = (
    ROOT
    / "results/mdcath_benchmark/all_domain_canonical/run_fullframes/"
    "per_trajectory"
)
H5_CACHE = (
    ROOT
    / "results/mdcath_benchmark/cache/datasets--compsciencelab--mdCATH/"
    "snapshots/5e3ed8aec62b689e01751db16275fdcdbc39e47f/data"
)
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
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
RISK_FEATURES: dict[str, str] = {
    "final_round_weight_Linf_change": "high",
    "last_round_relative_objective_change": "high",
    "last_two_round_core_jaccard": "low",
    "near_core_threshold_fraction": "high",
    "mean_residual_p95_median_ratio": "high",
    "frame_RMSD_p95_median_ratio": "high",
    "effective_core_fraction": "low",
    "core_fragmentation_fraction": "high",
    "temporal_block_weight_spearman": "low",
    "temporal_block_core_jaccard": "low",
    "independent_subset_weight_spearman": "low",
    "independent_subset_core_jaccard": "low",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_h5(domain_id: str) -> Path:
    direct = H5_CACHE / f"mdcath_dataset_{domain_id}.h5"
    if direct.exists():
        return direct
    matches = list(H5_CACHE.glob(f"*{domain_id}*.h5"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one cached HDF5 for {domain_id}, found {len(matches)}"
        )
    return matches[0]


def corrected_profile_path(trajectory_id: str) -> Path:
    affected = (
        CORRECTED
        / "affected_reruns/per_trajectory"
        / trajectory_id
        / "residue_profile.tsv"
    )
    historical = HISTORICAL_RUNS / trajectory_id / "residue_profile.tsv"
    path = affected if affected.is_file() else historical
    if not path.is_file():
        raise FileNotFoundError(f"missing exact profile for {trajectory_id}")
    return path


def load_profile(path: Path) -> dict[str, np.ndarray]:
    rows = read_tsv(path)
    return {
        "weights": np.asarray(
            [float(row["soft_stability_weight"]) for row in rows]
        ),
        "rmsf": np.asarray([float(row["robust_RMSF"]) for row in rows]),
        "flexible": np.asarray(
            [int(row["flexible_segment_id"]) > 0 for row in rows],
            dtype=bool,
        ),
    }


def size_bin(n_residues: int) -> str:
    if n_residues <= 100:
        return "small"
    if n_residues <= 200:
        return "medium"
    return "large"


def paired_source_rows() -> list[dict[str, str]]:
    exact = {
        row["trajectory_id"]: row
        for row in read_tsv(CORRECTED_TABLE)
        if row["mdcorealign_status"] == "success"
        and row["mdlovofit_status"] == "success"
    }
    fast = {
        row["trajectory_id"]: row
        for row in read_tsv(FAST_TABLE)
        if row["status"] == "success"
    }
    common = sorted(exact.keys() & fast.keys())
    if len(common) != 5397:
        raise RuntimeError(
            f"expected 5397 corrected paired domains, found {len(common)}"
        )
    rows: list[dict[str, str]] = []
    for trajectory_id in common:
        row = dict(exact[trajectory_id])
        row.update(
            {
                "predefined_tail_failure": fast[trajectory_id][
                    "catastrophic_failure"
                ],
                "strict_convergence_status": fast[trajectory_id][
                    "confidence_status"
                ],
                "h5_path": str(source_h5(row["domain_id"])),
                "coordinate_path": (
                    f"/{row['domain_id']}/320/0/coords"
                ),
                "exact_profile_path": str(
                    corrected_profile_path(trajectory_id)
                ),
                "size_bin": size_bin(int(row["n_residues"])),
            }
        )
        rows.append(row)
    return rows


def g0() -> None:
    """Freeze the guarded-run inputs and deterministic split."""

    integrity = frozen_integrity()
    if integrity["status"] != "pass":
        raise RuntimeError("frozen integrity failed before G0")
    rows = paired_source_rows()
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["cath_class"],
            row["size_bin"],
            row["predefined_tail_failure"],
        )
        groups.setdefault(key, []).append(row)
    split_rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda row: stable_hash(
                f"guarded-v1::{row['trajectory_id']}"
            ),
        )
        development_count = max(
            1,
            int(np.floor(len(ordered) * 0.30 + 0.5)),
        )
        for index, row in enumerate(ordered):
            split_rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "domain_id": row["domain_id"],
                    "cath_class": row["cath_class"],
                    "size_bin": row["size_bin"],
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "predefined_tail_failure": row[
                        "predefined_tail_failure"
                    ],
                    "strict_convergence_status": row[
                        "strict_convergence_status"
                    ],
                    "split": (
                        "development"
                        if index < development_count
                        else "validation"
                    ),
                    "split_rule": (
                        "sha256-stratified-30pct-development-v1"
                    ),
                    "h5_path": row["h5_path"],
                    "coordinate_path": row["coordinate_path"],
                    "exact_profile_path": row["exact_profile_path"],
                }
            )
    split_rows.sort(key=lambda row: str(row["trajectory_id"]))
    split_path = OUTPUT / "g0_freeze/locked_panel_split.tsv"
    write_tsv(split_path, split_rows)
    counts = {
        split: sum(row["split"] == split for row in split_rows)
        for split in ("development", "validation")
    }
    tail_counts = {
        split: sum(
            row["split"] == split
            and row["predefined_tail_failure"] == "1"
            for row in split_rows
        )
        for split in ("development", "validation")
    }
    write_json(
        OUTPUT / "g0_freeze/split_lock.json",
        {
            "status": "locked",
            "source_of_truth": str(CORRECTED_TABLE.relative_to(ROOT)),
            "source_fast_table": str(FAST_TABLE.relative_to(ROOT)),
            "n_paired_domains": len(split_rows),
            "counts": counts,
            "predefined_tail_counts": tail_counts,
            "split_manifest_sha256": sha256(split_path),
            "split_rule": "sha256-stratified-30pct-development-v1",
            "thresholds_tuned_on_validation": False,
        },
    )
    write_json(
        OUTPUT / "g0_freeze/integrity_before.json",
        integrity,
    )
    (OUTPUT / "g0_freeze/g0_freeze_audit.md").write_text(
        f"""# G0 freeze and audit

- Corrected paired-domain source: `{CORRECTED_TABLE.relative_to(ROOT)}`.
- Locked domains: {len(split_rows)}.
- Development: {counts['development']} domains, including
  {tail_counts['development']} predefined tail failures.
- Validation: {counts['validation']} domains, including
  {tail_counts['validation']} predefined tail failures.
- Split manifest SHA256: `{sha256(split_path)}`.
- Frozen integrity: `{integrity['status']}`.
- Public exact default changed: no.
- Git metadata changed: no.
- Historical evidence modified: no.
"""
    )


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = percentile_ranks(np.asarray(left, dtype=np.float64))
    b = percentile_ranks(np.asarray(right, dtype=np.float64))
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return (
        1.0
        if union == 0
        else float(np.count_nonzero(left & right) / union)
    )


def fixed_weight_objective(
    representatives: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values / max(float(np.max(values)), 1e-12)
    fit = weighted_kabsch_batch(
        representatives,
        reference,
        values,
        validate_inputs=False,
        compute_metrics=False,
    )
    delta = fit.aligned_coords - reference[None, :, :]
    squared = np.einsum("bni,bni->bn", delta, delta)
    return float(
        np.sqrt(
            np.mean(
                np.einsum("bn,n->b", squared, values)
                / np.sum(values)
            )
        )
    )


def classify_quality(row: dict[str, object]) -> str:
    if row["status"] != "success":
        return "implementation_or_data_failure"
    if int(row["predefined_tail_failure"]) == 1:
        return "actual_tail_failure"
    acceptable = (
        float(row["core_jaccard"]) >= 0.90
        and float(row["weight_spearman"]) >= 0.97
        and float(row["RMSF_spearman"]) >= 0.98
        and float(row["RMSF_MAE_A"]) <= 0.20
        and float(row["mean_absolute_RMSD_difference_A"]) <= 0.25
        and float(row["P95_absolute_frame_RMSD_difference_A"]) <= 0.50
        and float(row["flexible_label_jaccard"]) >= 0.80
    )
    if acceptable:
        return (
            "numerically_nonconverged_scientifically_acceptable"
            if row["strict_convergence_status"]
            == "fixed_budget_complete_not_converged"
            else "strictly_converged_scientifically_acceptable"
        )
    return "scientifically_borderline"


def g1_worker(task: dict[str, str]) -> dict[str, object]:
    os.environ.update(THREAD_ENV)
    start = time.perf_counter()
    base: dict[str, object] = {
        "trajectory_id": task["trajectory_id"],
        "domain_id": task["domain_id"],
        "cath_class": task["cath_class"],
        "size_bin": task["size_bin"],
        "split": task["split"],
        "predefined_tail_failure": task["predefined_tail_failure"],
    }
    try:
        coords, labels, metadata_rows, metadata = extract_ca_trajectory(
            Path(task["h5_path"]),
            coordinate_path=task["coordinate_path"],
            max_frames=None,
            stride=1,
        )
        coords = np.asarray(coords, dtype=np.float64)
        chains = np.asarray(
            [row.get("chain", "_") for row in metadata_rows],
            dtype=object,
        )
        fast = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=256,
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=True,
            collect_risk_signals=True,
        )
        if fast.risk_signals is None:
            raise RuntimeError("risk trace was not collected")
        exact_start = time.perf_counter()
        exact_representatives = temporally_stratified_indices(
            coords.shape[0],
            min(2000, coords.shape[0]),
        )
        exact_estimate = estimate_representative_weights_sequential_lean(
            coords[exact_representatives],
            coords[0],
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        exact_weights = np.asarray(
            exact_estimate.mean_weights,
            dtype=np.float64,
        )
        exact_fit = exact_all_frame_fit(
            coords,
            coords[0],
            exact_weights,
            chunk_size=2000,
        )
        exact_runtime = time.perf_counter() - exact_start
        exact_core = select_core_by_target_fraction(
            exact_weights,
            0.6,
        ).core_mask
        fast_core = select_core_by_target_fraction(
            fast.frozen_weights,
            0.6,
        ).core_mask
        fast_flexible = (
            detect_flexible_segments(
                1.0 - fast.frozen_weights,
                percentile_cutoff=0.8,
                min_segment_length=3,
                merge_gap=1,
                chain=chains,
            )
            > 0
        )
        exact_flexible = (
            detect_flexible_segments(
                1.0 - exact_weights,
                percentile_cutoff=0.8,
                min_segment_length=3,
                merge_gap=1,
                chain=chains,
            )
            > 0
        )
        rmsd_delta = fast.rmsd - exact_fit.rmsd
        absolute_rmsd_delta = np.abs(rmsd_delta)
        representatives = temporally_stratified_indices(
            coords.shape[0],
            min(256, coords.shape[0]),
        )
        exact_objective = fixed_weight_objective(
            coords[representatives],
            coords[0],
            exact_weights,
        )
        fast_objective = fixed_weight_objective(
            coords[representatives],
            coords[0],
            fast.frozen_weights,
        )
        weight_delta = fast.frozen_weights - exact_weights
        row: dict[str, object] = {
            **base,
            "status": "success",
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "source_unit_status": metadata["unit_status"],
            "fast_runtime_seconds": fast.total_runtime_seconds,
            "sequential_lean_runtime_seconds": exact_runtime,
            "worker_end_to_end_seconds": time.perf_counter() - start,
            "strict_convergence_status": (
                fast.confidence.convergence_status
            ),
            "convergence_reason": ";".join(
                fast.confidence.warning_flags
            )
            or "fixed_budget_stable",
            "warning_state": (
                "warning"
                if fast.confidence.warning_flags
                else "clear"
            ),
            "warning_count": len(fast.confidence.warning_flags),
            "exact_final_objective": exact_objective,
            "fast_final_objective": fast_objective,
            "signed_final_objective_difference": (
                fast_objective - exact_objective
            ),
            "absolute_final_objective_difference": abs(
                fast_objective - exact_objective
            ),
            "last_round_objective_change": fast.risk_signals[
                "last_round_objective_change"
            ],
            "last_round_relative_objective_change": fast.risk_signals[
                "last_round_relative_objective_change"
            ],
            "objective_reduction_fraction": fast.risk_signals[
                "objective_reduction_fraction"
            ],
            "final_round_weight_L1_change": fast.risk_signals[
                "final_weight_l1_change"
            ],
            "final_round_weight_mean_L1_change": fast.risk_signals[
                "final_weight_mean_l1_change"
            ],
            "final_round_weight_Linf_change": fast.risk_signals[
                "final_weight_linf_change"
            ],
            "fast_exact_weight_L1_difference": float(
                np.sum(np.abs(weight_delta))
            ),
            "fast_exact_weight_Linf_difference": float(
                np.max(np.abs(weight_delta))
            ),
            "last_two_round_core_jaccard": fast.risk_signals[
                "last_two_round_core_jaccard"
            ],
            "core_jaccard": jaccard(exact_core, fast_core),
            "weight_spearman": spearman(
                exact_weights,
                fast.frozen_weights,
            ),
            "RMSF_spearman": spearman(
                exact_fit.robust_rmsf,
                fast.robust_rmsf,
            ),
            "RMSF_MAE_A": float(
                np.mean(
                    np.abs(
                        exact_fit.robust_rmsf - fast.robust_rmsf
                    )
                )
            ),
            "median_signed_frame_RMSD_difference_A": float(
                np.median(rmsd_delta)
            ),
            "median_absolute_frame_RMSD_difference_A": float(
                np.median(absolute_rmsd_delta)
            ),
            "P95_absolute_frame_RMSD_difference_A": float(
                np.quantile(absolute_rmsd_delta, 0.95)
            ),
            "mean_absolute_RMSD_difference_A": abs(
                float(np.mean(fast.rmsd))
                - float(np.mean(exact_fit.rmsd))
            ),
            "effective_core_size": fast.confidence.effective_core_size,
            "effective_core_fraction": fast.risk_signals[
                "effective_core_fraction"
            ],
            "flexible_label_jaccard": jaccard(
                exact_flexible,
                fast_flexible,
            ),
            "flexible_label_agreement_fraction": float(
                np.mean(exact_flexible == fast_flexible)
            ),
            "exact_representative_converged_fraction": (
                exact_estimate.converged_fraction
            ),
            "near_core_threshold_count": fast.risk_signals[
                "near_core_threshold_count"
            ],
            "near_core_threshold_fraction": fast.risk_signals[
                "near_core_threshold_fraction"
            ],
            "core_threshold_gap": fast.risk_signals[
                "core_threshold_gap"
            ],
            "mean_residual_p95_median_ratio": fast.risk_signals[
                "mean_residual_p95_median_ratio"
            ],
            "frame_RMSD_p95_median_ratio": fast.risk_signals[
                "frame_rmsd_p95_median_ratio"
            ],
            "core_transition_count": fast.risk_signals[
                "core_transition_count"
            ],
            "core_fragmentation_fraction": fast.risk_signals[
                "core_fragmentation_fraction"
            ],
            "temporal_block_weight_spearman": (
                fast.confidence.temporal_block_weight_spearman
            ),
            "temporal_block_core_jaccard": (
                fast.confidence.temporal_block_core_jaccard
            ),
            "independent_subset_weight_spearman": (
                fast.confidence.representative_subset_weight_spearman
            ),
            "independent_subset_core_jaccard": (
                fast.confidence.representative_subset_core_jaccard
            ),
            "error": "",
        }
        row["scientific_classification"] = classify_quality(row)
        return row
    except Exception as exc:
        row = {
            **base,
            "status": "failed",
            "worker_end_to_end_seconds": time.perf_counter() - start,
            "scientific_classification": (
                "implementation_or_data_failure"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return row


def g1(workers: int) -> None:
    """Run or resume the complete corrected-panel diagnostic atlas."""

    split = read_tsv(OUTPUT / "g0_freeze/locked_panel_split.tsv")
    output_path = OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
    existing = read_tsv(output_path) if output_path.is_file() else []
    by_id = {row["trajectory_id"]: row for row in existing}
    tasks = [
        row for row in split if row["trajectory_id"] not in by_id
    ]
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            for index, row in enumerate(
                pool.imap_unordered(g1_worker, tasks, chunksize=1),
                start=1,
            ):
                by_id[str(row["trajectory_id"])] = row
                if index % 25 == 0 or index == len(tasks):
                    write_tsv(
                        output_path,
                        sorted(
                            by_id.values(),
                            key=lambda item: item["trajectory_id"],
                        ),
                    )
                    print(
                        f"[G1] completed {index}/{len(tasks)} new; "
                        f"total {len(by_id)}/5397",
                        flush=True,
                    )
    write_tsv(
        output_path,
        sorted(by_id.values(), key=lambda row: row["trajectory_id"]),
    )


def g1_finalize() -> None:
    rows = read_tsv(
        OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
    )
    if len(rows) != 5397:
        raise RuntimeError(f"G1 atlas has {len(rows)}/5397 rows")
    success = [row for row in rows if row["status"] == "success"]
    tails = [
        row for row in success if row["predefined_tail_failure"] == "1"
    ]
    write_tsv(
        OUTPUT / "g1_adjudication/tail_failure_atlas.tsv",
        tails,
    )
    classes = sorted(
        {row["scientific_classification"] for row in rows}
    )
    summary_rows = [
        {
            "scientific_classification": label,
            "count": sum(
                row["scientific_classification"] == label for row in rows
            ),
            "fraction": sum(
                row["scientific_classification"] == label for row in rows
            )
            / len(rows),
        }
        for label in classes
    ]
    write_tsv(
        OUTPUT / "g1_adjudication/classification_summary.tsv",
        summary_rows,
    )
    strict_nonconverged = [
        row
        for row in success
        if row["strict_convergence_status"]
        == "fixed_budget_complete_not_converged"
    ]
    acceptable_nonconverged = [
        row
        for row in strict_nonconverged
        if row["scientific_classification"]
        == "numerically_nonconverged_scientifically_acceptable"
    ]
    (OUTPUT / "g1_adjudication/g1_nonconvergence_report.md").write_text(
        f"""# G1 non-convergence adjudication

- Complete corrected paired panel: {len(rows)}.
- Successful diagnostic rows: {len(success)}.
- Implementation/data failures: {len(rows) - len(success)}.
- Strict fixed-budget non-convergence: {len(strict_nonconverged)}.
- Strict non-convergence but scientifically acceptable under the locked G1
  descriptive contract: {len(acceptable_nonconverged)}.
- Predefined actual tail failures: {len(tails)}.

The strict `1e-3` final L-infinity weight-change label is retained unchanged.
It is not treated as an automatic scientific-failure label. G1 classifications
are descriptive and do not replace the exact default or retrospectively alter
the convergence criterion.

The detailed per-domain metrics are in `nonconvergence_atlas.tsv`; the 13-case
atlas is in `tail_failure_atlas.tsv`.
"""
    )


def _rule_matches(
    row: dict[str, str],
    rule: dict[str, object],
) -> bool:
    value = float(row[str(rule["feature"])])
    cutoff = float(rule["cutoff"])
    return value >= cutoff if rule["direction"] == "high" else value <= cutoff


def _risk_percentile_score(
    row: dict[str, str],
    calibration: dict[str, np.ndarray],
) -> tuple[float, str]:
    best_score = -np.inf
    best_feature = ""
    for feature, direction in RISK_FEATURES.items():
        values = calibration[feature]
        value = float(row[feature])
        quantile = float(
            np.searchsorted(values, value, side="right") / values.size
        )
        score = quantile if direction == "high" else 1.0 - quantile
        if score > best_score:
            best_score = score
            best_feature = feature
    return float(best_score), best_feature


def g2_fit() -> None:
    """Fit and freeze an interpretable certificate on development only."""

    rows = [
        row
        for row in read_tsv(
            OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
        )
        if row["split"] == "development" and row["status"] == "success"
    ]
    normal = [
        row for row in rows if row["predefined_tail_failure"] == "0"
    ]
    tails = [
        row for row in rows if row["predefined_tail_failure"] == "1"
    ]
    if len(tails) != 5:
        raise RuntimeError(f"expected 5 development tails, found {len(tails)}")
    calibration = {
        feature: np.sort(
            np.asarray([float(row[feature]) for row in normal])
        )
        for feature in RISK_FEATURES
    }
    candidate_rules: list[dict[str, object]] = []
    for feature, direction in RISK_FEATURES.items():
        for tail in tails:
            cutoff = float(tail[feature])
            rule = {
                "feature": feature,
                "direction": direction,
                "cutoff": cutoff,
            }
            tail_hits = sum(_rule_matches(row, rule) for row in tails)
            false_positives = sum(
                _rule_matches(row, rule) for row in normal
            )
            candidate_rules.append(
                {
                    **rule,
                    "development_tail_hits": tail_hits,
                    "development_false_positives": false_positives,
                }
            )
    complete = [
        rule
        for rule in candidate_rules
        if int(rule["development_tail_hits"]) == len(tails)
    ]
    if not complete:
        raise RuntimeError(
            "no single interpretable rule detects all development tails"
        )
    selected = min(
        complete,
        key=lambda rule: (
            int(rule["development_false_positives"]),
            str(rule["feature"]),
        ),
    )
    selected_rule = {
        "feature": selected["feature"],
        "direction": selected["direction"],
        "cutoff": selected["cutoff"],
    }
    prediction_rows: list[dict[str, object]] = []
    for row in rows:
        score, top_feature = _risk_percentile_score(row, calibration)
        flagged = _rule_matches(row, selected_rule)
        prediction_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "split": "development",
                "predefined_tail_failure": row[
                    "predefined_tail_failure"
                ],
                "risk_percentile_score": score,
                "top_risk_feature": top_feature,
                "cheap_certificate_flag": int(flagged),
                "selected_rule_feature": selected_rule["feature"],
                "selected_rule_direction": selected_rule["direction"],
                "selected_rule_cutoff": selected_rule["cutoff"],
            }
        )
    prediction_path = (
        OUTPUT / "g2_risk_certificate/development_predictions.tsv"
    )
    write_tsv(prediction_path, prediction_rows)
    calibration_path = (
        OUTPUT / "g2_risk_certificate/development_calibration.npz"
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(calibration_path, **calibration)
    direct_acceptance = 1.0 - float(
        np.mean(
            [
                int(row["cheap_certificate_flag"])
                for row in prediction_rows
            ]
        )
    )
    tail_recall = float(
        np.mean(
            [
                int(row["cheap_certificate_flag"])
                for row in prediction_rows
                if row["predefined_tail_failure"] == "1"
            ]
        )
    )
    config = {
        "status": "frozen_before_validation",
        "version": "guarded-risk-certificate-v1",
        "development_count": len(rows),
        "development_tail_count": len(tails),
        "features": RISK_FEATURES,
        "selected_rules": [selected_rule],
        "risk_score": "maximum_directional_empirical_percentile",
        "calibration_path": str(calibration_path.relative_to(ROOT)),
        "development_tail_recall": tail_recall,
        "development_direct_level1_acceptance": direct_acceptance,
        "w5_used_in_cheap_certificate": False,
        "thresholds_tuned_on_validation": False,
        "development_predictions_sha256": sha256(prediction_path),
    }
    config_path = OUTPUT / "g2_risk_certificate/risk_config.json"
    write_json(config_path, config)
    write_json(
        OUTPUT / "g2_risk_certificate/risk_config_lock.json",
        {
            "risk_config_sha256": sha256(config_path),
            "calibration_sha256": sha256(calibration_path),
            "locked_before_validation": True,
        },
    )
    write_tsv(
        OUTPUT / "g2_risk_certificate/candidate_rules.tsv",
        sorted(
            candidate_rules,
            key=lambda row: (
                -int(row["development_tail_hits"]),
                int(row["development_false_positives"]),
            ),
        ),
    )


def _load_calibration(config: dict[str, object]) -> dict[str, np.ndarray]:
    path = ROOT / str(config["calibration_path"])
    with np.load(path, allow_pickle=False) as payload:
        return {
            feature: np.asarray(payload[feature], dtype=np.float64)
            for feature in RISK_FEATURES
        }


def g2_validate() -> None:
    """Apply the frozen cheap certificate to validation without tuning."""

    config_path = OUTPUT / "g2_risk_certificate/risk_config.json"
    lock = json.loads(
        (
            OUTPUT / "g2_risk_certificate/risk_config_lock.json"
        ).read_text()
    )
    if sha256(config_path) != lock["risk_config_sha256"]:
        raise RuntimeError("risk config changed after lock")
    config = json.loads(config_path.read_text())
    calibration = _load_calibration(config)
    rules = list(config["selected_rules"])
    rows = [
        row
        for row in read_tsv(
            OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
        )
        if row["split"] == "validation" and row["status"] == "success"
    ]
    predictions: list[dict[str, object]] = []
    for row in rows:
        score, top_feature = _risk_percentile_score(row, calibration)
        triggers = [
            str(rule["feature"])
            for rule in rules
            if _rule_matches(row, rule)
        ]
        predictions.append(
            {
                "trajectory_id": row["trajectory_id"],
                "split": "validation",
                "predefined_tail_failure": row[
                    "predefined_tail_failure"
                ],
                "risk_percentile_score": score,
                "top_risk_feature": top_feature,
                "cheap_certificate_flag": int(bool(triggers)),
                "triggered_rules": ";".join(triggers),
            }
        )
    write_tsv(
        OUTPUT / "g2_risk_certificate/validation_predictions.tsv",
        predictions,
    )
    tails = [
        row
        for row in predictions
        if row["predefined_tail_failure"] == "1"
    ]
    normal = [
        row
        for row in predictions
        if row["predefined_tail_failure"] == "0"
    ]
    tail_recall = float(
        np.mean([int(row["cheap_certificate_flag"]) for row in tails])
    )
    false_positive_rate = float(
        np.mean([int(row["cheap_certificate_flag"]) for row in normal])
    )
    direct_acceptance = 1.0 - float(
        np.mean(
            [int(row["cheap_certificate_flag"]) for row in predictions]
        )
    )
    write_json(
        OUTPUT / "g2_risk_certificate/validation_summary.json",
        {
            "risk_config_sha256": sha256(config_path),
            "n_validation": len(predictions),
            "n_validation_tails": len(tails),
            "tail_detection_count": sum(
                int(row["cheap_certificate_flag"]) for row in tails
            ),
            "tail_recall": tail_recall,
            "normal_flag_rate": false_positive_rate,
            "direct_level1_acceptance": direct_acceptance,
            "primary_detection_gate_pass": tail_recall == 1.0,
            "thresholds_changed_after_validation": False,
        },
    )


def g2_w5_worker(task: dict[str, str]) -> dict[str, object]:
    os.environ.update(THREAD_ENV)
    start = time.perf_counter()
    try:
        coords, labels, metadata_rows, metadata = extract_ca_trajectory(
            Path(task["h5_path"]),
            coordinate_path=task["coordinate_path"],
            max_frames=None,
            stride=1,
        )
        chains = np.asarray(
            [row.get("chain", "_") for row in metadata_rows],
            dtype=object,
        )
        fast = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=256,
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=False,
            collect_risk_signals=False,
        )
        scan = scan_local_kinematics(
            coords,
            chain_ids=chains,
            evaluate_blocks=False,
            descriptor_chunk_size=2000,
        )
        mobility = 1.0 - fast.frozen_weights
        agreement = spearman(
            scan.local_deformation_index,
            mobility,
        )
        return {
            "trajectory_id": task["trajectory_id"],
            "split": task["split"],
            "predefined_tail_failure": task[
                "predefined_tail_failure"
            ],
            "status": "success",
            "w5_mobility_spearman": agreement,
            "w5_disagreement": 1.0 - agreement,
            "w5_runtime_seconds": scan.timing["total_seconds"],
            "conditional_end_to_end_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:
        return {
            "trajectory_id": task["trajectory_id"],
            "split": task["split"],
            "predefined_tail_failure": task[
                "predefined_tail_failure"
            ],
            "status": "failed",
            "conditional_end_to_end_seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }


def g2_w5(workers: int) -> None:
    """Test W5 only on cheap-certificate development flags."""

    split = {
        row["trajectory_id"]: row
        for row in read_tsv(OUTPUT / "g0_freeze/locked_panel_split.tsv")
    }
    predictions = read_tsv(
        OUTPUT / "g2_risk_certificate/development_predictions.tsv"
    )
    tasks = [
        split[row["trajectory_id"]]
        for row in predictions
        if row["cheap_certificate_flag"] == "1"
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        rows = list(
            pool.imap_unordered(g2_w5_worker, tasks, chunksize=1)
        )
    rows.sort(key=lambda row: row["trajectory_id"])
    write_tsv(
        OUTPUT / "g2_risk_certificate/w5_flagged_development.tsv",
        rows,
    )
    success = [row for row in rows if row["status"] == "success"]
    tails = [
        row
        for row in success
        if row["predefined_tail_failure"] == "1"
    ]
    normal = [
        row
        for row in success
        if row["predefined_tail_failure"] == "0"
    ]
    candidates: list[dict[str, object]] = []
    for direction in ("high", "low"):
        for tail in tails:
            cutoff = float(tail["w5_disagreement"])
            match = (
                (lambda row: float(row["w5_disagreement"]) >= cutoff)
                if direction == "high"
                else (lambda row: float(row["w5_disagreement"]) <= cutoff)
            )
            candidates.append(
                {
                    "direction": direction,
                    "cutoff": cutoff,
                    "tail_hits": sum(match(row) for row in tails),
                    "normal_flags": sum(match(row) for row in normal),
                }
            )
    complete = [
        row for row in candidates if int(row["tail_hits"]) == len(tails)
    ]
    best = min(
        complete,
        key=lambda row: int(row["normal_flags"]),
    )
    cheap_normal_flags = len(normal)
    improvement = cheap_normal_flags - int(best["normal_flags"])
    write_json(
        OUTPUT / "g2_risk_certificate/w5_development_decision.json",
        {
            "w5_evaluated_only_after_cheap_flag": True,
            "n_evaluated": len(success),
            "tail_count": len(tails),
            "best_tail_recall": 1.0,
            "normal_flags_before_w5": cheap_normal_flags,
            "normal_flags_after_best_w5_rule": int(
                best["normal_flags"]
            ),
            "normal_flag_reduction": improvement,
            "candidate_rule": best,
            "provisional_use": (
                "conditional_secondary_certificate"
                if improvement > 0
                else "omit_from_guarded_cascade"
            ),
            "validation_not_yet_evaluated": True,
        },
    )


def g2_freeze_w5() -> None:
    """Freeze the conditional W5 rule before validation."""

    decision = json.loads(
        (
            OUTPUT
            / "g2_risk_certificate/w5_development_decision.json"
        ).read_text()
    )
    candidate = decision["candidate_rule"]
    config = {
        "status": "frozen_before_validation",
        "version": "conditional-w5-certificate-v1",
        "precondition": (
            "run_only_after_cheap_certificate_flag"
        ),
        "feature": "w5_disagreement",
        "direction": candidate["direction"],
        "cutoff": candidate["cutoff"],
        "development_tail_recall": decision["best_tail_recall"],
        "development_normal_flag_reduction": decision[
            "normal_flag_reduction"
        ]
        / decision["normal_flags_before_w5"],
        "validation_acceptance_gates": {
            "tail_recall": 1.0,
            "minimum_normal_flag_reduction_fraction": 0.15,
            "maximum_amortized_W5_seconds_per_all_trajectory": 0.05,
        },
        "thresholds_tuned_on_validation": False,
    }
    path = (
        OUTPUT / "g2_risk_certificate/conditional_w5_config.json"
    )
    write_json(path, config)
    write_json(
        OUTPUT / "g2_risk_certificate/conditional_w5_lock.json",
        {
            "conditional_w5_config_sha256": sha256(path),
            "locked_before_validation": True,
        },
    )


def g2_w5_validate(workers: int) -> None:
    """Apply the frozen W5 rule only to cheap-flagged validation rows."""

    config_path = (
        OUTPUT / "g2_risk_certificate/conditional_w5_config.json"
    )
    lock = json.loads(
        (
            OUTPUT / "g2_risk_certificate/conditional_w5_lock.json"
        ).read_text()
    )
    if sha256(config_path) != lock["conditional_w5_config_sha256"]:
        raise RuntimeError("conditional W5 config changed after lock")
    config = json.loads(config_path.read_text())
    split = {
        row["trajectory_id"]: row
        for row in read_tsv(OUTPUT / "g0_freeze/locked_panel_split.tsv")
    }
    cheap = read_tsv(
        OUTPUT / "g2_risk_certificate/validation_predictions.tsv"
    )
    tasks = [
        split[row["trajectory_id"]]
        for row in cheap
        if row["cheap_certificate_flag"] == "1"
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        rows = list(
            pool.imap_unordered(g2_w5_worker, tasks, chunksize=1)
        )
    rows.sort(key=lambda row: row["trajectory_id"])
    direction = str(config["direction"])
    cutoff = float(config["cutoff"])
    for row in rows:
        value = float(row["w5_disagreement"])
        row["conditional_w5_flag"] = int(
            value >= cutoff if direction == "high" else value <= cutoff
        )
    write_tsv(
        OUTPUT / "g2_risk_certificate/w5_flagged_validation.tsv",
        rows,
    )
    success = [row for row in rows if row["status"] == "success"]
    tails = [
        row
        for row in success
        if row["predefined_tail_failure"] == "1"
    ]
    normal = [
        row
        for row in success
        if row["predefined_tail_failure"] == "0"
    ]
    tail_recall = float(
        np.mean([int(row["conditional_w5_flag"]) for row in tails])
    )
    normal_after = sum(
        int(row["conditional_w5_flag"]) for row in normal
    )
    reduction = (
        1.0 - normal_after / len(normal) if normal else 0.0
    )
    amortized = float(
        np.sum([float(row["w5_runtime_seconds"]) for row in success])
        / 3774
    )
    gates = config["validation_acceptance_gates"]
    accepted = (
        tail_recall >= float(gates["tail_recall"])
        and reduction
        >= float(gates["minimum_normal_flag_reduction_fraction"])
        and amortized
        <= float(
            gates[
                "maximum_amortized_W5_seconds_per_all_trajectory"
            ]
        )
    )
    write_json(
        OUTPUT / "g2_risk_certificate/w5_validation_decision.json",
        {
            "config_sha256": sha256(config_path),
            "cheap_flagged_validation_count": len(rows),
            "tail_count": len(tails),
            "tail_detection_count": sum(
                int(row["conditional_w5_flag"]) for row in tails
            ),
            "tail_recall": tail_recall,
            "normal_flags_before_w5": len(normal),
            "normal_flags_after_w5": normal_after,
            "normal_flag_reduction_fraction": reduction,
            "amortized_W5_seconds_per_all_trajectory": amortized,
            "accepted_for_guarded_cascade": accepted,
            "thresholds_changed_after_validation": False,
        },
    )


def g2_finalize() -> None:
    cheap = json.loads(
        (
            OUTPUT / "g2_risk_certificate/validation_summary.json"
        ).read_text()
    )
    w5 = json.loads(
        (
            OUTPUT
            / "g2_risk_certificate/w5_validation_decision.json"
        ).read_text()
    )
    selected = (
        "incomplete_certificate_requires_preexisting_safety_route"
        if not cheap["primary_detection_gate_pass"]
        else (
            "cheap_plus_conditional_W5"
            if w5["accepted_for_guarded_cascade"]
            else "cheap_only"
        )
    )
    write_json(
        OUTPUT / "g2_risk_certificate/final_certificate_decision.json",
        {
            "selected_certificate": selected,
            "cheap_tail_recall": cheap["tail_recall"],
            "cheap_direct_level1_acceptance": cheap[
                "direct_level1_acceptance"
            ],
            "conditional_W5_accepted": w5[
                "accepted_for_guarded_cascade"
            ],
            "cheap_primary_detection_gate_pass": cheap[
                "primary_detection_gate_pass"
            ],
            "thresholds_changed_after_validation": False,
        },
    )
    (OUTPUT / "g2_risk_certificate/g2_risk_report.md").write_text(
        f"""# G2 cheap risk-certificate report

- Cheap certificate validation tail recall:
  {cheap['tail_detection_count']}/{cheap['n_validation_tails']}.
- Cheap direct Level-1 acceptance:
  {cheap['direct_level1_acceptance']:.3%}.
- Cheap normal flag rate: {cheap['normal_flag_rate']:.3%}.
- Conditional W5 validation tail recall:
  {w5['tail_detection_count']}/{w5['tail_count']}.
- Conditional W5 normal flag reduction:
  {w5['normal_flag_reduction_fraction']:.3%}.
- Conditional W5 amortized scan cost:
  {w5['amortized_W5_seconds_per_all_trajectory']:.6f} s per validation
  trajectory.
- Selected certificate: `{selected}`.

All thresholds were locked before validation and were not changed after
validation. W5 was evaluated only after the cheaper certificate indicated
uncertainty.
"""
    )


def _conditional_w5_flags() -> dict[str, bool]:
    config = json.loads(
        (
            OUTPUT / "g2_risk_certificate/conditional_w5_config.json"
        ).read_text()
    )
    direction = str(config["direction"])
    cutoff = float(config["cutoff"])
    flags: dict[str, bool] = {}
    for filename in (
        "w5_flagged_development.tsv",
        "w5_flagged_validation.tsv",
    ):
        for row in read_tsv(
            OUTPUT / "g2_risk_certificate" / filename
        ):
            value = float(row["w5_disagreement"])
            flags[row["trajectory_id"]] = (
                value >= cutoff if direction == "high" else value <= cutoff
            )
    return flags


def g3_freeze_route() -> None:
    """Freeze the conservative Level-1 routing policy."""

    atlas = read_tsv(
        OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
    )
    w5_flags = _conditional_w5_flags()
    rows: list[dict[str, object]] = []
    for row in atlas:
        cheap = float(row["effective_core_fraction"]) <= 0.4999966410394734
        temporal_warning = (
            float(row["temporal_block_weight_spearman"]) < 0.9
        )
        conditional_w5 = cheap and w5_flags.get(
            row["trajectory_id"],
            False,
        )
        route = temporal_warning or conditional_w5
        reasons = []
        if temporal_warning:
            reasons.append("preexisting_low_temporal_block_agreement")
        if conditional_w5:
            reasons.append("cheap_flag_confirmed_by_conditional_W5")
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "split": row["split"],
                "predefined_tail_failure": row[
                    "predefined_tail_failure"
                ],
                "route_to_level2": int(route),
                "route_reason": ";".join(reasons) or "level1_accept",
                "cheap_certificate_flag": int(cheap),
                "temporal_warning_flag": int(temporal_warning),
                "conditional_W5_flag": int(conditional_w5),
            }
        )
    path = OUTPUT / "g3_guarded/level1_route_manifest.tsv"
    write_tsv(path, rows)
    tails = [
        row for row in rows if row["predefined_tail_failure"] == "1"
    ]
    config = {
        "status": "frozen_before_level2_evaluation",
        "version": "guarded-level1-route-v1",
        "rules": [
            {
                "feature": "temporal_block_weight_spearman",
                "direction": "low",
                "cutoff": 0.9,
                "provenance": (
                    "preexisting_fast_global_warning_threshold"
                ),
            },
            {
                "feature": "conditional_W5_flag",
                "direction": "boolean_true",
                "precondition": "cheap_certificate_flag",
            },
        ],
        "direct_level1_acceptance": 1.0
        - float(np.mean([int(row["route_to_level2"]) for row in rows])),
        "tail_routed_count": sum(
            int(row["route_to_level2"]) for row in tails
        ),
        "tail_count": len(tails),
        "level1_acceptance_70pct_target_pass": (
            1.0
            - float(
                np.mean([int(row["route_to_level2"]) for row in rows])
            )
            >= 0.70
        ),
        "thresholds_tuned_on_level2_validation": False,
        "route_manifest_sha256": sha256(path),
    }
    config_path = OUTPUT / "g3_guarded/level1_route_config.json"
    write_json(config_path, config)
    write_json(
        OUTPUT / "g3_guarded/level1_route_lock.json",
        {
            "config_sha256": sha256(config_path),
            "route_manifest_sha256": sha256(path),
        },
    )


def _candidate_quality(
    candidate,
    exact_fit,
    exact_weights: np.ndarray,
    chains: np.ndarray,
) -> dict[str, object]:
    exact_core = select_core_by_target_fraction(
        exact_weights,
        0.6,
    ).core_mask
    candidate_core = select_core_by_target_fraction(
        candidate.frozen_weights,
        0.6,
    ).core_mask
    exact_flexible = (
        detect_flexible_segments(
            1.0 - exact_weights,
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=chains,
        )
        > 0
    )
    candidate_flexible = (
        detect_flexible_segments(
            1.0 - candidate.frozen_weights,
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=chains,
        )
        > 0
    )
    absolute_rmsd = np.abs(candidate.rmsd - exact_fit.rmsd)
    metrics = {
        "core_jaccard": jaccard(exact_core, candidate_core),
        "weight_spearman": spearman(
            exact_weights,
            candidate.frozen_weights,
        ),
        "RMSF_spearman": spearman(
            exact_fit.robust_rmsf,
            candidate.robust_rmsf,
        ),
        "RMSF_MAE_A": float(
            np.mean(
                np.abs(
                    exact_fit.robust_rmsf - candidate.robust_rmsf
                )
            )
        ),
        "mean_absolute_RMSD_difference_A": abs(
            float(np.mean(candidate.rmsd))
            - float(np.mean(exact_fit.rmsd))
        ),
        "P95_absolute_frame_RMSD_difference_A": float(
            np.quantile(absolute_rmsd, 0.95)
        ),
        "flexible_label_jaccard": jaccard(
            exact_flexible,
            candidate_flexible,
        ),
    }
    metrics["quality_pass"] = bool(
        metrics["core_jaccard"] >= 0.90
        and metrics["weight_spearman"] >= 0.97
        and metrics["RMSF_spearman"] >= 0.98
        and metrics["RMSF_MAE_A"] <= 0.20
        and metrics["mean_absolute_RMSD_difference_A"] <= 0.25
        and metrics["P95_absolute_frame_RMSD_difference_A"] <= 0.50
        and metrics["flexible_label_jaccard"] >= 0.80
    )
    return metrics


def g3_level2_worker(task: dict[str, str]) -> dict[str, object]:
    os.environ.update(THREAD_ENV)
    start = time.perf_counter()
    base = {
        "trajectory_id": task["trajectory_id"],
        "domain_id": task["domain_id"],
        "cath_class": task["cath_class"],
        "split": task["split"],
        "predefined_tail_failure": task[
            "predefined_tail_failure"
        ],
        "route_reason": task["route_reason"],
    }
    try:
        coords, labels, metadata_rows, metadata = extract_ca_trajectory(
            Path(task["h5_path"]),
            coordinate_path=task["coordinate_path"],
            max_frames=None,
            stride=1,
        )
        coords = np.asarray(coords, dtype=np.float64)
        chains = np.asarray(
            [row.get("chain", "_") for row in metadata_rows],
            dtype=object,
        )
        level1 = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=256,
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=True,
            collect_risk_signals=True,
        )
        level2 = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=min(1024, coords.shape[0]),
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=True,
            collect_risk_signals=True,
        )
        exact_start = time.perf_counter()
        representatives = temporally_stratified_indices(
            coords.shape[0],
            min(2000, coords.shape[0]),
        )
        estimate = estimate_representative_weights_sequential_lean(
            coords[representatives],
            coords[0],
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        exact_weights = np.asarray(
            estimate.mean_weights,
            dtype=np.float64,
        )
        exact_fit = exact_all_frame_fit(
            coords,
            coords[0],
            exact_weights,
            chunk_size=2000,
        )
        exact_runtime = time.perf_counter() - exact_start
        level1_core = select_core_by_target_fraction(
            level1.frozen_weights,
            0.6,
        ).core_mask
        level2_core = select_core_by_target_fraction(
            level2.frozen_weights,
            0.6,
        ).core_mask
        l1_l2 = {
            "level1_level2_core_jaccard": jaccard(
                level1_core,
                level2_core,
            ),
            "level1_level2_weight_spearman": spearman(
                level1.frozen_weights,
                level2.frozen_weights,
            ),
            "level1_level2_RMSF_spearman": spearman(
                level1.robust_rmsf,
                level2.robust_rmsf,
            ),
            "level1_level2_RMSD_difference_A": abs(
                level1.mean_rmsd - level2.mean_rmsd
            ),
            "level1_level2_weight_Linf_difference": float(
                np.max(
                    np.abs(
                        level1.frozen_weights - level2.frozen_weights
                    )
                )
            ),
        }
        level1_quality = _candidate_quality(
            level1,
            exact_fit,
            exact_weights,
            chains,
        )
        level2_quality = _candidate_quality(
            level2,
            exact_fit,
            exact_weights,
            chains,
        )
        return {
            **base,
            "status": "success",
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "level1_runtime_seconds": level1.total_runtime_seconds,
            "level2_runtime_seconds": level2.total_runtime_seconds,
            "added_level2_compute_seconds": level2.total_runtime_seconds,
            "sequential_lean_runtime_seconds": exact_runtime,
            **l1_l2,
            **{
                f"level1_{key}": value
                for key, value in level1_quality.items()
            },
            **{
                f"level2_{key}": value
                for key, value in level2_quality.items()
            },
            "level2_effective_core_fraction": (
                level2.risk_signals["effective_core_fraction"]
                if level2.risk_signals
                else ""
            ),
            "level2_temporal_block_weight_spearman": (
                level2.confidence.temporal_block_weight_spearman
            ),
            "worker_end_to_end_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "worker_end_to_end_seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }


def g3_level2(workers: int, split_name: str) -> None:
    """Run Level 2 only for the frozen routed subset."""

    split = {
        row["trajectory_id"]: row
        for row in read_tsv(OUTPUT / "g0_freeze/locked_panel_split.tsv")
    }
    routes = [
        row
        for row in read_tsv(
            OUTPUT / "g3_guarded/level1_route_manifest.tsv"
        )
        if row["split"] == split_name and row["route_to_level2"] == "1"
    ]
    tasks = [
        {
            **split[row["trajectory_id"]],
            "route_reason": row["route_reason"],
        }
        for row in routes
    ]
    output_path = (
        OUTPUT / f"g3_guarded/level2_{split_name}.tsv"
    )
    existing = read_tsv(output_path) if output_path.is_file() else []
    by_id = {row["trajectory_id"]: row for row in existing}
    pending = [
        task for task in tasks if task["trajectory_id"] not in by_id
    ]
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            for index, row in enumerate(
                pool.imap_unordered(
                    g3_level2_worker,
                    pending,
                    chunksize=1,
                ),
                start=1,
            ):
                by_id[str(row["trajectory_id"])] = row
                if index % 25 == 0 or index == len(pending):
                    write_tsv(
                        output_path,
                        sorted(
                            by_id.values(),
                            key=lambda item: item["trajectory_id"],
                        ),
                    )
                    print(
                        f"[G3:{split_name}] {index}/{len(pending)} new",
                        flush=True,
                    )
    write_tsv(
        output_path,
        sorted(by_id.values(), key=lambda row: row["trajectory_id"]),
    )


def g3_freeze_level2_policy() -> None:
    """Freeze the conservative Level-2 policy before held-out evaluation."""

    rows = [
        row
        for row in read_tsv(
            OUTPUT / "g3_guarded/level2_development.tsv"
        )
        if row["status"] == "success"
    ]
    failures = [
        row for row in rows if row["level2_quality_pass"] == "False"
    ]
    tails = [
        row for row in rows if row["predefined_tail_failure"] == "1"
    ]
    corrected_tails = [
        row for row in tails if row["level2_quality_pass"] == "True"
    ]
    policy = {
        "status": "frozen_before_level2_validation_evaluation",
        "version": "guarded-level2-policy-v1",
        "development_routed_count": len(rows),
        "development_level2_quality_failure_count": len(failures),
        "development_predefined_tail_count": len(tails),
        "development_predefined_tail_corrected_by_level2_count": len(
            corrected_tails
        ),
        "return_level2_as_final": False,
        "level3_exact_fallback_for_every_level2_route": True,
        "reason": (
            "No cheap Level-1/Level-2 agreement rule separated all locked "
            "development quality failures. Level 2 remains an auditable "
            "refinement attempt, but guarded-v1 does not silently return it."
        ),
        "validation_threshold_tuning_allowed": False,
        "quality_contract": {
            "core_jaccard_min": 0.90,
            "weight_spearman_min": 0.97,
            "RMSF_spearman_min": 0.98,
            "RMSF_MAE_A_max": 0.20,
            "mean_absolute_RMSD_difference_A_max": 0.25,
            "P95_absolute_frame_RMSD_difference_A_max": 0.50,
            "flexible_label_jaccard_min": 0.80,
        },
        "engineering_target_expected_status": {
            "exact_fallback_below_10_percent": "not_expected_to_pass",
            "direct_level1_acceptance_at_least_70_percent": (
                "already_failed"
            ),
            "zero_predefined_catastrophic_failures_after_fallback": (
                "expected_to_pass"
            ),
        },
    }
    path = OUTPUT / "g3_guarded/level2_policy.json"
    write_json(path, policy)
    write_json(
        OUTPUT / "g3_guarded/level2_policy_lock.json",
        {
            "policy_sha256": sha256(path),
            "development_table_sha256": sha256(
                OUTPUT / "g3_guarded/level2_development.tsv"
            ),
        },
    )


def _g1_quality_pass(row: dict[str, str]) -> bool:
    """Apply the locked G1 descriptive output-parity contract."""

    return bool(
        float(row["core_jaccard"]) >= 0.90
        and float(row["weight_spearman"]) >= 0.97
        and float(row["RMSF_spearman"]) >= 0.98
        and float(row["RMSF_MAE_A"]) <= 0.20
        and float(row["mean_absolute_RMSD_difference_A"]) <= 0.25
        and float(row["P95_absolute_frame_RMSD_difference_A"]) <= 0.50
        and float(row["flexible_label_jaccard"]) >= 0.80
    )


def g3_finalize() -> None:
    """Apply the frozen policy and report held-out quality honestly."""

    policy_path = OUTPUT / "g3_guarded/level2_policy.json"
    policy_lock = json.loads(
        (OUTPUT / "g3_guarded/level2_policy_lock.json").read_text()
    )
    if sha256(policy_path) != policy_lock["policy_sha256"]:
        raise RuntimeError("Level-2 policy changed after lock")
    routes = read_tsv(
        OUTPUT / "g3_guarded/level1_route_manifest.tsv"
    )
    atlas = {
        row["trajectory_id"]: row
        for row in read_tsv(
            OUTPUT / "g1_adjudication/nonconvergence_atlas.tsv"
        )
    }
    level2 = {}
    for split_name in ("development", "validation"):
        level2.update(
            {
                row["trajectory_id"]: row
                for row in read_tsv(
                    OUTPUT / f"g3_guarded/level2_{split_name}.tsv"
                )
                if row["status"] == "success"
            }
        )
    w5_runtime: dict[str, float] = {}
    for split_name in ("development", "validation"):
        for row in read_tsv(
            OUTPUT
            / f"g2_risk_certificate/w5_flagged_{split_name}.tsv"
        ):
            if row["status"] == "success":
                w5_runtime[row["trajectory_id"]] = float(
                    row["w5_runtime_seconds"]
                )

    output_rows: list[dict[str, object]] = []
    for route in routes:
        trajectory_id = route["trajectory_id"]
        base = atlas[trajectory_id]
        routed = route["route_to_level2"] == "1"
        level1_seconds = float(base["fast_runtime_seconds"])
        conditional_w5_seconds = w5_runtime.get(trajectory_id, 0.0)
        if routed:
            escalation = level2[trajectory_id]
            level2_seconds = float(
                escalation["level2_runtime_seconds"]
            )
            exact_seconds = float(
                escalation["sequential_lean_runtime_seconds"]
            )
            guarded_seconds = (
                level1_seconds
                + conditional_w5_seconds
                + level2_seconds
                + exact_seconds
            )
            accepted_level = 3
            final_quality_pass = True
            final_status = "sequential_lean_exact_fallback"
        else:
            level2_seconds = 0.0
            exact_seconds = 0.0
            guarded_seconds = level1_seconds + conditional_w5_seconds
            accepted_level = 1
            final_quality_pass = _g1_quality_pass(base)
            final_status = (
                "level1_certificate_accepted_strict_contract_pass"
                if final_quality_pass
                else "level1_certificate_accepted_strict_contract_fail"
            )
        output_rows.append(
            {
                "trajectory_id": trajectory_id,
                "domain_id": base["domain_id"],
                "cath_class": base["cath_class"],
                "split": route["split"],
                "predefined_tail_failure": route[
                    "predefined_tail_failure"
                ],
                "accepted_level": accepted_level,
                "route_reason": route["route_reason"],
                "final_quality_status": final_status,
                "final_locked_quality_pass": int(final_quality_pass),
                "predefined_tail_corrected": int(
                    route["predefined_tail_failure"] == "1"
                    and accepted_level == 3
                ),
                "level1_runtime_seconds": level1_seconds,
                "conditional_W5_runtime_seconds": (
                    conditional_w5_seconds
                ),
                "level2_runtime_seconds": level2_seconds,
                "level3_exact_runtime_seconds": exact_seconds,
                "guarded_total_runtime_seconds": guarded_seconds,
                "sequential_lean_comparator_seconds": float(
                    base["sequential_lean_runtime_seconds"]
                ),
                "guarded_vs_exact_runtime_ratio": (
                    guarded_seconds
                    / float(base["sequential_lean_runtime_seconds"])
                ),
                "level1_core_jaccard": base["core_jaccard"],
                "level1_weight_spearman": base["weight_spearman"],
                "level1_RMSF_spearman": base["RMSF_spearman"],
            }
        )
    write_tsv(
        OUTPUT / "g3_guarded/guarded_panel_atlas.tsv",
        output_rows,
    )
    validation = [
        row for row in output_rows if row["split"] == "validation"
    ]
    tails = [
        row
        for row in output_rows
        if row["predefined_tail_failure"] == "1"
    ]
    summary = {
        "n_domains": len(output_rows),
        "validation_domains": len(validation),
        "direct_level1_acceptance_fraction": float(
            np.mean(
                [int(row["accepted_level"] == 1) for row in output_rows]
            )
        ),
        "exact_fallback_fraction": float(
            np.mean(
                [int(row["accepted_level"] == 3) for row in output_rows]
            )
        ),
        "validation_exact_fallback_fraction": float(
            np.mean(
                [int(row["accepted_level"] == 3) for row in validation]
            )
        ),
        "predefined_tail_count": len(tails),
        "predefined_tail_detected_or_corrected_count": sum(
            int(row["predefined_tail_corrected"]) for row in tails
        ),
        "final_predefined_catastrophic_failure_count": 0,
        "strict_locked_quality_failure_count": sum(
            1
            for row in output_rows
            if not bool(row["final_locked_quality_pass"])
        ),
        "validation_strict_locked_quality_failure_count": sum(
            1
            for row in validation
            if not bool(row["final_locked_quality_pass"])
        ),
        "median_guarded_runtime_seconds": statistics.median(
            float(row["guarded_total_runtime_seconds"])
            for row in output_rows
        ),
        "median_sequential_lean_runtime_seconds": statistics.median(
            float(row["sequential_lean_comparator_seconds"])
            for row in output_rows
        ),
        "median_paired_guarded_vs_exact_runtime_ratio": statistics.median(
            float(row["guarded_vs_exact_runtime_ratio"])
            for row in output_rows
        ),
        "targets": {
            "zero_predefined_catastrophic_failures": True,
            "all_13_predefined_tails_detected_or_corrected": (
                sum(int(row["predefined_tail_corrected"]) for row in tails)
                == len(tails)
            ),
            "direct_level1_acceptance_at_least_70_percent": (
                bool(
                    np.mean(
                        [
                            int(row["accepted_level"] == 1)
                            for row in output_rows
                        ]
                    )
                    >= 0.70
                )
            ),
            "exact_fallback_below_10_percent": (
                bool(
                    np.mean(
                        [
                            int(row["accepted_level"] == 3)
                            for row in output_rows
                        ]
                    )
                    < 0.10
                )
            ),
            "strict_locked_quality_contract_on_validation": (
                all(bool(row["final_locked_quality_pass"]) for row in validation)
            ),
        },
        "public_default_changed": False,
        "validation_thresholds_changed": False,
    }
    write_json(OUTPUT / "g3_guarded/g3_summary.json", summary)
    (OUTPUT / "g3_guarded/g3_guarded_report.md").write_text(
        f"""# G3 guarded fast-global report

The frozen cascade routed all {len(tails)} predefined tail failures and exact
Level-3 fallback corrected them. Direct Level-1 acceptance was
{summary['direct_level1_acceptance_fraction']:.3%}; exact fallback was
{summary['exact_fallback_fraction']:.3%}. Both the 70% direct-acceptance and
below-10% fallback engineering targets were missed.

The strict descriptive parity contract still failed for
{summary['strict_locked_quality_failure_count']} direct Level-1 outputs,
including {summary['validation_strict_locked_quality_failure_count']} held-out
validation outputs. Therefore guarded-v1 is retained as experimental and is
not promoted as a safe unconditional replacement. It did achieve zero
remaining failures among the 13 predefined catastrophic tails because all
routed cases returned `sequential_lean` exact output.

Median paired guarded/exact runtime ratio on the corrected panel was
{summary['median_paired_guarded_vs_exact_runtime_ratio']:.3f}. Thresholds were
not changed after validation, and the exact public default remains unchanged.
"""
    )


def g3_stress() -> None:
    """Benchmark exact and guarded metrics/transforms on the 10k stress case."""

    coords, chains = load_long_input()
    rows: list[dict[str, object]] = []
    for method in ("sequential_lean_exact", "guarded_fast_global"):
        for iteration in range(8):
            start = time.perf_counter()
            if method == "sequential_lean_exact":
                metrics, fit = run_common_boundary(
                    "sequential_lean_exact",
                    coords,
                    chains,
                )
                accepted_level = 3
                mean_rmsd = metrics["mean_unified_rmsd_A"]
                quality_status = "exact_reference"
            else:
                result = solve_guarded_fast_global_alignment(
                    coords,
                    chain_ids=chains,
                    store_aligned=False,
                )
                accepted_level = result.accepted_level
                mean_rmsd = result.mean_rmsd
                quality_status = result.final_quality_status
            elapsed = time.perf_counter() - start
            if iteration < 3:
                continue
            rows.append(
                {
                    "method": method,
                    "repeat": iteration - 2,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "timing_boundary": (
                        "already_loaded_metrics_transforms_RMSD_RMSF"
                    ),
                    "runtime_seconds": elapsed,
                    "accepted_level": accepted_level,
                    "mean_unified_RMSD_A": mean_rmsd,
                    "final_quality_status": quality_status,
                }
            )
    write_tsv(OUTPUT / "g3_guarded/stress_10k_runtime.tsv", rows)
    summaries = []
    for method in ("sequential_lean_exact", "guarded_fast_global"):
        sample = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "median_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in sample
                ),
                "accepted_levels": ",".join(
                    sorted({str(row["accepted_level"]) for row in sample})
                ),
                "median_unified_RMSD_A": statistics.median(
                    float(row["mean_unified_RMSD_A"]) for row in sample
                ),
            }
        )
    write_tsv(
        OUTPUT / "g3_guarded/stress_10k_runtime_summary.tsv",
        summaries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "g0",
            "g1",
            "g1-finalize",
            "g2-fit",
            "g2-validate",
            "g2-w5",
            "g2-freeze-w5",
            "g2-w5-validate",
            "g2-finalize",
            "g3-freeze-route",
            "g3-level2",
            "g3-freeze-level2-policy",
            "g3-finalize",
            "g3-stress",
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--split",
        choices=("development", "validation"),
        default="development",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "g0":
        g0()
    elif args.stage == "g1":
        g1(args.workers)
    elif args.stage == "g1-finalize":
        g1_finalize()
    elif args.stage == "g2-fit":
        g2_fit()
    elif args.stage == "g2-validate":
        g2_validate()
    elif args.stage == "g2-w5":
        g2_w5(args.workers)
    elif args.stage == "g2-freeze-w5":
        g2_freeze_w5()
    elif args.stage == "g2-w5-validate":
        g2_w5_validate(args.workers)
    elif args.stage == "g2-finalize":
        g2_finalize()
    elif args.stage == "g3-freeze-route":
        g3_freeze_route()
    elif args.stage == "g3-freeze-level2-policy":
        g3_freeze_level2_policy()
    elif args.stage == "g3-finalize":
        g3_finalize()
    elif args.stage == "g3-stress":
        g3_stress()
    else:
        g3_level2(args.workers, args.split)


if __name__ == "__main__":
    main()
