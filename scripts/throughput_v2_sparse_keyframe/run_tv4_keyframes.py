#!/usr/bin/env python3
"""Run TV4 keyframe/signature/interpolation development ablations."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdcorealign.experimental_throughput_v2 import (  # noqa: E402
    build_frame_signature,
    fit_keyframe_alignment,
    fit_keyframe_transforms,
    select_keyframes,
    select_weighted_landmarks,
)
from mdsuperpose.fast_kabsch import apply_transforms_batch  # noqa: E402
from scripts.throughput_v2_sparse_keyframe.study_utils import (  # noqa: E402
    OUT,
    compare_alignment,
    frozen_current_throughput,
    load_study_case,
    read_tsv,
    write_tsv,
)


PHASE = OUT / "04_keyframes"
SIGNATURES = (
    "centered_coordinates",
    "pair_distances",
    "anchor_distances",
    "pca_centered",
    "hybrid",
)
KEYFRAME_METHODS = ("uniform", "novelty", "hierarchical", "max_gap_novelty")
BUDGETS = (
    (0.20, 10),
    (0.10, 25),
    (0.05, 50),
    (0.02, 100),
    (0.01, 250),
    (0.005, 500),
)


def development_cases() -> list[dict[str, str]]:
    rows = read_tsv(OUT / "07_fresh_validation/development_panel.tsv")
    atlas = [row for row in rows if row["source_class"].startswith("ATLAS")][:2]
    mdcath = [row for row in rows if row["source_class"].startswith("mdCATH")][:4]
    control = {
        "case_id": "transition_control_4oelA02_10000",
        "family_id": "MDCATH:4oelA02",
        "source_class": "transition_aligned_synthetic_scaling_control",
        "input_path": "registered_raw_replicates",
        "input_key": "transition_aligned_cycle",
        "n_frames": "10000",
        "n_residues": "153",
    }
    return atlas + mdcath + [control]


def audit_transform_quality(
    coords: np.ndarray,
    baseline,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> dict[str, object]:
    count = min(coords.shape[0], max(64, int(np.ceil(0.01 * coords.shape[0]))))
    audit = np.unique(np.rint(np.linspace(0, coords.shape[0] - 1, count)).astype(int))
    relative = rotations[audit] @ np.swapaxes(baseline.fit.rotations[audit], 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    candidate_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64), rotations[audit], translations[audit]
    )
    baseline_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64),
        baseline.fit.rotations[audit],
        baseline.fit.translations[audit],
    )
    coordinate = np.sqrt(
        np.mean(np.sum((candidate_aligned - baseline_aligned) ** 2, axis=2), axis=1)
    )
    reference = np.asarray(coords[0], dtype=np.float64)
    candidate_rmsd = np.sqrt(
        np.mean(np.sum((candidate_aligned - reference[None, :, :]) ** 2, axis=2), axis=1)
    )
    rmsd_difference = np.abs(candidate_rmsd - baseline.fit.rmsd[audit])
    catastrophic = bool(
        not np.isfinite(rotations).all()
        or np.any(np.linalg.det(rotations) <= 0.0)
        or float(np.median(coordinate)) > 2.0
        or float(np.quantile(rmsd_difference, 0.95)) > 2.0
    )
    return {
        "audit_frame_count": audit.size,
        "rotation_median_degrees": float(np.median(angle)),
        "rotation_p95_degrees": float(np.quantile(angle, 0.95)),
        "aligned_coordinate_audit_median_A": float(np.median(coordinate)),
        "aligned_coordinate_audit_p95_A": float(np.quantile(coordinate, 0.95)),
        "audit_rmsd_abs_diff_median_A": float(np.median(rmsd_difference)),
        "audit_rmsd_abs_diff_p95_A": float(np.quantile(rmsd_difference, 0.95)),
        "audit_catastrophic": catastrophic,
    }


def config_id(signature: str, method: str, fraction: float, gap: int, interpolation: str) -> str:
    return f"{signature}__{method}__F{fraction:g}__G{gap}__{interpolation}"


def run_ablation() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    signature_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    interpolation_rows: list[dict[str, object]] = []
    panel = development_cases()
    for case_number, row in enumerate(panel, start=1):
        coords, chains = load_study_case(row)
        baseline = frozen_current_throughput(coords, chains)
        landmarks = select_weighted_landmarks(
            coords[0],
            baseline.weights,
            count=min(64, coords.shape[1]),
            method="spatial_farthest",
            pool_fraction=1.0,
            chain_ids=chains,
        )
        signatures: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
        for signature_method in SIGNATURES:
            signature, metadata = build_frame_signature(
                coords, landmarks.indices, method=signature_method
            )
            signatures[signature_method] = (signature, metadata)
            signature_rows.append(
                {
                    "case_id": row["case_id"],
                    "signature": signature_method,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    **metadata,
                }
            )
        for signature_method, (signature, metadata) in signatures.items():
            del metadata
            for method in KEYFRAME_METHODS:
                for fraction, gap in BUDGETS:
                    keys, key_metadata = select_keyframes(
                        signature,
                        method=method,
                        fraction=fraction,
                        maximum_gap=gap,
                    )
                    transformed = fit_keyframe_transforms(
                        coords,
                        coords[0],
                        baseline.weights,
                        landmarks,
                        keys,
                        interpolation="slerp_linear",
                    )
                    selection_rows.append(
                        {
                            "case_id": row["case_id"],
                            "config_id": config_id(signature_method, method, fraction, gap, "slerp_linear"),
                            "signature": signature_method,
                            "keyframe_method": method,
                            "target_fraction": fraction,
                            "registered_maximum_gap": gap,
                            **key_metadata,
                            **transformed.timing,
                            **audit_transform_quality(
                                coords,
                                baseline,
                                transformed.rotations,
                                transformed.translations,
                            ),
                        }
                    )
        signature = signatures["pair_distances"][0]
        interpolation_designs = (
            (0.10, 25),
            (0.02, 100),
            (0.005, 500),
        )
        for fraction, gap in interpolation_designs:
            keys, key_metadata = select_keyframes(
                signature,
                method="max_gap_novelty",
                fraction=fraction,
                maximum_gap=gap,
            )
            for interpolation in ("piecewise_constant", "slerp_linear", "se3"):
                transformed = fit_keyframe_transforms(
                    coords,
                    coords[0],
                    baseline.weights,
                    landmarks,
                    keys,
                    interpolation=interpolation,
                )
                interpolation_rows.append(
                    {
                        "case_id": row["case_id"],
                        "config_id": config_id("pair_distances", "max_gap_novelty", fraction, gap, interpolation),
                        "signature": "pair_distances",
                        "keyframe_method": "max_gap_novelty",
                        "target_fraction": fraction,
                        "registered_maximum_gap": gap,
                        "interpolation": interpolation,
                        **key_metadata,
                        **transformed.timing,
                        **audit_transform_quality(
                            coords,
                            baseline,
                            transformed.rotations,
                            transformed.translations,
                        ),
                    }
                )
        write_tsv(PHASE / "signature_ablation.tsv", signature_rows)
        write_tsv(PHASE / "keyframe_selection_ablation.tsv", selection_rows)
        write_tsv(PHASE / "interpolation_ablation.tsv", interpolation_rows)
        print(f"[TV4] {case_number}/{len(panel)} {row['case_id']}", flush=True)


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)
    output: list[dict[str, object]] = []
    for candidate, values in grouped.items():
        output.append(
            {
                "config_id": candidate,
                "signature": values[0]["signature"],
                "keyframe_method": values[0]["keyframe_method"],
                "target_fraction": values[0]["target_fraction"],
                "registered_maximum_gap": values[0]["registered_maximum_gap"],
                "interpolation": values[0].get("interpolation", "slerp_linear"),
                "n_cases": len(values),
                "catastrophic_count": sum(row["audit_catastrophic"] == "True" for row in values),
                "median_keyframe_fraction": float(np.median([float(row["actual_fraction"]) for row in values])),
                "median_transform_seconds": float(np.median([float(row["total_transform_seconds"]) for row in values])),
                "median_audit_rmsd_diff_A": float(np.median([float(row["audit_rmsd_abs_diff_median_A"]) for row in values])),
                "p95_of_case_audit_p95_RMSD_A": float(np.quantile([float(row["audit_rmsd_abs_diff_p95_A"]) for row in values], 0.95)),
                "median_coordinate_audit_A": float(np.median([float(row["aligned_coordinate_audit_median_A"]) for row in values])),
            }
        )
    return output


def lock_and_full_evaluate() -> None:
    selection = read_tsv(PHASE / "keyframe_selection_ablation.tsv")
    interpolation = read_tsv(PHASE / "interpolation_ablation.tsv")
    summary = aggregate(selection + interpolation)
    safe = [row for row in summary if row["catastrophic_count"] == 0]
    if safe:
        balanced = min(
            safe,
            key=lambda row: (
                row["p95_of_case_audit_p95_RMSD_A"],
                row["median_coordinate_audit_A"],
                row["median_transform_seconds"],
            ),
        )
        fast = min(
            safe,
            key=lambda row: (
                row["median_transform_seconds"],
                row["p95_of_case_audit_p95_RMSD_A"],
            ),
        )
    else:
        balanced = min(summary, key=lambda row: (row["catastrophic_count"], row["p95_of_case_audit_p95_RMSD_A"]))
        fast = min(summary, key=lambda row: (row["catastrophic_count"], row["median_transform_seconds"]))
    finalists = (("keyframe_balanced", balanced), ("keyframe_fast", fast))
    full_rows: list[dict[str, object]] = []
    for row in development_cases():
        coords, chains = load_study_case(row)
        baseline = frozen_current_throughput(coords, chains)
        landmarks = select_weighted_landmarks(
            coords[0], baseline.weights, count=min(64, coords.shape[1]), method="spatial_farthest", pool_fraction=1.0, chain_ids=chains
        )
        for label, config in finalists:
            signature, signature_metadata = build_frame_signature(
                coords, landmarks.indices, method=str(config["signature"])
            )
            keys, key_metadata = select_keyframes(
                signature,
                method=str(config["keyframe_method"]),
                fraction=float(config["target_fraction"]),
                maximum_gap=int(config["registered_maximum_gap"]),
            )
            started = time.perf_counter()
            candidate = fit_keyframe_alignment(
                coords,
                coords[0],
                baseline.weights,
                landmarks,
                keys,
                interpolation=str(config["interpolation"]),
            )
            candidate_seconds = time.perf_counter() - started
            total = baseline.weight_seconds + float(signature_metadata["construction_seconds"]) + candidate_seconds
            full_rows.append(
                {
                    "case_id": row["case_id"],
                    "candidate": label,
                    **config,
                    **key_metadata,
                    "baseline_total_seconds": baseline.total_seconds,
                    "candidate_total_seconds": total,
                    "speedup_vs_current_throughput": baseline.total_seconds / total,
                    **compare_alignment(candidate, baseline, coords),
                }
            )
    write_tsv(PHASE / "full_output_finalist_evaluation.tsv", full_rows)
    locks: list[dict[str, object]] = []
    for label, config in finalists:
        values = [row for row in full_rows if row["candidate"] == label]
        catastrophic = sum(bool(row["catastrophic_mismatch"]) for row in values)
        locks.append(
            {
                "candidate": label,
                **config,
                "full_output_cases": len(values),
                "full_output_catastrophic_count": catastrophic,
                "median_full_output_speedup": float(np.median([float(row["speedup_vs_current_throughput"]) for row in values])),
                "median_full_output_rmsd_diff_A": float(np.median([float(row["rmsd_abs_diff_median_A"]) for row in values])),
                "median_full_output_rmsf_spearman": float(np.median([float(row["stable_rmsf_spearman"]) for row in values])),
                "lock_status": "development_safe" if catastrophic == 0 else "rejected_development_catastrophic",
                "fresh_results_seen": False,
            }
        )
    write_tsv(PHASE / "locked_keyframe_configs.tsv", locks)
    write_tsv(PHASE / "quality_speed_pareto.tsv", summary)
    report = [
        "# Keyframe Method Report",
        "",
        "Five landmark-only signatures, four keyframe selectors, all six registered fractions/gaps, and three interpolation methods were assessed on development data.",
        "",
    ]
    for row in locks:
        report.append(
            f"- {row['candidate']}: `{row['config_id']}`, status `{row['lock_status']}`, median full-output speedup {row['median_full_output_speedup']:.3f}x."
        )
    report.extend(
        [
            "",
            "Piecewise-constant interpolation is retained only as a negative control. Transform-only screening did not replace matched full-output evaluation for the two finalists.",
        ]
    )
    (PHASE / "keyframe_method_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    run_ablation()
    lock_and_full_evaluate()
    print(json.dumps({"phase": "TV4", "status": "complete"}))


if __name__ == "__main__":
    main()
