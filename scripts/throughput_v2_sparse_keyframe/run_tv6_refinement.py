#!/usr/bin/env python3
"""Run TV6 deterministic audit and bounded local keyframe refinement."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdcorealign.experimental_throughput_v2 import (  # noqa: E402
    audit_against_full_fits,
    build_frame_signature,
    deterministic_audit_indices,
    fit_keyframe_alignment,
    select_keyframes,
    select_weighted_landmarks,
)
from scripts.throughput_v2_sparse_keyframe.run_tv4_keyframes import development_cases  # noqa: E402
from scripts.throughput_v2_sparse_keyframe.study_utils import (  # noqa: E402
    OUT,
    compare_alignment,
    frozen_current_throughput,
    load_study_case,
    write_tsv,
)


PHASE = OUT / "06_adaptive_refinement"


def interval_rows(keys: np.ndarray, flagged: np.ndarray) -> list[tuple[int, int, int]]:
    rows = []
    for frame in flagged:
        right_position = int(np.searchsorted(keys, frame, side="right"))
        left_position = max(0, right_position - 1)
        right_position = min(right_position, keys.size - 1)
        rows.append((int(keys[left_position]), int(keys[right_position]), int(frame)))
    return rows


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    audit_registry: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
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
        signature, _ = build_frame_signature(
            coords, landmarks.indices, method="pair_distances"
        )
        curvature = np.zeros(coords.shape[0], dtype=np.float64)
        if coords.shape[0] > 2:
            curvature[1:-1] = np.linalg.norm(
                signature[2:] - 2.0 * signature[1:-1] + signature[:-2],
                axis=1,
            )
        keys, key_metadata = select_keyframes(
            signature,
            method="max_gap_novelty",
            fraction=0.20,
            maximum_gap=10,
        )
        original_count = keys.size
        audit = deterministic_audit_indices(
            coords.shape[0], keyframes=keys, curvature=curvature
        )
        audit_registry.extend(
            {
                "case_id": row["case_id"],
                "audit_index": int(index),
                "temporal_fraction": float(index / max(1, coords.shape[0] - 1)),
                "signature_curvature": float(curvature[index]),
                "excluded_initial_keyframes": True,
            }
            for index in audit
        )
        final_candidate = None
        final_audit = None
        for round_index in range(4):
            candidate = fit_keyframe_alignment(
                coords,
                coords[0],
                baseline.weights,
                landmarks,
                keys,
                interpolation="slerp_linear",
            )
            audited = audit_against_full_fits(
                coords,
                coords[0],
                baseline.weights,
                candidate,
                audit,
            )
            risk_rows.append(
                {
                    "case_id": row["case_id"],
                    "round": round_index,
                    "keyframe_count": keys.size,
                    "keyframe_fraction": keys.size / coords.shape[0],
                    "audit_count": audit.size,
                    "flagged_count": audited.flagged_indices.size,
                    "flagged_fraction": audited.unresolved_rate,
                    "rotation_p95_degrees": float(np.quantile(audited.rotation_geodesic_degrees, 0.95)),
                    "coordinate_p95_A": float(np.quantile(audited.aligned_coordinate_rmsd, 0.95)),
                    "rmsd_difference_p95_A": float(np.quantile(audited.rmsd_absolute_difference, 0.95)),
                    "signature_curvature_flagged_median": float(
                        np.median(curvature[audited.flagged_indices])
                        if audited.flagged_indices.size
                        else 0.0
                    ),
                }
            )
            quality = compare_alignment(candidate, baseline, coords)
            ablation_rows.append(
                {
                    "case_id": row["case_id"],
                    "round": round_index,
                    "initial_keyframe_count": original_count,
                    "final_keyframe_count": keys.size,
                    "keyframe_fraction": keys.size / coords.shape[0],
                    "audit_unresolved_rate": audited.unresolved_rate,
                    "candidate_seconds_excluding_weight_estimation": candidate.timing["total_excluding_weight_estimation"],
                    "speedup_vs_current_throughput": baseline.total_seconds
                    / (
                        baseline.weight_seconds
                        + candidate.timing["total_excluding_weight_estimation"]
                    ),
                    **quality,
                }
            )
            final_candidate = candidate
            final_audit = audited
            if audited.flagged_indices.size == 0 or round_index == 3:
                break
            additions = audited.flagged_indices
            for left, right, frame in interval_rows(keys, additions):
                path_rows.append(
                    {
                        "case_id": row["case_id"],
                        "round": round_index + 1,
                        "interval_start": left,
                        "interval_end": right,
                        "inserted_keyframe": frame,
                        "action": "insert_flagged_audit_frame",
                    }
                )
            keys = np.unique(np.concatenate((keys, additions))).astype(np.int64)
        assert final_candidate is not None and final_audit is not None
        for left, right, frame in interval_rows(keys, final_audit.flagged_indices):
            unresolved_rows.append(
                {
                    "case_id": row["case_id"],
                    "interval_start": left,
                    "interval_end": right,
                    "audit_frame": frame,
                    "final_round": min(3, len([item for item in ablation_rows if item["case_id"] == row["case_id"]]) - 1),
                    "status": "unresolved_after_registered_max_rounds",
                }
            )
        final_quality = compare_alignment(final_candidate, baseline, coords)
        final_rows.append(
            {
                "case_id": row["case_id"],
                "initial_keyframe_fraction": key_metadata["actual_fraction"],
                "final_keyframe_fraction": keys.size / coords.shape[0],
                "refinement_rounds": max(
                    int(item["round"])
                    for item in ablation_rows
                    if item["case_id"] == row["case_id"]
                ),
                "final_unresolved_rate": final_audit.unresolved_rate,
                **final_quality,
            }
        )
        write_tsv(PHASE / "audit_frame_registry.tsv", audit_registry)
        write_tsv(PHASE / "risk_features.tsv", risk_rows)
        write_tsv(PHASE / "refinement_ablation.tsv", ablation_rows)
        write_tsv(PHASE / "refinement_paths.tsv", path_rows)
        write_tsv(PHASE / "unresolved_intervals.tsv", unresolved_rows)
        print(f"[TV6] {case_number}/{len(panel)} {row['case_id']}", flush=True)
    catastrophic = sum(bool(row["catastrophic_mismatch"]) for row in final_rows)
    unresolved = float(np.mean([float(row["final_unresolved_rate"]) for row in final_rows]))
    lock = {
        "candidate": "adaptive_keyframe_local_refinement",
        "initial_signature": "pair_distances",
        "initial_keyframe_method": "max_gap_novelty",
        "initial_fraction": 0.20,
        "initial_maximum_gap": 10,
        "maximum_refinement_rounds": 3,
        "development_cases": len(final_rows),
        "catastrophic_count": catastrophic,
        "mean_unresolved_interval_rate": unresolved,
        "lock_status": (
            "development_safe"
            if catastrophic == 0 and unresolved <= 0.02
            else "rejected_development_quality_or_unresolved_gate"
        ),
        "fresh_results_seen": False,
    }
    write_tsv(PHASE / "locked_adaptive_configs.tsv", [lock])
    (PHASE / "adaptive_refinement_report.md").write_text(
        "# Adaptive Refinement Report\n\n"
        f"The bounded three-round local insertion cascade ended with {catastrophic} "
        f"catastrophic full-output cases and mean unresolved audit rate {unresolved:.3%}. "
        f"Status: `{lock['lock_status']}`. No whole-trajectory current-throughput "
        "fallback was used inside the candidate.\n",
        encoding="utf-8",
    )
    print(json.dumps({"phase": "TV6", "status": "complete", **lock}))


if __name__ == "__main__":
    main()
