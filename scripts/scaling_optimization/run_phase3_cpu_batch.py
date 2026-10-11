#!/usr/bin/env python
"""Validate and benchmark the Phase 3 batched representative estimator."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from build_phase2_regression import (
    PROJECT_ROOT,
    REFERENCE_ROOT,
    ROBUST_KWARGS,
    build_cases,
    repo_path,
)
from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.long_mode import (
    build_long_flexibility_profile,
    solve_long_trajectory_alignment,
)
from mdsuperpose.representative_weights import estimate_representative_weights
from profile_support import MemorySampler


RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v2/cpu_batch"
PRIMARY_INPUT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
    / "runs/4oelA02/10000/transition_aligned_coords.npz"
)
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REPRESENTATIVE_BATCH_SIZE = 512
WARM_REFINEMENT_PASSES = 2


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable TSV from heterogeneous result rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.sum(first | second)
    return float(np.sum(first & second) / union) if union else 1.0


def candidate_arrays(
    coords: np.ndarray,
    chain: np.ndarray | None,
    representative_batch_size: int,
) -> tuple[dict[str, np.ndarray], object]:
    """Run the explicit batched long-mode candidate and compact its arrays."""

    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=min(2000, coords.shape[0]),
        frame_selection="uniform",
        random_seed=42,
        chunk_size=min(2000, coords.shape[0]),
        store_aligned=False,
        representative_weight_estimator="batched",
        representative_batch_size=representative_batch_size,
        representative_warm_refinement_passes=WARM_REFINEMENT_PASSES,
        **ROBUST_KWARGS,
    )
    labels = (
        [f"{value}:{index + 1}:UNK" for index, value in enumerate(chain)]
        if chain is not None
        else None
    )
    profile = build_long_flexibility_profile(
        result,
        residue_labels=labels,
        min_segment_length=3,
        merge_gap=1,
    )
    core = select_core_by_target_fraction(result.frozen_weights, 0.6)
    arrays = {
        "frozen_weights": result.frozen_weights,
        "core_labels": core.core_mask.astype(np.uint8),
        "rotations": result.rotations,
        "translations": result.translations,
        "rmsd": result.rmsd,
        "weighted_rmsd": result.weighted_rmsd,
        "robust_rmsf": result.robust_rmsf,
        "ordinary_rmsf": result.ordinary_rmsf,
        "flexible_labels": (profile.flexible_segment_id > 0).astype(np.uint8),
    }
    return arrays, result


def parity_panel() -> list[dict[str, object]]:
    """Compare the candidate against all compact golden legacy outputs."""

    rows: list[dict[str, object]] = []
    for case in build_cases():
        golden_path = REFERENCE_ROOT / f"{case.case_id}.npz"
        if not golden_path.is_file():
            rows.append(
                {
                    "case_id": case.case_id,
                    "status": "missing_golden",
                    "strict_parity_pass": 0,
                    "quality_pass": 0,
                }
            )
            continue
        print(f"[parity] {case.case_id}", flush=True)
        coords, chain = case.loader()
        with np.load(golden_path, allow_pickle=False) as payload:
            golden = {
                key.removeprefix("legacy_long_"): np.asarray(payload[key])
                for key in payload.files
                if key.startswith("legacy_long_")
            }
        start = time.perf_counter()
        candidate, result = candidate_arrays(
            coords,
            chain,
            min(REPRESENTATIVE_BATCH_SIZE, coords.shape[0]),
        )
        runtime = time.perf_counter() - start
        weight_max = float(
            np.max(np.abs(candidate["frozen_weights"] - golden["frozen_weights"]))
        )
        rmsd_max = float(np.max(np.abs(candidate["rmsd"] - golden["rmsd"])))
        weighted_rmsd_max = float(
            np.max(
                np.abs(
                    candidate["weighted_rmsd"]
                    - golden["weighted_rmsd"]
                )
            )
        )
        rmsf_max = float(
            np.max(np.abs(candidate["robust_rmsf"] - golden["robust_rmsf"]))
        )
        ordinary_rmsf_max = float(
            np.max(
                np.abs(
                    candidate["ordinary_rmsf"]
                    - golden["ordinary_rmsf"]
                )
            )
        )
        rotation_max = float(
            np.max(np.abs(candidate["rotations"] - golden["rotations"]))
        )
        translation_max = float(
            np.max(np.abs(candidate["translations"] - golden["translations"]))
        )
        core_jaccard = jaccard(candidate["core_labels"], golden["core_labels"])
        flexible_jaccard = jaccard(
            candidate["flexible_labels"],
            golden["flexible_labels"],
        )
        strict = (
            weight_max <= 1e-8
            and rmsd_max <= 1e-8
            and weighted_rmsd_max <= 1e-8
            and rmsf_max <= 1e-8
            and ordinary_rmsf_max <= 1e-8
            and rotation_max <= 1e-8
            and translation_max <= 1e-8
            and core_jaccard == 1.0
            and flexible_jaccard == 1.0
        )
        quality = (
            weight_max <= 1e-4
            and rmsd_max <= 1e-3
            and weighted_rmsd_max <= 1e-3
            and rmsf_max <= 1e-3
            and core_jaccard >= 0.98
            and flexible_jaccard >= 0.95
        )
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "candidate_runtime_seconds": runtime,
                "weight_max_abs_difference": weight_max,
                "core_jaccard": core_jaccard,
                "rotation_max_abs_difference": rotation_max,
                "translation_max_abs_difference": translation_max,
                "rmsd_max_abs_difference_angstrom": rmsd_max,
                "weighted_rmsd_max_abs_difference_angstrom": weighted_rmsd_max,
                "robust_rmsf_max_abs_difference_angstrom": rmsf_max,
                "ordinary_rmsf_max_abs_difference_angstrom": ordinary_rmsf_max,
                "flexible_label_jaccard": flexible_jaccard,
                "candidate_average_iterations": result.average_iterations_per_frame,
                "candidate_converged_fraction": result.converged_fraction,
                "strict_parity_pass": int(strict),
                "quality_pass": int(quality),
                "status": "pass" if quality else "quality_regression",
                "interpretation": (
                    "Strict legacy parity includes sequential warm-start semantics; "
                    "the candidate uses independent batched initialization."
                ),
            }
        )
    return rows


def _run_primary(method: str, *, instrumented: bool) -> dict[str, object]:
    """Run one complete in-child 10k timing, including NPZ decode."""

    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    with np.load(PRIMARY_INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=float)
    decode_seconds = time.perf_counter() - decode_start
    callback_totals = {
        "weighted_fit": 0.0,
        "residual_compute": 0.0,
        "weight_update": 0.0,
        "sequence_regularization": 0.0,
        "contact_regularization": 0.0,
        "convergence": 0.0,
    }

    kwargs = dict(ROBUST_KWARGS)
    if instrumented and method == "legacy":
        def callback(payload: dict[str, object]) -> None:
            callback_totals["weighted_fit"] += float(
                payload.get("weighted_kabsch_or_refit_seconds", 0.0)
            )
            callback_totals["residual_compute"] += float(
                payload.get("residual_computation_seconds", 0.0)
            )
            callback_totals["weight_update"] += float(
                payload.get("weight_update_seconds", 0.0)
            )
            callback_totals["sequence_regularization"] += float(
                payload.get("sequence_regularization_seconds", 0.0)
            )
            callback_totals["contact_regularization"] += float(
                payload.get("contact_regularization_seconds", 0.0)
            )
            callback_totals["convergence"] += float(
                payload.get("convergence_check_seconds", 0.0)
            )

        kwargs["diagnostic_callback"] = callback

    sampler = MemorySampler()
    sampler.start()
    try:
        solve_start = time.perf_counter()
        result = solve_long_trajectory_alignment(
            coords,
            mode="long",
            max_weight_fit_frames=2000,
            frame_selection="uniform",
            random_seed=42,
            chunk_size=2000,
            store_aligned=False,
            representative_weight_estimator=method,
            representative_batch_size=REPRESENTATIVE_BATCH_SIZE,
            representative_warm_refinement_passes=(
                WARM_REFINEMENT_PASSES if method == "batched" else 0
            ),
            **kwargs,
        )
        solve_seconds = time.perf_counter() - solve_start
    finally:
        sampler.stop()
    representative_timing = getattr(result.subset_result, "timing", None) or {}
    call_counts = getattr(result.subset_result, "call_counts", None) or {}
    if method == "legacy":
        iteration_count = sum(
            frame.iteration_count for frame in result.subset_result.frame_results
        )
        call_counts = {
            "per_frame_kabsch_calls": iteration_count
            + len(result.subset_result.frame_results),
            "stacked_svd_calls": 0,
            "per_frame_result_objects": len(result.subset_result.frame_results),
            "sequence_regularization_calls": iteration_count,
        }
    return {
        "method": method,
        "profile_mode": "stage_instrumented" if instrumented else "clean",
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "representative_batch_size": (
            REPRESENTATIVE_BATCH_SIZE if method == "batched" else ""
        ),
        "warm_start_refinement_passes": (
            WARM_REFINEMENT_PASSES if method == "batched" else ""
        ),
        "input_decode_seconds": decode_seconds,
        "solve_seconds": solve_seconds,
        "in_child_end_to_end_seconds": time.perf_counter() - total_start,
        "representative_weight_estimation_seconds": result.timing[
            "iterative_weight_estimation"
        ],
        "all_frame_alignment_seconds": result.timing["weighted_fit"],
        "rmsf_seconds": result.timing["RMSF"],
        "representative_batched_kabsch_seconds": representative_timing.get(
            "batched_kabsch",
            "",
        ),
        "representative_residual_seconds": representative_timing.get(
            "residual_compute",
            "",
        ),
        "representative_weight_update_seconds": representative_timing.get(
            "weight_update",
            "",
        ),
        "representative_sequence_seconds": representative_timing.get(
            "sequence_regularization",
            "",
        ),
        "legacy_callback_weighted_fit_seconds": callback_totals["weighted_fit"],
        "legacy_callback_residual_seconds": callback_totals["residual_compute"],
        "legacy_callback_weight_update_seconds": callback_totals["weight_update"],
        "legacy_callback_sequence_seconds": callback_totals[
            "sequence_regularization"
        ],
        "peak_rss_mb": sampler.to_dict().get("peak_rss_mb", ""),
        "average_iterations": result.average_iterations_per_frame,
        "converged_fraction": result.converged_fraction,
        "mean_rmsd": result.mean_rmsd,
        "mean_weighted_rmsd": result.mean_weighted_rmsd,
        **call_counts,
    }


def child_main(args: argparse.Namespace) -> int:
    payload = _run_primary(args.method, instrumented=args.instrumented)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def subprocess_timings() -> list[dict[str, object]]:
    """Run warm-up, clean repeats and one instrumented repeat in fresh processes."""

    raw_dir = RESULT_ROOT / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["PYTHONPATH"] = "src:."
    script = Path(__file__).resolve()
    for method in ("legacy", "batched"):
        for repeat in range(4):
            warmup = repeat == 0
            output = raw_dir / f"{method}_clean_repeat_{repeat}.json"
            command = [
                sys.executable,
                str(script),
                "--child",
                "--method",
                method,
                "--output",
                str(output),
            ]
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            fresh = time.perf_counter() - start
            if completed.returncode != 0:
                rows.append(
                    {
                        "method": method,
                        "profile_mode": "clean",
                        "repeat": repeat,
                        "warmup": int(warmup),
                        "status": "failed",
                        "failure": completed.stderr,
                        "fresh_process_seconds": fresh,
                    }
                )
                continue
            payload = json.loads(output.read_text())
            payload.update(
                {
                    "repeat": repeat,
                    "warmup": int(warmup),
                    "status": "success",
                    "fresh_process_seconds": fresh,
                }
            )
            rows.append(payload)

        output = raw_dir / f"{method}_instrumented.json"
        command = [
            sys.executable,
            str(script),
            "--child",
            "--method",
            method,
            "--instrumented",
            "--output",
            str(output),
        ]
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        fresh = time.perf_counter() - start
        if completed.returncode == 0:
            payload = json.loads(output.read_text())
            payload.update(
                {
                    "repeat": 1,
                    "warmup": 0,
                    "status": "success",
                    "fresh_process_seconds": fresh,
                }
            )
            rows.append(payload)
        else:
            rows.append(
                {
                    "method": method,
                    "profile_mode": "stage_instrumented",
                    "repeat": 1,
                    "warmup": 0,
                    "status": "failed",
                    "failure": completed.stderr,
                    "fresh_process_seconds": fresh,
                }
            )
    return rows


def add_performance_summary(rows: list[dict[str, object]]) -> None:
    """Append median clean timing and speedup fields to each successful row."""

    medians: dict[str, float] = {}
    for method in ("legacy", "batched"):
        values = [
            float(row["in_child_end_to_end_seconds"])
            for row in rows
            if row.get("method") == method
            and row.get("profile_mode") == "clean"
            and not row.get("warmup")
            and row.get("status") == "success"
        ]
        if values:
            medians[method] = statistics.median(values)
    for row in rows:
        row["clean_median_end_to_end_seconds"] = medians.get(
            str(row.get("method")),
            "",
        )
        if "legacy" in medians and "batched" in medians:
            row["batched_speedup_vs_legacy"] = (
                medians["legacy"] / medians["batched"]
            )
            row["performance_gate_below_4_30_seconds"] = int(
                medians["batched"] < 4.30
            )


def chunk_sensitivity() -> list[dict[str, object]]:
    """Measure representative-only chunk sizes against one 2,000-frame batch."""

    with np.load(PRIMARY_INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=float)
    with np.load(
        REFERENCE_ROOT / "synthetic_scaling_4oelA02_10000.npz",
        allow_pickle=False,
    ) as payload:
        selected = np.asarray(payload["legacy_long_selected_indices"], dtype=int)
    representatives = coords[selected]
    reference = coords[0]
    baseline = estimate_representative_weights(
        representatives,
        reference,
        batch_size=2000,
        warm_start_refinement_passes=WARM_REFINEMENT_PASSES,
        **ROBUST_KWARGS,
    )
    rows: list[dict[str, object]] = []
    for batch_size in (128, 256, 512, 1000, 2000):
        runtimes = []
        candidate = None
        for _ in range(3):
            start = time.perf_counter()
            candidate = estimate_representative_weights(
                representatives,
                reference,
                batch_size=batch_size,
                warm_start_refinement_passes=WARM_REFINEMENT_PASSES,
                **ROBUST_KWARGS,
            )
            runtimes.append(time.perf_counter() - start)
        assert candidate is not None
        rows.append(
            {
                "batch_size": batch_size,
                "n_representatives": representatives.shape[0],
                "median_runtime_seconds": statistics.median(runtimes),
                "min_runtime_seconds": min(runtimes),
                "max_runtime_seconds": max(runtimes),
                "weight_max_abs_difference_vs_batch_2000": float(
                    np.max(np.abs(candidate.weights - baseline.weights))
                ),
                "mean_weight_max_abs_difference_vs_batch_2000": float(
                    np.max(
                        np.abs(candidate.mean_weights - baseline.mean_weights)
                    )
                ),
                "stacked_svd_calls": candidate.call_counts["stacked_svd_calls"],
                "representative_chunks": candidate.call_counts[
                    "representative_chunks"
                ],
                "warm_start_refinement_passes": WARM_REFINEMENT_PASSES,
            }
        )
    return rows


def call_count_rows(performance: list[dict[str, object]]) -> list[dict[str, object]]:
    """Extract one clean 10k call-count comparison."""

    rows = []
    for method in ("legacy", "batched"):
        row = next(
            (
                value
                for value in performance
                if value.get("method") == method
                and value.get("profile_mode") == "clean"
                and not value.get("warmup")
                and value.get("status") == "success"
            ),
            None,
        )
        if row is None:
            rows.append(
                {
                    "method": method,
                    "n_representatives": 2000,
                    "status": "unavailable_no_successful_clean_run",
                }
            )
            continue
        rows.append(
            {
                "method": method,
                "n_representatives": 2000,
                "status": "success",
                "per_frame_kabsch_calls": row.get("per_frame_kabsch_calls", 0),
                "stacked_svd_calls": row.get("stacked_svd_calls", 0),
                "per_frame_result_objects": row.get(
                    "per_frame_result_objects",
                    0,
                ),
                "sequence_regularization_calls": row.get(
                    "sequence_regularization_calls",
                    0,
                ),
                "average_iterations": row.get("average_iterations", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=("legacy", "batched"))
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    if args.child:
        if not args.method or not args.output:
            parser.error("--child requires --method and --output")
        return child_main(args)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    parity_path = RESULT_ROOT / "batch_parity.tsv"
    if args.skip_parity and parity_path.is_file():
        with parity_path.open(newline="") as handle:
            parity = list(csv.DictReader(handle, delimiter="\t"))
    else:
        parity = parity_panel()
    if parity:
        write_tsv(parity_path, parity)
    performance = subprocess_timings()
    add_performance_summary(performance)
    write_tsv(RESULT_ROOT / "batch_performance.tsv", performance)
    write_tsv(
        RESULT_ROOT / "call_count_before_after.tsv",
        call_count_rows(performance),
    )
    chunks = chunk_sensitivity()
    write_tsv(RESULT_ROOT / "chunk_size_sensitivity.tsv", chunks)

    strict_passes = sum(int(row["strict_parity_pass"]) for row in parity)
    quality_passes = sum(int(row["quality_pass"]) for row in parity)
    clean_batch = [
        float(row["in_child_end_to_end_seconds"])
        for row in performance
        if row.get("method") == "batched"
        and row.get("profile_mode") == "clean"
        and not row.get("warmup")
        and row.get("status") == "success"
    ]
    ledger = [
        {
            "change": "variable-weight batched Kabsch",
            "decision": "keep",
            "reason": "Strict scalar-independent parity tests pass.",
        },
        {
            "change": "chain-aware batched sequence regularization",
            "decision": "keep",
            "reason": "Matches scalar rule within chains and blocks cross-chain smoothing.",
        },
        {
            "change": "batched estimator as public default",
            "decision": "do_not_promote_yet",
            "reason": (
                f"strict legacy parity passed {strict_passes}/{len(parity)} cases; "
                "legacy sequential warm-start semantics are not exactly reproduced."
            ),
        },
        {
            "change": "legacy estimator",
            "decision": "retain",
            "reason": "Frozen reference and current CLI default remain unchanged.",
        },
    ]
    write_tsv(RESULT_ROOT / "optimization_ledger.tsv", ledger)
    summary = {
        "parity_case_count": len(parity),
        "strict_parity_pass_count": strict_passes,
        "quality_pass_count": quality_passes,
        "batched_clean_median_seconds": (
            statistics.median(clean_batch) if clean_batch else None
        ),
        "performance_gate_seconds": 4.30,
        "performance_gate_pass": (
            bool(clean_batch and statistics.median(clean_batch) < 4.30)
        ),
        "primary_input": repo_path(PRIMARY_INPUT),
        "data_semantics": "transition_aligned_synthetic_long_trajectory",
    }
    (RESULT_ROOT / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
