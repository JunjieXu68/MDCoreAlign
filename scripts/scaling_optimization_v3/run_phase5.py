"""Validate and benchmark the exact sequential_lean long-mode path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v3/sequential_lean"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v3"
REFERENCE_ROOT = PROJECT_ROOT / "results/scaling_optimization_v2/regression/reference_outputs"
PRIMARY_INPUT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
    / "runs/4oelA02/10000/transition_aligned_coords.npz"
)
CANONICAL_PYTHON = Path("/home/jjxu/miniconda3/bin/python")
THREAD_COUNT = 2
THREAD_ENV = {
    "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
    "OPENBLAS_NUM_THREADS": str(THREAD_COUNT),
    "OMP_NUM_THREADS": str(THREAD_COUNT),
    "MKL_NUM_THREADS": str(THREAD_COUNT),
    "NUMEXPR_NUM_THREADS": str(THREAD_COUNT),
    "VECLIB_MAXIMUM_THREADS": str(THREAD_COUNT),
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


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, str]:
    values = os.environ.copy()
    values.update(THREAD_ENV)
    return values


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.sum(first | second))
    return float(np.sum(first & second) / union) if union else 1.0


def candidate_arrays(
    coords: np.ndarray,
    chain: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], object]:
    from mdsuperpose.core_selection import select_core_by_target_fraction
    from mdsuperpose.long_mode import (
        build_long_flexibility_profile,
        solve_long_trajectory_alignment,
    )

    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=min(2000, coords.shape[0]),
        frame_selection="uniform",
        random_seed=42,
        chunk_size=min(2000, coords.shape[0]),
        store_aligned=False,
        representative_weight_estimator="sequential_lean",
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
    return (
        {
            "frozen_weights": result.frozen_weights,
            "core_labels": core.core_mask.astype(np.uint8),
            "rotations": result.rotations,
            "translations": result.translations,
            "rmsd": result.rmsd,
            "weighted_rmsd": result.weighted_rmsd,
            "robust_rmsf": result.robust_rmsf,
            "ordinary_rmsf": result.ordinary_rmsf,
            "flexible_labels": (
                profile.flexible_segment_id > 0
            ).astype(np.uint8),
        },
        result,
    )


def parity_panel() -> list[dict[str, object]]:
    scripts_root = PROJECT_ROOT / "scripts/scaling_optimization"
    sys.path.insert(0, str(scripts_root))
    from build_phase2_regression import build_cases

    rows: list[dict[str, object]] = []
    for case in build_cases():
        print(f"[parity] {case.case_id}", flush=True)
        coords, chain = case.loader()
        with np.load(REFERENCE_ROOT / f"{case.case_id}.npz", allow_pickle=False) as payload:
            golden = {
                key.removeprefix("legacy_long_"): np.asarray(payload[key])
                for key in payload.files
                if key.startswith("legacy_long_")
            }
        start = time.perf_counter()
        candidate, result = candidate_arrays(coords, chain)
        runtime = time.perf_counter() - start
        differences = {
            "weight_max_abs_difference": float(
                np.max(
                    np.abs(
                        candidate["frozen_weights"]
                        - golden["frozen_weights"]
                    )
                )
            ),
            "rotation_max_abs_difference": float(
                np.max(np.abs(candidate["rotations"] - golden["rotations"]))
            ),
            "translation_max_abs_difference": float(
                np.max(
                    np.abs(
                        candidate["translations"] - golden["translations"]
                    )
                )
            ),
            "rmsd_max_abs_difference_angstrom": float(
                np.max(np.abs(candidate["rmsd"] - golden["rmsd"]))
            ),
            "weighted_rmsd_max_abs_difference_angstrom": float(
                np.max(
                    np.abs(
                        candidate["weighted_rmsd"]
                        - golden["weighted_rmsd"]
                    )
                )
            ),
            "robust_rmsf_max_abs_difference_angstrom": float(
                np.max(
                    np.abs(
                        candidate["robust_rmsf"]
                        - golden["robust_rmsf"]
                    )
                )
            ),
            "ordinary_rmsf_max_abs_difference_angstrom": float(
                np.max(
                    np.abs(
                        candidate["ordinary_rmsf"]
                        - golden["ordinary_rmsf"]
                    )
                )
            ),
        }
        core_jaccard = jaccard(
            candidate["core_labels"], golden["core_labels"]
        )
        flexible_jaccard = jaccard(
            candidate["flexible_labels"], golden["flexible_labels"]
        )
        exact = (
            max(differences.values()) <= 1e-8
            and core_jaccard == 1.0
            and flexible_jaccard == 1.0
        )
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "runtime_seconds": runtime,
                **differences,
                "core_jaccard": core_jaccard,
                "flexible_label_jaccard": flexible_jaccard,
                "average_iterations": result.average_iterations_per_frame,
                "converged_fraction": result.converged_fraction,
                "per_representative_result_objects": result.subset_result.call_counts[
                    "per_representative_result_objects"
                ],
                "retained_aligned_coordinate_arrays": result.subset_result.call_counts[
                    "retained_aligned_coordinate_arrays"
                ],
                "quality_gate_pass": int(exact),
                "status": "pass" if exact else "failed",
            }
        )
    return rows


def child_run(
    method: str,
    n_frames: int,
    *,
    instrumented: bool,
    output: Path,
) -> int:
    scripts_root = PROJECT_ROOT / "scripts/scaling_optimization"
    sys.path.insert(0, str(scripts_root))
    from profile_support import MemorySampler
    from mdsuperpose.long_mode import solve_long_trajectory_alignment

    total_start = time.perf_counter()
    sampler = MemorySampler(interval_seconds=0.005)
    sampler.start()
    decode_start = time.perf_counter()
    with np.load(PRIMARY_INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"][:n_frames], dtype=float)
    decode_seconds = time.perf_counter() - decode_start
    kwargs = dict(ROBUST_KWARGS)
    callback_timing = {
        "weighted_kabsch": 0.0,
        "residual_compute": 0.0,
        "weight_update": 0.0,
        "sequence_regularization": 0.0,
        "contact_regularization": 0.0,
        "convergence": 0.0,
    }
    if method == "legacy_object" and instrumented:
        def callback(payload: dict[str, object]) -> None:
            callback_timing["weighted_kabsch"] += float(
                payload.get("weighted_kabsch_or_refit_seconds", 0.0)
            )
            callback_timing["residual_compute"] += float(
                payload.get("residual_computation_seconds", 0.0)
            )
            callback_timing["weight_update"] += float(
                payload.get("weight_update_seconds", 0.0)
            )
            callback_timing["sequence_regularization"] += float(
                payload.get("sequence_regularization_seconds", 0.0)
            )
            callback_timing["contact_regularization"] += float(
                payload.get("contact_regularization_seconds", 0.0)
            )
            callback_timing["convergence"] += float(
                payload.get("convergence_check_seconds", 0.0)
            )

        kwargs["diagnostic_callback"] = callback
        estimator = "legacy"
    elif method == "sequential_lean":
        estimator = "sequential_lean"
        kwargs["collect_timing"] = instrumented
    else:
        estimator = "legacy"

    try:
        solve_start = time.perf_counter()
        result = solve_long_trajectory_alignment(
            coords,
            mode="long",
            max_weight_fit_frames=min(2000, n_frames),
            frame_selection="uniform",
            random_seed=42,
            chunk_size=min(2000, n_frames),
            store_aligned=False,
            representative_weight_estimator=estimator,
            **kwargs,
        )
        solve_seconds = time.perf_counter() - solve_start
    finally:
        sampler.stop()
    subset = result.subset_result
    if method == "legacy_object":
        iterations = sum(frame.iteration_count for frame in subset.frame_results)
        call_counts = {
            "kabsch_calls": iterations + len(subset.frame_results),
            "svd_calls": iterations + len(subset.frame_results),
            "per_representative_result_objects": len(subset.frame_results),
            "retained_aligned_coordinate_arrays": len(subset.frame_results),
        }
        stage_timing = callback_timing
    else:
        call_counts = dict(subset.call_counts)
        stage_timing = dict(subset.timing)
    record = {
        "method": method,
        "profile_mode": "stage_instrumented" if instrumented else "clean",
        "n_frames": n_frames,
        "n_residues": coords.shape[1],
        "input_decode_seconds": decode_seconds,
        "solve_seconds": solve_seconds,
        "in_child_end_to_end_seconds": time.perf_counter() - total_start,
        "representative_weight_seconds": result.timing[
            "iterative_weight_estimation"
        ],
        "all_frame_fit_seconds": result.timing["weighted_fit"],
        "rmsf_seconds": result.timing["RMSF"],
        "peak_rss_mb": sampler.to_dict()["peak_rss_mb"],
        "average_iterations": result.average_iterations_per_frame,
        "converged_fraction": result.converged_fraction,
        "mean_rmsd": result.mean_rmsd,
        "mean_weighted_rmsd": result.mean_weighted_rmsd,
        **{f"representative_{key}_seconds": value for key, value in stage_timing.items()},
        **call_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


def fresh_run(
    method: str,
    n_frames: int,
    repeat: int,
    *,
    warmup: bool,
    instrumented: bool,
) -> dict[str, object]:
    raw = RESULT_ROOT / "raw_runs"
    suffix = "instrumented" if instrumented else (
        "warmup" if warmup else f"repeat_{repeat}"
    )
    output = raw / f"{method}__f{n_frames}__{suffix}.json"
    command = [
        str(CANONICAL_PYTHON),
        str(Path(__file__).resolve()),
        "--child",
        "--method",
        method,
        "--n-frames",
        str(n_frames),
        "--output",
        str(output),
    ]
    if instrumented:
        command.append("--instrumented")
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    fresh_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        return {
            "method": method,
            "n_frames": n_frames,
            "repeat": repeat,
            "warmup": int(warmup),
            "profile_mode": "stage_instrumented" if instrumented else "clean",
            "status": "failed",
            "fresh_process_seconds": fresh_seconds,
            "stderr": completed.stderr.strip(),
            "command": " ".join(command),
        }
    row = json.loads(output.read_text())
    row.update(
        {
            "repeat": repeat,
            "warmup": int(warmup),
            "status": "success",
            "fresh_process_seconds": fresh_seconds,
            "command": " ".join(command),
        }
    )
    return row


def performance_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_frames, clean_repeats in ((500, 3), (10000, 5)):
        for method in ("legacy_object", "sequential_lean"):
            rows.append(
                fresh_run(
                    method,
                    n_frames,
                    0,
                    warmup=True,
                    instrumented=False,
                )
            )
            for repeat in range(1, clean_repeats + 1):
                row = fresh_run(
                    method,
                    n_frames,
                    repeat,
                    warmup=False,
                    instrumented=False,
                )
                rows.append(row)
                print(
                    f"[performance] f={n_frames} {method} r={repeat} "
                    f"{row['status']}",
                    flush=True,
                )
            if n_frames == 10000:
                rows.append(
                    fresh_run(
                        method,
                        n_frames,
                        1,
                        warmup=False,
                        instrumented=True,
                    )
                )
    medians: dict[tuple[int, str], tuple[float, float]] = {}
    for n_frames in (500, 10000):
        for method in ("legacy_object", "sequential_lean"):
            selected = [
                row
                for row in rows
                if row["n_frames"] == n_frames
                and row["method"] == method
                and row["profile_mode"] == "clean"
                and not row["warmup"]
                and row["status"] == "success"
            ]
            medians[(n_frames, method)] = (
                statistics.median(
                    float(row["in_child_end_to_end_seconds"])
                    for row in selected
                ),
                statistics.median(
                    float(row["fresh_process_seconds"]) for row in selected
                ),
            )
    for row in rows:
        key = (int(row["n_frames"]), str(row["method"]))
        if key in medians:
            row["median_in_child_seconds"] = medians[key][0]
            row["median_fresh_process_seconds"] = medians[key][1]
            legacy = medians[(int(row["n_frames"]), "legacy_object")]
            row["speedup_vs_legacy_in_child"] = legacy[0] / medians[key][0]
            row["speedup_vs_legacy_fresh"] = legacy[1] / medians[key][1]
    return rows


def verify_hashes() -> dict[str, object]:
    project_manifest = (
        PROJECT_ROOT / "results/scaling_optimization_v1/baseline/key_output_hashes.tsv"
    )
    project_failures: list[str] = []
    with project_manifest.open(newline="") as handle:
        project_rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in project_rows:
        path = PROJECT_ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            project_failures.append(row["path"])
    golden_manifest = (
        PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
    )
    golden_failures: list[str] = []
    with golden_manifest.open(newline="") as handle:
        golden_rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in golden_rows:
        if row.get("status") != "success":
            continue
        path = PROJECT_ROOT / row["reference_npz"]
        if not path.is_file() or sha256(path) != row["reference_sha256"]:
            golden_failures.append(row["reference_npz"])
    return {
        "project_completion_checked": len(project_rows),
        "project_completion_failures": project_failures,
        "golden_checked": sum(
            row.get("status") == "success" for row in golden_rows
        ),
        "golden_failures": golden_failures,
        "status": (
            "pass"
            if not project_failures and not golden_failures
            else "failed"
        ),
    }


def run_tests() -> dict[str, object]:
    output = RESULT_ROOT / "test_results.txt"
    command = [str(CANONICAL_PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output.write_text(completed.stdout + completed.stderr)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": (completed.stdout.strip().splitlines() or [""])[-1],
    }


def condensed_tables(
    performance: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    memory_rows: list[dict[str, object]] = []
    call_rows: list[dict[str, object]] = []
    for row in performance:
        if row["status"] != "success":
            continue
        memory_rows.append(
            {
                "method": row["method"],
                "n_frames": row["n_frames"],
                "profile_mode": row["profile_mode"],
                "repeat": row["repeat"],
                "warmup": row["warmup"],
                "peak_rss_mb": row["peak_rss_mb"],
            }
        )
        call_rows.append(
            {
                "method": row["method"],
                "n_frames": row["n_frames"],
                "profile_mode": row["profile_mode"],
                "repeat": row["repeat"],
                "kabsch_calls": row.get("kabsch_calls", ""),
                "svd_calls": row.get("svd_calls", ""),
                "per_representative_result_objects": row.get(
                    "per_representative_result_objects", ""
                ),
                "retained_aligned_coordinate_arrays": row.get(
                    "retained_aligned_coordinate_arrays", ""
                ),
            }
        )
    return memory_rows, call_rows


def median_for(
    rows: list[dict[str, object]],
    n_frames: int,
    method: str,
    field: str,
) -> float:
    values = [
        float(row[field])
        for row in rows
        if row["n_frames"] == n_frames
        and row["method"] == method
        and row["profile_mode"] == "clean"
        and not row["warmup"]
        and row["status"] == "success"
    ]
    return statistics.median(values)


def write_report(
    parity: list[dict[str, object]],
    performance: list[dict[str, object]],
    tests: dict[str, object],
    integrity: dict[str, object],
) -> dict[str, object]:
    parity_passes = sum(int(row["quality_gate_pass"]) for row in parity)
    fresh_10k = median_for(
        performance, 10000, "sequential_lean", "fresh_process_seconds"
    )
    child_10k = median_for(
        performance,
        10000,
        "sequential_lean",
        "in_child_end_to_end_seconds",
    )
    legacy_fresh = median_for(
        performance, 10000, "legacy_object", "fresh_process_seconds"
    )
    lean_500 = median_for(
        performance, 500, "sequential_lean", "fresh_process_seconds"
    )
    legacy_500 = median_for(
        performance, 500, "legacy_object", "fresh_process_seconds"
    )
    legacy_rss = max(
        float(row["peak_rss_mb"])
        for row in performance
        if row["n_frames"] == 10000
        and row["method"] == "legacy_object"
        and row["profile_mode"] == "clean"
        and not row["warmup"]
    )
    lean_rss = max(
        float(row["peak_rss_mb"])
        for row in performance
        if row["n_frames"] == 10000
        and row["method"] == "sequential_lean"
        and row["profile_mode"] == "clean"
        and not row["warmup"]
    )
    correctness = (
        parity_passes == 34
        and tests["status"] == "pass"
        and integrity["status"] == "pass"
    )
    performance_gate = fresh_10k < 4.30
    short_gate = lean_500 <= legacy_500 * 1.05
    rss_gate = lean_rss <= legacy_rss
    accepted = correctness and performance_gate and short_gate and rss_gate
    instrumented = next(
        row
        for row in performance
        if row["method"] == "sequential_lean"
        and row["n_frames"] == 10000
        and row["profile_mode"] == "stage_instrumented"
        and row["status"] == "success"
    )
    stages = [
        (
            "weighted Kabsch/SVD",
            float(instrumented.get("representative_weighted_kabsch_seconds", 0.0)),
        ),
        (
            "sequence regularisation",
            float(
                instrumented.get(
                    "representative_sequence_regularization_seconds", 0.0
                )
            ),
        ),
        (
            "residual computation",
            float(instrumented.get("representative_residual_compute_seconds", 0.0)),
        ),
        (
            "weight update",
            float(instrumented.get("representative_weight_update_seconds", 0.0)),
        ),
        (
            "all-frame fitting",
            float(instrumented.get("all_frame_fit_seconds", 0.0)),
        ),
    ]
    stages.sort(key=lambda item: item[1], reverse=True)
    report = f"""# Phase 5 sequential_lean report

## Decision

`sequential_lean` {"passed" if accepted else "did not pass"} the complete Phase
5 acceptance contract. The public default was not changed in this phase;
`legacy_object` remains available as the frozen reference.

## Correctness

- Exact quality gate: {parity_passes}/34 cases.
- `mdcath_3pio100`: {next(row["status"] for row in parity if row["case_id"] == "mdcath_3pio100")}.
- Full tests: {tests["summary"]}.
- Frozen project-completion and golden hashes: {integrity["status"]}.

## Performance

- 10k fresh-process median, legacy_object: {legacy_fresh:.6f} s.
- 10k fresh-process median, sequential_lean: {fresh_10k:.6f} s.
- 10k in-child median, sequential_lean: {child_10k:.6f} s.
- Fresh-process speedup: {legacy_fresh / fresh_10k:.3f}x.
- Mandatory <4.30 s gate: {"pass" if performance_gate else "fail"}.
- Aspirational <3.50 s gate: {"pass" if fresh_10k < 3.50 else "not reached"}.
- 500-frame no-regression gate: {"pass" if short_gate else "fail"}
  ({legacy_500:.6f} s legacy vs {lean_500:.6f} s lean).

## Memory and object graph

- 10k peak RSS, legacy_object: {legacy_rss:.3f} MiB.
- 10k peak RSS, sequential_lean: {lean_rss:.3f} MiB.
- RSS gate: {"pass" if rss_gate else "fail"}.
- Per-representative public result objects in sequential_lean: 0.
- Retained representative aligned-coordinate arrays in sequential_lean: 0.

## Measured hotspots

| Rank | Stage | Instrumented seconds |
|---:|---|---:|
"""
    for rank, (label, seconds) in enumerate(stages, start=1):
        report += f"| {rank} | {label} | {seconds:.6f} |\n"
    report += f"""

Instrumentation is optional and its timing is not used for the clean speed
claim. The dominant call count remains one small 3x3 SVD per IRLS iteration
plus final refit; reducing this count would alter convergence semantics.
The canonical environment includes SciPy 1.11.4, so `sequential_lean` uses the
public SciPy `gesvd` driver for small 3x3 decompositions. SciPy remains optional:
the implementation falls back to NumPy SVD with identical scientific
semantics, but the 4.30 s performance result is specific to the recorded
canonical environment and must not be assumed for the fallback.

## Exact commands

```bash
PY={CANONICAL_PYTHON}
OPENBLAS_NUM_THREADS={THREAD_COUNT} OMP_NUM_THREADS={THREAD_COUNT} \\
MKL_NUM_THREADS={THREAD_COUNT} NUMEXPR_NUM_THREADS={THREAD_COUNT} \\
PYTHONPATH=src:. "$PY" scripts/scaling_optimization_v3/run_phase5.py
```

## Scientific interpretation

Phase 4B found that sequential warm-start is measurably order-dependent without
a consistent held-out advantage. Phase 5 nevertheless preserves it exactly,
providing a faster safe reference/default candidate. The scientifically
distinct `batched_independent` path remains explicit and experimental; no
guarded hybrid, GPU, FastScan or ANN path was introduced.
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "02_sequential_lean_report.md").write_text(report)
    return {
        "accepted": accepted,
        "correctness_gate": correctness,
        "parity_passes": parity_passes,
        "performance_gate": performance_gate,
        "short_gate": short_gate,
        "rss_gate": rss_gate,
        "fresh_10k_median_seconds": fresh_10k,
        "in_child_10k_median_seconds": child_10k,
        "legacy_fresh_10k_median_seconds": legacy_fresh,
        "test_status": tests,
        "integrity": integrity,
    }


def parent_main() -> int:
    if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
        raise RuntimeError(f"Use {CANONICAL_PYTHON}; got {sys.executable}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    parity = parity_panel()
    write_tsv(RESULT_ROOT / "parity.tsv", parity)
    performance = performance_matrix()
    write_tsv(RESULT_ROOT / "performance.tsv", performance)
    memory, calls = condensed_tables(performance)
    write_tsv(RESULT_ROOT / "memory.tsv", memory)
    write_tsv(RESULT_ROOT / "call_counts.tsv", calls)
    tests = run_tests()
    (RESULT_ROOT / "test_status.json").write_text(
        json.dumps(tests, indent=2, sort_keys=True) + "\n"
    )
    integrity = verify_hashes()
    (RESULT_ROOT / "integrity_after_phase5.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    summary = write_report(parity, performance, tests, integrity)
    (RESULT_ROOT / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0 if summary["accepted"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=("legacy_object", "sequential_lean"))
    parser.add_argument("--n-frames", type=int)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child:
        if args.method is None or args.n_frames is None or args.output is None:
            raise ValueError("--method, --n-frames and --output are required")
        return child_run(
            args.method,
            args.n_frames,
            instrumented=args.instrumented,
            output=args.output,
        )
    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
