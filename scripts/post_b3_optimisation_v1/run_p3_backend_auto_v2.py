#!/usr/bin/env python3
"""Register and benchmark a warmed, domain-separated backend-auto v2 panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from scripts.database_scale_benchmark_v2.run_b3_full_mdcath import load_task
from scripts.database_scale_gpu_v1.common import (
    core_mask,
    jaccard,
    safe_spearman,
)
from scripts.post_b3_optimisation_v1.common import (
    B3,
    GPU_PYTHON,
    OUT,
    ROOT,
    THREAD_ENV,
    domain_split,
    iter_tsv,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


P3 = OUT / "p3_backend_auto_v2"
MANIFEST = P3 / "backend_auto_v2_panel_locked.tsv"
CHECKPOINTS = P3 / "checkpoints"
THRESHOLDS = (1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 150_000)


def _quantile_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (int(row["workload_residue_frames"]), row["trajectory_id"]))
    indices = np.unique(
        np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    )
    return [ordered[index] for index in indices]


def register() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite locked panel: {MANIFEST}")
    candidates = [
        row
        for row in iter_tsv(B3 / "b3_all_trajectory_results.tsv")
        if row["status"] == "success"
        and row["temperature_K"] == "320"
        and row["replicate"] == "1"
    ]
    development = [row for row in candidates if domain_split(row["domain_id"]) == "development"]
    validation = [row for row in candidates if domain_split(row["domain_id"]) == "validation"]
    selected = _quantile_sample(development, 800) + _quantile_sample(validation, 200)
    rows = []
    for row in selected:
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "domain_split": domain_split(row["domain_id"]),
                "temperature_K": row["temperature_K"],
                "replicate": row["replicate"],
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "coordinate_path": row["coordinate_path"],
                "expected_n_residues": row["expected_n_residues"],
                "n_frames_frozen": row["n_frames"],
                "n_residues_frozen": row["n_residues"],
                "workload_residue_frames": row["workload_residue_frames"],
                "panel_selection": "domain_split_then_workload_quantile",
                "panel_locked_before_new_timing": True,
            }
        )
    if len(rows) != 1000 or len({row["domain_id"] for row in rows}) != 1000:
        raise RuntimeError("P3 panel must contain 1,000 distinct domains")
    write_tsv(MANIFEST, sorted(rows, key=lambda row: (row["domain_split"], row["trajectory_id"])))
    write_json(
        P3 / "registration.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "development": sum(row["domain_split"] == "development" for row in rows),
            "validation": sum(row["domain_split"] == "validation" for row in rows),
            "replicate_leakage": 0,
            "candidate_thresholds_residue_frames": list(THRESHOLDS),
            "timing_contract": "loaded_coordinates_metrics_transforms_RMSD_RMSF_weights",
            "public_default_change": False,
        },
    )


def _worker() -> PersistentGPUWorker:
    return PersistentGPUWorker.create(
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )


def _cpu(coords: np.ndarray, chains: np.ndarray):
    return solve_fast_global_alignment(
        coords,
        chain_ids=chains,
        representative_count=min(256, coords.shape[0]),
        update_rounds=4,
        chunk_size=1000,
        store_aligned=False,
        compute_confidence=False,
        collect_risk_signals=False,
    )


def _load(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, float]:
    loaded = load_task(row)
    if loaded["error"]:
        raise RuntimeError(str(loaded["error"]))
    coords = np.asarray(loaded["coords"], dtype=np.float64)
    chains = np.asarray(
        [item.get("chain", "_") for item in loaded["residue_metadata"]],
        dtype=object,
    )
    return coords, chains, float(loaded["shared_decode_seconds"])


def cold_probe() -> None:
    row = read_tsv(MANIFEST)[0]
    total = time.perf_counter()
    coords, chains, decode = _load(row)
    create_start = time.perf_counter()
    worker = _worker()
    create_seconds = time.perf_counter() - create_start
    compute_start = time.perf_counter()
    result = worker.throughput(coords)
    compute_seconds = time.perf_counter() - compute_start
    print(
        json.dumps(
            {
                "trajectory_id": row["trajectory_id"],
                "decode_seconds": decode,
                "worker_create_seconds": create_seconds,
                "first_gpu_call_seconds": compute_seconds,
                "reported_gpu_seconds": result.total_runtime_seconds,
                "child_total_seconds": time.perf_counter() - total,
            }
        )
    )


def run() -> None:
    rows = read_tsv(MANIFEST)
    pending = [row for row in rows if not (CHECKPOINTS / f"{row['trajectory_id']}.json").is_file()]
    if not rows:
        raise RuntimeError("P3 manifest is empty")
    first_coords, first_chains, _ = _load(rows[0])
    worker = _worker()
    worker.chain_ids = first_chains
    _cpu(first_coords, first_chains)
    worker.throughput(first_coords)
    worker.throughput(first_coords)
    cold_command = [str(GPU_PYTHON), str(Path(__file__).resolve()), "cold-probe"]
    cold_env = {**os.environ, **THREAD_ENV, "PYTHONPATH": "src:."}
    cold_start = time.perf_counter()
    cold = subprocess.run(
        cold_command,
        cwd=ROOT,
        env=cold_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    cold_wall = time.perf_counter() - cold_start
    cold_payload = json.loads(cold.stdout.strip().splitlines()[-1])
    write_json(
        P3 / "cold_warm_timing.json",
        {"fresh_process_wall_seconds": cold_wall, **cold_payload},
    )
    start = time.perf_counter()
    for index, row in enumerate(pending, start=1):
        task_start = time.perf_counter()
        try:
            coords, chains, decode = _load(row)
            cpu_start = time.perf_counter()
            cpu = _cpu(coords, chains)
            cpu_seconds = time.perf_counter() - cpu_start
            worker.chain_ids = chains
            gpu_start = time.perf_counter()
            gpu = worker.throughput(coords)
            gpu_seconds = time.perf_counter() - gpu_start
            cpu_core = core_mask(cpu.frozen_weights)
            gpu_core = core_mask(gpu.frozen_weights)
            output = {
                **row,
                "status": "success",
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "workload_residue_frames_observed": coords.shape[0] * coords.shape[1],
                "shared_decode_seconds": decode,
                "cpu_loaded_seconds": cpu_seconds,
                "gpu_loaded_seconds": gpu_seconds,
                "gpu_reported_seconds": gpu.total_runtime_seconds,
                "gpu_H2D_seconds": gpu.timing["representative_H2D"] + gpu.timing["all_frame_H2D"],
                "gpu_compute_seconds": gpu.timing["GPU_compute_total"],
                "gpu_D2H_seconds": gpu.timing["weight_D2H"] + gpu.timing["all_frame_D2H"],
                "cpu_gpu_core_jaccard": jaccard(cpu_core, gpu_core),
                "cpu_gpu_weight_spearman": safe_spearman(cpu.frozen_weights, gpu.frozen_weights),
                "cpu_gpu_RMSF_spearman": safe_spearman(cpu.robust_rmsf, gpu.robust_rmsf),
                "cpu_gpu_RMSF_MAE_A": float(np.mean(np.abs(cpu.robust_rmsf - gpu.robust_rmsf))),
                "peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "peak_VRAM_bytes": gpu.metadata["peak_GPU_memory_bytes"],
                "task_wall_seconds": time.perf_counter() - task_start,
            }
        except Exception as exc:
            output = {
                **row,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "task_wall_seconds": time.perf_counter() - task_start,
            }
        write_json(CHECKPOINTS / f"{row['trajectory_id']}.json", output)
        if index % 50 == 0 or index == len(pending):
            elapsed = time.perf_counter() - start
            print(f"[P3] {index}/{len(pending)} new; {3600 * index / max(elapsed, 1e-9):.1f} trajectories/hour", flush=True)
    write_json(
        P3 / "latest_run_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "new_rows": len(pending),
            "wall_seconds": time.perf_counter() - start,
            "GPU_fully_warmed_before_measured_panel": True,
        },
    )


def _model_coefficients(rows: list[dict[str, str]], field: str) -> np.ndarray:
    x = np.asarray(
        [
            [1.0, float(row["workload_residue_frames_observed"]), float(row["n_frames"]), float(row["n_residues"])]
            for row in rows
        ]
    )
    y = np.asarray([float(row[field]) for row in rows])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def _predict(row: dict[str, str], coefficients: np.ndarray) -> float:
    values = np.asarray([1.0, float(row["workload_residue_frames_observed"]), float(row["n_frames"]), float(row["n_residues"])])
    return float(max(0.0, values @ coefficients))


def evaluate_policy(rows: list[dict[str, str]], *, threshold: int | None, cpu_model=None, gpu_model=None) -> dict[str, object]:
    total = 0.0
    incorrect = gpu_count = 0
    for row in rows:
        cpu = float(row["cpu_loaded_seconds"])
        gpu = float(row["gpu_loaded_seconds"])
        if threshold is None and cpu_model is None:
            selected = "cpu"
        elif threshold is None:
            selected = "gpu" if _predict(row, gpu_model) < _predict(row, cpu_model) else "cpu"
        else:
            selected = "gpu" if int(row["workload_residue_frames_observed"]) >= threshold else "cpu"
        chosen = gpu if selected == "gpu" else cpu
        total += chosen
        gpu_count += selected == "gpu"
        incorrect += chosen > min(cpu, gpu) * 1.05
    return {
        "total_compute_seconds": total,
        "incorrect_backend_choices_gt_5pct": incorrect,
        "gpu_choices": gpu_count,
        "cpu_choices": len(rows) - gpu_count,
    }


def finalize() -> None:
    manifest = read_tsv(MANIFEST)
    paths = [CHECKPOINTS / f"{row['trajectory_id']}.json" for row in manifest]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"P3 has {len(missing)} missing checkpoints")
    rows = [json.loads(path.read_text()) for path in paths]
    failed = [row for row in rows if row["status"] != "success"]
    if failed:
        write_tsv(P3 / "backend_auto_v2_failures.tsv", failed)
        raise RuntimeError(f"P3 has {len(failed)} failed rows")
    write_tsv(P3 / "backend_auto_v2_timing.tsv", rows)
    development = [row for row in rows if row["domain_split"] == "development"]
    validation = [row for row in rows if row["domain_split"] == "validation"]
    cpu_model = _model_coefficients(development, "cpu_loaded_seconds")
    gpu_model = _model_coefficients(development, "gpu_loaded_seconds")
    policy_rows = []
    policies: list[tuple[str, int | None]] = [("always_CPU", None), ("always_warm_GPU", 0)]
    policies.extend((f"threshold_{threshold}", threshold) for threshold in THRESHOLDS)
    for split, split_rows in (("development", development), ("validation_locked", validation)):
        for name, threshold in policies:
            metrics = evaluate_policy(split_rows, threshold=threshold)
            policy_rows.append({"split": split, "policy": name, "threshold_residue_frames": "" if threshold is None else threshold, **metrics})
        policy_rows.append({"split": split, "policy": "simple_fitted_latency_model", "threshold_residue_frames": "not_applicable", **evaluate_policy(split_rows, threshold=None, cpu_model=cpu_model, gpu_model=gpu_model)})
        oracle = sum(min(float(row["cpu_loaded_seconds"]), float(row["gpu_loaded_seconds"])) for row in split_rows)
        policy_rows.append({"split": split, "policy": "per_trajectory_oracle_reporting_only", "threshold_residue_frames": "not_deployable", "total_compute_seconds": oracle, "incorrect_backend_choices_gt_5pct": 0, "gpu_choices": sum(float(row["gpu_loaded_seconds"]) < float(row["cpu_loaded_seconds"]) for row in split_rows), "cpu_choices": sum(float(row["gpu_loaded_seconds"]) >= float(row["cpu_loaded_seconds"]) for row in split_rows)})
    development_candidates = [row for row in policy_rows if row["split"] == "development" and row["policy"] != "per_trajectory_oracle_reporting_only"]
    selected = min(development_candidates, key=lambda row: float(row["total_compute_seconds"]))["policy"]
    for row in policy_rows:
        row["selected_on_development"] = row["policy"] == selected
    write_tsv(P3 / "backend_auto_v2_policy_comparison.tsv", policy_rows)
    write_json(P3 / "simple_latency_model.json", {"features": ["intercept", "residue_frames", "n_frames", "n_residues"], "cpu_coefficients": cpu_model.tolist(), "gpu_coefficients": gpu_model.tolist(), "selected_policy_on_development": selected, "validation_not_used_for_selection": True})
    validation_selected = next(row for row in policy_rows if row["split"] == "validation_locked" and row["policy"] == selected)
    write_text(
        P3 / "backend_auto_v2_report.md",
        "\n".join(
            [
                "# Backend-auto v2 report",
                "",
                f"- Timed trajectories: `{len(rows):,}` (`{len(development)}` development, `{len(validation)}` locked validation).",
                "- CUDA context, kernels, solver and memory pool were exercised twice before panel timing.",
                "- Candidate rules include always CPU/GPU, thresholds from 1k to 50k residue-frames, the legacy 150k rule and a simple linear latency model.",
                f"- Development-selected experimental policy: `{selected}`.",
                f"- Locked-validation total compute: `{float(validation_selected['total_compute_seconds']):.6f}` s; incorrect choices >5%: `{validation_selected['incorrect_backend_choices_gt_5pct']}/{len(validation)}`.",
                "- The public default was not changed; this is experimental evidence requiring ATLAS validation.",
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run", "cold-probe", "finalize"))
    args = parser.parse_args()
    if args.mode == "register":
        register()
    elif args.mode == "run":
        run()
    elif args.mode == "cold-probe":
        cold_probe()
    else:
        finalize()


if __name__ == "__main__":
    main()
