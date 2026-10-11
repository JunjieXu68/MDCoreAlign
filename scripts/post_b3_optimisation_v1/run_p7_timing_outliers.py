#!/usr/bin/env python3
"""Rerun prospectively defined D1 GPU timing outliers under warm control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np

from mdcorealign.gpu_backend import PersistentGPUWorker
from scripts.database_scale_benchmark_v2.run_b3_full_mdcath import load_task
from scripts.post_b3_optimisation_v1.common import (
    OUT,
    ROOT,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


P7 = OUT / "p7_timing_atlas"
D1 = ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv"
MANIFEST = P7 / "gpu_timing_outliers_locked.tsv"
THRESHOLD = 0.1
REPEATS = 5


def register() -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    rows = [
        row
        for row in read_tsv(D1)
        if row["status"] == "success" and float(row["gpu_compute_seconds"]) > THRESHOLD
    ]
    if len(rows) != 109:
        raise RuntimeError(f"expected 109 D1 outliers, observed {len(rows)}")
    write_tsv(
        MANIFEST,
        [
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["system_id"],
                "temperature_K": row["temperature_K"],
                "replicate": row["replicate"],
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "coordinate_path": row["coordinate_path"],
                "expected_n_residues": row["expected_n_residues"],
                "original_gpu_compute_seconds": row["gpu_compute_seconds"],
                "original_gpu_loaded_seconds": row["gpu_throughput_loaded_compute_seconds"],
                "prospective_outlier_rule": "original_gpu_compute_seconds_gt_0.1",
            }
            for row in rows
        ],
    )
    write_json(P7 / "timing_outlier_registration.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "threshold_seconds": THRESHOLD, "outlier_count": len(rows), "warmups_per_case": 1, "measured_repeats_per_case": REPEATS, "original_timings_preserved": True})


def run() -> None:
    tasks = read_tsv(MANIFEST)
    worker = PersistentGPUWorker.create(representative_count=256, update_rounds=4, chunk_size=1000, precision="float32", rotation_solver="horn")
    output = []
    for index, task in enumerate(tasks, start=1):
        loaded = load_task(task)
        if loaded["error"]:
            output.append({**task, "status": "failed", "failure_reason": loaded["error"]})
            continue
        coords = np.asarray(loaded["coords"], dtype=np.float64)
        chains = np.asarray([row.get("chain", "_") for row in loaded["residue_metadata"]], dtype=object)
        worker.chain_ids = chains
        worker.throughput(coords)
        for repeat in range(REPEATS):
            start = time.perf_counter()
            result = worker.throughput(coords)
            wall = time.perf_counter() - start
            output.append({**task, "status": "success", "repeat_index": repeat + 1, "shared_decode_seconds_once": loaded["shared_decode_seconds"] if repeat == 0 else 0.0, "controlled_warm_wall_seconds": wall, "controlled_warm_reported_seconds": result.total_runtime_seconds, "controlled_gpu_compute_seconds": result.timing["GPU_compute_total"], "controlled_H2D_seconds": result.timing["representative_H2D"] + result.timing["all_frame_H2D"], "controlled_D2H_seconds": result.timing["weight_D2H"] + result.timing["all_frame_D2H"], "peak_VRAM_bytes": result.metadata["peak_GPU_memory_bytes"], "explicit_GPU_synchronization": result.metadata["explicit_CUDA_synchronization"]})
        if index % 10 == 0 or index == len(tasks):
            print(f"[P7 timing] {index}/{len(tasks)} cases", flush=True)
    write_tsv(P7 / "controlled_gpu_timing_repeats.tsv", output)


def finalize() -> None:
    rows = read_tsv(P7 / "controlled_gpu_timing_repeats.tsv")
    failures = [row for row in rows if row["status"] != "success"]
    grouped = {}
    for row in rows:
        if row["status"] == "success":
            grouped.setdefault(row["trajectory_id"], []).append(row)
    summary = []
    for trajectory_id, values in sorted(grouped.items()):
        compute = [float(row["controlled_gpu_compute_seconds"]) for row in values]
        wall = [float(row["controlled_warm_wall_seconds"]) for row in values]
        summary.append({"trajectory_id": trajectory_id, "domain_id": values[0]["domain_id"], "original_gpu_compute_seconds": values[0]["original_gpu_compute_seconds"], "original_gpu_loaded_seconds": values[0]["original_gpu_loaded_seconds"], "controlled_compute_median_seconds": statistics.median(compute), "controlled_compute_min_seconds": min(compute), "controlled_compute_max_seconds": max(compute), "controlled_wall_median_seconds": statistics.median(wall), "controlled_still_gt_0.1_seconds": statistics.median(compute) > THRESHOLD, "measured_repeats": len(values)})
    write_tsv(P7 / "controlled_gpu_timing_summary.tsv", summary)
    persistent = sum(str(row["controlled_still_gt_0.1_seconds"]).lower() == "true" for row in summary)
    write_text(P7 / "timing_outlier_report.md", "\n".join(["# Warm GPU timing-outlier audit", "", f"- Prospective rule: original D1 `gpu_compute_seconds > {THRESHOLD}` s.", f"- Affected cases: `{len(summary)}`.", "- Every case used one unreported warmup followed by five synchronized measurements in one persistent GPU process.", f"- Cases whose controlled median remained above 0.1 s: `{persistent}/{len(summary)}`.", f"- Failed controlled cases: `{len(failures)}`.", "- Original D1 timings are preserved; both original and controlled values are reported."]))
    write_json(P7 / "timing_outlier_status.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "registered_outliers": 109, "completed": len(summary), "failed_rows": len(failures), "controlled_persistent_outliers": persistent})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run", "finalize"))
    args = parser.parse_args()
    {"register": register, "run": run, "finalize": finalize}[args.mode]()


if __name__ == "__main__":
    main()
