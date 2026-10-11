#!/usr/bin/env python3
"""Benchmark true same-shape multi-trajectory GPU batching."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import resource
import statistics
import time

import numpy as np

from mdcorealign.experimental_gpu_batch import solve_same_shape_gpu_batch
from mdcorealign.gpu_backend import PersistentGPUWorker, scan_w5_gpu
from scripts.database_scale_benchmark_v2.run_b3_full_mdcath import load_task
from scripts.database_scale_gpu_v1.common import core_mask, jaccard, safe_spearman
from scripts.post_b3_optimisation_v1.common import (
    B3,
    OUT,
    iter_tsv,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


P4 = OUT / "p4_gpu_batching"
MANIFEST = P4 / "gpu_batching_panel_locked.tsv"
BATCH_SIZES = (1, 4, 8, 16, 25, 50)
REPEATS = 5


def register() -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    by_domain = defaultdict(list)
    for row in iter_tsv(B3 / "b3_all_trajectory_results.tsv"):
        if row["status"] == "success":
            by_domain[row["domain_id"]].append(row)
    eligible = []
    for domain, rows in by_domain.items():
        shapes = {(int(row["n_frames"]), int(row["n_residues"])) for row in rows}
        if len(rows) == 25 and len(shapes) == 1:
            n_frames, n_residues = next(iter(shapes))
            eligible.append((n_residues, n_frames, domain, rows))
    eligible.sort()
    indices = [int(round(value)) for value in np.linspace(0, len(eligible) - 1, 3)]
    selected = [eligible[index] for index in indices]
    output = []
    for _, _, _, rows in selected:
        for row in sorted(rows, key=lambda item: (int(item["temperature_K"]), int(item["replicate"]))):
            output.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "domain_id": row["domain_id"],
                    "temperature_K": row["temperature_K"],
                    "replicate": row["replicate"],
                    "source_file": row["source_file"],
                    "source_sha256": row["source_sha256"],
                    "coordinate_path": row["coordinate_path"],
                    "expected_n_residues": row["expected_n_residues"],
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "same_shape_within_domain": True,
                }
            )
    write_tsv(MANIFEST, output)
    write_json(P4 / "registration.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "domains": [item[2] for item in selected], "rows": len(output), "batch_sizes": list(BATCH_SIZES), "repeats": REPEATS, "batch_50_reuses_raw_inputs_in_a_second_cycle": True, "public_default_change": False})


def _load_domain(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, float]:
    arrays = []
    chains = None
    decode = 0.0
    for row in rows:
        loaded = load_task(row)
        if loaded["error"]:
            raise RuntimeError(str(loaded["error"]))
        arrays.append(np.asarray(loaded["coords"], dtype=np.float64))
        observed = np.asarray([item.get("chain", "_") for item in loaded["residue_metadata"]], dtype=object)
        if chains is None:
            chains = observed
        elif not np.array_equal(chains, observed):
            raise RuntimeError("chain metadata differs inside one domain")
        decode += float(loaded["shared_decode_seconds"])
    return np.stack(arrays), chains, decode


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    ordered = sorted(values)
    return statistics.median(ordered), float(np.quantile(ordered, 0.25)), float(np.quantile(ordered, 0.75)), min(ordered)


def run() -> None:
    rows = read_tsv(MANIFEST)
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain_id"]].append(row)
    timing_rows = []
    quality_rows = []
    for domain, tasks in sorted(by_domain.items()):
        coords, chains, decode = _load_domain(tasks)
        serial_worker = PersistentGPUWorker.create(chain_ids=chains, representative_count=256, update_rounds=4, chunk_size=1000, precision="float32", rotation_solver="horn")
        serial_worker.throughput(coords[0])
        serial_throughput = [serial_worker.throughput(frame_set) for frame_set in coords]
        serial_w5 = [scan_w5_gpu(frame_set, chain_ids=chains, precision="float32") for frame_set in coords]
        for batch_size in BATCH_SIZES:
            batch = np.stack([coords[index % len(coords)] for index in range(batch_size)])
            for contract, kwargs in (
                ("throughput_only", {"run_throughput": True, "run_w5": False}),
                ("W5_only", {"run_throughput": False, "run_w5": True}),
                ("fused_throughput_plus_W5", {"run_throughput": True, "run_w5": True}),
            ):
                solve_same_shape_gpu_batch(batch, chain_ids=chains, **kwargs)
                times = []
                result = None
                for _ in range(REPEATS):
                    start = time.perf_counter()
                    result = solve_same_shape_gpu_batch(batch, chain_ids=chains, **kwargs)
                    times.append(time.perf_counter() - start)
                assert result is not None
                median, q1, q3, minimum = _summary(times)
                timing_rows.append({"domain_id": domain, "n_frames": batch.shape[1], "n_residues": batch.shape[2], "batch_size": batch_size, "contract": contract, "measured_repeats": REPEATS, "median_total_seconds": median, "q1_total_seconds": q1, "q3_total_seconds": q3, "minimum_total_seconds": minimum, "median_trajectories_per_second": batch_size / median, "selector_and_transfer_included": True, "H2D_seconds_last_repeat": result.timing["H2D_seconds"], "GPU_compute_seconds_last_repeat": result.timing["throughput_GPU_compute_seconds"] + result.timing["W5_GPU_compute_seconds"], "D2H_seconds_last_repeat": result.timing["D2H_seconds"], "CPU_postprocess_seconds_last_repeat": result.timing["CPU_postprocess_seconds"], "peak_VRAM_bytes": result.peak_vram_bytes, "process_peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "shared_decode_seconds_domain_once": decode})
                if result.weights is not None:
                    core_values = []
                    weight_values = []
                    rmsf_values = []
                    rmsf_mae = []
                    for index in range(batch_size):
                        reference = serial_throughput[index % len(coords)]
                        core_values.append(jaccard(core_mask(result.weights[index]), core_mask(reference.frozen_weights)))
                        weight_values.append(safe_spearman(result.weights[index], reference.frozen_weights))
                        rmsf_values.append(safe_spearman(result.robust_rmsf[index], reference.robust_rmsf))
                        rmsf_mae.append(float(np.mean(np.abs(result.robust_rmsf[index] - reference.robust_rmsf))))
                    quality_rows.append({"domain_id": domain, "batch_size": batch_size, "contract": contract, "median_core_jaccard_vs_individual_GPU": statistics.median(core_values), "minimum_core_jaccard_vs_individual_GPU": min(core_values), "median_weight_spearman_vs_individual_GPU": statistics.median(weight_values), "median_RMSF_spearman_vs_individual_GPU": statistics.median(rmsf_values), "maximum_RMSF_MAE_A_vs_individual_GPU": max(rmsf_mae), "finite_outputs": bool(np.isfinite(result.rmsd).all())})
                if result.local_deformation_index is not None:
                    local_spearman = []
                    local_label = []
                    for index in range(batch_size):
                        reference = serial_w5[index % len(coords)]
                        local_spearman.append(safe_spearman(result.local_deformation_index[index], reference.local_deformation_index))
                        local_label.append(jaccard(result.local_stable_labels[index], reference.scan.candidate_stable_label))
                    quality_rows.append({"domain_id": domain, "batch_size": batch_size, "contract": contract + "_local", "median_local_deformation_spearman_vs_individual_GPU": statistics.median(local_spearman), "minimum_local_label_jaccard_vs_individual_GPU": min(local_label), "finite_outputs": bool(np.isfinite(result.local_deformation_index).all())})
                write_tsv(P4 / "gpu_batching_timing.tsv", timing_rows)
                write_tsv(P4 / "gpu_batching_quality.tsv", quality_rows)
    write_json(P4 / "run_status.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "domains": len(by_domain), "timing_rows": len(timing_rows), "quality_rows": len(quality_rows), "true_same_shape_kernel_batching": True})


def finalize() -> None:
    timing = read_tsv(P4 / "gpu_batching_timing.tsv")
    quality = read_tsv(P4 / "gpu_batching_quality.tsv")
    throughput = [row for row in timing if row["contract"] == "throughput_only"]
    batch1 = statistics.median(float(row["median_trajectories_per_second"]) for row in throughput if row["batch_size"] == "1")
    batch25 = statistics.median(float(row["median_trajectories_per_second"]) for row in throughput if row["batch_size"] == "25")
    min_core = min(float(row["minimum_core_jaccard_vs_individual_GPU"]) for row in quality if row.get("minimum_core_jaccard_vs_individual_GPU"))
    min_rmsf = min(float(row["median_RMSF_spearman_vs_individual_GPU"]) for row in quality if row.get("median_RMSF_spearman_vs_individual_GPU"))
    write_text(P4 / "GPU_batching_report.md", "\n".join(["# Multi-trajectory GPU batching", "", "- This is an experimental same-shape, same-residue-mapping path; the public default is unchanged.", f"- Median throughput gain, batch 25 versus batch 1: `{batch25 / batch1:.3f}x`.", f"- Minimum core Jaccard versus individual GPU execution: `{min_core:.6f}`.", f"- Minimum median RMSF Spearman versus individual GPU execution: `{min_rmsf:.6f}`.", "- Timing includes H2D, synchronized GPU compute, D2H and CPU W5 post-processing.", "- Batch size 50 repeats the locked 25 raw domain trajectories in a second cycle; it is an engineering saturation test, not additional biological evidence."]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run", "finalize"))
    args = parser.parse_args()
    {"register": register, "run": run, "finalize": finalize}[args.mode]()


if __name__ == "__main__":
    main()
