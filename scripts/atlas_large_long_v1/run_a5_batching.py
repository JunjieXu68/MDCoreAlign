#!/usr/bin/env python3
"""Benchmark same-shape ATLAS GPU batches as computational stress controls."""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import time

import numpy as np

from mdcorealign.experimental_gpu_batch import solve_same_shape_gpu_batch
from scripts.atlas_large_long_v1.common import (
    OUT,
    REPEATS,
    ROOT,
    WARMUPS,
    read_tsv,
    write_json,
    write_tsv,
)


BATCH_SIZES = (1, 2, 4, 8)
CHECKPOINTS = OUT / "a5_batching/checkpoints"


def _load(system: str) -> tuple[np.ndarray, np.ndarray]:
    manifest = {row["system"]: row for row in read_tsv(OUT / "atlas_manifest.tsv")}
    row = manifest[system]
    coords = np.load(ROOT / row["coordinate_cache_path"], mmap_mode="r")
    with np.load(ROOT / row["residue_metadata_path"], allow_pickle=False) as archive:
        chains = np.asarray(archive["chain_ids"])
    return coords, chains


def _core(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * 0.6 + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def _mode_options(mode: str) -> tuple[bool, bool]:
    if mode == "GPU_throughput_only_batched":
        return True, False
    if mode == "GPU_W5_only_batched":
        return False, True
    if mode == "GPU_throughput_plus_W5_batched":
        return True, True
    raise ValueError(mode)


def run(system: str) -> None:
    coords, chains = _load(system)
    values = np.asarray(coords, dtype=np.float32)
    exact_cache = np.load(
        OUT / "a3_loaded_analysis/result_cache" / f"{system}__f10001.npz",
        allow_pickle=False,
    )
    modes = (
        "GPU_throughput_only_batched",
        "GPU_W5_only_batched",
        "GPU_throughput_plus_W5_batched",
    )
    for batch_size in BATCH_SIZES:
        batch = np.broadcast_to(values, (batch_size, *values.shape))
        for mode in modes:
            checkpoint = CHECKPOINTS / f"{system}__b{batch_size:02d}__{mode}.tsv"
            quality_path = checkpoint.with_name(checkpoint.stem + "__quality.tsv")
            if checkpoint.exists() and quality_path.exists():
                print(f"[A5 batch] skip {system} b={batch_size} {mode}", flush=True)
                continue
            run_throughput, run_w5 = _mode_options(mode)
            rows: list[dict[str, object]] = []
            quality: dict[str, object] | None = None
            for index in range(WARMUPS + REPEATS):
                started = time.perf_counter()
                try:
                    result = solve_same_shape_gpu_batch(
                        batch,
                        chain_ids=chains,
                        representative_count=256,
                        update_rounds=4,
                        run_throughput=run_throughput,
                        run_w5=run_w5,
                    )
                    wall = time.perf_counter() - started
                    timing = result.timing
                    rows.append(
                        {
                            "system": system,
                            "mode": mode,
                            "batch_size": batch_size,
                            "n_frames_per_trajectory": values.shape[0],
                            "n_residues_per_trajectory": values.shape[1],
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "success",
                            "batch_wall_seconds": wall,
                            "warm_seconds_per_trajectory": wall / batch_size,
                            "trajectories_per_second": batch_size / wall,
                            "residue_frames_per_second": (
                                batch_size * values.shape[0] * values.shape[1] / wall
                            ),
                            "H2D_seconds": timing["H2D_seconds"],
                            "throughput_GPU_compute_seconds": timing[
                                "throughput_GPU_compute_seconds"
                            ],
                            "W5_GPU_compute_seconds": timing[
                                "W5_GPU_compute_seconds"
                            ],
                            "D2H_seconds": timing["D2H_seconds"],
                            "CPU_postprocess_seconds": timing[
                                "CPU_postprocess_seconds"
                            ],
                            "peak_RSS_mb_process": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / 1024.0,
                            "peak_VRAM_bytes": result.peak_vram_bytes,
                            "duplicated_input_computational_stress_only": True,
                            "biological_replicate_interpretation_permitted": False,
                            "notes": (
                                "Same decoded trajectory duplicated only to measure "
                                "same-shape GPU computational throughput."
                            ),
                        }
                    )
                    if index == WARMUPS:
                        quality = {
                            "system": system,
                            "mode": mode,
                            "batch_size": batch_size,
                            "all_outputs_finite": bool(
                                (result.rmsd is None or np.isfinite(result.rmsd).all())
                                and (
                                    result.local_deformation_index is None
                                    or np.isfinite(result.local_deformation_index).all()
                                )
                            ),
                            "identical_duplicate_throughput_weights": (
                                "not_applicable"
                                if result.weights is None
                                else bool(
                                    np.allclose(
                                        result.weights,
                                        result.weights[0][None, :],
                                        atol=1e-7,
                                        rtol=1e-7,
                                    )
                                )
                            ),
                            "first_trajectory_core_Jaccard_vs_exact": (
                                "not_applicable"
                                if result.weights is None
                                else _jaccard(
                                    _core(result.weights[0]),
                                    np.asarray(exact_cache["exact_core"], dtype=bool),
                                )
                            ),
                            "W5_duplicate_labels_identical": (
                                "not_applicable"
                                if result.local_stable_labels is None
                                else bool(
                                    np.array_equal(
                                        result.local_stable_labels,
                                        np.broadcast_to(
                                            result.local_stable_labels[0],
                                            result.local_stable_labels.shape,
                                        ),
                                    )
                                )
                            ),
                        }
                    del result
                except Exception as exc:
                    rows.append(
                        {
                            "system": system,
                            "mode": mode,
                            "batch_size": batch_size,
                            "n_frames_per_trajectory": values.shape[0],
                            "n_residues_per_trajectory": values.shape[1],
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "failure",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "duplicated_input_computational_stress_only": True,
                        }
                    )
                    quality = {
                        "system": system,
                        "mode": mode,
                        "batch_size": batch_size,
                        "all_outputs_finite": False,
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                    break
            write_tsv(checkpoint, rows)
            write_tsv(quality_path, [quality or {"status": "not_measured"}])
            print(f"[A5 batch] complete {system} b={batch_size} {mode}", flush=True)
            gc.collect()
        del batch
    exact_cache.close()
    write_json(
        OUT / f"a5_batching/{system}_status.json",
        {"system": system, "status": "complete", "batch_sizes": list(BATCH_SIZES)},
    )


def finalize() -> None:
    raw: list[dict[str, str]] = []
    quality: list[dict[str, str]] = []
    for item in read_tsv(OUT / "atlas_manifest.tsv"):
        system = item["system"]
        for batch_size in BATCH_SIZES:
            for mode in (
                "GPU_throughput_only_batched",
                "GPU_W5_only_batched",
                "GPU_throughput_plus_W5_batched",
            ):
                checkpoint = CHECKPOINTS / f"{system}__b{batch_size:02d}__{mode}.tsv"
                raw.extend(read_tsv(checkpoint))
                quality.extend(
                    read_tsv(checkpoint.with_name(checkpoint.stem + "__quality.tsv"))
                )
    write_tsv(OUT / "a5_batching/atlas_batching_raw_repeats.tsv", raw)
    write_tsv(OUT / "a5_batching/atlas_batching_quality.tsv", quality)
    summary = []
    keys = sorted(
        {(row["system"], row["mode"], int(row["batch_size"])) for row in raw}
    )
    for system, mode, batch_size in keys:
        selected_all = [
            row
            for row in raw
            if row["system"] == system
            and row["mode"] == mode
            and int(row["batch_size"]) == batch_size
        ]
        selected = [
            row
            for row in selected_all
            if row.get("status") == "success"
            and row.get("is_warmup", "").lower() == "false"
        ]
        failures = [row for row in selected_all if row.get("status") == "failure"]
        values = [float(row["batch_wall_seconds"]) for row in selected]
        summary.append(
            {
                "system": system,
                "mode": mode,
                "batch_size": batch_size,
                "status": (
                    "success"
                    if len(values) == REPEATS
                    else "failed"
                    if failures
                    else "incomplete"
                ),
                "measured_repeats": len(values),
                "median_batch_wall_seconds": statistics.median(values) if values else "",
                "median_seconds_per_trajectory": (
                    statistics.median(values) / batch_size if values else ""
                ),
                "median_trajectories_per_second": (
                    batch_size / statistics.median(values) if values else ""
                ),
                "minimum_batch_wall_seconds": min(values) if values else "",
                "maximum_batch_wall_seconds": max(values) if values else "",
                "failure_count": len(failures),
                "failure_reason": " | ".join(
                    sorted({row.get("failure_reason", "") for row in failures})
                ),
                "duplicated_input_computational_stress_only": True,
            }
        )
    write_tsv(OUT / "a5_batching/atlas_batching_summary.tsv", summary)
    write_json(
        OUT / "a5_batching/atlas_batching_status.json",
        {"status": "complete", "raw_rows": len(raw), "summary_rows": len(summary)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "finalize"))
    parser.add_argument("--system")
    args = parser.parse_args()
    if args.mode == "finalize":
        finalize()
        return
    systems = [row["system"] for row in read_tsv(OUT / "atlas_manifest.tsv")]
    for system in systems if args.system is None else [args.system]:
        run(system)


if __name__ == "__main__":
    main()
