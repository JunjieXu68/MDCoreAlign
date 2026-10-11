#!/usr/bin/env python3
"""Run ATLAS internal materialised-output timing without changing algorithms."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from mdcorealign.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.atlas_large_long_v1.common import (
    OUT,
    PREFIXES,
    REPEATS,
    ROOT,
    WARMUPS,
    prefix_label,
    read_tsv,
    write_json,
    write_tsv,
)


CHECKPOINTS = OUT / "a3_materialised_output/checkpoints"


def _load(system: str) -> tuple[np.ndarray, np.ndarray]:
    manifest = {row["system"]: row for row in read_tsv(OUT / "atlas_manifest.tsv")}
    row = manifest[system]
    coords = np.load(ROOT / row["coordinate_cache_path"], mmap_mode="r")
    with np.load(ROOT / row["residue_metadata_path"], allow_pickle=False) as archive:
        chains = np.asarray(archive["chain_ids"])
    return coords, chains


def _row(
    *,
    system: str,
    frames: int,
    method: str,
    repeat_index: int,
    is_warmup: bool,
    result: object,
    write_seconds: float,
    output_size: int,
    full_seconds: float,
) -> dict[str, object]:
    timing = getattr(result, "timing", {})
    metadata = getattr(result, "metadata", {})
    transform_seconds = float(timing.get("aligned_coordinate_materialization", 0.0))
    algorithm_seconds = max(0.0, float(result.total_runtime_seconds) - transform_seconds)
    return {
        "system": system,
        "prefix_label": prefix_label(frames),
        "n_frames": frames,
        "n_residues": int(result.frozen_weights.size),
        "residue_frames": frames * int(result.frozen_weights.size),
        "contract": "C_materialised_output_internal_aligned_CA_PDB",
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": is_warmup,
        "status": "success",
        "total_seconds": full_seconds,
        "analysis_metrics_transforms_seconds": algorithm_seconds,
        "coordinate_transform_materialization_seconds": transform_seconds,
        "aligned_PDB_write_seconds": write_seconds,
        "output_size_bytes": output_size,
        "H2D_seconds": timing.get("representative_H2D", 0.0)
        + timing.get("all_frame_H2D", 0.0),
        "GPU_compute_seconds": timing.get("GPU_compute_total", 0.0),
        "D2H_seconds": timing.get("weight_D2H", 0.0)
        + timing.get("all_frame_D2H", 0.0),
        "peak_RSS_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "peak_VRAM_bytes": metadata.get("peak_GPU_memory_bytes", 0),
        "mean_unified_RMSD_A": float(np.mean(result.rmsd)),
        "mean_robust_RMSF_A": float(np.mean(result.robust_rmsf)),
        "explicit_CUDA_synchronization": metadata.get(
            "explicit_CUDA_synchronization", False
        ),
        "input_decode_included": False,
        "notes": (
            "Loaded CA coordinates through analysis/transforms, full aligned-coordinate "
            "materialisation and streaming CA-only PDB writing."
        ),
    }


def _failure_row(
    system: str,
    frames: int,
    method: str,
    repeat_index: int,
    is_warmup: bool,
    exc: Exception,
) -> dict[str, object]:
    return {
        "system": system,
        "prefix_label": prefix_label(frames),
        "n_frames": frames,
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": is_warmup,
        "contract": "C_materialised_output_internal_aligned_CA_PDB",
        "status": "failure",
        "failure_reason": f"{type(exc).__name__}: {exc}",
    }


def run(system: str) -> None:
    coords, chains = _load(system)
    gpu_worker = PersistentGPUWorker.create(
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
        store_aligned=True,
    )
    gpu_worker.throughput(np.asarray(coords[:250]))
    methods = (
        (
            "CPU_exact_sequential_lean",
            lambda values: solve_exact_contract(
                values,
                chain_ids=chains,
                representative_count=min(2000, values.shape[0]),
                chunk_size=1000,
                store_aligned=True,
            ),
        ),
        (
            "CPU_throughput",
            lambda values: solve_fast_global_alignment(
                values,
                chain_ids=chains,
                representative_count=min(256, values.shape[0]),
                update_rounds=4,
                chunk_size=1000,
                store_aligned=True,
                compute_confidence=False,
            ),
        ),
        (
            "GPU_throughput_float32_Horn",
            lambda values: gpu_worker.throughput(values, store_aligned=True),
        ),
        (
            "backend_auto_v2_threshold_25000",
            lambda values: (
                gpu_worker.throughput(values, store_aligned=True)
                if values.shape[0] * values.shape[1] >= 25000
                else solve_fast_global_alignment(
                    values,
                    chain_ids=chains,
                    representative_count=min(256, values.shape[0]),
                    update_rounds=4,
                    chunk_size=1000,
                    store_aligned=True,
                    compute_confidence=False,
                )
            ),
        ),
    )
    for frames in PREFIXES:
        checkpoint = CHECKPOINTS / f"{system}__{prefix_label(frames)}.tsv"
        if checkpoint.exists():
            print(f"[A3 materialised] skip {system} {frames}", flush=True)
            continue
        values = np.asarray(coords[:frames])
        rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="atlas_materialised_", dir="/tmp") as work:
            workdir = Path(work)
            for method, function in methods:
                for index in range(WARMUPS + REPEATS):
                    warmup = index < WARMUPS
                    output = workdir / f"{method}.pdb"
                    started = time.perf_counter()
                    try:
                        result = function(values)
                        write_started = time.perf_counter()
                        write_multimodel_ca_pdb_streaming(result.aligned_coords, output)
                        write_seconds = time.perf_counter() - write_started
                        full_seconds = time.perf_counter() - started
                        rows.append(
                            _row(
                                system=system,
                                frames=frames,
                                method=method,
                                repeat_index=index - WARMUPS,
                                is_warmup=warmup,
                                result=result,
                                write_seconds=write_seconds,
                                output_size=output.stat().st_size,
                                full_seconds=full_seconds,
                            )
                        )
                        output.unlink(missing_ok=True)
                        del result
                    except Exception as exc:
                        rows.append(
                            _failure_row(
                                system,
                                frames,
                                method,
                                index - WARMUPS,
                                warmup,
                                exc,
                            )
                        )
                        break
        write_tsv(checkpoint, rows)
        print(f"[A3 materialised] complete {system} {frames}", flush=True)
        del values
        gc.collect()
    write_json(
        OUT / f"a3_materialised_output/{system}_status.json",
        {"system": system, "status": "complete", "prefixes": list(PREFIXES)},
    )


def finalize() -> None:
    rows: list[dict[str, str]] = []
    manifest = read_tsv(OUT / "atlas_manifest.tsv")
    for item in manifest:
        for frames in PREFIXES:
            rows.extend(
                read_tsv(
                    CHECKPOINTS / f"{item['system']}__{prefix_label(frames)}.tsv"
                )
            )
    write_tsv(
        OUT / "a3_materialised_output/materialised_internal_raw_repeats.tsv", rows
    )
    write_json(
        OUT / "a3_materialised_output/materialised_internal_status.json",
        {"status": "complete", "rows": len(rows)},
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
