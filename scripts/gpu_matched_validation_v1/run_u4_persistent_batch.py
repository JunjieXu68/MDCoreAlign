#!/usr/bin/env python3
"""Benchmark persistent CPU/GPU batch throughput and external subprocess cost."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import numpy as np

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker, scan_w5_gpu
from mdsuperpose.mdlovofit import (
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.window_kinematics import scan_local_kinematics
from scripts.gpu_matched_validation_v1.common import (
    RESULTS,
    core_mask,
    jaccard,
    load_coords,
    load_panel,
    timing_summary,
    write_tsv,
)


OUTPUT = RESULTS / "u4_persistent_batch"
PYTHON = Path(
    "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python"
)
BATCH_SIZES = (1, 10, 100, 1000)
REPEATS = 5


def _inputs() -> tuple[list[np.ndarray], np.ndarray]:
    rows = {
        row["system"]: row
        for row in load_panel()
    }
    first, chains = load_coords(rows["4a18O00__T320__R1"])
    second, second_chains = load_coords(rows["4a18O00__T320__R2"])
    if not np.array_equal(chains, second_chains):
        raise ValueError("paired persistent-worker replicates have mismatched chains")
    return [first[:100], second[:100]], chains


def _gpu_options() -> dict[str, object]:
    return {
        "representative_count": 64,
        "update_rounds": 4,
        "chunk_size": 100,
        "precision": "float32",
        "rotation_solver": "horn",
        "store_aligned": False,
    }


def cold_child() -> None:
    inputs, chains = _inputs()
    start = time.perf_counter()
    worker = PersistentGPUWorker.create(chain_ids=chains, **_gpu_options())
    result = worker.throughput(inputs[0])
    print(
        json.dumps(
            {
                "child_compute_seconds": result.total_runtime_seconds,
                "child_import_worker_and_first_task_seconds": (
                    time.perf_counter() - start
                ),
                "peak_vram_bytes": result.metadata[
                    "peak_GPU_memory_bytes"
                ],
                "cache_summary": worker.cache_summary,
            }
        )
    )


def _run_batch(function, batch_size: int) -> tuple[float, object]:
    result = None
    start = time.perf_counter()
    for index in range(batch_size):
        result = function(index)
    assert result is not None
    return time.perf_counter() - start, result


def _row(
    *,
    method: str,
    batch_size: int,
    times: list[float],
    first_result,
    notes: str,
    contract: str = "C_persistent_batch",
) -> dict[str, object]:
    summary = timing_summary(times)
    median = summary["median_seconds"]
    metadata = getattr(first_result, "metadata", {})
    return {
        "contract": contract,
        "method": method,
        "batch_size": batch_size,
        "n_frames_per_task": 100,
        "n_residues_per_task": 134,
        **summary,
        "warm_per_task_seconds": median / batch_size,
        "amortised_tasks_per_second": batch_size / median,
        "peak_vram_mb": float(
            metadata.get("peak_GPU_memory_bytes", 0)
        )
        / (1024.0 * 1024.0),
        "process_peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        / 1024.0,
        "fallback_rate": 0.0 if "throughput" in method else "not_applicable",
        "measured_repeats": REPEATS,
        "trajectory_data_retained_between_jobs": False,
        "notes": notes,
    }


def _cold_latency() -> dict[str, object]:
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--cold-child",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "src:.",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    wall = time.perf_counter() - start
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "method": "GPU_throughput_persistent_worker",
        "fresh_process_wall_seconds": wall,
        **payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-child", action="store_true")
    args = parser.parse_args()
    if args.cold_child:
        cold_child()
        return

    inputs, chains = _inputs()
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    cold = _cold_latency()
    write_tsv(OUTPUT / "cold_start_latency.tsv", [cold])

    exact_references = [
        solve_exact_contract(
            coords,
            chain_ids=chains,
            representative_count=100,
            chunk_size=100,
            store_aligned=False,
        )
        for coords in inputs
    ]

    for batch_size in BATCH_SIZES:
        if batch_size <= 10:
            times = []
            final = None
            _run_batch(
                lambda index: solve_exact_contract(
                    inputs[index % 2],
                    chain_ids=chains,
                    representative_count=100,
                    chunk_size=100,
                    store_aligned=False,
                ),
                batch_size,
            )
            for _ in range(REPEATS):
                elapsed, final = _run_batch(
                    lambda index: solve_exact_contract(
                        inputs[index % 2],
                        chain_ids=chains,
                        representative_count=100,
                        chunk_size=100,
                        store_aligned=False,
                    ),
                    batch_size,
                )
                times.append(elapsed)
            rows.append(
                _row(
                    method="CPU_exact_sequential_lean",
                    batch_size=batch_size,
                    times=times,
                    first_result=final,
                    notes="Imports reused; exact sequential estimator per task.",
                )
            )

        cpu_times = []
        cpu_result = None
        _run_batch(
            lambda index: solve_fast_global_alignment(
                inputs[index % 2],
                chain_ids=chains,
                representative_count=64,
                update_rounds=4,
                chunk_size=100,
                store_aligned=False,
                compute_confidence=False,
            ),
            batch_size,
        )
        for _ in range(REPEATS):
            elapsed, cpu_result = _run_batch(
                lambda index: solve_fast_global_alignment(
                    inputs[index % 2],
                    chain_ids=chains,
                    representative_count=64,
                    update_rounds=4,
                    chunk_size=100,
                    store_aligned=False,
                    compute_confidence=False,
                ),
                batch_size,
            )
            cpu_times.append(elapsed)
        rows.append(
            _row(
                method="CPU_throughput_persistent_process",
                batch_size=batch_size,
                times=cpu_times,
                first_result=cpu_result,
                notes="Python imports and thread pools reused.",
            )
        )

        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            **_gpu_options(),
        )
        _run_batch(
            lambda index: worker.throughput(inputs[index % 2]),
            batch_size,
        )
        gpu_times = []
        gpu_result = None
        for _ in range(REPEATS):
            elapsed, gpu_result = _run_batch(
                lambda index: worker.throughput(inputs[index % 2]),
                batch_size,
            )
            gpu_times.append(elapsed)
        gpu_row = _row(
            method="GPU_throughput_persistent_worker",
            batch_size=batch_size,
            times=gpu_times,
            first_result=gpu_result,
            notes=(
                "CUDA context, CuPy pool, pinned host/device buffers, streams, "
                "scratch arrays and representative indices reused. Reusable "
                "coordinate buffers are zeroed after each task."
            ),
        )
        gpu_row["workspace_allocation_generation"] = (
            worker.cache_summary["workspace_allocation_generation"]
        )
        gpu_row["cache_summary"] = json.dumps(
            worker.cache_summary,
            sort_keys=True,
        )
        rows.append(gpu_row)
        quality_rows.append(
            {
                "batch_size": batch_size,
                "core_jaccard_gpu_vs_exact_first_task": jaccard(
                    core_mask(gpu_result.frozen_weights),
                    core_mask(
                        exact_references[(batch_size - 1) % 2].frozen_weights
                    ),
                ),
                "finite_outputs": bool(
                    np.isfinite(gpu_result.rmsd).all()
                ),
                "workspace_allocation_generation": worker.cache_summary[
                    "workspace_allocation_generation"
                ],
                "trajectory_data_retained_between_jobs": (
                    worker.cache_summary[
                        "trajectory_data_retained_between_jobs"
                    ]
                ),
            }
        )
        if batch_size <= 100:
            for method, function in (
                (
                    "CPU_W5",
                    lambda index: scan_local_kinematics(
                        inputs[index % 2],
                        chain_ids=chains,
                        evaluate_blocks=False,
                    ),
                ),
                (
                    "GPU_W5",
                    lambda index: scan_w5_gpu(
                        inputs[index % 2],
                        chain_ids=chains,
                        precision="float32",
                    ),
                ),
            ):
                _run_batch(function, batch_size)
                times = []
                final = None
                for _ in range(REPEATS):
                    elapsed, final = _run_batch(function, batch_size)
                    times.append(elapsed)
                rows.append(
                    _row(
                        method=method,
                        batch_size=batch_size,
                        times=times,
                        first_result=final,
                        contract=(
                            "C_persistent_batch_W5_local_deformation_"
                            "not_alignment_equivalent"
                        ),
                        notes=(
                            "Local-deformation batch contract; not compared "
                            "as an MDLovoFit-equivalent alignment workflow."
                        ),
                    )
                )
        write_tsv(OUTPUT / "persistent_batch_summary.tsv", rows)
        write_tsv(OUTPUT / "persistent_batch_quality.tsv", quality_rows)

    with tempfile.TemporaryDirectory(prefix="mdca_u4_") as temporary:
        workdir = Path(temporary)
        input_paths = []
        for index, coords in enumerate(inputs):
            path = workdir / f"input_{index}.pdb"
            write_multimodel_pdb(coords, path)
            input_paths.append(path)
        for batch_size in (1, 10):
            times = []
            for repeat in range(REPEATS + 1):
                start = time.perf_counter()
                for index in range(batch_size):
                    run_mdlovofit_on_files(
                        [input_paths[index % 2]],
                        fraction=0.6,
                        output_path=workdir / "mdlovofit_aligned.pdb",
                        executable="/usr/local/bin/mdlovofit",
                        timeout_seconds=120,
                    )
                elapsed = time.perf_counter() - start
                if repeat > 0:
                    times.append(elapsed)
            rows.append(
                _row(
                    method="MDLovoFit_subprocess_no_persistent_API",
                    batch_size=batch_size,
                    times=times,
                    first_result=object(),
                    notes=(
                        "Cached PDB input excluded. A new unavoidable external "
                        "subprocess is launched for each task because no "
                        "persistent MDLovoFit API is available."
                    ),
                )
            )
            write_tsv(OUTPUT / "persistent_batch_summary.tsv", rows)

    write_tsv(OUTPUT / "persistent_batch_summary.tsv", rows)
    write_tsv(OUTPUT / "persistent_batch_quality.tsv", quality_rows)
    (OUTPUT / "persistent_worker_report.md").write_text(
        "\n".join(
            [
                "# U4 persistent GPU worker",
                "",
                f"- Fresh-process wall latency: `{cold['fresh_process_wall_seconds']:.6f}` s",
                "- Batch sizes: `1`, `10`, `100`, `1000` for CPU/GPU throughput.",
                "- Exact and external subprocess baselines are restricted to "
                "feasible batch sizes.",
                "- The GPU worker reuses context, memory pool, pinned buffers, "
                "streams, scratch arrays and representative indices.",
                "- Reusable coordinate-bearing buffers are zeroed after each "
                "task; trajectory arrays are not stored by the worker.",
                "- MDLovoFit has no persistent in-process API, so each task "
                "retains unavoidable subprocess startup.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
