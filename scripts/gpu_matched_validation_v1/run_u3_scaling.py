#!/usr/bin/env python3
"""Run a two-dimensional frames-by-residues CPU/GPU scaling matrix."""

from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from scripts.gpu_matched_validation_v1.common import (
    RESULTS,
    core_mask,
    jaccard,
    read_tsv,
    spearman,
    timing_summary,
    write_tsv,
)


OUTPUT = RESULTS / "u3_scaling"
FRAMES = (1000, 5000, 10_000, 50_000, 100_000, 250_000)
RESIDUES = (150, 300, 500, 1000, 2000, 3000)
MAX_RESIDUE_FRAMES = 50_000_000
WARMUPS = 1
REPEATS = 5
MDLOVOFIT_CELLS = {
    (1000, 150),
    (5000, 150),
    (10_000, 150),
    (1000, 300),
    (1000, 500),
}


def _trajectory(n_frames: int, n_residues: int) -> np.ndarray:
    rng = np.random.default_rng(20260801 + n_residues)
    reference = np.cumsum(
        rng.normal(scale=0.9, size=(n_residues, 3)),
        axis=0,
    )
    coords = np.empty((n_frames, n_residues, 3), dtype=np.float64)
    phases = np.linspace(0.0, 2.0 * np.pi, n_residues, endpoint=False)
    mobile_start = n_residues // 3
    mobile_stop = min(n_residues, mobile_start + max(8, n_residues // 5))
    for start in range(0, n_frames, 2000):
        stop = min(start + 2000, n_frames)
        t = np.linspace(
            start / max(1, n_frames - 1),
            (stop - 1) / max(1, n_frames - 1),
            stop - start,
        )
        chunk = coords[start:stop]
        chunk[:] = reference
        wave = np.sin(8.0 * np.pi * t[:, None] + phases[None, :])
        chunk[:, :, 0] += 0.015 * wave
        chunk[:, :, 1] += 0.012 * np.cos(
            6.0 * np.pi * t[:, None] + phases[None, :]
        )
        hinge = 0.40 * np.sin(4.0 * np.pi * t)
        chunk[:, mobile_start:mobile_stop, 2] += hinge[:, None]
        chunk[:, :, 0] += (0.5 * np.sin(2.0 * np.pi * t))[:, None]
    return coords


def _rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


class _MemorySampler:
    def __init__(self) -> None:
        self.peak = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.005):
            self.peak = max(self.peak, _rss_mb())

    def __enter__(self) -> "_MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_mb())


def _repeat(function) -> tuple[object, list[float], float]:
    function()
    result = None
    times = []
    with _MemorySampler() as memory:
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = function()
            times.append(time.perf_counter() - start)
    assert result is not None
    return result, times, memory.peak


def _row(
    *,
    n_frames: int,
    n_residues: int,
    method: str,
    result,
    times: list[float],
    peak_rss_mb: float,
) -> dict[str, object]:
    summary = timing_summary(times)
    median = summary["median_seconds"]
    timing = getattr(result, "timing", {})
    metadata = getattr(result, "metadata", {})
    return {
        "n_frames": n_frames,
        "n_residues": n_residues,
        "residue_frames": n_frames * n_residues,
        "method": method,
        "status": "success",
        **summary,
        "frames_per_second": n_frames / median,
        "residue_frames_per_second": n_frames * n_residues / median,
        "peak_rss_mb_sampled": peak_rss_mb,
        "peak_vram_mb": float(
            metadata.get("peak_GPU_memory_bytes", 0)
        )
        / (1024.0 * 1024.0),
        "chunk_size": metadata.get("chunk_size", 2000),
        "transfer_seconds": timing.get("CPU_GPU_transfer_total", 0.0),
        "compute_seconds": timing.get(
            "GPU_compute_total",
            median,
        ),
        "end_to_end_seconds": median,
        "warmups": WARMUPS,
        "measured_repeats": REPEATS,
    }


def _mdlovofit_cell(
    coords: np.ndarray,
    n_frames: int,
    n_residues: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mdca_u3_") as temporary:
        workdir = Path(temporary)
        input_path = workdir / "input.pdb"
        output_path = workdir / "aligned.pdb"
        write_multimodel_pdb(coords, input_path)

        def run():
            result = run_mdlovofit_on_files(
                [input_path],
                fraction=0.6,
                output_path=output_path,
                executable="/usr/local/bin/mdlovofit",
                timeout_seconds=1800,
            )
            aligned, _ = read_multimodel_pdb_with_occupancy(
                output_path,
                n_residues,
            )
            delta = aligned - coords[0][None, :, :]
            rmsd = np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))
            return result, float(np.mean(rmsd))

        run()
        times: list[float] = []
        native = None
        unified = None
        with _MemorySampler() as memory:
            for _ in range(REPEATS):
                start = time.perf_counter()
                native, unified = run()
                times.append(time.perf_counter() - start)
        row = _row(
            n_frames=n_frames,
            n_residues=n_residues,
            method="MDLovoFit_20.1.1",
            result=native,
            times=times,
            peak_rss_mb=memory.peak,
        )
        row["unified_posthoc_rmsd_A"] = unified
        row["native_all_atom_rmsd"] = (
            native.parsed_output.average_rmsd_all_atoms
        )
        row["native_core_rmsd"] = (
            native.parsed_output.average_rmsd_least_mobile
        )
        row["timing_boundary"] = (
            "Cached PDB input excluded; subprocess, native aligned-PDB output "
            "and output parse included."
        )
        return row


def main() -> None:
    scaling_path = OUTPUT / "frames_by_residues_scaling.tsv"
    quality_path = OUTPUT / "scaling_quality.tsv"
    rows: list[dict[str, object]] = (
        read_tsv(scaling_path) if scaling_path.exists() else []
    )
    quality_rows: list[dict[str, object]] = (
        read_tsv(quality_path) if quality_path.exists() else []
    )
    completed_cells = {
        (int(row["n_frames"]), int(row["n_residues"]))
        for row in rows
        if row.get("method") == "GPU_throughput_float32_Horn"
        and row.get("status") == "success"
    }
    guarded_cells = {
        (int(row["n_frames"]), int(row["n_residues"]))
        for row in rows
        if row.get("status") == "not_run_resource_guard"
    }
    for n_frames in FRAMES:
        for n_residues in RESIDUES:
            if (n_frames, n_residues) in completed_cells:
                continue
            residue_frames = n_frames * n_residues
            if residue_frames > MAX_RESIDUE_FRAMES:
                if (n_frames, n_residues) not in guarded_cells:
                    rows.append(
                        {
                            "n_frames": n_frames,
                            "n_residues": n_residues,
                            "residue_frames": residue_frames,
                            "method": "CPU_and_GPU_throughput",
                            "status": "not_run_resource_guard",
                            "reason": (
                                f"residue_frames>{MAX_RESIDUE_FRAMES}; no "
                                "performance value was inferred."
                            ),
                        }
                    )
                    write_tsv(scaling_path, rows)
                continue
            print(f"[U3] {n_frames} x {n_residues}", flush=True)
            coords = _trajectory(n_frames, n_residues)
            chains = np.asarray(["A"] * n_residues)
            cpu, cpu_times, cpu_rss = _repeat(
                lambda: solve_fast_global_alignment(
                    coords,
                    chain_ids=chains,
                    representative_count=min(256, n_frames),
                    update_rounds=4,
                    chunk_size=2000,
                    store_aligned=False,
                    compute_confidence=False,
                )
            )
            cpu_row = _row(
                n_frames=n_frames,
                n_residues=n_residues,
                method="CPU_throughput",
                result=cpu,
                times=cpu_times,
                peak_rss_mb=cpu_rss,
            )
            rows.append(cpu_row)

            worker = PersistentGPUWorker.create(
                chain_ids=chains,
                representative_count=min(256, n_frames),
                update_rounds=4,
                chunk_size=2000,
                precision="float32",
                rotation_solver="horn",
            )
            gpu, gpu_times, gpu_rss = _repeat(
                lambda: worker.throughput(coords)
            )
            gpu_row = _row(
                n_frames=n_frames,
                n_residues=n_residues,
                method="GPU_throughput_float32_Horn",
                result=gpu,
                times=gpu_times,
                peak_rss_mb=gpu_rss,
            )
            gpu_row["gpu_over_cpu_speedup"] = (
                float(cpu_row["median_seconds"])
                / float(gpu_row["median_seconds"])
            )
            rows.append(gpu_row)
            quality_rows.append(
                {
                    "n_frames": n_frames,
                    "n_residues": n_residues,
                    "core_jaccard_gpu_vs_cpu": jaccard(
                        core_mask(gpu.frozen_weights),
                        core_mask(cpu.frozen_weights),
                    ),
                    "weight_spearman_gpu_vs_cpu": spearman(
                        gpu.frozen_weights,
                        cpu.frozen_weights,
                    ),
                    "rmsf_spearman_gpu_vs_cpu": spearman(
                        gpu.robust_rmsf,
                        cpu.robust_rmsf,
                    ),
                    "rmsf_mae_A_gpu_vs_cpu": float(
                        np.mean(np.abs(gpu.robust_rmsf - cpu.robust_rmsf))
                    ),
                    "median_abs_rmsd_delta_A_gpu_vs_cpu": float(
                        np.median(np.abs(gpu.rmsd - cpu.rmsd))
                    ),
                    "finite_outputs": bool(
                        np.isfinite(gpu.rmsd).all()
                        and np.isfinite(gpu.frozen_weights).all()
                    ),
                }
            )
            if (n_frames, n_residues) in MDLOVOFIT_CELLS:
                mdlovo_row = _mdlovofit_cell(
                    coords,
                    n_frames,
                    n_residues,
                )
                mdlovo_row["gpu_over_mdlovofit_speedup"] = (
                    float(mdlovo_row["median_seconds"])
                    / float(gpu_row["median_seconds"])
                )
                rows.append(mdlovo_row)
            write_tsv(scaling_path, rows)
            write_tsv(quality_path, quality_rows)
            del coords, cpu, gpu, worker
            gc.collect()
    write_tsv(scaling_path, rows)
    write_tsv(quality_path, quality_rows)

    completed_gpu = [
        row
        for row in rows
        if row.get("method") == "GPU_throughput_float32_Horn"
        and row.get("status") == "success"
    ]
    crossover = sorted(
        completed_gpu,
        key=lambda row: int(row["residue_frames"]),
    )
    first_faster = next(
        (
            row
            for row in crossover
            if float(row["gpu_over_cpu_speedup"]) > 1.0
        ),
        None,
    )
    summary_rows = [
        {
            "item": "completed_matrix_cells",
            "value": len(completed_gpu),
            "notes": (
                f"{len(FRAMES) * len(RESIDUES)} requested grid cells; "
                "resource-guarded cells are explicit."
            ),
        },
        {
            "item": "cpu_gpu_crossover_residue_frames",
            "value": (
                first_faster["residue_frames"]
                if first_faster is not None
                else "not_observed"
            ),
            "notes": (
                "Smallest measured cell where GPU end-to-end median was "
                "lower than CPU throughput median."
            ),
        },
        {
            "item": "larger_streaming_case",
            "value": "250000_frames_x_150_residues",
            "notes": (
                "Largest frame-count case is bounded at 37.5 million "
                "residue-frames and is synthetic scaling evidence only."
            ),
        },
    ]
    write_tsv(OUTPUT / "cpu_gpu_crossover_summary.tsv", summary_rows)


if __name__ == "__main__":
    main()
