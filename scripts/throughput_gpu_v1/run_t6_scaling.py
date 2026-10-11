#!/usr/bin/env python3
"""Run the T6 CPU/GPU scaling matrix with matched in-memory inputs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.stats import spearmanr

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import (
    PersistentGPUWorker,
    scan_w5_gpu,
)
from mdsuperpose.window_kinematics import scan_local_kinematics


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "throughput_gpu_v1" / "t6_scaling"


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("output table cannot be empty")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _trajectory(n_frames: int, n_residues: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_residues, 3))
    steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-12)
    reference = np.cumsum(3.8 * steps, axis=0)
    coords = np.empty((n_frames, n_residues, 3), dtype=np.float64)
    mobile_start = n_residues // 3
    mobile_stop = min(n_residues, mobile_start + max(8, n_residues // 5))
    for start in range(0, n_frames, 2000):
        stop = min(start + 2000, n_frames)
        count = stop - start
        values = np.repeat(reference[None], count, axis=0)
        values += rng.normal(scale=0.025, size=values.shape)
        phase = np.arange(start, stop) * (2.0 * np.pi / 997.0)
        values[:, mobile_start:mobile_stop, 0] += (
            1.8 * np.sin(phase)[:, None]
        )
        values[:, mobile_start:mobile_stop, 2] += (
            0.7 * np.cos(phase * 0.61)[:, None]
        )
        coords[start:stop] = values
    return coords


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


class _MemoryMonitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak = _rss_bytes()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, _rss_bytes())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, _rss_bytes())


def _measure(function, repeats: int) -> tuple[object, list[float], int]:
    results = []
    runtimes = []
    peak = 0
    for _ in range(repeats):
        with _MemoryMonitor() as monitor:
            start = time.perf_counter()
            result = function()
            runtimes.append(time.perf_counter() - start)
        peak = max(peak, monitor.peak)
        results.append(result)
    index = int(np.argsort(runtimes)[len(runtimes) // 2])
    return results[index], runtimes, peak


def _core(weights: np.ndarray) -> np.ndarray:
    count = max(1, int(np.floor(weights.size * 0.6 + 0.5)))
    order = np.lexsort((np.arange(weights.size), -weights))
    mask = np.zeros(weights.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _cold_start_rows() -> list[dict[str, object]]:
    rows = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    script = ROOT / "scripts" / "throughput_gpu_v1" / "cold_start_probe.py"
    for method in (
        "CPU_exact",
        "CPU_throughput",
        "GPU_throughput",
        "CPU_W5",
        "GPU_W5",
    ):
        completed = subprocess.run(
            [sys.executable, str(script), method],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=300,
        )
        if completed.returncode == 0:
            rows.append(json.loads(completed.stdout))
        else:
            rows.append(
                {
                    "method": method,
                    "python": sys.executable,
                    "python_version": "",
                    "frames": 10_000,
                    "residues": 150,
                    "import_seconds": "",
                    "first_call_seconds": "",
                    "fresh_process_total_after_input_seconds": "",
                    "thread_environment": "",
                    "result_available": False,
                    "error": completed.stderr.strip(),
                }
            )
    return rows


def _base_row(case: dict[str, object], method: str) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "scaling_axis": case["axis"],
        "n_frames": case["frames"],
        "n_residues": case["residues"],
        "residue_frames": int(case["frames"]) * int(case["residues"]),
        "method": method,
        "status": "success",
        "skip_reason": "",
        "backend": "gpu" if method.startswith("GPU") else "cpu",
        "precision": (
            "float32" if method.startswith("GPU") else "float64"
        ),
        "rotation_solver": (
            "horn" if method == "GPU_throughput" else ""
        ),
    }


def _alignment_quality(result, exact) -> dict[str, object]:
    return {
        "core_jaccard_vs_exact": _jaccard(
            _core(result.frozen_weights),
            _core(exact.frozen_weights),
        ),
        "weight_spearman_vs_exact": _spearman(
            result.frozen_weights,
            exact.frozen_weights,
        ),
        "RMSF_spearman_vs_exact": _spearman(
            result.robust_rmsf,
            exact.robust_rmsf,
        ),
        "RMSF_MAE_vs_exact": float(
            np.mean(np.abs(result.robust_rmsf - exact.robust_rmsf))
        ),
        "median_abs_RMSD_delta_vs_exact": float(
            np.median(np.abs(result.rmsd - exact.rmsd))
        ),
        "P95_abs_RMSD_delta_vs_exact": float(
            np.quantile(np.abs(result.rmsd - exact.rmsd), 0.95)
        ),
        "quality_status": (
            "pass"
            if (
                _jaccard(
                    _core(result.frozen_weights),
                    _core(exact.frozen_weights),
                )
                >= 0.90
                and _spearman(result.robust_rmsf, exact.robust_rmsf)
                >= 0.98
            )
            else "warning"
        ),
    }


def _fit_models(
    rows: list[dict[str, object]],
    *,
    axis: str,
) -> list[dict[str, object]]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [
            row
            for row in rows
            if row["scaling_axis"] == axis
            and row["method"] == method
            and row["status"] == "success"
        ]
        if len(subset) < 3:
            continue
        x_name = "n_frames" if axis == "frames" else "n_residues"
        x = np.asarray([float(row[x_name]) for row in subset])
        y = np.asarray([float(row["warm_loaded_seconds"]) for row in subset])
        for degree in (1, 2):
            coefficients = np.polyfit(x, y, degree)
            prediction = np.polyval(coefficients, x)
            total = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = (
                1.0
                if total == 0.0
                else 1.0 - float(np.sum((y - prediction) ** 2)) / total
            )
            output.append(
                {
                    "scaling_axis": axis,
                    "method": method,
                    "model": "linear" if degree == 1 else "quadratic",
                    "n_points": len(subset),
                    "coefficient_quadratic": (
                        coefficients[0] if degree == 2 else 0.0
                    ),
                    "coefficient_linear": (
                        coefficients[-2]
                    ),
                    "intercept": coefficients[-1],
                    "R_squared": r_squared,
                }
            )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cold_rows = _cold_start_rows()
    _write_table(OUTPUT / "cold_start_summary.tsv", cold_rows)
    cold_lookup = {
        row["method"]: row.get(
            "fresh_process_total_after_input_seconds", ""
        )
        for row in cold_rows
    }
    cases = [
        {
            "case_id": f"frames_{frames}_residues_150",
            "axis": "frames",
            "frames": frames,
            "residues": 150,
        }
        for frames in (1_000, 5_000, 10_000, 50_000, 100_000, 250_000)
    ]
    cases.extend(
        {
            "case_id": f"frames_5000_residues_{residues}",
            "axis": "residues",
            "frames": 5_000,
            "residues": residues,
        }
        for residues in (300, 500, 1_000, 2_000, 3_000)
    )
    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        coords = _trajectory(
            int(case["frames"]),
            int(case["residues"]),
            1000 + case_index,
        )
        chains = np.asarray(["A"] * int(case["residues"]))
        exact, exact_times, exact_peak = _measure(
            lambda: solve_exact_contract(
                coords,
                chain_ids=chains,
                representative_count=min(2000, int(case["frames"])),
                chunk_size=2000,
            ),
            1,
        )
        exact_row = _base_row(case, "CPU_exact")
        exact_row.update(
            {
                "cold_start_seconds": cold_lookup["CPU_exact"],
                "warm_loaded_seconds": exact_times[0],
                "repeat_runtimes_seconds": f"{exact_times[0]:.9g}",
                "H2D_seconds": 0.0,
                "GPU_compute_seconds": 0.0,
                "D2H_seconds": 0.0,
                "end_to_end_seconds": exact_times[0],
                "frames_per_second": int(case["frames"]) / exact_times[0],
                "residue_frames_per_second": (
                    int(case["frames"])
                    * int(case["residues"])
                    / exact_times[0]
                ),
                "peak_RSS_bytes": exact_peak,
                "peak_VRAM_bytes": 0,
                "core_jaccard_vs_exact": 1.0,
                "weight_spearman_vs_exact": 1.0,
                "RMSF_spearman_vs_exact": 1.0,
                "RMSF_MAE_vs_exact": 0.0,
                "median_abs_RMSD_delta_vs_exact": 0.0,
                "P95_abs_RMSD_delta_vs_exact": 0.0,
                "quality_status": "exact_reference",
            }
        )
        rows.append(exact_row)

        cpu, cpu_times, cpu_peak = _measure(
            lambda: solve_fast_global_alignment(
                coords,
                chain_ids=chains,
                representative_count=min(256, int(case["frames"])),
                update_rounds=4,
                chunk_size=2000,
                store_aligned=False,
                compute_confidence=False,
            ),
            2,
        )
        cpu_seconds = float(np.median(cpu_times))
        cpu_row = _base_row(case, "CPU_throughput")
        cpu_row.update(
            {
                "cold_start_seconds": cold_lookup["CPU_throughput"],
                "warm_loaded_seconds": cpu_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in cpu_times
                ),
                "H2D_seconds": 0.0,
                "GPU_compute_seconds": 0.0,
                "D2H_seconds": 0.0,
                "end_to_end_seconds": cpu_seconds,
                "frames_per_second": int(case["frames"]) / cpu_seconds,
                "residue_frames_per_second": (
                    int(case["frames"])
                    * int(case["residues"])
                    / cpu_seconds
                ),
                "peak_RSS_bytes": cpu_peak,
                "peak_VRAM_bytes": 0,
                **_alignment_quality(cpu, exact),
            }
        )
        rows.append(cpu_row)

        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            representative_count=min(256, int(case["frames"])),
            update_rounds=4,
            chunk_size=2000,
            precision="float32",
            rotation_solver="horn",
        )
        worker.throughput(
            coords[: min(64, int(case["frames"]))],
            representative_count=min(32, int(case["frames"])),
            chunk_size=64,
        )
        gpu, gpu_times, gpu_peak = _measure(
            lambda: worker.throughput(coords),
            2,
        )
        gpu_seconds = float(np.median(gpu_times))
        gpu_row = _base_row(case, "GPU_throughput")
        gpu_row.update(
            {
                "cold_start_seconds": cold_lookup["GPU_throughput"],
                "warm_loaded_seconds": gpu_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in gpu_times
                ),
                "H2D_seconds": gpu.timing["all_frame_H2D"]
                + gpu.timing["representative_H2D"],
                "GPU_compute_seconds": gpu.timing["GPU_compute_total"],
                "D2H_seconds": gpu.timing["all_frame_D2H"]
                + gpu.timing["weight_D2H"],
                "end_to_end_seconds": gpu_seconds,
                "frames_per_second": int(case["frames"]) / gpu_seconds,
                "residue_frames_per_second": (
                    int(case["frames"])
                    * int(case["residues"])
                    / gpu_seconds
                ),
                "peak_RSS_bytes": gpu_peak,
                "peak_VRAM_bytes": gpu.metadata[
                    "peak_GPU_memory_bytes"
                ],
                **_alignment_quality(gpu, exact),
            }
        )
        rows.append(gpu_row)

        cpu_w5, cpu_w5_times, cpu_w5_peak = _measure(
            lambda: scan_local_kinematics(
                coords,
                chain_ids=chains,
                evaluate_blocks=False,
            ),
            2,
        )
        cpu_w5_seconds = float(np.median(cpu_w5_times))
        cpu_w5_row = _base_row(case, "CPU_W5")
        cpu_w5_row.update(
            {
                "cold_start_seconds": cold_lookup["CPU_W5"],
                "warm_loaded_seconds": cpu_w5_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in cpu_w5_times
                ),
                "H2D_seconds": 0.0,
                "GPU_compute_seconds": 0.0,
                "D2H_seconds": 0.0,
                "end_to_end_seconds": cpu_w5_seconds,
                "frames_per_second": int(case["frames"]) / cpu_w5_seconds,
                "residue_frames_per_second": (
                    int(case["frames"])
                    * int(case["residues"])
                    / cpu_w5_seconds
                ),
                "peak_RSS_bytes": cpu_w5_peak,
                "peak_VRAM_bytes": 0,
                "core_jaccard_vs_exact": _jaccard(
                    _core(cpu_w5.local_rigidity_score),
                    _core(exact.frozen_weights),
                ),
                "weight_spearman_vs_exact": _spearman(
                    cpu_w5.local_rigidity_score,
                    exact.frozen_weights,
                ),
                "RMSF_spearman_vs_exact": "",
                "RMSF_MAE_vs_exact": "",
                "median_abs_RMSD_delta_vs_exact": "",
                "P95_abs_RMSD_delta_vs_exact": "",
                "quality_status": "local_diagnostic_not_alignment",
            }
        )
        rows.append(cpu_w5_row)

        gpu_w5, gpu_w5_times, gpu_w5_peak = _measure(
            lambda: scan_w5_gpu(
                coords,
                chain_ids=chains,
                precision="float32",
            ),
            2,
        )
        gpu_w5_seconds = float(np.median(gpu_w5_times))
        gpu_w5_row = _base_row(case, "GPU_W5")
        gpu_w5_row.update(
            {
                "cold_start_seconds": cold_lookup["GPU_W5"],
                "warm_loaded_seconds": gpu_w5_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in gpu_w5_times
                ),
                "H2D_seconds": gpu_w5.scan.timing["H2D"],
                "GPU_compute_seconds": gpu_w5.scan.timing["GPU_compute"],
                "D2H_seconds": (
                    gpu_w5.timing["total_seconds"]
                    - gpu_w5.scan.timing["GPU_compute"]
                    - gpu_w5.scan.timing["H2D"]
                ),
                "end_to_end_seconds": gpu_w5_seconds,
                "frames_per_second": int(case["frames"]) / gpu_w5_seconds,
                "residue_frames_per_second": (
                    int(case["frames"])
                    * int(case["residues"])
                    / gpu_w5_seconds
                ),
                "peak_RSS_bytes": gpu_w5_peak,
                "peak_VRAM_bytes": gpu_w5.scan.metadata[
                    "peak_GPU_memory_bytes"
                ],
                "core_jaccard_vs_exact": _jaccard(
                    _core(gpu_w5.local_rigidity_score),
                    _core(exact.frozen_weights),
                ),
                "weight_spearman_vs_exact": _spearman(
                    gpu_w5.local_rigidity_score,
                    exact.frozen_weights,
                ),
                "RMSF_spearman_vs_exact": "",
                "RMSF_MAE_vs_exact": "",
                "median_abs_RMSD_delta_vs_exact": "",
                "P95_abs_RMSD_delta_vs_exact": "",
                "quality_status": "local_diagnostic_not_alignment",
                "W5_spearman_vs_CPU_W5": _spearman(
                    gpu_w5.local_deformation_index,
                    cpu_w5.local_deformation_index,
                ),
            }
        )
        rows.append(gpu_w5_row)
        _write_table(OUTPUT / "cpu_gpu_scaling_matrix.tsv", rows)
        del coords, exact, cpu, gpu, cpu_w5, gpu_w5, worker
        gc.collect()

    model_rows = _fit_models(rows, axis="frames") + _fit_models(
        rows,
        axis="residues",
    )
    _write_table(OUTPUT / "scaling_model_summary.tsv", model_rows)
    crossover_rows = []
    frame_rows = [row for row in rows if row["scaling_axis"] == "frames"]
    for gpu_method, cpu_method in (
        ("GPU_throughput", "CPU_throughput"),
        ("GPU_W5", "CPU_W5"),
    ):
        speedups = []
        for n_frames in sorted(
            {int(row["n_frames"]) for row in frame_rows}
        ):
            gpu_row = next(
                row
                for row in frame_rows
                if row["method"] == gpu_method
                and int(row["n_frames"]) == n_frames
            )
            cpu_row = next(
                row
                for row in frame_rows
                if row["method"] == cpu_method
                and int(row["n_frames"]) == n_frames
            )
            speedups.append(
                (
                    n_frames,
                    float(cpu_row["warm_loaded_seconds"])
                    / float(gpu_row["warm_loaded_seconds"]),
                )
            )
        crossover = next(
            (frames for frames, speedup in speedups if speedup >= 1.5),
            None,
        )
        crossover_rows.append(
            {
                "GPU_method": gpu_method,
                "CPU_method": cpu_method,
                "first_frame_count_with_1.5x_end_to_end_speedup": (
                    "" if crossover is None else crossover
                ),
                "speedup_by_frame_count": ";".join(
                    f"{frames}:{speedup:.4f}"
                    for frames, speedup in speedups
                ),
                "small_input_GPU_win_required": False,
            }
        )
    _write_table(
        OUTPUT / "cpu_gpu_crossover_summary.tsv",
        crossover_rows,
    )
    gpu_rows = [
        row for row in rows if row["method"] == "GPU_throughput"
    ]
    target_compute = sum(
        float(row["GPU_compute_seconds"])
        and (
            next(
                float(candidate["warm_loaded_seconds"])
                for candidate in rows
                if candidate["case_id"] == row["case_id"]
                and candidate["method"] == "CPU_throughput"
            )
            / float(row["GPU_compute_seconds"])
            >= 2.0
        )
        for row in gpu_rows
    )
    target_e2e = sum(
        next(
            float(candidate["warm_loaded_seconds"])
            for candidate in rows
            if candidate["case_id"] == row["case_id"]
            and candidate["method"] == "CPU_throughput"
        )
        / float(row["warm_loaded_seconds"])
        >= 1.5
        for row in gpu_rows
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "case_count": len(cases),
        "row_count": len(rows),
        "GPU_compute_target_pass_cases": target_compute,
        "GPU_end_to_end_target_pass_cases": target_e2e,
        "GPU_case_count": len(gpu_rows),
        "inputs": "deterministic synthetic scaling inputs",
        "biological_continuity_claimed": False,
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    report = f"""# T6 CPU/GPU scaling matrix

This matrix uses deterministic synthetic inputs solely to characterize
computational scaling. It does not claim MD or biological continuity.

- Scaling cases: `{len(cases)}`
- Method rows: `{len(rows)}`
- Frame axis: 1k, 5k, 10k, 50k, 100k and 250k at 150 residues.
- Residue axis: 150, 300, 500, 1,000, 2,000 and 3,000 where feasible.
- GPU throughput: Horn/float32 after T5 implementation-parity validation.
- CPU exact: sequential_lean reference.
- W5: alignment-free local-deformation diagnostic, not an alignment workflow.
- GPU compute target (>=2x) passes: `{target_compute}/{len(gpu_rows)}`
- GPU end-to-end target (>=1.5x) passes: `{target_e2e}/{len(gpu_rows)}`

Cold-start results include imports and first CUDA call after the input has
already been generated. Warm results reuse imports and, for GPU throughput,
the CUDA context and CuPy memory pool. Peak RSS is sampled from `/proc`, while
peak VRAM is reported by the backend's CuPy memory pool.

GPU is not required to win on small inputs. See
`cpu_gpu_crossover_summary.tsv` for measured crossover points and
`scaling_model_summary.tsv` for linear/quadratic fits.
"""
    (OUTPUT / "t6_scaling_report.md").write_text(report)


if __name__ == "__main__":
    main()
