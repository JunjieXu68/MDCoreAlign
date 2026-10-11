#!/usr/bin/env python3
"""Audit the isolated CUDA environment and validate GPU numerical designs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import (
    PersistentGPUWorker,
    gpu_environment,
    scan_w5_gpu,
    solve_fast_global_gpu,
)
from mdsuperpose.window_kinematics import scan_local_kinematics


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "results" / "throughput_gpu_v1"
T4_ROOT = OUTPUT_ROOT / "t4_gpu_environment"
T5_ROOT = OUTPUT_ROOT / "t5_gpu_design"


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("at least one output row is required")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _command(args: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": args,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": args,
            "return_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _synthetic_trajectory(
    n_frames: int,
    n_residues: int,
    *,
    seed: int = 20260731,
) -> np.ndarray:
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
        values = np.repeat(reference[None, :, :], count, axis=0)
        values += rng.normal(scale=0.025, size=values.shape)
        phase = np.linspace(start, stop - 1, count) * (2.0 * np.pi / 997.0)
        values[:, mobile_start:mobile_stop, 0] += (
            1.8 * np.sin(phase)[:, None]
        )
        values[:, mobile_start:mobile_stop, 2] += (
            0.7 * np.cos(phase * 0.61)[:, None]
        )
        angle = 0.08 * np.sin(phase * 0.17)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        x = values[:, :, 0].copy()
        y = values[:, :, 1].copy()
        values[:, :, 0] = cosine[:, None] * x - sine[:, None] * y
        values[:, :, 1] = sine[:, None] * x + cosine[:, None] * y
        values += np.stack(
            (
                0.4 * np.sin(phase * 0.11),
                0.3 * np.cos(phase * 0.13),
                0.2 * np.sin(phase * 0.07),
            ),
            axis=1,
        )[:, None, :]
        coords[start:stop] = values
    return coords


def _core(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    count = max(1, int(np.floor(weights.size * fraction + 0.5)))
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


def run_t4() -> None:
    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT.relative_to(ROOT)),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "canonical_cpu_environment_modified": False,
        "isolated_environment": ".venv-gpu-cu12",
        "gpu": gpu_environment(),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,"
                "compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": _command(["nvcc", "--version"]),
        "pip_freeze": _command(
            [sys.executable, "-m", "pip", "freeze"]
        ),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "public_default_changed": False,
        "sequential_lean_gpu_ported": False,
        "hard_motion_domains_gpu_ported": False,
    }
    _json(T4_ROOT / "gpu_environment.json", environment)
    report = f"""# T4 isolated GPU backend

- Python: `{environment['python_executable']}`
- Isolated environment: `.venv-gpu-cu12`
- Device: `{environment['gpu']['device_name']}`
- Compute capability: `{environment['gpu']['compute_capability']}`
- CUDA runtime seen by CuPy: `{environment['gpu']['cuda_runtime_version']}`
- CuPy: `{environment['gpu']['cupy_version']}`
- Canonical CPU environment modified: **no**
- Public exact/default path changed: **no**
- GPU scope: experimental fast-global, one bounded refinement, W5,
  all-frame transforms/RMSD/RMSF and optional transform materialisation.
- Not ported: sequential_lean and hard motion-domain segmentation.

The CUDA environment is intentionally isolated. Core CPU commands remain
usable when CuPy is absent. GPU timings must use explicit synchronization or
CUDA events.
"""
    (T4_ROOT / "t4_gpu_environment_report.md").write_text(report)


def _comparison_row(
    *,
    method: str,
    precision: str,
    rotation_solver: str,
    result,
    cpu,
    exact,
) -> dict[str, object]:
    return {
        "method": method,
        "precision": precision,
        "rotation_solver": rotation_solver,
        "n_frames": result.n_frames,
        "n_residues": result.n_atoms,
        "total_runtime_seconds": result.total_runtime_seconds,
        "GPU_compute_seconds": result.timing.get("GPU_compute_total", ""),
        "H2D_seconds": result.timing.get("all_frame_H2D", ""),
        "D2H_seconds": result.timing.get("all_frame_D2H", ""),
        "peak_VRAM_bytes": result.metadata.get(
            "peak_GPU_memory_bytes", ""
        ),
        "pinned_host_memory": result.metadata.get(
            "pinned_host_memory", False
        ),
        "double_buffering": result.metadata.get("double_buffering", False),
        "core_jaccard_vs_CPU_throughput": _jaccard(
            _core(result.frozen_weights),
            _core(cpu.frozen_weights),
        ),
        "weight_spearman_vs_CPU_throughput": _spearman(
            result.frozen_weights,
            cpu.frozen_weights,
        ),
        "RMSF_spearman_vs_CPU_throughput": _spearman(
            result.robust_rmsf,
            cpu.robust_rmsf,
        ),
        "RMSF_MAE_vs_CPU_throughput": float(
            np.mean(np.abs(result.robust_rmsf - cpu.robust_rmsf))
        ),
        "median_abs_RMSD_delta_vs_CPU_throughput": float(
            np.median(np.abs(result.rmsd - cpu.rmsd))
        ),
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
        "mean_unified_RMSD": result.mean_rmsd,
        "quality_status": (
            "implementation_parity_pass"
            if (
                _jaccard(
                    _core(result.frozen_weights),
                    _core(cpu.frozen_weights),
                )
                >= 0.999
                and _spearman(
                    result.robust_rmsf,
                    cpu.robust_rmsf,
                )
                >= 0.99999
                and float(
                    np.median(np.abs(result.rmsd - cpu.rmsd))
                )
                <= 1e-4
            )
            else "implementation_parity_warning"
        ),
    }


def run_t5() -> None:
    T5_ROOT.mkdir(parents=True, exist_ok=True)
    coords = _synthetic_trajectory(10_000, 153)
    chains = np.asarray(["A"] * 77 + ["B"] * 76)
    exact = solve_exact_contract(
        coords,
        chain_ids=chains,
        representative_count=2000,
        chunk_size=2000,
    )
    cpu_start = time.perf_counter()
    cpu = solve_fast_global_alignment(
        coords,
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=2000,
        store_aligned=False,
        compute_confidence=False,
    )
    cpu_wall = time.perf_counter() - cpu_start

    worker = PersistentGPUWorker.create(
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=2000,
    )
    worker.throughput(
        coords[:64],
        representative_count=32,
        chunk_size=32,
        precision="mixed",
        rotation_solver="horn",
    )
    rows: list[dict[str, object]] = []
    for rotation_solver in ("svd", "cusolver_svd", "horn"):
        for precision in ("float64", "float32", "mixed"):
            repeats = []
            results = []
            for _ in range(3):
                result = worker.throughput(
                    coords,
                    precision=precision,
                    rotation_solver=rotation_solver,
                )
                repeats.append(result.total_runtime_seconds)
                results.append(result)
            median_index = int(np.argsort(repeats)[len(repeats) // 2])
            result = results[median_index]
            row = _comparison_row(
                method="GPU_throughput",
                precision=precision,
                rotation_solver=rotation_solver,
                result=result,
                cpu=cpu,
                exact=exact,
            )
            row["repeat_runtimes_seconds"] = ";".join(
                f"{value:.9g}" for value in repeats
            )
            row["median_runtime_seconds"] = float(np.median(repeats))
            row["GPU_compute_speedup_vs_CPU"] = (
                cpu_wall / max(float(row["GPU_compute_seconds"]), 1e-12)
            )
            row["end_to_end_speedup_vs_CPU"] = (
                cpu_wall / max(float(row["median_runtime_seconds"]), 1e-12)
            )
            rows.append(row)
    _table(T5_ROOT / "gpu_solver_precision_comparison.tsv", rows)

    cpu_w5_results = []
    cpu_w5_times = []
    for _ in range(2):
        cpu_w5_start = time.perf_counter()
        cpu_w5_results.append(
            scan_local_kinematics(
                coords,
                chain_ids=chains,
                evaluate_blocks=False,
            )
        )
        cpu_w5_times.append(time.perf_counter() - cpu_w5_start)
    cpu_w5 = cpu_w5_results[
        int(np.argsort(cpu_w5_times)[len(cpu_w5_times) // 2])
    ]
    cpu_w5_seconds = float(np.median(cpu_w5_times))
    w5_rows = []
    for precision in ("float64", "float32", "mixed"):
        scan_w5_gpu(
            coords[:1000],
            chain_ids=chains,
            precision=precision,
        )
        gpu_w5_results = []
        gpu_w5_times = []
        for _ in range(3):
            gpu_w5_start = time.perf_counter()
            gpu_w5_results.append(
                scan_w5_gpu(
                    coords,
                    chain_ids=chains,
                    precision=precision,
                )
            )
            gpu_w5_times.append(time.perf_counter() - gpu_w5_start)
        gpu_w5 = gpu_w5_results[
            int(np.argsort(gpu_w5_times)[len(gpu_w5_times) // 2])
        ]
        w5_rows.append(
            {
                "precision": precision,
                "CPU_W5_seconds": cpu_w5_seconds,
                "CPU_W5_repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in cpu_w5_times
                ),
                "GPU_W5_seconds": float(np.median(gpu_w5_times)),
                "GPU_W5_repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in gpu_w5_times
                ),
                "GPU_compute_seconds": gpu_w5.scan.timing["GPU_compute"],
                "H2D_seconds": gpu_w5.scan.timing["H2D"],
                "local_deformation_spearman_vs_CPU": _spearman(
                    gpu_w5.local_deformation_index,
                    cpu_w5.local_deformation_index,
                ),
                "local_deformation_max_abs_delta_vs_CPU": float(
                    np.max(
                        np.abs(
                            gpu_w5.local_deformation_index
                            - cpu_w5.local_deformation_index
                        )
                    )
                ),
                "score_called_RMSD": gpu_w5.metadata["score_called_rmsd"],
            }
        )
    _table(T5_ROOT / "gpu_w5_precision_comparison.tsv", w5_rows)

    best = max(rows, key=lambda row: row["end_to_end_speedup_vs_CPU"])
    parity_pass = sum(
        row["quality_status"] == "implementation_parity_pass"
        for row in rows
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": "transition-like deterministic synthetic scaling input",
        "biological_continuity_claimed": False,
        "n_frames": 10_000,
        "n_residues": 153,
        "CPU_exact_seconds": exact.total_runtime_seconds,
        "CPU_throughput_seconds": cpu_wall,
        "implementation_parity_pass_count": parity_pass,
        "implementation_parity_total": len(rows),
        "best_GPU_setting": {
            key: best[key]
            for key in (
                "precision",
                "rotation_solver",
                "median_runtime_seconds",
                "GPU_compute_speedup_vs_CPU",
                "end_to_end_speedup_vs_CPU",
                "quality_status",
            )
        },
        "persistent_worker": worker.cache_summary,
    }
    _json(T5_ROOT / "t5_summary.json", summary)
    report = f"""# T5 GPU compute-design validation

The implementation was tested on a deterministic 10,000-frame,
153-residue synthetic scaling input. This is a computational validation input,
not a claim of biological continuity.

- CPU exact sequential_lean: `{exact.total_runtime_seconds:.6f} s`
- CPU throughput: `{cpu_wall:.6f} s`
- GPU parity passes: `{parity_pass}/{len(rows)}`
- Best measured GPU setting: `{best['rotation_solver']}` /
  `{best['precision']}`
- Best median end-to-end runtime: `{best['median_runtime_seconds']:.6f} s`
- End-to-end speedup over CPU throughput:
  `{best['end_to_end_speedup_vs_CPU']:.3f}x`
- GPU-compute speedup over CPU throughput:
  `{best['GPU_compute_speedup_vs_CPU']:.3f}x`

Pinned host staging and two-buffer H2D prefetch are enabled when allocation
succeeds. The persistent worker reuses the CUDA context and CuPy memory pool,
but it does not yet reuse pinned host buffers across trajectories.

`cusolver_svd` currently maps to CuPy's batched SVD/cuSOLVER path and is
reported separately for interface clarity; it is not a distinct custom kernel.
All failures or slower settings remain in the TSV rather than being filtered.
"""
    (T5_ROOT / "t5_gpu_design_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["t4", "t5", "all"])
    args = parser.parse_args()
    if args.stage in {"t4", "all"}:
        run_t4()
    if args.stage in {"t5", "all"}:
        run_t5()


if __name__ == "__main__":
    main()
