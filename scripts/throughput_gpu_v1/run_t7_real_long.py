#!/usr/bin/env python3
"""Validate matched CPU/GPU contracts on real continuous long trajectories."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
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
OUTPUT = ROOT / "results" / "throughput_gpu_v1" / "t7_real_long"


def _read_manifest() -> list[dict[str, str]]:
    with (OUTPUT / "real_long_input_manifest.tsv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _table(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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


def _alignment_quality(result, exact) -> dict[str, object]:
    rmsd_delta = np.abs(result.rmsd - exact.rmsd)
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
        "RMSF_MAE_A_vs_exact": float(
            np.mean(np.abs(result.robust_rmsf - exact.robust_rmsf))
        ),
        "median_abs_RMSD_delta_A_vs_exact": float(np.median(rmsd_delta)),
        "P95_abs_RMSD_delta_A_vs_exact": float(
            np.quantile(rmsd_delta, 0.95)
        ),
        "mean_unified_RMSD_A": result.mean_rmsd,
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


def _method_row(
    system: str,
    n_frames: int,
    n_residues: int,
    contract: str,
    method: str,
    runtimes: list[float],
    result,
) -> dict[str, object]:
    timing = getattr(result, "timing", {})
    metadata = getattr(result, "metadata", {})
    runtime = float(np.median(runtimes))
    return {
        "system": system,
        "n_frames": n_frames,
        "n_residues": n_residues,
        "contract": contract,
        "method": method,
        "runtime_seconds": runtime,
        "repeat_runtimes_seconds": ";".join(
            f"{value:.9g}" for value in runtimes
        ),
        "frames_per_second": n_frames / runtime,
        "residue_frames_per_second": n_frames * n_residues / runtime,
        "H2D_seconds": timing.get("all_frame_H2D", 0.0)
        + timing.get("representative_H2D", 0.0),
        "GPU_compute_seconds": timing.get("GPU_compute_total", 0.0),
        "D2H_seconds": timing.get("all_frame_D2H", 0.0)
        + timing.get("weight_D2H", 0.0),
        "peak_VRAM_bytes": metadata.get("peak_GPU_memory_bytes", 0),
        "aligned_coordinates_materialised": bool(
            getattr(result, "aligned_coords", None) is not None
        ),
        "MDLovoFit_status": (
            "not_run_no_in_memory_matched_contract"
        ),
        "MDLovoFit_interpretation": (
            "No MDLovoFit timing is reported because its subprocess/PDB "
            "boundary is not matched to this warm in-memory contract."
        ),
    }


def _repeat(function, count: int) -> tuple[object, list[float]]:
    result = None
    runtimes = []
    for _ in range(count):
        start = time.perf_counter()
        result = function()
        runtimes.append(time.perf_counter() - start)
    return result, runtimes


def main() -> None:
    method_rows = []
    quality_rows = []
    agreement_rows = []
    for item in _read_manifest():
        if item["status"] != "ready":
            continue
        cache = ROOT / item["cache_path"]
        with np.load(cache, allow_pickle=False) as archive:
            coords = np.asarray(archive["coords"], dtype=np.float64)
            chains = np.asarray(archive["chain_ids"])
        system = item["system"]
        n_frames, n_residues, _ = coords.shape

        exact, exact_times = _repeat(
            lambda: solve_exact_contract(
                coords,
                chain_ids=chains,
                representative_count=min(2000, n_frames),
                chunk_size=1000,
                store_aligned=False,
            ),
            1,
        )
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "metrics_transforms",
                "CPU_exact_sequential_lean",
                exact_times,
                exact,
            )
        )

        cpu, cpu_times = _repeat(
            lambda: solve_fast_global_alignment(
                coords,
                chain_ids=chains,
                representative_count=min(256, n_frames),
                update_rounds=4,
                chunk_size=1000,
                store_aligned=False,
                compute_confidence=False,
            ),
            2,
        )
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "metrics_transforms",
                "CPU_throughput",
                cpu_times,
                cpu,
            )
        )

        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            representative_count=min(256, n_frames),
            update_rounds=4,
            chunk_size=1000,
            precision="float32",
            rotation_solver="horn",
        )
        worker.throughput(
            coords[: min(1000, n_frames)],
            representative_count=min(256, n_frames),
            chunk_size=1000,
        )
        gpu, gpu_times = _repeat(lambda: worker.throughput(coords), 3)
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "metrics_transforms",
                "GPU_throughput_float32_Horn",
                gpu_times,
                gpu,
            )
        )

        worker.throughput(
            coords[: min(1000, n_frames)],
            representative_count=min(256, n_frames),
            chunk_size=1000,
            precision="mixed",
        )
        gpu_mixed, gpu_mixed_times = _repeat(
            lambda: worker.throughput(coords, precision="mixed"),
            2,
        )
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "metrics_transforms",
                "GPU_throughput_mixed_Horn",
                gpu_mixed_times,
                gpu_mixed,
            )
        )

        cpu_materialised, cpu_materialised_times = _repeat(
            lambda: solve_fast_global_alignment(
                coords,
                chain_ids=chains,
                representative_count=min(256, n_frames),
                update_rounds=4,
                chunk_size=1000,
                store_aligned=True,
                compute_confidence=False,
            ),
            2,
        )
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "aligned_trajectory_materialisation",
                "CPU_throughput",
                cpu_materialised_times,
                cpu_materialised,
            )
        )

        gpu_materialised, gpu_materialised_times = _repeat(
            lambda: worker.throughput(coords, store_aligned=True),
            2,
        )
        method_rows.append(
            _method_row(
                system,
                n_frames,
                n_residues,
                "aligned_trajectory_materialisation",
                "GPU_throughput_float32_Horn",
                gpu_materialised_times,
                gpu_materialised,
            )
        )

        cpu_w5, cpu_w5_times = _repeat(
            lambda: scan_local_kinematics(
                coords,
                chain_ids=chains,
                evaluate_blocks=False,
            ),
            2,
        )
        cpu_w5_seconds = float(np.median(cpu_w5_times))
        method_rows.append(
            {
                "system": system,
                "n_frames": n_frames,
                "n_residues": n_residues,
                "contract": "local_deformation_scan",
                "method": "CPU_W5",
                "runtime_seconds": cpu_w5_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in cpu_w5_times
                ),
                "frames_per_second": n_frames / cpu_w5_seconds,
                "residue_frames_per_second": (
                    n_frames * n_residues / cpu_w5_seconds
                ),
                "H2D_seconds": 0.0,
                "GPU_compute_seconds": 0.0,
                "D2H_seconds": 0.0,
                "peak_VRAM_bytes": 0,
                "aligned_coordinates_materialised": False,
                "MDLovoFit_status": "not_equivalent_contract",
                "MDLovoFit_interpretation": (
                    "W5 is alignment-free and is not compared as an "
                    "equivalent MDLovoFit workflow."
                ),
            }
        )
        scan_w5_gpu(
            coords[: min(1000, n_frames)],
            chain_ids=chains,
            precision="float32",
        )
        gpu_w5, gpu_w5_times = _repeat(
            lambda: scan_w5_gpu(
                coords,
                chain_ids=chains,
                precision="float32",
            ),
            2,
        )
        gpu_w5_seconds = float(np.median(gpu_w5_times))
        method_rows.append(
            {
                "system": system,
                "n_frames": n_frames,
                "n_residues": n_residues,
                "contract": "local_deformation_scan",
                "method": "GPU_W5",
                "runtime_seconds": gpu_w5_seconds,
                "repeat_runtimes_seconds": ";".join(
                    f"{value:.9g}" for value in gpu_w5_times
                ),
                "frames_per_second": n_frames / gpu_w5_seconds,
                "residue_frames_per_second": (
                    n_frames * n_residues / gpu_w5_seconds
                ),
                "H2D_seconds": gpu_w5.scan.timing["H2D"],
                "GPU_compute_seconds": gpu_w5.scan.timing["GPU_compute"],
                "D2H_seconds": (
                    gpu_w5_seconds
                    - gpu_w5.scan.timing["H2D"]
                    - gpu_w5.scan.timing["GPU_compute"]
                ),
                "peak_VRAM_bytes": gpu_w5.scan.metadata[
                    "peak_GPU_memory_bytes"
                ],
                "aligned_coordinates_materialised": False,
                "MDLovoFit_status": "not_equivalent_contract",
                "MDLovoFit_interpretation": (
                    "W5 is alignment-free and is not compared as an "
                    "equivalent MDLovoFit workflow."
                ),
            }
        )

        for method, result in (
            ("CPU_throughput", cpu),
            ("GPU_throughput_float32_Horn", gpu),
            ("GPU_throughput_mixed_Horn", gpu_mixed),
        ):
            quality_rows.append(
                {
                    "system": system,
                    "method": method,
                    **_alignment_quality(result, exact),
                }
            )
        agreement_rows.append(
            {
                "system": system,
                "core_jaccard_CPU_vs_GPU": _jaccard(
                    _core(cpu.frozen_weights),
                    _core(gpu.frozen_weights),
                ),
                "weight_spearman_CPU_vs_GPU": _spearman(
                    cpu.frozen_weights,
                    gpu.frozen_weights,
                ),
                "RMSF_spearman_CPU_vs_GPU": _spearman(
                    cpu.robust_rmsf,
                    gpu.robust_rmsf,
                ),
                "RMSF_MAE_A_CPU_vs_GPU": float(
                    np.mean(np.abs(cpu.robust_rmsf - gpu.robust_rmsf))
                ),
                "median_abs_RMSD_delta_A_CPU_vs_GPU": float(
                    np.median(np.abs(cpu.rmsd - gpu.rmsd))
                ),
                "transform_rotation_max_abs_delta_CPU_vs_GPU": float(
                    np.max(np.abs(cpu.rotations - gpu.rotations))
                ),
                "transform_translation_max_abs_delta_A_CPU_vs_GPU": float(
                    np.max(np.abs(cpu.translations - gpu.translations))
                ),
                "W5_local_deformation_spearman_CPU_vs_GPU": _spearman(
                    cpu_w5.local_deformation_index,
                    gpu_w5.local_deformation_index,
                ),
            }
        )
        _table(OUTPUT / "real_long_matched_contracts.tsv", method_rows)
        _table(OUTPUT / "real_long_quality_vs_exact.tsv", quality_rows)
        _table(
            OUTPUT / "cpu_gpu_numerical_agreement.tsv",
            agreement_rows,
        )
        del (
            coords,
            exact,
            cpu,
            gpu,
            gpu_mixed,
            cpu_materialised,
            gpu_materialised,
            cpu_w5,
            gpu_w5,
            worker,
        )
        gc.collect()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "systems": len(agreement_rows),
        "all_inputs_single_source_trajectory": True,
        "stitched_inputs_used": False,
        "minimum_frames": min(
            int(row["n_frames"])
            for row in method_rows
            if row["contract"] == "metrics_transforms"
        ),
        "maximum_frames": max(
            int(row["n_frames"])
            for row in method_rows
            if row["contract"] == "metrics_transforms"
        ),
        "all_CPU_GPU_core_jaccard_at_least_0_99": all(
            float(row["core_jaccard_CPU_vs_GPU"]) >= 0.99
            for row in agreement_rows
        ),
        "all_CPU_GPU_RMSF_spearman_at_least_0_999": all(
            float(row["RMSF_spearman_CPU_vs_GPU"]) >= 0.999
            for row in agreement_rows
        ),
        "public_default_changed": False,
        "MDLovoFit_warm_in_memory_timing_reported": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = f"""# T7 matched contracts on real continuous trajectories

Three local, single-file, time-ordered trajectories were evaluated after
CA extraction: ADK, NhaA and the full 90 ns YiiP trajectory. No replicate
stitching or synthetic extension was used.

- Systems: `{summary['systems']}`
- Frame range: `{summary['minimum_frames']}` to
  `{summary['maximum_frames']}`
- CPU/GPU core Jaccard >= 0.99 for every system:
  `{summary['all_CPU_GPU_core_jaccard_at_least_0_99']}`
- CPU/GPU RMSF Spearman >= 0.999 for every system:
  `{summary['all_CPU_GPU_RMSF_spearman_at_least_0_999']}`

Separate contracts report metrics/transforms only, W5 local-deformation scan,
and materialised aligned coordinates. Input decode/extraction is excluded from
warm loaded timing and documented in `input_preparation.json`.

MDLovoFit is not assigned a warm in-memory runtime because its external
subprocess/PDB boundary is not matched to this contract. Existing formal
MDLovoFit evidence remains frozen and is not reused as though it were a new
matched GPU timing. W5 is not compared to MDLovoFit as an equivalent alignment
workflow.

These files are genuine continuous trajectory sources in the technical sense
of one time-ordered trajectory file. This audit does not independently prove
PBC/nojump preprocessing provenance or uninterrupted physical sampling beyond
the available source metadata.
"""
    (OUTPUT / "t7_real_long_report.md").write_text(report)


if __name__ == "__main__":
    main()
