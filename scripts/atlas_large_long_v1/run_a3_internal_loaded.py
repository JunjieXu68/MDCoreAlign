#!/usr/bin/env python3
"""Run prospectively registered in-memory ATLAS prefix benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import spearmanr

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import ExperimentalBatchWorker, solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker, scan_w5_gpu, solve_fast_global_gpu
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


RAW_DIR = OUT / "a3_loaded_analysis/checkpoints"
CACHE_DIR = OUT / "a3_loaded_analysis/result_cache"


def _core(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _flexible(rmsf: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    values = np.asarray(rmsf, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _valid_transforms(result: object) -> bool:
    rotations = np.asarray(result.rotations, dtype=np.float64)
    translations = np.asarray(result.translations, dtype=np.float64)
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        return False
    identity = np.eye(3)[None, :, :]
    orthogonal = np.max(np.abs(rotations @ np.swapaxes(rotations, 1, 2) - identity))
    determinants = np.linalg.det(rotations)
    return bool(orthogonal < 5e-3 and np.max(np.abs(determinants - 1.0)) < 5e-3)


def _load(system: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    manifest = {row["system"]: row for row in read_tsv(OUT / "atlas_manifest.tsv")}
    row = manifest[system]
    coords = np.load(ROOT / row["coordinate_cache_path"], mmap_mode="r")
    with np.load(ROOT / row["residue_metadata_path"], allow_pickle=False) as archive:
        chains = np.asarray(archive["chain_ids"])
    return coords, chains, row


def _alignment_row(
    *,
    system: str,
    frames: int,
    residues: int,
    method: str,
    repeat_index: int,
    warmup: bool,
    wall: float,
    result: object,
    backend_selected: str,
) -> dict[str, object]:
    timing = getattr(result, "timing", {})
    metadata = getattr(result, "metadata", {})
    weights = np.asarray(result.frozen_weights)
    return {
        "system": system,
        "prefix_label": prefix_label(frames),
        "n_frames": frames,
        "n_residues": residues,
        "residue_frames": frames * residues,
        "contract": "A_loaded_analysis",
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": warmup,
        "status": "success",
        "backend_selected": backend_selected,
        "wall_seconds": wall,
        "reported_method_seconds": getattr(result, "total_runtime_seconds", wall),
        "H2D_seconds": timing.get("representative_H2D", 0.0) + timing.get("all_frame_H2D", 0.0),
        "GPU_compute_seconds": timing.get("GPU_compute_total", 0.0),
        "D2H_seconds": timing.get("weight_D2H", 0.0) + timing.get("all_frame_D2H", 0.0),
        "weight_estimation_seconds": timing.get("weight_estimation", timing.get("representative_gpu_fit", 0.0) + timing.get("representative_gpu_weight_update", 0.0)),
        "all_frame_fit_RMSD_RMSF_seconds": timing.get("all_frame_fit_metrics_RMSF", timing.get("all_frame_gpu_compute", 0.0)),
        "post_task_buffer_sanitization_seconds": timing.get("post_task_buffer_sanitization", 0.0),
        "frames_per_second": frames / wall,
        "residue_frames_per_second": frames * residues / wall,
        "peak_RSS_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_VRAM_bytes": metadata.get("peak_GPU_memory_bytes", 0),
        "mean_unified_RMSD_A": float(np.mean(result.rmsd)),
        "mean_weighted_RMSD_A": float(np.mean(result.weighted_rmsd)),
        "mean_robust_RMSF_A": float(np.mean(result.robust_rmsf)),
        "core_size": int(np.count_nonzero(_core(weights))),
        "core_fraction": float(np.mean(_core(weights))),
        "finite_outputs": bool(
            np.isfinite(weights).all()
            and np.isfinite(result.robust_rmsf).all()
            and np.isfinite(result.rmsd).all()
        ),
        "valid_transforms": _valid_transforms(result),
        "explicit_CUDA_synchronization": metadata.get("explicit_CUDA_synchronization", False),
    }


def _w5_row(
    *,
    system: str,
    frames: int,
    residues: int,
    method: str,
    repeat_index: int,
    warmup: bool,
    wall: float,
    result: object,
) -> dict[str, object]:
    if method == "CPU_W5":
        movement = np.asarray(result.movement_score)
        timing = result.timing
        metadata = result.metadata
        h2d = gpu = d2h = 0.0
    else:
        movement = np.asarray(result.local_deformation_index)
        timing = result.scan.timing
        metadata = result.scan.metadata
        h2d = timing.get("H2D", 0.0)
        gpu = timing.get("GPU_compute", 0.0)
        d2h = max(0.0, wall - h2d - gpu)
    return {
        "system": system,
        "prefix_label": prefix_label(frames),
        "n_frames": frames,
        "n_residues": residues,
        "residue_frames": frames * residues,
        "contract": "A_loaded_W5_local_deformation_not_alignment_equivalent",
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": warmup,
        "status": "success",
        "backend_selected": "cpu" if method == "CPU_W5" else "gpu",
        "wall_seconds": wall,
        "reported_method_seconds": timing.get("total_scan_only", timing.get("total_seconds", wall)),
        "H2D_seconds": h2d,
        "GPU_compute_seconds": gpu,
        "D2H_seconds": d2h,
        "frames_per_second": frames / wall,
        "residue_frames_per_second": frames * residues / wall,
        "peak_RSS_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_VRAM_bytes": metadata.get("peak_GPU_memory_bytes", 0),
        "finite_outputs": bool(np.isfinite(movement).all()),
        "valid_transforms": "not_applicable_alignment_free",
        "explicit_CUDA_synchronization": method == "GPU_W5",
        "notes": "W5 is an alignment-free local-deformation diagnostic and is not RMSD.",
    }


def _repeat_alignment(system: str, frames: int, method: str, function, backend: str):
    rows = []
    measured = None
    for index in range(WARMUPS + REPEATS):
        started = time.perf_counter()
        result = function()
        wall = time.perf_counter() - started
        rows.append(
            _alignment_row(
                system=system,
                frames=frames,
                residues=result.frozen_weights.size,
                method=method,
                repeat_index=index - WARMUPS,
                warmup=index < WARMUPS,
                wall=wall,
                result=result,
                backend_selected=backend,
            )
        )
        if index >= WARMUPS:
            measured = result
    return rows, measured


def _repeat_w5(system: str, frames: int, residues: int, method: str, function):
    rows = []
    measured = None
    for index in range(WARMUPS + REPEATS):
        started = time.perf_counter()
        result = function()
        wall = time.perf_counter() - started
        rows.append(
            _w5_row(
                system=system,
                frames=frames,
                residues=residues,
                method=method,
                repeat_index=index - WARMUPS,
                warmup=index < WARMUPS,
                wall=wall,
                result=result,
            )
        )
        if index >= WARMUPS:
            measured = result
    return rows, measured


def _quality(system: str, frames: int, exact, cpu, gpu, auto, tie: dict[str, object]):
    exact_core = _core(exact.frozen_weights)
    cpu_core = _core(cpu.frozen_weights)
    gpu_core = np.asarray(tie.get("final_gpu_core", _core(gpu.frozen_weights)), dtype=bool)
    rows = []
    for method, result, core in (
        ("CPU_throughput", cpu, cpu_core),
        ("GPU_throughput_float32_Horn", gpu, gpu_core),
        ("backend_auto_v2_threshold_25000", auto, _core(auto.frozen_weights)),
    ):
        rmsd_delta = np.abs(np.asarray(result.rmsd) - np.asarray(exact.rmsd))
        core_jaccard = _jaccard(core, exact_core)
        rmsf_spearman = _spearman(result.robust_rmsf, exact.robust_rmsf)
        registered_catastrophic = bool(
            not np.isfinite(result.frozen_weights).all()
            or not np.isfinite(result.rmsd).all()
            or not _valid_transforms(result)
            or np.count_nonzero(core) == 0
            or core_jaccard < 0.5
            or rmsf_spearman < 0.9
        )
        rows.append(
            {
                "system": system,
                "prefix_label": prefix_label(frames),
                "n_frames": frames,
                "n_residues": exact.frozen_weights.size,
                "residue_frames": frames * exact.frozen_weights.size,
                "method": method,
                "exact_relative_core_Jaccard": core_jaccard,
                "exact_relative_weight_Spearman": _spearman(result.frozen_weights, exact.frozen_weights),
                "exact_relative_RMSF_Spearman": rmsf_spearman,
                "exact_relative_RMSF_MAE_A": float(np.mean(np.abs(result.robust_rmsf - exact.robust_rmsf))),
                "exact_relative_flexible_label_agreement": float(np.mean(_flexible(result.robust_rmsf) == _flexible(exact.robust_rmsf))),
                "median_absolute_RMSD_difference_A": float(np.median(rmsd_delta)),
                "P95_absolute_RMSD_difference_A": float(np.quantile(rmsd_delta, 0.95)),
                "registered_catastrophic_quality_failure": registered_catastrophic,
                "large_geometric_tail_median_RMSD_gt_1A": float(np.median(rmsd_delta)) > 1.0,
                "core_gate_pass": core_jaccard >= 0.95,
                "RMSF_gate_pass": rmsf_spearman >= 0.995,
                "quality_contract_pass": core_jaccard >= 0.95 and rmsf_spearman >= 0.995 and not registered_catastrophic,
            }
        )
    implementation_jaccard = _jaccard(cpu_core, gpu_core)
    agreement = {
        "system": system,
        "prefix_label": prefix_label(frames),
        "n_frames": frames,
        "n_residues": exact.frozen_weights.size,
        "CPU_GPU_core_Jaccard": implementation_jaccard,
        "CPU_GPU_weight_Spearman": _spearman(cpu.frozen_weights, gpu.frozen_weights),
        "CPU_GPU_RMSF_Spearman": _spearman(cpu.robust_rmsf, gpu.robust_rmsf),
        "CPU_GPU_RMSF_MAE_A": float(np.mean(np.abs(cpu.robust_rmsf - gpu.robust_rmsf))),
        "tie_adjudication_performed": tie.get("performed", False),
        "tie_adjudication_seconds": tie.get("seconds", 0.0),
        "CPU_GPU_implementation_gate_pass": implementation_jaccard == 1.0,
    }
    return rows, agreement


def run(system: str) -> None:
    coords, chains, _manifest = _load(system)
    cpu_w5_worker = ExperimentalBatchWorker.create(coords.shape[1], chain_ids=chains)
    gpu_worker = PersistentGPUWorker.create(
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )
    gpu_worker.throughput(np.asarray(coords[:250]))
    for frames in PREFIXES:
        checkpoint = RAW_DIR / f"{system}__{prefix_label(frames)}.tsv"
        quality_path = RAW_DIR / f"{system}__{prefix_label(frames)}__quality.tsv"
        agreement_path = RAW_DIR / f"{system}__{prefix_label(frames)}__agreement.tsv"
        cache_path = CACHE_DIR / f"{system}__{prefix_label(frames)}.npz"
        if checkpoint.exists() and quality_path.exists() and agreement_path.exists() and cache_path.exists():
            print(f"[A3 loaded] skip complete {system} {frames}", flush=True)
            continue
        values = np.asarray(coords[:frames])
        exact_rows, exact = _repeat_alignment(
            system,
            frames,
            "CPU_exact_sequential_lean",
            lambda: solve_exact_contract(
                values,
                chain_ids=chains,
                representative_count=min(2000, frames),
                chunk_size=1000,
                store_aligned=False,
            ),
            "cpu",
        )
        cpu_rows, cpu = _repeat_alignment(
            system,
            frames,
            "CPU_throughput",
            lambda: solve_fast_global_alignment(
                values,
                chain_ids=chains,
                representative_count=min(256, frames),
                update_rounds=4,
                chunk_size=1000,
                store_aligned=False,
                compute_confidence=False,
            ),
            "cpu",
        )
        gpu_rows, gpu = _repeat_alignment(
            system,
            frames,
            "GPU_throughput_float32_Horn",
            lambda: gpu_worker.throughput(values),
            "gpu",
        )
        selected_backend = "gpu" if frames * values.shape[1] >= 25000 else "cpu"
        auto_function = (
            (lambda: gpu_worker.throughput(values))
            if selected_backend == "gpu"
            else (
                lambda: solve_fast_global_alignment(
                    values,
                    chain_ids=chains,
                    representative_count=min(256, frames),
                    update_rounds=4,
                    chunk_size=1000,
                    store_aligned=False,
                    compute_confidence=False,
                )
            )
        )
        auto_rows, auto = _repeat_alignment(
            system,
            frames,
            "backend_auto_v2_threshold_25000",
            auto_function,
            selected_backend,
        )
        cpu_w5_rows, cpu_w5 = _repeat_w5(
            system,
            frames,
            values.shape[1],
            "CPU_W5",
            lambda: cpu_w5_worker.scan_w5(values),
        )
        gpu_w5_rows, gpu_w5 = _repeat_w5(
            system,
            frames,
            values.shape[1],
            "GPU_W5",
            lambda: scan_w5_gpu(values, chain_ids=chains, precision="float32"),
        )
        tie: dict[str, object] = {"performed": False}
        raw_jaccard = _jaccard(_core(cpu.frozen_weights), _core(gpu.frozen_weights))
        if raw_jaccard != 1.0:
            started = time.perf_counter()
            float64 = solve_fast_global_gpu(
                values,
                chain_ids=chains,
                representative_count=min(256, frames),
                update_rounds=4,
                chunk_size=1000,
                precision="float64",
                rotation_solver="horn",
                store_aligned=False,
            )
            tie = {
                "performed": True,
                "seconds": time.perf_counter() - started,
                "final_gpu_core": _core(float64.frozen_weights),
            }
        quality_rows, agreement = _quality(system, frames, exact, cpu, gpu, auto, tie)
        write_tsv(checkpoint, [*exact_rows, *cpu_rows, *gpu_rows, *auto_rows, *cpu_w5_rows, *gpu_w5_rows])
        write_tsv(quality_path, quality_rows)
        write_tsv(agreement_path, [agreement])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            exact_weights=exact.frozen_weights,
            exact_rmsf=exact.robust_rmsf,
            exact_rmsd=exact.rmsd,
            exact_core=_core(exact.frozen_weights),
            cpu_weights=cpu.frozen_weights,
            cpu_rmsf=cpu.robust_rmsf,
            cpu_rmsd=cpu.rmsd,
            cpu_core=_core(cpu.frozen_weights),
            gpu_weights=gpu.frozen_weights,
            gpu_rmsf=gpu.robust_rmsf,
            gpu_rmsd=gpu.rmsd,
            gpu_core=np.asarray(tie.get("final_gpu_core", _core(gpu.frozen_weights))),
            auto_weights=auto.frozen_weights,
            auto_rmsf=auto.robust_rmsf,
            auto_rmsd=auto.rmsd,
            cpu_w5=cpu_w5.movement_score,
            gpu_w5=gpu_w5.local_deformation_index,
        )
        print(f"[A3 loaded] complete {system} {frames}", flush=True)
        del values, exact, cpu, gpu, auto, cpu_w5, gpu_w5
        gc.collect()
    write_json(
        OUT / f"a3_loaded_analysis/{system}_status.json",
        {"system": system, "prefixes_complete": list(PREFIXES), "status": "complete"},
    )


def finalize() -> None:
    manifests = read_tsv(OUT / "atlas_manifest.tsv")
    raw = []
    quality = []
    agreement = []
    for row in manifests:
        system = row["system"]
        for frames in PREFIXES:
            raw.extend(read_tsv(RAW_DIR / f"{system}__{prefix_label(frames)}.tsv"))
            quality.extend(read_tsv(RAW_DIR / f"{system}__{prefix_label(frames)}__quality.tsv"))
            agreement.extend(read_tsv(RAW_DIR / f"{system}__{prefix_label(frames)}__agreement.tsv"))
    write_tsv(OUT / "a3_loaded_analysis/loaded_analysis_raw_repeats.tsv", raw)
    write_tsv(OUT / "a3_loaded_analysis/loaded_quality_per_case.tsv", quality)
    write_tsv(OUT / "a3_loaded_analysis/CPU_GPU_implementation_agreement.tsv", agreement)
    write_json(
        OUT / "a3_loaded_analysis/loaded_analysis_status.json",
        {
            "systems": len(manifests),
            "prefixes_per_system": len(PREFIXES),
            "raw_rows": len(raw),
            "quality_rows": len(quality),
            "agreement_rows": len(agreement),
            "status": "complete",
        },
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
    selected = systems if args.system is None else [args.system]
    for system in selected:
        run(system)


if __name__ == "__main__":
    main()
