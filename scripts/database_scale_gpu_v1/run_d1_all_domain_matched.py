#!/usr/bin/env python3
"""Run the resumable corrected mdCATH D1 matched all-domain panel."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker, scan_w5_gpu
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_refine import exact_all_frame_fit
from scripts.database_scale_gpu_v1.common import (
    D0_ROOT,
    D1_ROOT,
    REGISTRY_PATH,
    ROOT,
    core_mask,
    corrected_profile_path,
    cpu_gpu_agreement,
    flexible_mask,
    jaccard,
    load_exact_profile,
    mapping_digest,
    normalized_weights,
    quality_against_exact,
    read_tsv,
    repository_path,
    safe_spearman,
    write_json,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)


CHECKPOINT_ROOT = D1_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse incremental and cold-probe options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N pending eligible rows for smoke testing.",
    )
    parser.add_argument(
        "--trajectory-id",
        action="append",
        default=[],
        help="Restrict execution to one or more trajectory IDs.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Replace checkpoints for selected rows.",
    )
    parser.add_argument(
        "--cold-probe",
        help="Run one fresh-process GPU-throughput probe and emit JSON.",
    )
    return parser.parse_args()


def current_rss_mb() -> float:
    """Read current resident memory from procfs."""

    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def peak_rss_mb() -> float:
    """Return the process high-water RSS on Linux."""

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def checkpoint_path(trajectory_id: str) -> Path:
    """Return one atomic per-domain checkpoint path."""

    return CHECKPOINT_ROOT / f"{trajectory_id}.json"


def normalize_mapping(
    residue_metadata: list[dict[str, str]],
) -> list[tuple[str, str, str]]:
    """Return an ordered mapping in the exact-profile schema."""

    return [
        (
            str(row.get("chain", "_")),
            str(row.get("residue_id", "")),
            str(row.get("resname", "UNK")),
        )
        for row in residue_metadata
    ]


def base_row(registry: dict[str, str]) -> dict[str, object]:
    """Return immutable registry fields copied into every checkpoint."""

    return {
        "dataset": registry["dataset"],
        "dataset_release": registry["dataset_release"],
        "system_id": registry["system_id"],
        "trajectory_id": registry["trajectory_id"],
        "cath_class": registry["cath_class"],
        "cath_architecture": registry["cath_architecture"],
        "cath_topology": registry["cath_topology"],
        "cath_homologous_superfamily": registry[
            "cath_homologous_superfamily"
        ],
        "temperature_K": registry["temperature_K"],
        "replicate": registry["replicate"],
        "source_file": registry["source_file"],
        "source_sha256": registry["source_sha256"],
        "coordinate_path": registry["coordinate_path"],
        "expected_n_frames": registry["n_frames"],
        "expected_n_residues": registry["n_residues"],
        "chain_count": registry["chain_count"],
        "fixed_core_fraction": registry["fixed_core_fraction"],
        "effective_core_fraction_exact": registry[
            "effective_core_fraction"
        ],
        "motion_proxy_noncore_core_RMSF_ratio": registry[
            "motion_proxy_noncore_core_RMSF_ratio"
        ],
        "output_contract": (
            "analysis_metrics_transforms_RMSD_RMSF_plus_W5_diagnostic"
        ),
        "exact_core_source": (
            "frozen_unit_corrected_sequential_lean_mean_weights"
        ),
        "exact_fit_boundary": (
            "matched_fixed_weight_all_frame_fit_recomputed"
        ),
    }


def unavailable_checkpoint(registry: dict[str, str]) -> dict[str, object]:
    """Represent the retained unavailable 5,398th domain."""

    return {
        **base_row(registry),
        "status": "explicitly_unavailable",
        "failure_class": "input_unavailable",
        "failure_reason": registry["unavailable_reason"],
        "algorithm_failure": False,
        "eligible_for_D1_quality": False,
        "backend_selected": "",
        "backend_reason": "",
        "task_end_to_end_seconds": 0.0,
    }


def w5_agreement(cpu_w5: object, gpu_w5: object) -> dict[str, object]:
    """Compare alignment-free CPU and GPU W5 outputs."""

    cpu_segments = np.asarray(cpu_w5.locally_deforming_segments) > 0
    gpu_segments = np.asarray(gpu_w5.locally_deforming_segments) > 0
    return {
        "w5_cpu_gpu_local_deformation_spearman": safe_spearman(
            np.asarray(cpu_w5.local_deformation_index),
            np.asarray(gpu_w5.local_deformation_index),
        ),
        "w5_cpu_gpu_local_rigidity_spearman": safe_spearman(
            np.asarray(cpu_w5.local_rigidity_score),
            np.asarray(gpu_w5.local_rigidity_score),
        ),
        "w5_cpu_gpu_segment_jaccard": jaccard(
            cpu_segments,
            gpu_segments,
        ),
        "w5_output_called_RMSD": False,
    }


def load_coordinates(registry: dict[str, str]) -> dict[str, object]:
    """Read one input for the bounded two-thread prefetch pipeline."""

    start = time.perf_counter()
    try:
        source = ROOT / registry["source_file"]
        coords, labels, residue_metadata, metadata = extract_ca_trajectory(
            source,
            coordinate_path=registry["coordinate_path"],
            min_ca_atoms=30,
            min_frames=20,
            direct_contiguous_read=True,
        )
        return {
            "coords": coords,
            "labels": labels,
            "residue_metadata": residue_metadata,
            "metadata": metadata,
            "extraction_seconds": time.perf_counter() - start,
            "error": None,
        }
    except Exception as exc:
        return {
            "coords": None,
            "labels": None,
            "residue_metadata": None,
            "metadata": None,
            "extraction_seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def process_row(
    registry: dict[str, str],
    worker: PersistentGPUWorker,
    preloaded: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one locally eligible domain under every D1 contract."""

    task_start = time.perf_counter()
    initial_rss = current_rss_mb()
    row = base_row(registry)
    try:
        loaded = load_coordinates(registry) if preloaded is None else preloaded
        if loaded["error"] is not None:
            raise RuntimeError(str(loaded["error"]))
        coords = np.asarray(loaded["coords"], dtype=np.float64)
        labels = loaded["labels"]
        residue_metadata = loaded["residue_metadata"]
        metadata = loaded["metadata"]
        extraction_seconds = float(loaded["extraction_seconds"])
        if metadata["unit_status"] != "verified_angstrom_metadata":
            raise RuntimeError(
                f"unit integrity failed: {metadata['unit_status']}"
            )
        if coords.shape != (
            int(registry["n_frames"]),
            int(registry["n_residues"]),
            3,
        ):
            raise RuntimeError(
                f"shape mismatch: {coords.shape} versus registry "
                f"({registry['n_frames']}, {registry['n_residues']}, 3)"
            )
        observed_mapping = normalize_mapping(residue_metadata)
        observed_mapping_hash = mapping_digest(observed_mapping)
        if observed_mapping_hash != registry["chain_residue_mapping_sha256"]:
            raise RuntimeError(
                "chain/residue mapping digest differs from locked registry"
            )
        chains = np.asarray([item[0] for item in observed_mapping])

        profile_start = time.perf_counter()
        profile_path = corrected_profile_path(registry["trajectory_id"])
        profile = load_exact_profile(profile_path)
        profile_seconds = time.perf_counter() - profile_start
        if profile["mapping"] != observed_mapping:
            raise RuntimeError("profile and HDF5 residue mappings differ")
        exact_weights = np.asarray(profile["weights"], dtype=np.float64)
        exact_core = np.asarray(profile["core"], dtype=bool)
        if not np.array_equal(exact_core, core_mask(exact_weights)):
            raise RuntimeError("frozen exact core is not the locked 60% core")
        exact_flexible = flexible_mask(exact_weights, chains)

        exact_fit_start = time.perf_counter()
        exact_fit = exact_all_frame_fit(
            coords,
            coords[0],
            exact_weights,
            chunk_size=1000,
        )
        exact_fit_seconds = time.perf_counter() - exact_fit_start

        cpu_start = time.perf_counter()
        cpu = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=min(256, coords.shape[0]),
            update_rounds=4,
            chunk_size=1000,
            store_aligned=False,
            compute_confidence=False,
        )
        cpu_wall_seconds = time.perf_counter() - cpu_start

        worker.chain_ids = chains
        gpu_start = time.perf_counter()
        gpu = worker.throughput(
            coords,
            representative_count=min(256, coords.shape[0]),
        )
        gpu_wall_seconds = time.perf_counter() - gpu_start

        cpu_w5_start = time.perf_counter()
        cpu_w5 = scan_local_kinematics(
            coords,
            chain_ids=chains,
            evaluate_blocks=False,
        )
        cpu_w5_seconds = time.perf_counter() - cpu_w5_start
        gpu_w5_start = time.perf_counter()
        gpu_w5 = scan_w5_gpu(
            coords,
            chain_ids=chains,
            precision="float32",
        )
        gpu_w5_seconds = time.perf_counter() - gpu_w5_start

        exact_rmsf = np.asarray(exact_fit.robust_rmsf, dtype=np.float64)
        exact_rmsd = np.asarray(exact_fit.rmsd, dtype=np.float64)
        cpu_quality = quality_against_exact(
            cpu,
            exact_weights=exact_weights,
            exact_rmsd=exact_rmsd,
            exact_rmsf=exact_rmsf,
            exact_core=exact_core,
            exact_flexible=exact_flexible,
            chains=chains,
        )
        gpu_quality = quality_against_exact(
            gpu,
            exact_weights=exact_weights,
            exact_rmsd=exact_rmsd,
            exact_rmsf=exact_rmsf,
            exact_core=exact_core,
            exact_flexible=exact_flexible,
            chains=chains,
        )
        implementation = cpu_gpu_agreement(cpu, gpu, chains=chains)
        local_agreement = w5_agreement(cpu_w5, gpu_w5)
        task_seconds = time.perf_counter() - task_start
        gpu_timing = dict(gpu.timing)
        gpu_metadata = dict(gpu.metadata)
        gpu_w5_metadata = dict(gpu_w5.scan.metadata)
        output = {
            **row,
            "status": "success",
            "failure_class": "",
            "failure_reason": "",
            "algorithm_failure": False,
            "eligible_for_D1_quality": True,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "observed_chain_count": len(set(chains.tolist())),
            "source_unit_status": metadata["unit_status"],
            "input_read_strategy": metadata["read_strategy"],
            "unit_integrity": "pass",
            "mapping_integrity": "pass",
            "observed_mapping_sha256": observed_mapping_hash,
            "input_decode_preprocessing_seconds": extraction_seconds,
            "exact_profile_load_seconds": profile_seconds,
            "exact_weight_estimation_reused": True,
            "exact_weight_estimation_rerun": False,
            "exact_fixed_weight_fit_seconds": exact_fit_seconds,
            "exact_mean_unified_posthoc_rmsd_A": float(
                np.mean(exact_rmsd)
            ),
            "exact_mean_weighted_rmsd_A": float(
                np.mean(exact_fit.weighted_rmsd)
            ),
            "exact_profile_vs_matched_fit_RMSF_MAE_A": float(
                np.mean(
                    np.abs(
                        np.asarray(profile["rmsf"], dtype=np.float64)
                        - exact_rmsf
                    )
                )
            ),
            "cpu_throughput_loaded_compute_seconds": cpu_wall_seconds,
            "cpu_throughput_reported_algorithm_seconds": (
                cpu.total_runtime_seconds
            ),
            "gpu_throughput_loaded_compute_seconds": gpu_wall_seconds,
            "gpu_throughput_reported_algorithm_seconds": (
                gpu.total_runtime_seconds
            ),
            "cpu_w5_loaded_compute_seconds": cpu_w5_seconds,
            "gpu_w5_loaded_compute_seconds": gpu_w5_seconds,
            "cpu_throughput_end_to_end_seconds": (
                extraction_seconds + profile_seconds + cpu_wall_seconds
            ),
            "gpu_throughput_end_to_end_seconds": (
                extraction_seconds + profile_seconds + gpu_wall_seconds
            ),
            "cpu_w5_end_to_end_seconds": (
                extraction_seconds + cpu_w5_seconds
            ),
            "gpu_w5_end_to_end_seconds": (
                extraction_seconds + gpu_w5_seconds
            ),
            "task_end_to_end_seconds": task_seconds,
            "backend_selected": (
                "CPU_exact_reused;CPU_throughput;GPU_persistent;"
                "CPU_W5;GPU_W5"
            ),
            "backend_reason": (
                "D1 matched all-domain implementation and quality audit; "
                "no automatic crossover policy applied"
            ),
            "gpu_persistent_worker": True,
            "gpu_context_reused": True,
            "gpu_buffer_reused_for_same_shape": gpu_metadata.get(
                "workspace_reused", False
            ),
            "gpu_H2D_seconds": gpu_timing.get(
                "representative_H2D", 0.0
            )
            + gpu_timing.get("all_frame_H2D", 0.0),
            "gpu_D2H_seconds": gpu_timing.get("weight_D2H", 0.0)
            + gpu_timing.get("all_frame_D2H", 0.0),
            "gpu_compute_seconds": gpu_timing.get("GPU_compute_total", 0.0),
            "gpu_post_task_sanitization_seconds": gpu_timing.get(
                "post_task_buffer_sanitization", 0.0
            ),
            "peak_RSS_mb": peak_rss_mb(),
            "RSS_at_task_start_mb": initial_rss,
            "RSS_at_task_end_mb": current_rss_mb(),
            "gpu_throughput_peak_VRAM_bytes": gpu_metadata.get(
                "peak_GPU_memory_bytes", 0
            ),
            "gpu_w5_peak_VRAM_bytes": gpu_w5_metadata.get(
                "peak_GPU_memory_bytes", 0
            ),
            **{f"cpu_{key}": value for key, value in cpu_quality.items()},
            **{f"gpu_{key}": value for key, value in gpu_quality.items()},
            **implementation,
            **local_agreement,
            "quality_warning": bool(
                cpu_quality["catastrophic_quality_failure"]
                or gpu_quality["catastrophic_quality_failure"]
                or implementation["cpu_gpu_core_jaccard"] < 1.0
            ),
        }
        return output
    except Exception as exc:
        return {
            **row,
            "status": "failed",
            "failure_class": (
                "mapping_failure"
                if "mapping" in str(exc).lower()
                else "unit_failure"
                if "unit" in str(exc).lower()
                else "implementation_or_data_failure"
            ),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "algorithm_failure": True,
            "eligible_for_D1_quality": True,
            "task_end_to_end_seconds": time.perf_counter() - task_start,
            "peak_RSS_mb": peak_rss_mb(),
            "RSS_at_task_start_mb": initial_rss,
            "RSS_at_task_end_mb": current_rss_mb(),
        }


def load_registry() -> list[dict[str, str]]:
    """Load D0 only after its gate has passed."""

    status_path = D0_ROOT / "d0_status.json"
    if not status_path.is_file():
        raise RuntimeError("D0 status is missing; run D0 first")
    status = json.loads(status_path.read_text())
    if status["status"] != "pass" or not status["D1_authorized"]:
        raise RuntimeError("D0 did not authorize D1")
    return read_tsv(REGISTRY_PATH)


def run_cold_probe(trajectory_id: str) -> None:
    """Measure one forced fresh-process GPU call."""

    registry = {
        row["trajectory_id"]: row for row in load_registry()
    }
    if trajectory_id not in registry:
        raise KeyError(f"trajectory not in D0 registry: {trajectory_id}")
    row = registry[trajectory_id]
    total_start = time.perf_counter()
    source = ROOT / row["source_file"]
    decode_start = time.perf_counter()
    coords, _, residue_metadata, metadata = extract_ca_trajectory(
        source,
        coordinate_path=row["coordinate_path"],
        direct_contiguous_read=True,
    )
    decode_seconds = time.perf_counter() - decode_start
    chains = np.asarray(
        [str(item.get("chain", "_")) for item in residue_metadata]
    )
    worker_start = time.perf_counter()
    worker = PersistentGPUWorker.create(
        chain_ids=chains,
        representative_count=min(256, coords.shape[0]),
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )
    initialization_seconds = time.perf_counter() - worker_start
    compute_start = time.perf_counter()
    result = worker.throughput(coords)
    compute_seconds = time.perf_counter() - compute_start
    payload = {
        "trajectory_id": trajectory_id,
        "status": "success",
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "source_unit_status": metadata["unit_status"],
        "decode_seconds": decode_seconds,
        "GPU_worker_initialization_seconds": initialization_seconds,
        "GPU_loaded_compute_seconds": compute_seconds,
        "GPU_reported_algorithm_seconds": result.total_runtime_seconds,
        "fresh_process_internal_end_to_end_seconds": (
            time.perf_counter() - total_start
        ),
        "peak_RSS_mb": peak_rss_mb(),
        "peak_VRAM_bytes": result.metadata.get(
            "peak_GPU_memory_bytes", 0
        ),
    }
    print("D1_COLD_JSON=" + json.dumps(payload, allow_nan=False))


def main() -> None:
    """Run or resume D1 checkpoints."""

    args = parse_args()
    if args.cold_probe:
        run_cold_probe(args.cold_probe)
        return
    registry = load_registry()
    selected_ids = set(args.trajectory_id)
    if selected_ids:
        registry = [
            row for row in registry if row["trajectory_id"] in selected_ids
        ]
        missing = selected_ids - {row["trajectory_id"] for row in registry}
        if missing:
            raise KeyError(f"unknown trajectory IDs: {sorted(missing)}")

    unavailable = [
        row
        for row in registry
        if row["registry_status"] == "explicitly_unavailable"
    ]
    for row in unavailable:
        path = checkpoint_path(row["trajectory_id"])
        if args.rerun or not path.is_file():
            write_json(path, unavailable_checkpoint(row))

    eligible = [
        row for row in registry if row["registry_status"] == "eligible"
    ]
    eligible.sort(
        key=lambda row: (int(row["n_residues"]), row["trajectory_id"])
    )
    pending = [
        row
        for row in eligible
        if args.rerun or not checkpoint_path(row["trajectory_id"]).is_file()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("D1 has no pending rows for this selection.")
        return

    worker = PersistentGPUWorker.create(
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )
    run_start = time.perf_counter()
    successes = 0
    failures = 0
    input_loader_workers = 2
    with ThreadPoolExecutor(max_workers=input_loader_workers) as executor:
        loaded_inputs = executor.map(load_coordinates, pending)
        for index, (row, loaded) in enumerate(
            zip(pending, loaded_inputs),
            start=1,
        ):
            result = process_row(row, worker, loaded)
            write_json(checkpoint_path(row["trajectory_id"]), result)
            if result["status"] == "success":
                successes += 1
            else:
                failures += 1
            if (
                index == 1
                or index % 25 == 0
                or result["status"] != "success"
            ):
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed * 3600.0
                print(
                    f"[D1] {index}/{len(pending)} new; "
                    f"{successes} success, {failures} failed; "
                    f"{rate:.1f} trajectories/hour; "
                    f"{row['trajectory_id']} {result['status']}",
                    flush=True,
                )
    run_status = {
        "interpreter": sys.executable,
        "pid": os.getpid(),
        "new_rows": len(pending),
        "new_successes": successes,
        "new_failures": failures,
        "wall_seconds": time.perf_counter() - run_start,
        "input_loader_workers": input_loader_workers,
        "worker_cache_summary": worker.cache_summary,
        "historical_outputs_modified": False,
        "public_default_changed": False,
        "git_metadata_modified": False,
    }
    write_json(D1_ROOT / "latest_run_status.json", run_status)
    print(json.dumps(run_status, indent=2))


if __name__ == "__main__":
    main()
