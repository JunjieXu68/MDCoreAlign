#!/usr/bin/env python3
"""Run the resumable B3 full mdCATH deployment benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import traceback
from typing import Iterator

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker, scan_w5_gpu
from mdsuperpose.sequential_weights import estimate_representative_weights_sequential_lean
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_refine import exact_all_frame_fit
from scripts.database_scale_benchmark_v2.common import (
    B2_ROOT,
    B3_ROOT,
    ROOT,
    TAIL_DOMAINS,
    read_tsv,
    write_json,
    write_tsv,
)
from scripts.database_scale_gpu_v1.common import (
    core_mask,
    cpu_gpu_agreement,
    flexible_mask,
    jaccard,
    normalized_weights,
    quality_against_exact,
    safe_spearman,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory


PILOT_CHECKPOINTS = B3_ROOT / "pilot_checkpoints"
FULL_CHECKPOINTS = B3_ROOT / "domain_checkpoints"
ROBUST_KWARGS: dict[str, object] = {
    "rule": "huber",
    "delta": 1.0,
    "max_iterations": 20,
    "tolerance": 1e-6,
    "objective_tolerance": 1e-5,
    "objective_patience": 2,
    "sequence_continuity_strength": 0.05,
    "contact_consistency_strength": 0.0,
    "record_history": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "finalize-pilot", "full", "finalize-full"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--input-workers", type=int, default=2)
    return parser.parse_args()


def current_rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def transform_hash(result: object) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(result.rotations).view(np.uint8))
    digest.update(np.ascontiguousarray(result.translations).view(np.uint8))
    return digest.hexdigest()


def finite_transform(result: object) -> bool:
    rotations = np.asarray(result.rotations, dtype=np.float64)
    translations = np.asarray(result.translations, dtype=np.float64)
    return bool(
        np.isfinite(rotations).all()
        and np.isfinite(translations).all()
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3)
    )


def load_task(task: dict[str, str]) -> dict[str, object]:
    """Decode one canonical CA array once for every downstream method."""

    start = time.perf_counter()
    try:
        coords, _, residue_metadata, metadata = extract_ca_trajectory(
            ROOT / task["source_file"],
            coordinate_path=task["coordinate_path"],
            direct_contiguous_read=True,
        )
        return {
            "coords": np.asarray(coords, dtype=np.float64),
            "residue_metadata": residue_metadata,
            "metadata": metadata,
            "shared_decode_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:
        return {
            "coords": None,
            "residue_metadata": None,
            "metadata": None,
            "shared_decode_seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def confidence_status(cpu: object, *, valid: bool) -> tuple[str, str]:
    """Apply the prospective B1 warning rule using within-run signals only."""

    if not valid:
        return "failure", "non_finite_or_invalid_transform"
    risk = cpu.risk_signals or {}
    effective = float(np.sum(normalized_weights(cpu.frozen_weights)) / cpu.frozen_weights.size)
    if effective < 0.10:
        return "failure", "effective_core_fraction_below_0.10"
    low_signals = []
    checks = (
        (cpu.confidence.representative_subset_weight_spearman, 0.90, "subset_weight_spearman"),
        (cpu.confidence.representative_subset_core_jaccard, 0.80, "subset_core_jaccard"),
        (cpu.confidence.temporal_block_weight_spearman, 0.85, "temporal_weight_spearman"),
        (cpu.confidence.temporal_block_core_jaccard, 0.75, "temporal_core_jaccard"),
        (risk.get("last_two_round_core_jaccard"), 0.80, "last_two_round_core_jaccard"),
    )
    for value, threshold, name in checks:
        if value is not None and np.isfinite(float(value)) and float(value) < threshold:
            low_signals.append(name)
    if low_signals:
        return "low_confidence", ";".join(low_signals)
    moderate = list(cpu.confidence.warning_flags)
    if float(risk.get("near_core_threshold_fraction", 0.0)) >= 0.20:
        moderate.append("near_core_threshold_fraction_at_least_0.20")
    if moderate:
        return "moderate_confidence", ";".join(sorted(set(moderate)))
    return "high_confidence", "all_registered_internal_checks_pass"


def w5_agreement(cpu: object, gpu: object) -> dict[str, object]:
    return {
        "w5_cpu_gpu_local_deformation_spearman": safe_spearman(
            cpu.local_deformation_index, gpu.local_deformation_index
        ),
        "w5_cpu_gpu_local_rigidity_spearman": safe_spearman(
            cpu.local_rigidity_score, gpu.local_rigidity_score
        ),
        "w5_cpu_gpu_segment_jaccard": jaccard(
            np.asarray(cpu.locally_deforming_segments) > 0,
            np.asarray(gpu.locally_deforming_segments) > 0,
        ),
    }


def exact_analysis(coords: np.ndarray, chains: np.ndarray) -> tuple[object, object, float, float]:
    """Run exact sequential core estimation and the matched fixed-core fit."""

    estimate_start = time.perf_counter()
    estimate = estimate_representative_weights_sequential_lean(
        coords,
        coords[0],
        sequence_chain_ids=chains,
        **ROBUST_KWARGS,
    )
    estimate_seconds = time.perf_counter() - estimate_start
    fit_start = time.perf_counter()
    fit = exact_all_frame_fit(coords, coords[0], estimate.mean_weights, chunk_size=1000)
    fit_seconds = time.perf_counter() - fit_start
    return estimate, fit, estimate_seconds, fit_seconds


def process_task(
    task: dict[str, str],
    loaded: dict[str, object],
    worker: PersistentGPUWorker,
    *,
    run_exact: bool,
) -> dict[str, object]:
    """Run all matched B3 contracts from one decoded coordinate array."""

    task_start = time.perf_counter()
    base: dict[str, object] = {
        "domain_id": task["domain_id"],
        "trajectory_id": task["trajectory_id"],
        "temperature_K": task["temperature_K"],
        "replicate": task["replicate"],
        "cath_class": task["cath_class"],
        "cath_architecture": task["cath_architecture"],
        "cath_topology": task["cath_topology"],
        "cath_homologous_superfamily": task["cath_homologous_superfamily"],
        "source_file": task["source_file"],
        "source_sha256": task["source_sha256"],
        "coordinate_path": task["coordinate_path"],
        "shared_input_decode_seconds": loaded["shared_decode_seconds"],
        "shared_decode_counted_once": True,
        "exact_requested": run_exact,
    }
    try:
        if loaded["error"]:
            raise RuntimeError(str(loaded["error"]))
        coords = np.asarray(loaded["coords"], dtype=np.float64)
        metadata = loaded["metadata"]
        residue_metadata = loaded["residue_metadata"]
        if metadata["unit_status"] != "verified_angstrom_metadata":
            raise RuntimeError(f"unit_integrity: {metadata['unit_status']}")
        if coords.shape[1] != int(task["expected_n_residues"]):
            raise RuntimeError(
                f"mapping_integrity: {coords.shape[1]} versus {task['expected_n_residues']} residues"
            )
        chains = np.asarray([row.get("chain", "_") for row in residue_metadata], dtype=object)
        cpu_start = time.perf_counter()
        cpu = solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=min(256, coords.shape[0]),
            update_rounds=4,
            chunk_size=1000,
            store_aligned=False,
            compute_confidence=True,
            collect_risk_signals=True,
        )
        cpu_seconds = time.perf_counter() - cpu_start
        worker.chain_ids = chains
        gpu_start = time.perf_counter()
        gpu = worker.throughput(coords, representative_count=min(256, coords.shape[0]))
        gpu_seconds = time.perf_counter() - gpu_start
        cpu_w5_start = time.perf_counter()
        cpu_w5 = scan_local_kinematics(coords, chain_ids=chains, evaluate_blocks=True)
        cpu_w5_seconds = time.perf_counter() - cpu_w5_start
        gpu_w5_start = time.perf_counter()
        gpu_w5 = scan_w5_gpu(coords, chain_ids=chains, precision="float32")
        gpu_w5_seconds = time.perf_counter() - gpu_w5_start
        implementation = cpu_gpu_agreement(cpu, gpu, chains=chains)
        local_agreement = w5_agreement(cpu_w5, gpu_w5)
        valid = finite_transform(cpu) and finite_transform(gpu)
        confidence, warning_reason = confidence_status(cpu, valid=valid)
        workload = int(coords.shape[0] * coords.shape[1])
        auto_backend = "gpu" if workload >= 150_000 else "cpu"
        auto_result = gpu if auto_backend == "gpu" else cpu
        auto_seconds = gpu_seconds if auto_backend == "gpu" else cpu_seconds
        risk = cpu.risk_signals or {}
        output: dict[str, object] = {
            **base,
            "status": "success",
            "failure_class": "",
            "failure_reason": "",
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "n_chains": len(set(chains.tolist())),
            "unit_integrity": "pass",
            "mapping_integrity": "pass",
            "input_read_strategy": metadata["read_strategy"],
            "workload_residue_frames": workload,
            "cpu_throughput_loaded_seconds": cpu_seconds,
            "gpu_throughput_loaded_seconds": gpu_seconds,
            "backend_auto_selected": auto_backend,
            "backend_auto_loaded_seconds": auto_seconds,
            "backend_auto_reason": "residue_frames_at_least_150000" if auto_backend == "gpu" else "residue_frames_below_150000",
            "cpu_w5_loaded_seconds": cpu_w5_seconds,
            "gpu_w5_loaded_seconds": gpu_w5_seconds,
            "cpu_transform_sha256": transform_hash(cpu),
            "gpu_transform_sha256": transform_hash(gpu),
            "auto_transform_sha256": transform_hash(auto_result),
            "cpu_mean_unified_RMSD_A": float(np.mean(cpu.rmsd)),
            "gpu_mean_unified_RMSD_A": float(np.mean(gpu.rmsd)),
            "auto_mean_unified_RMSD_A": float(np.mean(auto_result.rmsd)),
            "cpu_mean_weighted_RMSD_A": float(np.mean(cpu.weighted_rmsd)),
            "gpu_mean_weighted_RMSD_A": float(np.mean(gpu.weighted_rmsd)),
            "cpu_effective_core_fraction": float(np.sum(normalized_weights(cpu.frozen_weights)) / cpu.frozen_weights.size),
            "gpu_effective_core_fraction": float(np.sum(normalized_weights(gpu.frozen_weights)) / gpu.frozen_weights.size),
            "cpu_RMSF_mean_A": float(np.mean(cpu.robust_rmsf)),
            "gpu_RMSF_mean_A": float(np.mean(gpu.robust_rmsf)),
            "confidence_category": confidence,
            "warning_reason": warning_reason,
            "strict_fixed_budget_status": cpu.confidence.convergence_status,
            "temporal_block_weight_spearman": cpu.confidence.temporal_block_weight_spearman,
            "temporal_block_core_jaccard": cpu.confidence.temporal_block_core_jaccard,
            "representative_subset_weight_spearman": cpu.confidence.representative_subset_weight_spearman,
            "representative_subset_core_jaccard": cpu.confidence.representative_subset_core_jaccard,
            "last_two_round_core_jaccard": risk.get("last_two_round_core_jaccard", ""),
            "final_round_weight_linf_change": risk.get("final_weight_linf_change", ""),
            "near_core_threshold_fraction": risk.get("near_core_threshold_fraction", ""),
            "cpu_w5_temporal_block_reproducibility": cpu_w5.temporal_block_reproducibility,
            "cpu_w5_temporal_block_segment_jaccard": cpu_w5.temporal_block_segment_jaccard,
            "cpu_w5_mean_local_deformation_index": float(np.mean(cpu_w5.local_deformation_index)),
            "gpu_w5_mean_local_deformation_index": float(np.mean(gpu_w5.local_deformation_index)),
            "peak_RSS_mb": peak_rss_mb(),
            "RSS_at_task_end_mb": current_rss_mb(),
            "gpu_peak_VRAM_bytes": gpu.metadata.get("peak_GPU_memory_bytes", 0),
            "gpu_w5_peak_VRAM_bytes": gpu_w5.scan.metadata.get("peak_GPU_memory_bytes", 0),
            "gpu_H2D_seconds": gpu.timing.get("representative_H2D", 0.0) + gpu.timing.get("all_frame_H2D", 0.0),
            "gpu_D2H_seconds": gpu.timing.get("weight_D2H", 0.0) + gpu.timing.get("all_frame_D2H", 0.0),
            "gpu_compute_seconds": gpu.timing.get("GPU_compute_total", 0.0),
            **implementation,
            **local_agreement,
        }
        if run_exact:
            estimate, exact_fit, estimate_seconds, fit_seconds = exact_analysis(coords, chains)
            exact_weights = np.asarray(estimate.mean_weights, dtype=np.float64)
            exact_core = core_mask(exact_weights)
            exact_flexible = flexible_mask(exact_weights, chains)
            cpu_quality = quality_against_exact(
                cpu,
                exact_weights=exact_weights,
                exact_rmsd=exact_fit.rmsd,
                exact_rmsf=exact_fit.robust_rmsf,
                exact_core=exact_core,
                exact_flexible=exact_flexible,
                chains=chains,
            )
            gpu_quality = quality_against_exact(
                gpu,
                exact_weights=exact_weights,
                exact_rmsd=exact_fit.rmsd,
                exact_rmsf=exact_fit.robust_rmsf,
                exact_core=exact_core,
                exact_flexible=exact_flexible,
                chains=chains,
            )
            output.update(
                {
                    "exact_completed": True,
                    "exact_weight_estimation_seconds": estimate_seconds,
                    "exact_all_frame_fit_seconds": fit_seconds,
                    "exact_total_loaded_seconds": estimate_seconds + fit_seconds,
                    "exact_mean_unified_RMSD_A": float(np.mean(exact_fit.rmsd)),
                    "exact_mean_weighted_RMSD_A": float(np.mean(exact_fit.weighted_rmsd)),
                    "exact_effective_core_fraction": float(np.sum(normalized_weights(exact_weights)) / exact_weights.size),
                    "exact_transform_sha256": transform_hash(exact_fit),
                    **{f"cpu_exact_{key}": value for key, value in cpu_quality.items()},
                    **{f"gpu_exact_{key}": value for key, value in gpu_quality.items()},
                }
            )
        else:
            output.update(
                {
                    "exact_completed": False,
                    "exact_weight_estimation_seconds": 0.0,
                    "exact_all_frame_fit_seconds": 0.0,
                    "exact_total_loaded_seconds": 0.0,
                }
            )
        output["task_end_to_end_seconds"] = time.perf_counter() - task_start
        return output
    except Exception as exc:
        text = str(exc).lower()
        failure_class = (
            "unit_integrity_failure" if "unit_integrity" in text else
            "mapping_integrity_failure" if "mapping_integrity" in text else
            "numerical_or_implementation_failure"
        )
        return {
            **base,
            "status": "failed",
            "failure_class": failure_class,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "confidence_category": "failure",
            "task_end_to_end_seconds": time.perf_counter() - task_start,
            "peak_RSS_mb": peak_rss_mb(),
        }


def task_prefetch(
    tasks: list[dict[str, str]],
    worker: PersistentGPUWorker,
    *,
    run_exact_ids: set[str],
    input_workers: int,
) -> Iterator[dict[str, object]]:
    """Run bounded ordered prefetch while preserving source frame order."""

    with ThreadPoolExecutor(max_workers=max(1, input_workers)) as executor:
        futures: dict[int, Future] = {}
        next_submit = 0
        while next_submit < min(len(tasks), input_workers):
            futures[next_submit] = executor.submit(load_task, tasks[next_submit])
            next_submit += 1
        for index, task in enumerate(tasks):
            loaded = futures.pop(index).result()
            if next_submit < len(tasks):
                futures[next_submit] = executor.submit(load_task, tasks[next_submit])
                next_submit += 1
            yield process_task(
                task,
                loaded,
                worker,
                run_exact=task["trajectory_id"] in run_exact_ids,
            )


def create_worker() -> PersistentGPUWorker:
    return PersistentGPUWorker.create(
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
    )


def pilot(args: argparse.Namespace) -> None:
    status = json.loads((B2_ROOT / "b2_status.json").read_text())
    if not status["B3_pilot_authorized"]:
        raise RuntimeError("B2 did not authorize B3 pilot")
    manifest = {row["trajectory_id"]: row for row in read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")}
    locked = read_tsv(B2_ROOT / "b3_exact_pilot_2000_locked.tsv")
    tasks = [manifest[row["trajectory_id"]] for row in locked]
    pending = [
        task for task in tasks
        if args.rerun or not (PILOT_CHECKPOINTS / f"{task['trajectory_id']}.json").is_file()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    start = time.perf_counter()
    worker = create_worker()
    rows: list[dict[str, object]] = []
    total_pending = len(pending)
    stream = task_prefetch(
        pending,
        worker,
        run_exact_ids={task["trajectory_id"] for task in pending},
        input_workers=args.input_workers,
    )
    for index, row in enumerate(stream, start=1):
        rows.append(row)
        write_json(PILOT_CHECKPOINTS / f"{row['trajectory_id']}.json", row)
        if index % 25 == 0 or index == total_pending:
            print(f"[B3 pilot] {index}/{total_pending} new", flush=True)
    write_json(
        B3_ROOT / "latest_pilot_run_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "new_rows": len(rows),
            "new_successes": sum(row["status"] == "success" for row in rows),
            "new_failures": sum(row["status"] != "success" for row in rows),
            "wall_seconds": time.perf_counter() - start,
            "input_workers": args.input_workers,
            "persistent_GPU_worker": True,
            "trajectory_data_retained_between_tasks": False,
        },
    )


def exact_subset(rows: list[dict[str, str]], target: int = 10_000) -> list[dict[str, str]]:
    """Lock a >=10k exact subset covering every available domain and condition."""

    available = [row for row in rows if row["locally_available"].lower() == "true"]
    selected = {
        row["trajectory_id"]: row
        for row in available
        if (row["temperature_K"] == "320" and row["replicate"] == "1")
        or row["domain_id"] in TAIL_DOMAINS
    }
    pilot_ids = {
        row["trajectory_id"]
        for row in read_tsv(B2_ROOT / "b3_exact_pilot_2000_locked.tsv")
    }
    selected.update({row["trajectory_id"]: row for row in available if row["trajectory_id"] in pilot_ids})
    ordered = sorted(
        available,
        key=lambda row: hashlib.sha256(row["trajectory_id"].encode()).hexdigest(),
    )
    for row in ordered:
        if len(selected) >= target:
            break
        selected.setdefault(row["trajectory_id"], row)
    return sorted(selected.values(), key=lambda row: row["trajectory_id"])


def finalize_pilot() -> None:
    locked = read_tsv(B2_ROOT / "b3_exact_pilot_2000_locked.tsv")
    paths = [PILOT_CHECKPOINTS / f"{row['trajectory_id']}.json" for row in locked]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"pilot has {len(missing)} missing checkpoints")
    rows = [json.loads(path.read_text()) for path in paths]
    write_tsv(B3_ROOT / "b3_exact_pilot_results.tsv", rows)
    success = [row for row in rows if row["status"] == "success"]
    integrity_failures = [
        row for row in rows
        if row["failure_class"] in {"unit_integrity_failure", "mapping_integrity_failure"}
    ]
    numerical_failures = [
        row for row in rows if row["failure_class"] == "numerical_or_implementation_failure"
    ]
    exact_seconds = sum(float(row["exact_total_loaded_seconds"]) for row in success)
    serial_boundary = sum(float(row["task_end_to_end_seconds"]) for row in success)
    incremental_fraction = exact_seconds / serial_boundary if serial_boundary else math.inf
    projected_storage_bytes = int(
        (B3_ROOT / "b3_exact_pilot_results.tsv").stat().st_size
        / max(1, len(rows))
        * 134_950
    )
    run_full_exact = (
        not integrity_failures
        and not numerical_failures
        and incremental_fraction <= 0.25
        and projected_storage_bytes <= 10 * 1024**3
    )
    manifest = read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")
    subset = (
        [row for row in manifest if row["locally_available"].lower() == "true"]
        if run_full_exact
        else exact_subset(manifest)
    )
    write_tsv(
        B3_ROOT / ("b3_exact_full_locked.tsv" if run_full_exact else "b3_exact_quality_subset_locked.tsv"),
        subset,
    )
    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_rows": len(rows),
        "pilot_successes": len(success),
        "integrity_failures": len(integrity_failures),
        "numerical_or_implementation_failures": len(numerical_failures),
        "exact_incremental_loaded_seconds": exact_seconds,
        "pilot_serial_task_boundary_seconds": serial_boundary,
        "exact_incremental_fraction_of_pilot_wall_boundary": incremental_fraction,
        "threshold_max": 0.25,
        "projected_compact_summary_storage_bytes": projected_storage_bytes,
        "storage_manageable": projected_storage_bytes <= 10 * 1024**3,
        "decision": "exact_all_available" if run_full_exact else "exact_locked_subset",
        "exact_trajectory_count": len(subset),
        "B3_full_authorized": not integrity_failures and not numerical_failures,
        "post_hoc_threshold_tuning": False,
    }
    write_json(B3_ROOT / "exact_execution_decision.json", decision)
    print(json.dumps(decision, indent=2))


def full(args: argparse.Namespace) -> None:
    decision = json.loads((B3_ROOT / "exact_execution_decision.json").read_text())
    if not decision["B3_full_authorized"]:
        raise RuntimeError("B3 pilot did not authorize the full run")
    manifest = read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")
    exact_path = B3_ROOT / (
        "b3_exact_full_locked.tsv"
        if decision["decision"] == "exact_all_available"
        else "b3_exact_quality_subset_locked.tsv"
    )
    exact_ids = {row["trajectory_id"] for row in read_tsv(exact_path)}
    pilot_rows = {
        path.stem: json.loads(path.read_text())
        for path in PILOT_CHECKPOINTS.glob("*.json")
    }
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        by_domain.setdefault(row["domain_id"], []).append(row)
    domains = sorted(by_domain)
    pending = [
        domain for domain in domains
        if args.rerun or not (FULL_CHECKPOINTS / f"{domain}.json").is_file()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    worker = create_worker()
    start = time.perf_counter()
    successes = failures = unavailable = 0
    for index, domain in enumerate(pending, start=1):
        tasks = sorted(
            by_domain[domain],
            key=lambda row: (int(row["temperature_K"]), int(row["replicate"])),
        )
        if tasks[0]["locally_available"].lower() != "true":
            rows = [
                {
                    **task,
                    "status": "explicitly_unavailable",
                    "failure_class": "input_unavailable",
                    "failure_reason": task["unavailable_reason"],
                    "confidence_category": "not_evaluated",
                }
                for task in tasks
            ]
            unavailable += len(rows)
        else:
            reusable = [task for task in tasks if task["trajectory_id"] in pilot_rows]
            new_tasks = [task for task in tasks if task["trajectory_id"] not in pilot_rows]
            rows = [pilot_rows[task["trajectory_id"]] for task in reusable]
            rows.extend(
                task_prefetch(
                    new_tasks,
                    worker,
                    run_exact_ids=exact_ids,
                    input_workers=args.input_workers,
                )
            )
            rows.sort(key=lambda row: (int(row["temperature_K"]), int(row["replicate"])))
            successes += sum(row["status"] == "success" for row in rows)
            failures += sum(row["status"] == "failed" for row in rows)
        write_json(FULL_CHECKPOINTS / f"{domain}.json", rows)
        if index % 10 == 0 or index == len(pending):
            elapsed = time.perf_counter() - start
            print(
                f"[B3 full] domains {index}/{len(pending)}; success={successes}; "
                f"failed={failures}; unavailable={unavailable}; "
                f"rate={3600 * (successes + failures) / max(elapsed, 1e-9):.1f} trajectories/hour",
                flush=True,
            )
        if index <= 10 and failures >= 3:
            recent = [row for domain_path in list(FULL_CHECKPOINTS.glob("*.json"))[-10:] for row in json.loads(domain_path.read_text())]
            integrity = sum(row.get("failure_class") in {"unit_integrity_failure", "mapping_integrity_failure"} for row in recent)
            if integrity >= 3:
                raise RuntimeError("systematic input integrity failures detected; B3 stopped")
    write_json(
        B3_ROOT / "latest_full_run_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "new_domains": len(pending),
            "new_successes": successes,
            "new_failures": failures,
            "new_unavailable": unavailable,
            "wall_seconds": time.perf_counter() - start,
            "input_workers": args.input_workers,
            "persistent_GPU_worker": True,
            "trajectory_data_retained_between_tasks": False,
        },
    )


def finalize_full() -> None:
    manifest = read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")
    domains = sorted({row["domain_id"] for row in manifest})
    missing = [domain for domain in domains if not (FULL_CHECKPOINTS / f"{domain}.json").is_file()]
    if missing:
        raise RuntimeError(f"full run has {len(missing)} missing domain checkpoints")
    rows = [
        row
        for domain in domains
        for row in json.loads((FULL_CHECKPOINTS / f"{domain}.json").read_text())
    ]
    if len(rows) != 134_950:
        raise RuntimeError(f"full aggregate has {len(rows):,}/134,950 rows")
    write_tsv(B3_ROOT / "b3_all_trajectory_results.tsv", rows)
    decisions = [
        {
            "trajectory_id": row["trajectory_id"],
            "domain_id": row["domain_id"],
            "workload_residue_frames": row.get("workload_residue_frames", ""),
            "backend_auto_selected": row.get("backend_auto_selected", ""),
            "backend_auto_reason": row.get("backend_auto_reason", ""),
            "cpu_throughput_loaded_seconds": row.get("cpu_throughput_loaded_seconds", ""),
            "gpu_throughput_loaded_seconds": row.get("gpu_throughput_loaded_seconds", ""),
            "auto_selected_faster_backend": (
                float(row["backend_auto_loaded_seconds"])
                <= min(float(row["cpu_throughput_loaded_seconds"]), float(row["gpu_throughput_loaded_seconds"])) * 1.05
                if row["status"] == "success" else ""
            ),
        }
        for row in rows
    ]
    write_tsv(B3_ROOT / "b3_backend_auto_decisions.tsv", decisions)
    exact_rows = [row for row in rows if str(row.get("exact_completed", "")).lower() == "true"]
    write_tsv(B3_ROOT / "b3_exact_quality_subset.tsv", exact_rows)
    failure_rows = [row for row in rows if row["status"] != "success" or row.get("confidence_category") != "high_confidence"]
    write_tsv(B3_ROOT / "b3_failure_warning_atlas.tsv", failure_rows)
    success = [row for row in rows if row["status"] == "success"]
    summary = {
        "manifest_rows": len(rows),
        "success": len(success),
        "failed": sum(row["status"] == "failed" for row in rows),
        "explicitly_unavailable": sum(row["status"] == "explicitly_unavailable" for row in rows),
        "exact_completed": len(exact_rows),
        "high_confidence": sum(row.get("confidence_category") == "high_confidence" for row in success),
        "moderate_confidence": sum(row.get("confidence_category") == "moderate_confidence" for row in success),
        "low_confidence": sum(row.get("confidence_category") == "low_confidence" for row in success),
        "failure_confidence": sum(row.get("confidence_category") == "failure" for row in success),
        "shared_decode_seconds_sum": sum(float(row["shared_input_decode_seconds"]) for row in success),
        "cpu_throughput_seconds_sum": sum(float(row["cpu_throughput_loaded_seconds"]) for row in success),
        "gpu_throughput_seconds_sum": sum(float(row["gpu_throughput_loaded_seconds"]) for row in success),
        "auto_seconds_sum": sum(float(row["backend_auto_loaded_seconds"]) for row in success),
        "cpu_w5_seconds_sum": sum(float(row["cpu_w5_loaded_seconds"]) for row in success),
        "gpu_w5_seconds_sum": sum(float(row["gpu_w5_loaded_seconds"]) for row in success),
        "exact_seconds_sum": sum(float(row["exact_total_loaded_seconds"]) for row in exact_rows),
        "peak_RSS_mb": max(float(row["peak_RSS_mb"]) for row in success),
        "peak_VRAM_bytes": max(float(row["gpu_peak_VRAM_bytes"]) for row in success),
    }
    write_tsv(B3_ROOT / "b3_database_throughput_summary.tsv", [{"metric": key, "value": value} for key, value in summary.items()])
    write_json(B3_ROOT / "b3_status.json", {"phase": "B3", "generated_at": datetime.now(timezone.utc).isoformat(), **summary})
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "pilot":
        pilot(args)
    elif args.mode == "finalize-pilot":
        finalize_pilot()
    elif args.mode == "full":
        full(args)
    else:
        finalize_full()


if __name__ == "__main__":
    main()
