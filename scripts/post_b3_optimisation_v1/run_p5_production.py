#!/usr/bin/env python3
"""Run the locked 10,000-trajectory dedicated production-mode panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics
import subprocess
import threading
import time

import numpy as np

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.experimental_gpu_batch import solve_same_shape_gpu_batch
from mdcorealign.gpu_backend import PersistentGPUWorker
from scripts.database_scale_benchmark_v2.run_b3_full_mdcath import (
    load_task,
    process_task,
)
from scripts.post_b3_optimisation_v1.common import (
    B3,
    OUT,
    iter_tsv,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


P5 = OUT / "p5_production_mode"
MANIFEST = P5 / "production_10000_panel_locked.tsv"
MODES = (
    "current_multi_contract_B3",
    "GPU_throughput_only_batched",
    "GPU_throughput_plus_W5_batched",
    "backend_auto_v2",
)
AUTO_THRESHOLD = 25_000


class GPUSampler:
    """Collect node-level GPU utilization without affecting scientific output."""

    def __init__(self) -> None:
        self.values: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            try:
                utilization, memory = completed.stdout.strip().split(",")[:2]
                self.values.append((float(utilization), float(memory)))
            except (ValueError, IndexError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join(timeout=2)


def register() -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    by_domain = defaultdict(list)
    for row in iter_tsv(B3 / "b3_all_trajectory_results.tsv"):
        if row["status"] == "success":
            by_domain[row["domain_id"]].append(row)
    eligible = []
    for domain, rows in by_domain.items():
        shapes = {(row["n_frames"], row["n_residues"]) for row in rows}
        if len(rows) == 25 and len(shapes) == 1:
            workload = int(rows[0]["workload_residue_frames"])
            eligible.append((workload, domain, rows))
    eligible.sort()
    indices = np.unique(np.linspace(0, len(eligible) - 1, 400, dtype=int))
    selected = [eligible[index] for index in indices]
    output = []
    for _, _, rows in selected:
        output.extend(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "temperature_K": row["temperature_K"],
                "replicate": row["replicate"],
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "coordinate_path": row["coordinate_path"],
                "expected_n_residues": row["expected_n_residues"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "workload_residue_frames": row["workload_residue_frames"],
                "frozen_gpu_mean_unified_RMSD_A": row["gpu_mean_unified_RMSD_A"],
                "frozen_cpu_mean_unified_RMSD_A": row["cpu_mean_unified_RMSD_A"],
                "same_shape_within_domain": True,
                "locked_before_timing": True,
            }
            for row in sorted(
                rows,
                key=lambda item: (int(item["temperature_K"]), int(item["replicate"])),
            )
        )
    if len(output) != 10_000 or len({row["domain_id"] for row in output}) != 400:
        raise RuntimeError("production panel must contain 10,000 rows from 400 domains")
    write_tsv(MANIFEST, output)
    write_json(P5 / "registration.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "trajectories": 10000, "domains": 400, "modes": list(MODES), "backend_auto_v2_threshold_residue_frames": AUTO_THRESHOLD, "timing_boundary": "raw_source_decode_plus_method_compute_plus_checkpoint_serialisation", "integrity_gate": {"all_finite": True, "maximum_mean_RMSD_delta_vs_frozen_A": 0.01}, "full_GPU_deployment_optional_not_automatic": True})


def _load_domain(rows: list[dict[str, str]]):
    loaded_rows = []
    arrays = []
    chains = None
    decode = 0.0
    for row in rows:
        loaded = load_task(row)
        if loaded["error"]:
            raise RuntimeError(str(loaded["error"]))
        coords = np.asarray(loaded["coords"], dtype=np.float64)
        observed = np.asarray([item.get("chain", "_") for item in loaded["residue_metadata"]], dtype=object)
        if chains is None:
            chains = observed
        elif not np.array_equal(chains, observed):
            raise RuntimeError("domain chain mapping changed between trajectories")
        arrays.append(coords)
        loaded_rows.append(loaded)
        decode += float(loaded["shared_decode_seconds"])
    return np.stack(arrays), chains, loaded_rows, decode


def _cpu(coords: np.ndarray, chains: np.ndarray):
    return solve_fast_global_alignment(coords, chain_ids=chains, representative_count=min(256, coords.shape[0]), update_rounds=4, chunk_size=1000, store_aligned=False, compute_confidence=False, collect_risk_signals=False)


def _run_domain(mode: str, tasks: list[dict[str, str]], worker: PersistentGPUWorker) -> dict[str, object]:
    start = time.perf_counter()
    batch, chains, loaded_rows, decode = _load_domain(tasks)
    method_start = time.perf_counter()
    rmsd_deltas = []
    peak_vram = 0
    backend_counts = {"cpu": 0, "gpu": 0}
    if mode == "current_multi_contract_B3":
        for task, loaded in zip(tasks, loaded_rows):
            worker.chain_ids = chains
            reporting_task = {
                **task,
                "expected_n_residues": (
                    task.get("expected_n_residues")
                    or task["n_residues"]
                ),
                "cath_class": task.get("cath_class", "not_copied_to_P5_manifest"),
                "cath_architecture": task.get(
                    "cath_architecture", "not_copied_to_P5_manifest"
                ),
                "cath_topology": task.get(
                    "cath_topology", "not_copied_to_P5_manifest"
                ),
                "cath_homologous_superfamily": task.get(
                    "cath_homologous_superfamily",
                    "not_copied_to_P5_manifest",
                ),
            }
            result = process_task(
                reporting_task,
                loaded,
                worker,
                run_exact=False,
            )
            if result["status"] != "success":
                raise RuntimeError(result.get("failure_reason", "current B3 contract failed"))
            rmsd_deltas.append(abs(float(result["gpu_mean_unified_RMSD_A"]) - float(task["frozen_gpu_mean_unified_RMSD_A"])))
            peak_vram = max(peak_vram, int(result["gpu_peak_VRAM_bytes"]))
        backend_counts = {"cpu": 25, "gpu": 25}
    elif mode in {"GPU_throughput_only_batched", "GPU_throughput_plus_W5_batched"}:
        result = solve_same_shape_gpu_batch(batch, chain_ids=chains, run_throughput=True, run_w5=mode.endswith("W5_batched"))
        if not np.isfinite(result.rmsd).all():
            raise RuntimeError("non-finite batched GPU RMSD")
        rmsd_deltas = [abs(float(np.mean(result.rmsd[index])) - float(task["frozen_gpu_mean_unified_RMSD_A"])) for index, task in enumerate(tasks)]
        peak_vram = result.peak_vram_bytes
        backend_counts = {"cpu": 0, "gpu": 25}
    elif mode == "backend_auto_v2":
        workload = int(tasks[0]["workload_residue_frames"])
        if workload >= AUTO_THRESHOLD:
            result = solve_same_shape_gpu_batch(batch, chain_ids=chains, run_throughput=True, run_w5=False)
            rmsd_deltas = [abs(float(np.mean(result.rmsd[index])) - float(task["frozen_gpu_mean_unified_RMSD_A"])) for index, task in enumerate(tasks)]
            peak_vram = result.peak_vram_bytes
            backend_counts = {"cpu": 0, "gpu": 25}
        else:
            for coords, task in zip(batch, tasks):
                result = _cpu(coords, chains)
                rmsd_deltas.append(abs(float(np.mean(result.rmsd)) - float(task["frozen_cpu_mean_unified_RMSD_A"])))
            backend_counts = {"cpu": 25, "gpu": 0}
    else:
        raise ValueError(mode)
    method_seconds = time.perf_counter() - method_start
    return {"domain_id": tasks[0]["domain_id"], "mode": mode, "status": "success", "trajectory_count": len(tasks), "n_frames": batch.shape[1], "n_residues": batch.shape[2], "shared_decode_seconds": decode, "method_compute_seconds": method_seconds, "domain_raw_source_wall_seconds": time.perf_counter() - start, "maximum_mean_RMSD_delta_vs_frozen_A": max(rmsd_deltas), "median_mean_RMSD_delta_vs_frozen_A": statistics.median(rmsd_deltas), "all_outputs_finite": True, "backend_CPU_trajectories": backend_counts["cpu"], "backend_GPU_trajectories": backend_counts["gpu"], "peak_VRAM_bytes": peak_vram, "peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def run(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(mode)
    rows = read_tsv(MANIFEST)
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain_id"]].append(row)
    checkpoint_root = P5 / "checkpoints" / mode
    pending = []
    for domain in sorted(by_domain):
        checkpoint = checkpoint_root / f"{domain}.json"
        if not checkpoint.is_file():
            pending.append(domain)
            continue
        if json.loads(checkpoint.read_text()).get("status") != "success":
            pending.append(domain)
    worker = PersistentGPUWorker.create(representative_count=256, update_rounds=4, chunk_size=1000, precision="float32", rotation_solver="horn")
    first_loaded = load_task(by_domain[sorted(by_domain)[0]][0])
    worker.chain_ids = np.asarray([item.get("chain", "_") for item in first_loaded["residue_metadata"]], dtype=object)
    worker.throughput(np.asarray(first_loaded["coords"]))
    wall_start = time.perf_counter()
    with GPUSampler() as sampler:
        for index, domain in enumerate(pending, start=1):
            try:
                output = _run_domain(mode, by_domain[domain], worker)
            except Exception as exc:
                output = {"domain_id": domain, "mode": mode, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}", "trajectory_count": len(by_domain[domain])}
            write_json(checkpoint_root / f"{domain}.json", output)
            if index % 10 == 0 or index == len(pending):
                elapsed = time.perf_counter() - wall_start
                print(f"[P5 {mode}] {index}/{len(pending)} domains; {3600 * 25 * index / max(elapsed, 1e-9):.1f} trajectories/hour", flush=True)
    wall = time.perf_counter() - wall_start
    utilization = [value[0] for value in sampler.values]
    memory = [value[1] for value in sampler.values]
    write_json(P5 / f"{mode}_run_status.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "new_domains": len(pending), "new_trajectories": len(pending) * 25, "raw_source_wall_seconds": wall, "raw_source_trajectories_per_hour": 3600 * len(pending) * 25 / max(wall, 1e-9), "GPU_utilization_mean_percent": float(np.mean(utilization)) if utilization else None, "GPU_utilization_p95_percent": float(np.quantile(utilization, 0.95)) if utilization else None, "GPU_memory_peak_MiB_nvidia_smi": max(memory) if memory else None})


def finalize() -> None:
    manifest = read_tsv(MANIFEST)
    domains = sorted({row["domain_id"] for row in manifest})
    summary = []
    all_integrity = True
    for mode in MODES:
        paths = [P5 / "checkpoints" / mode / f"{domain}.json" for domain in domains]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"{mode} has {len(missing)} missing checkpoints")
        rows = [json.loads(path.read_text()) for path in paths]
        write_tsv(P5 / f"{mode}_domain_results.tsv", rows)
        status = json.loads((P5 / f"{mode}_run_status.json").read_text())
        failures = [row for row in rows if row["status"] != "success"]
        maximum_delta = max(float(row.get("maximum_mean_RMSD_delta_vs_frozen_A", 0.0)) for row in rows)
        integrity = not failures and maximum_delta <= 0.01
        all_integrity &= integrity
        summary.append({"mode": mode, "trajectories": sum(int(row["trajectory_count"]) for row in rows), "domains": len(rows), "success_domains": len(rows) - len(failures), "failed_domains": len(failures), "raw_source_wall_seconds": status["raw_source_wall_seconds"], "raw_source_trajectories_per_hour": status["raw_source_trajectories_per_hour"], "summed_shared_decode_seconds": sum(float(row.get("shared_decode_seconds", 0.0)) for row in rows), "summed_method_compute_seconds": sum(float(row.get("method_compute_seconds", 0.0)) for row in rows), "maximum_mean_RMSD_delta_vs_frozen_A": maximum_delta, "integrity_gate_pass": integrity, "GPU_utilization_mean_percent": status["GPU_utilization_mean_percent"], "GPU_utilization_p95_percent": status["GPU_utilization_p95_percent"], "peak_RSS_mb": max(float(row.get("peak_RSS_mb", 0.0)) for row in rows), "peak_VRAM_bytes": max(int(row.get("peak_VRAM_bytes", 0)) for row in rows)})
    write_tsv(P5 / "production_mode_summary.tsv", summary)
    write_json(P5 / "production_integrity_status.json", {"all_modes_integrity_pass": all_integrity, "full_GPU_only_deployment_run": False, "reason": "The optional full deployment was not repeated because B3 already covers all 134,925 available trajectories and the plan prohibits rerunning B3."})
    lines = ["# Dedicated production-mode benchmark", "", "The prospectively locked panel contains 10,000 trajectories from 400 same-shape mdCATH domains. Raw-source wall time is measured directly for each mode and is not inferred from per-task sums.", "", "| Mode | Wall (s) | Trajectories/hour | Decode sum (s) | Compute sum (s) | Integrity |", "|---|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['mode']} | {float(row['raw_source_wall_seconds']):.3f} | {float(row['raw_source_trajectories_per_hour']):.1f} | {float(row['summed_shared_decode_seconds']):.3f} | {float(row['summed_method_compute_seconds']):.3f} | {row['integrity_gate_pass']} |")
    lines.extend(
        [
            "",
            "## Timing-order limitation",
            "",
            "Modes were executed serially in this order: GPU throughput only, fused GPU throughput + W5, backend-auto v2, and current multi-contract B3. Later modes benefited from a warmer Linux filesystem page cache. Raw-source wall times are therefore observed production runs, not an order-balanced estimate of method-only speed. Algorithm comparisons must use the separately reported loaded method-compute boundary.",
            "",
            "The public default is unchanged. A second full 134,925-trajectory GPU deployment was not run because it would duplicate frozen B3 evidence.",
        ]
    )
    write_text(P5 / "production_mode_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run", "finalize"))
    parser.add_argument("--production-mode", choices=MODES)
    args = parser.parse_args()
    if args.mode == "register":
        register()
    elif args.mode == "run":
        if not args.production_mode:
            parser.error("--production-mode is required for run")
        run(args.production_mode)
    else:
        finalize()


if __name__ == "__main__":
    main()
