#!/usr/bin/env python3
"""Remediate PDB-range failures and benchmark parallel MDLovoFit throughput."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import tempfile
import time

import numpy as np

from mdsuperpose.alignment import solve_rigid_alignment
from mdsuperpose.mdlovofit import (
    build_mdlovofit_command,
    parse_mdlovofit_stdout,
    read_multimodel_pdb_with_occupancy,
)
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.database_scale_benchmark_v2.run_mdlovofit_addendum import (
    EXTERNAL_EXECUTABLE,
    independent_frame_rmsd,
    load_coords,
)
from scripts.post_b3_optimisation_v1.common import (
    B2,
    MDL,
    OUT,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


P6 = OUT / "p6_mdlovofit_fairness"
RANGE_MANIFEST = P6 / "pdb_range_failures_45_locked.tsv"
PARALLEL_MANIFEST = P6 / "parallel_2000_panel_locked.tsv"
WORKERS = (1, 2, 4, 8, 16)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode(errors="replace")).hexdigest()


def center_for_pdb(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply an independent rigid COG translation to every frame."""

    values = np.asarray(coords, dtype=np.float64)
    translation = np.mean(values, axis=1)
    centered = values - translation[:, None, :]
    if np.any(np.abs(centered) >= 10_000.0):
        raise ValueError("common reference-COG centering did not satisfy PDB 8.3 range")
    return centered, translation


def _ordinary_invariance(raw: np.ndarray, centered: np.ndarray) -> float:
    differences = []
    for raw_frame, centered_frame in zip(raw, centered):
        raw_rmsd = solve_rigid_alignment(raw_frame, raw[0]).rmsd
        centered_rmsd = solve_rigid_alignment(centered_frame, centered[0]).rmsd
        differences.append(abs(raw_rmsd - centered_rmsd))
    return max(differences)


def register() -> None:
    if RANGE_MANIFEST.exists() or PARALLEL_MANIFEST.exists():
        raise FileExistsError("P6 manifests are already locked")
    all_tasks = {
        row["trajectory_id"]: row
        for row in read_tsv(B2 / "b3_all_trajectory_manifest_locked.tsv")
    }
    failures = read_tsv(MDL / "mdlovofit_failures_timeouts.tsv")
    range_ids = [
        row["trajectory_id"]
        for row in failures
        if "PDB 8.3 coordinate fields" in row.get("failure_reason", "")
    ]
    if len(range_ids) != 45:
        raise RuntimeError(f"expected 45 PDB-range failures, observed {len(range_ids)}")
    write_tsv(
        RANGE_MANIFEST,
        [
            {
                **all_tasks[trajectory_id],
                "original_failure_class": "PDB_coordinate_range_failure",
                "registered_centering": "subtract_reference_frame_center_of_geometry_from_every_frame",
            }
            for trajectory_id in range_ids
        ],
    )
    panel = read_tsv(MDL / "manifests/mdlovofit_pilot_2000_locked.tsv")
    if len(panel) != 2000:
        raise RuntimeError("parallel panel must contain the locked 2,000 trajectories")
    write_tsv(
        PARALLEL_MANIFEST,
        [
            {
                **row,
                "registered_centering": "subtract_reference_frame_center_of_geometry_from_every_frame",
                "parallel_worker_counts": ",".join(map(str, WORKERS)),
                "panel_reused_from_prospectively_locked_MDLovoFit_pilot": True,
            }
            for row in panel
        ],
    )
    write_json(P6 / "registration.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "PDB_range_failure_rows": 45, "parallel_panel_rows": 2000, "worker_counts": list(WORKERS), "original_serial_deployment_preserved": True, "centering_is_one_common_rigid_translation": True})


def run_centered(task: dict[str, str], *, check_invariance: bool) -> dict[str, object]:
    """Run one centered MDLovoFit workflow with temporary PDBs."""

    total_start = time.perf_counter()
    try:
        coords, metadata, decode = load_coords(task)
        centered, translation = center_for_pdb(coords)
        invariance = _ordinary_invariance(coords, centered) if check_invariance else ""
        with tempfile.TemporaryDirectory(prefix="mdca_p6_") as temporary:
            root = Path(temporary)
            input_path = root / "input.pdb"
            output_path = root / "aligned.pdb"
            prep_start = time.perf_counter()
            write_multimodel_ca_pdb_streaming(centered, input_path)
            prep = time.perf_counter() - prep_start
            command = build_mdlovofit_command(EXTERNAL_EXECUTABLE, fraction=0.60, output_path=output_path, input_paths=[input_path])
            external_start = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            external = time.perf_counter() - external_start
            if completed.returncode != 0:
                return {**task, "status": "failed", "failure_class": "external_return_code", "failure_reason": f"MDLovoFit return code {completed.returncode}: {completed.stderr[-500:]}", "shared_decode_seconds": decode, "input_preparation_seconds": prep, "subprocess_seconds": external, "complete_wall_seconds": time.perf_counter() - total_start}
            aligned, occupancies = read_multimodel_pdb_with_occupancy(output_path, n_atoms=centered.shape[1])
            per_frame, unified = independent_frame_rmsd(aligned, centered[0])
            parsed = parse_mdlovofit_stdout(completed.stdout)
            return {**task, "status": "success", "failure_class": "", "failure_reason": "", "n_frames": centered.shape[0], "n_residues": centered.shape[1], "shared_decode_seconds": decode, "input_preparation_seconds": prep, "subprocess_seconds": external, "complete_wall_seconds": time.perf_counter() - total_start, "centering_convention_used": "independent_per_frame_center_of_geometry", "reference_centering_translation_x": translation[0, 0], "reference_centering_translation_y": translation[0, 1], "reference_centering_translation_z": translation[0, 2], "maximum_centering_translation_norm": float(np.max(np.linalg.norm(translation, axis=1))), "centered_maximum_absolute_coordinate": float(np.max(np.abs(centered))), "ordinary_Kabsch_RMSD_invariance_max_abs_A": invariance, "unified_posthoc_RMSD_A": unified, "median_frame_RMSD_A": float(np.median(per_frame)), "native_all_atom_RMSD_A": parsed.average_rmsd_all_atoms, "native_least_mobile_RMSD_A": parsed.average_rmsd_least_mobile, "native_reported_fraction": parsed.fraction_of_best_aligned_atoms, "stdout_sha256": _digest(completed.stdout), "stderr_sha256": _digest(completed.stderr), "peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "PDB_input_and_output_retained": False}
    except subprocess.TimeoutExpired:
        return {**task, "status": "timeout", "failure_class": "timeout", "failure_reason": "MDLovoFit exceeded 120 seconds", "complete_wall_seconds": time.perf_counter() - total_start}
    except Exception as exc:
        return {**task, "status": "failed", "failure_class": "implementation_or_input_failure", "failure_reason": f"{type(exc).__name__}: {exc}", "complete_wall_seconds": time.perf_counter() - total_start}


def remediate() -> None:
    rows = read_tsv(RANGE_MANIFEST)
    write_json(
        P6 / "centering_v2_amendment.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "v1_result": "45/45 common-reference-COG attempts remained outside PDB 8.3 coordinate range",
            "v1_results_preserved_at": "pdb_range_failure_centered_reruns.tsv",
            "v2_convention": "subtract_each_frame_center_of_geometry_from_that_frame",
            "scientific_effect": "independent per-frame rigid translations only",
            "RMSD_invariance_must_be_verified": True,
            "amendment_made_after_observing_v1_range_failure": True,
        },
    )
    output = []
    for index, row in enumerate(rows, start=1):
        output.append(run_centered(row, check_invariance=True))
        print(f"[P6 remediation] {index}/{len(rows)}", flush=True)
    write_tsv(P6 / "pdb_range_failure_framewise_centered_reruns.tsv", output)


def _parallel_worker(task: dict[str, str]) -> dict[str, object]:
    return run_centered(task, check_invariance=False)


def _disk_sectors() -> tuple[int, int]:
    read = written = 0
    for line in Path("/proc/diskstats").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 10 and not fields[2].startswith(("loop", "ram")):
            read += int(fields[5])
            written += int(fields[9])
    return read, written


def parallel(worker_count: int) -> None:
    if worker_count not in WORKERS:
        raise ValueError(worker_count)
    tasks = read_tsv(PARALLEL_MANIFEST)
    checkpoint_root = P6 / "parallel_checkpoints" / f"workers_{worker_count}"
    pending = [task for task in tasks if not (checkpoint_root / f"{task['trajectory_id']}.json").is_file()]
    before = _disk_sectors()
    start = time.perf_counter()
    success = failed = timeout = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_parallel_worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            write_json(checkpoint_root / f"{row['trajectory_id']}.json", row)
            success += row["status"] == "success"
            failed += row["status"] == "failed"
            timeout += row["status"] == "timeout"
            if index % 50 == 0 or index == len(pending):
                elapsed = time.perf_counter() - start
                print(f"[P6 workers={worker_count}] {index}/{len(pending)}; {3600 * index / max(elapsed, 1e-9):.1f} trajectories/hour", flush=True)
    wall = time.perf_counter() - start
    after = _disk_sectors()
    write_json(P6 / f"parallel_workers_{worker_count}_status.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "workers": worker_count, "new_rows": len(pending), "success": success, "failed": failed, "timeout": timeout, "node_wall_seconds": wall, "trajectories_per_hour": 3600 * len(pending) / max(wall, 1e-9), "node_shared_disk_read_bytes_delta_approx": max(0, after[0] - before[0]) * 512, "node_shared_disk_write_bytes_delta_approx": max(0, after[1] - before[1]) * 512, "disk_counter_caveat": "Linux node-wide diskstats include unrelated activity."})


def finalize() -> None:
    remediation = read_tsv(
        P6 / "pdb_range_failure_framewise_centered_reruns.tsv"
    )
    remediation_success = [row for row in remediation if row["status"] == "success"]
    summary = []
    panel = read_tsv(PARALLEL_MANIFEST)
    for workers in WORKERS:
        root = P6 / "parallel_checkpoints" / f"workers_{workers}"
        paths = [root / f"{row['trajectory_id']}.json" for row in panel]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"workers={workers} has {len(missing)} missing rows")
        rows = [json.loads(path.read_text()) for path in paths]
        write_tsv(P6 / f"parallel_workers_{workers}_results.tsv", rows)
        status = json.loads((P6 / f"parallel_workers_{workers}_status.json").read_text())
        success = [row for row in rows if row["status"] == "success"]
        summary.append({"workers": workers, "panel_rows": len(rows), "success": len(success), "failed": sum(row["status"] == "failed" for row in rows), "timeouts": sum(row["status"] == "timeout" for row in rows), "failure_rate": 1.0 - len(success) / len(rows), "node_wall_seconds": status["node_wall_seconds"], "node_trajectories_per_hour": status["trajectories_per_hour"], "median_single_trajectory_complete_seconds": statistics.median(float(row["complete_wall_seconds"]) for row in success), "summed_decode_seconds": sum(float(row.get("shared_decode_seconds", 0)) for row in success), "summed_PDB_preparation_seconds": sum(float(row.get("input_preparation_seconds", 0)) for row in success), "summed_subprocess_seconds": sum(float(row.get("subprocess_seconds", 0)) for row in success), "peak_worker_RSS_mb": max(float(row.get("peak_RSS_mb", 0)) for row in success), "node_shared_disk_read_bytes_delta_approx": status["node_shared_disk_read_bytes_delta_approx"], "node_shared_disk_write_bytes_delta_approx": status["node_shared_disk_write_bytes_delta_approx"]})
    write_tsv(P6 / "parallel_mdlovofit_summary.tsv", summary)
    best = max(summary, key=lambda row: float(row["node_trajectories_per_hour"]))
    write_text(P6 / "parallel_MDLovoFit_report.md", "\n".join(["# MDLovoFit centering remediation and parallel throughput", "", f"- PDB-range remediation success: `{len(remediation_success)}/45`.", f"- Maximum ordinary-Kabsch RMSD change after common centering: `{max(float(row['ordinary_Kabsch_RMSD_invariance_max_abs_A']) for row in remediation_success):.3e} A`.", "- The original serial deployment is preserved and unchanged.", f"- Best observed node throughput: `{float(best['node_trajectories_per_hour']):.1f}` trajectories/hour with `{best['workers']}` workers.", "- Decode, PDB preparation and subprocess times are reported separately. Node-wide disk-sector deltas are included only as an I/O saturation proxy.", "- Parallel node throughput is not compared with single-trajectory loaded-analysis latency as if they were the same contract."]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "remediate", "parallel", "finalize"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.mode == "register":
        register()
    elif args.mode == "remediate":
        remediate()
    elif args.mode == "parallel":
        if args.workers is None:
            parser.error("--workers is required")
        parallel(args.workers)
    else:
        finalize()


if __name__ == "__main__":
    main()
