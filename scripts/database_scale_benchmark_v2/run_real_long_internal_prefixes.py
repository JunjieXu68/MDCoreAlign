#!/usr/bin/env python3
"""Run exact, CPU-throughput and GPU-throughput on the locked M3 prefixes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Callable

import numpy as np

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.database_scale_benchmark_v2.common import (
    M_ROOT,
    ROOT,
    read_tsv,
    write_json,
    write_tsv,
)
from scripts.database_scale_benchmark_v2.run_real_long_mdlovofit_addendum import (
    MANIFEST,
    _safe_id,
)


CHECKPOINTS = M_ROOT / "real_long_internal_checkpoints"
METHODS = (
    "CPU_exact_sequential_lean",
    "CPU_throughput",
    "GPU_throughput_float32_Horn",
)
CONTRACTS = (
    "A_analysis_metrics_transforms_rmsd_rmsf",
    "B_materialised_aligned_PDB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "finalize"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _load(task: dict[str, str]) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    with np.load(ROOT / task["cache_path"], allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"], dtype=np.float64)
        chain_ids = np.asarray(archive["chain_ids"])
    seconds = time.perf_counter() - start
    prefix = int(task["prefix_frames"])
    values = np.ascontiguousarray(coords[:prefix])
    if values.shape != (prefix, int(task["n_residues"]), 3):
        raise RuntimeError("cached coordinates do not match the locked prefix shape")
    if chain_ids.shape != (values.shape[1],):
        raise RuntimeError("cached chain IDs do not match the CA selection")
    if not np.isfinite(values).all():
        raise RuntimeError("cached coordinates contain non-finite values")
    return values, chain_ids, seconds


def _solver(
    method: str,
    coords: np.ndarray,
    chain_ids: np.ndarray,
    *,
    store_aligned: bool,
) -> tuple[Callable[[], Any], dict[str, object]]:
    if method == "CPU_exact_sequential_lean":
        return (
            lambda: solve_exact_contract(
                coords,
                chain_ids=chain_ids,
                representative_count=min(2_000, coords.shape[0]),
                chunk_size=500,
                store_aligned=store_aligned,
            ),
            {"representative_count": min(2_000, coords.shape[0]), "update_rounds": "exact"},
        )
    if method == "CPU_throughput":
        return (
            lambda: solve_fast_global_alignment(
                coords,
                chain_ids=chain_ids,
                representative_count=min(256, coords.shape[0]),
                update_rounds=4,
                chunk_size=500,
                store_aligned=store_aligned,
                compute_confidence=False,
            ),
            {"representative_count": min(256, coords.shape[0]), "update_rounds": 4},
        )
    worker = PersistentGPUWorker.create(
        chain_ids=chain_ids,
        representative_count=min(256, coords.shape[0]),
        update_rounds=4,
        chunk_size=500,
        precision="float32",
        rotation_solver="horn",
        store_aligned=store_aligned,
    )
    return (
        lambda: worker.throughput(coords),
        {"representative_count": min(256, coords.shape[0]), "update_rounds": 4},
    )


def _run_repeat(
    task: dict[str, str],
    *,
    method: str,
    contract: str,
    repeat_index: int,
    is_warmup: bool,
    coords: np.ndarray,
    chain_ids: np.ndarray,
    decode_seconds: float,
    solver: Callable[[], Any],
    solver_metadata: dict[str, object],
    workspace: Path,
) -> dict[str, Any]:
    store_aligned = contract == "B_materialised_aligned_PDB"
    base: dict[str, Any] = {
        **task,
        **solver_metadata,
        "method": method,
        "contract": contract,
        "repeat_index": repeat_index,
        "is_warmup": is_warmup,
        "shared_input_decode_seconds": decode_seconds,
    }
    try:
        start = time.perf_counter()
        result = solver()
        compute_seconds = time.perf_counter() - start
        write_seconds = 0.0
        output_bytes = 0
        if store_aligned:
            if result.aligned_coords is None:
                raise RuntimeError("materialised contract did not return aligned coordinates")
            output_path = workspace / f"{_safe_id(method)}_aligned.pdb"
            write_start = time.perf_counter()
            write_multimodel_ca_pdb_streaming(result.aligned_coords, output_path)
            write_seconds = time.perf_counter() - write_start
            output_bytes = output_path.stat().st_size
            output_path.unlink(missing_ok=True)
        loaded_seconds = compute_seconds + write_seconds
        timing = getattr(result, "timing", {})
        return {
            **base,
            "status": "success",
            "failure_class": "",
            "failure_reason": "",
            "method_compute_seconds": compute_seconds,
            "CPU_GPU_transfer_seconds": float(
                timing.get("CPU_GPU_transfer_total", 0.0)
            ),
            "coordinate_materialisation_output_write_seconds": write_seconds,
            "loaded_contract_seconds": loaded_seconds,
            "complete_end_to_end_seconds": decode_seconds + loaded_seconds,
            "unified_posthoc_RMSD_A": float(result.mean_rmsd),
            "mean_robust_RMSF_A": float(np.mean(result.robust_rmsf)),
            "aligned_PDB_bytes": output_bytes,
            "output_verified": (not store_aligned) or output_bytes > 0,
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "failure_class": "internal_method_failure",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
        }


def _checkpoint_path(
    task: dict[str, str], method: str, contract: str, repeat_index: int
) -> Path:
    return CHECKPOINTS / (
        f"{_safe_id(task['system'])}__{task['prefix_label']}__"
        f"{_safe_id(method)}__{contract[0]}__r{repeat_index}.json"
    )


def run(limit: int | None, rerun: bool) -> None:
    tasks = read_tsv(MANIFEST)
    if limit is not None:
        tasks = tasks[:limit]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    temp_root = M_ROOT / "temporary_workspaces"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real_long_internal_", dir=temp_root) as temporary:
        workspace = Path(temporary)
        for task_index, task in enumerate(tasks, start=1):
            coords, chain_ids, decode_seconds = _load(task)
            total_repeats = 1 + int(task["measured_repeats"])
            for method in METHODS:
                for contract in CONTRACTS:
                    pending = [
                        repeat_position
                        for repeat_position in range(total_repeats)
                        if rerun
                        or not _checkpoint_path(
                            task, method, contract, repeat_position - 1
                        ).is_file()
                    ]
                    if not pending:
                        continue
                    solver, solver_metadata = _solver(
                        method,
                        coords,
                        chain_ids,
                        store_aligned=contract == "B_materialised_aligned_PDB",
                    )
                    for repeat_position in pending:
                        repeat_index = repeat_position - 1
                        result = _run_repeat(
                            task,
                            method=method,
                            contract=contract,
                            repeat_index=repeat_index,
                            is_warmup=repeat_position == 0,
                            coords=coords,
                            chain_ids=chain_ids,
                            decode_seconds=decode_seconds,
                            solver=solver,
                            solver_metadata=solver_metadata,
                            workspace=workspace,
                        )
                        write_json(
                            _checkpoint_path(task, method, contract, repeat_index),
                            result,
                        )
            print(
                f"[internal real-long] {task_index}/{len(tasks)} "
                f"{task['system']} {task['prefix_label']}",
                flush=True,
            )


def _timing_summary(values: list[float], prefix: str) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        f"median_{prefix}": float(np.median(data)),
        f"q1_{prefix}": float(np.quantile(data, 0.25)),
        f"q3_{prefix}": float(np.quantile(data, 0.75)),
        f"iqr_{prefix}": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        f"min_{prefix}": float(np.min(data)),
        f"max_{prefix}": float(np.max(data)),
    }


def finalize() -> None:
    tasks = read_tsv(MANIFEST)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in tasks:
        for method in METHODS:
            for contract in CONTRACTS:
                for repeat_position in range(1 + int(task["measured_repeats"])):
                    checkpoint = _checkpoint_path(
                        task, method, contract, repeat_position - 1
                    )
                    if checkpoint.is_file():
                        rows.append(json.loads(checkpoint.read_text()))
                    else:
                        missing.append(checkpoint.name)
    if missing:
        raise RuntimeError(f"internal real-long panel has {len(missing)} missing checkpoints")
    write_tsv(M_ROOT / "internal_real_long_prefixes_raw_repeats.tsv", rows)
    measured = [row for row in rows if not row["is_warmup"]]
    summaries: list[dict[str, Any]] = []
    for task in tasks:
        for method in METHODS:
            for contract in CONTRACTS:
                group = [
                    row
                    for row in measured
                    if row["system"] == task["system"]
                    and row["prefix_label"] == task["prefix_label"]
                    and row["method"] == method
                    and row["contract"] == contract
                ]
                success = [row for row in group if row["status"] == "success"]
                summary: dict[str, Any] = {
                    **task,
                    "method": method,
                    "contract": contract,
                    "measured_completed": len(group),
                    "success": len(success),
                    "failed": len(group) - len(success),
                    "completion_fraction": len(success) / max(1, len(group)),
                }
                if success:
                    for field in (
                        "shared_input_decode_seconds",
                        "method_compute_seconds",
                        "CPU_GPU_transfer_seconds",
                        "coordinate_materialisation_output_write_seconds",
                        "loaded_contract_seconds",
                        "complete_end_to_end_seconds",
                    ):
                        summary.update(
                            _timing_summary(
                                [float(row[field]) for row in success], field
                            )
                        )
                    summary["median_unified_posthoc_RMSD_A"] = float(
                        np.median([float(row["unified_posthoc_RMSD_A"]) for row in success])
                    )
                    summary["median_mean_robust_RMSF_A"] = float(
                        np.median([float(row["mean_robust_RMSF_A"]) for row in success])
                    )
                summaries.append(summary)
    write_tsv(M_ROOT / "internal_real_long_prefixes.tsv", summaries)
    write_json(
        M_ROOT / "internal_real_long_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settings": len(tasks),
            "rows": len(measured),
            "success": sum(row["status"] == "success" for row in measured),
            "failed": sum(row["status"] != "success" for row in measured),
        },
    )


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run(args.limit, args.rerun)
    else:
        finalize()


if __name__ == "__main__":
    main()
