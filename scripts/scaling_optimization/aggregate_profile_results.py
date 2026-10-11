#!/usr/bin/env python
"""Aggregate Phase 1 raw JSON profiles and build the compact handoff bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import difflib
import json
import math
from pathlib import Path
import pstats
import shutil
import statistics
import tarfile
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v1"
STAGE_PREFIX = "stage_"


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Load every raw run, preserving malformed records as failures."""

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "profiling/raw_runs").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            payload["_raw_path"] = path.relative_to(PROJECT_ROOT).as_posix()
            rows.append(payload)
        except Exception as exc:
            rows.append(
                {
                    "run_id": path.stem,
                    "status": "malformed_json",
                    "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
                    "_raw_path": path.relative_to(PROJECT_ROOT).as_posix(),
                }
            )
    return rows


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def quantile(values: Iterable[float], fraction: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, fraction))


def median_iqr(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if not numbers:
        return None, None, None
    return statistics.median(numbers), quantile(numbers, 0.25), quantile(numbers, 0.75)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    """Write a stable TSV, retaining a header for an empty table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Flatten run metadata and stage timings into one wide table."""

    rows: list[dict[str, object]] = []
    all_stages = sorted(
        {
            stage
            for run in runs
            for stage in (run.get("stages_seconds") or {})
        }
    )
    for run in runs:
        stages = run.get("stages_seconds") or {}
        total = finite(stages.get("total"))
        row: dict[str, object] = {
            "run_id": run.get("run_id", ""),
            "dataset_id": run.get("dataset_id", ""),
            "data_semantics": run.get("data_semantics", ""),
            "n_frames": run.get("n_frames", ""),
            "n_residues": run.get("n_residues", ""),
            "method": run.get("method", ""),
            "config": run.get("config", ""),
            "write_aligned_trajectory": run.get("write_aligned_trajectory", ""),
            "repeat_id": run.get("repeat_id", ""),
            "warmup": run.get("warmup", ""),
            "status": run.get("status", ""),
            "total_seconds": total if total is not None else "",
            "algorithm_runtime_seconds": (
                run.get("details", {}).get("algorithm_runtime_seconds", "")
                if isinstance(run.get("details"), dict)
                else ""
            ),
            "fresh_process_wall_seconds": run.get("fresh_process_wall_seconds", ""),
            "failure_or_skip_reason": run.get("failure_or_skip_reason", ""),
            "raw_json": run.get("_raw_path", ""),
        }
        for stage in all_stages:
            value = finite(stages.get(stage))
            row[f"{STAGE_PREFIX}{stage}_seconds"] = value if value is not None else ""
            row[f"{STAGE_PREFIX}{stage}_fraction"] = (
                value / total if value is not None and total and stage != "total" else ""
            )
        rows.append(row)
    return rows


def memory_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Flatten peak RSS and major-array estimates."""

    rows: list[dict[str, object]] = []
    for run in runs:
        memory = run.get("memory") or {}
        arrays = run.get("largest_arrays") or []
        largest = arrays[0] if arrays else {}
        rows.append(
            {
                "run_id": run.get("run_id", ""),
                "dataset_id": run.get("dataset_id", ""),
                "data_semantics": run.get("data_semantics", ""),
                "n_frames": run.get("n_frames", ""),
                "n_residues": run.get("n_residues", ""),
                "method": run.get("method", ""),
                "config": run.get("config", ""),
                "repeat_id": run.get("repeat_id", ""),
                "warmup": run.get("warmup", ""),
                "status": run.get("status", ""),
                "initial_rss_mb": (
                    finite(memory.get("initial_rss_bytes")) / (1024**2)
                    if finite(memory.get("initial_rss_bytes")) is not None
                    else ""
                ),
                "final_rss_mb": (
                    finite(memory.get("final_rss_bytes")) / (1024**2)
                    if finite(memory.get("final_rss_bytes")) is not None
                    else ""
                ),
                "peak_rss_mb": memory.get("peak_rss_mb", ""),
                "rss_delta_mb": (
                    finite(memory.get("rss_delta_bytes")) / (1024**2)
                    if finite(memory.get("rss_delta_bytes")) is not None
                    else ""
                ),
                "peak_gpu_allocated_mb": (
                    finite(memory.get("peak_gpu_allocated_bytes")) / (1024**2)
                    if finite(memory.get("peak_gpu_allocated_bytes")) is not None
                    else ""
                ),
                "peak_gpu_reserved_mb": (
                    finite(memory.get("peak_gpu_reserved_bytes")) / (1024**2)
                    if finite(memory.get("peak_gpu_reserved_bytes")) is not None
                    else ""
                ),
                "largest_array_name": largest.get("name", ""),
                "largest_array_shape": json.dumps(largest.get("shape", [])),
                "largest_array_dtype": largest.get("dtype", ""),
                "largest_array_mb": (
                    finite(largest.get("nbytes")) / (1024**2)
                    if finite(largest.get("nbytes")) is not None
                    else ""
                ),
                "largest_array_is_view": largest.get("is_view", ""),
            }
        )
    return rows


def measured_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return successful non-warmup matrix rows, excluding profiler controls."""

    return [
        run
        for run in runs
        if run.get("status") == "success"
        and not run.get("warmup", False)
        and not str(run.get("run_id", "")).startswith("cprofile_")
    ]


def scaling_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Aggregate median/IQR timing and scaling metrics by configuration."""

    successful = measured_runs(runs)
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for run in successful:
        key = (
            run.get("dataset_id"),
            run.get("data_semantics"),
            run.get("n_frames"),
            run.get("n_residues"),
            run.get("method"),
            run.get("config"),
            run.get("write_aligned_trajectory"),
        )
        groups.setdefault(key, []).append(run)
    rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        totals = [
            value
            for value in (
                finite(run.get("stages_seconds", {}).get("total")) for run in group
            )
            if value is not None
        ]
        algorithms = [
            value
            for value in (
                finite(run.get("details", {}).get("algorithm_runtime_seconds")) for run in group
            )
            if value is not None
        ]
        peaks = [
            value
            for value in (
                finite(run.get("memory", {}).get("peak_rss_mb")) for run in group
            )
            if value is not None
        ]
        fresh_process_times = [
            value
            for value in (
                finite(run.get("fresh_process_wall_seconds")) for run in group
            )
            if value is not None
        ]
        total_median, total_q1, total_q3 = median_iqr(totals)
        algorithm_median, algorithm_q1, algorithm_q3 = median_iqr(algorithms)
        fresh_median, fresh_q1, fresh_q3 = median_iqr(fresh_process_times)
        peak_median, _, _ = median_iqr(peaks)
        n_frames = int(key[2])
        n_residues = int(key[3])
        rows.append(
            {
                "dataset_id": key[0],
                "data_semantics": key[1],
                "n_frames": n_frames,
                "n_residues": n_residues,
                "method": key[4],
                "config": key[5],
                "write_aligned_trajectory": key[6],
                "n_successful_repeats": len(group),
                "median_total_seconds": total_median if total_median is not None else "",
                "total_q1_seconds": total_q1 if total_q1 is not None else "",
                "total_q3_seconds": total_q3 if total_q3 is not None else "",
                "median_fresh_process_seconds": fresh_median if fresh_median is not None else "",
                "fresh_process_q1_seconds": fresh_q1 if fresh_q1 is not None else "",
                "fresh_process_q3_seconds": fresh_q3 if fresh_q3 is not None else "",
                "median_algorithm_seconds": algorithm_median if algorithm_median is not None else "",
                "algorithm_q1_seconds": algorithm_q1 if algorithm_q1 is not None else "",
                "algorithm_q3_seconds": algorithm_q3 if algorithm_q3 is not None else "",
                "frames_per_second": (
                    n_frames / total_median if total_median and total_median > 0 else ""
                ),
                "fresh_process_frames_per_second": (
                    n_frames / fresh_median if fresh_median and fresh_median > 0 else ""
                ),
                "residue_frames_per_second": (
                    n_frames * n_residues / total_median
                    if total_median and total_median > 0
                    else ""
                ),
                "median_peak_rss_mb": peak_median if peak_median is not None else "",
                "speedup_vs_mdlovofit": "",
                "speedup_vs_default_mdcorealign": "",
                "log_log_frame_scaling_slope": "",
            }
        )
    add_speedups(rows)
    add_scaling_slopes(rows)
    return rows


def add_speedups(rows: list[dict[str, object]]) -> None:
    """Add same-dataset/same-frame speed ratios to aggregated rows."""

    lookup = {
        (
            row["dataset_id"],
            row["n_frames"],
            row["config"],
        ): finite(row["median_total_seconds"])
        for row in rows
    }
    for row in rows:
        total = finite(row["median_total_seconds"])
        if total is None or total <= 0:
            continue
        mdlovo = lookup.get((row["dataset_id"], row["n_frames"], "mdlovofit"))
        default = lookup.get(
            (row["dataset_id"], row["n_frames"], "mdcorealign_default_no_write")
        )
        if mdlovo is not None:
            row["speedup_vs_mdlovofit"] = mdlovo / total
        if default is not None:
            row["speedup_vs_default_mdcorealign"] = default / total


def add_scaling_slopes(rows: list[dict[str, object]]) -> None:
    """Estimate log-log frame scaling slopes for each dataset/configuration."""

    groups: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["dataset_id"], row["config"]), []).append(row)
    for group in groups.values():
        valid = [
            row
            for row in group
            if finite(row["median_total_seconds"]) not in {None, 0.0}
            and int(row["n_frames"]) > 0
        ]
        if len(valid) < 3:
            continue
        x = np.log(np.asarray([int(row["n_frames"]) for row in valid], dtype=float))
        y = np.log(np.asarray([float(row["median_total_seconds"]) for row in valid]))
        slope = float(np.polyfit(x, y, 1)[0])
        for row in group:
            row["log_log_frame_scaling_slope"] = slope


def failure_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Keep every timeout, failure and unavailable execution path."""

    return [
        {
            "run_id": run.get("run_id", ""),
            "dataset_id": run.get("dataset_id", ""),
            "n_frames": run.get("n_frames", ""),
            "n_residues": run.get("n_residues", ""),
            "method": run.get("method", ""),
            "config": run.get("config", ""),
            "repeat_id": run.get("repeat_id", ""),
            "warmup": run.get("warmup", ""),
            "status": run.get("status", ""),
            "failure_or_skip_reason": run.get("failure_or_skip_reason", ""),
            "raw_json": run.get("_raw_path", ""),
        }
        for run in runs
        if run.get("status") != "success"
    ]


def bottleneck_summary(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Rank runtime stages by median fraction of end-to-end time."""

    fractions: dict[str, list[float]] = {}
    seconds: dict[str, list[float]] = {}
    for run in measured_runs(runs):
        stages = run.get("stages_seconds") or {}
        total = finite(stages.get("total"))
        if total is None or total <= 0:
            continue
        for stage, raw_value in stages.items():
            value = finite(raw_value)
            if value is None or value <= 0 or stage == "total":
                continue
            fractions.setdefault(stage, []).append(value / total)
            seconds.setdefault(stage, []).append(value)
    rows = []
    for stage, values in fractions.items():
        rows.append(
            {
                "stage": stage,
                "median_fraction_of_total": statistics.median(values),
                "median_seconds_when_present": statistics.median(seconds[stage]),
                "n_runs": len(values),
                "amdahl_max_speedup_if_stage_eliminated": 1.0 / max(
                    1e-12, 1.0 - statistics.median(values)
                ),
            }
        )
    return sorted(rows, key=lambda row: float(row["median_fraction_of_total"]), reverse=True)


def memory_consumer_summary(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Rank material arrays by maximum and median observed estimated size."""

    groups: dict[str, list[float]] = {}
    shapes: dict[str, set[str]] = {}
    for run in measured_runs(runs):
        for array in run.get("largest_arrays") or []:
            name = str(array.get("name", "unknown"))
            nbytes = finite(array.get("nbytes"))
            if nbytes is None:
                continue
            groups.setdefault(name, []).append(nbytes / (1024**2))
            shapes.setdefault(name, set()).add(str(array.get("shape", "")))
    rows = [
        {
            "array_or_allocation": name,
            "max_observed_mb": max(values),
            "median_observed_mb": statistics.median(values),
            "n_observations": len(values),
            "observed_shapes": "; ".join(sorted(shapes[name])[:8]),
        }
        for name, values in groups.items()
    ]
    return sorted(rows, key=lambda row: float(row["max_observed_mb"]), reverse=True)


def profiler_rows(root: Path) -> list[dict[str, object]]:
    """Extract top cumulative/self functions from the cProfile binary."""

    path = root / "profiling/cprofile_default_500.pstats"
    if not path.exists():
        return []
    stats = pstats.Stats(str(path))
    rows: list[dict[str, object]] = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_time, cumulative_time, _ = values
        rows.append(
            {
                "function": function,
                "file": filename,
                "line": line,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_time_seconds": self_time,
                "cumulative_time_seconds": cumulative_time,
            }
        )
    return sorted(rows, key=lambda row: float(row["cumulative_time_seconds"]), reverse=True)[:100]


def method_specific_hotspots(
    runs: list[dict[str, Any]],
    *,
    dataset_id: str,
    n_frames: int,
    config: str,
    limit: int = 5,
) -> list[tuple[str, float, float]]:
    """Return median stage seconds and fractions for one measured configuration."""

    matching = [
        run
        for run in measured_runs(runs)
        if run.get("dataset_id") == dataset_id
        and int(run.get("n_frames", -1)) == n_frames
        and run.get("config") == config
    ]
    stage_names = sorted(
        {
            stage
            for run in matching
            for stage in (run.get("stages_seconds") or {})
            if stage != "total"
        }
    )
    rows: list[tuple[str, float, float]] = []
    for stage in stage_names:
        values = [
            value
            for value in (
                finite((run.get("stages_seconds") or {}).get(stage)) for run in matching
            )
            if value is not None and value > 0.0
        ]
        totals = [
            value
            for value in (
                finite((run.get("stages_seconds") or {}).get("total")) for run in matching
            )
            if value is not None and value > 0.0
        ]
        if not values or not totals:
            continue
        seconds = statistics.median(values)
        total = statistics.median(totals)
        rows.append((stage, seconds, seconds / total))
    return sorted(rows, key=lambda row: row[1], reverse=True)[:limit]


def build_decision_report(
    runs: list[dict[str, Any]],
    bottlenecks: list[dict[str, object]],
    memory_consumers: list[dict[str, object]],
    scaling: list[dict[str, object]],
) -> str:
    """Build the required measured Phase 1 decision report."""

    failures = failure_rows(runs)
    top_runtime = bottlenecks[:5]
    top_memory = memory_consumers[:5]
    primary_scaling = [
        row
        for row in scaling
        if row["dataset_id"] == "4oelA02_transition_aligned"
        and row["config"] in {
            "mdcorealign_default_no_write",
            "mdcorealign_long_no_write",
            "mdlovofit",
            "theseus",
            "input_decode_only",
            "write_only",
        }
    ]
    stage_lines = [
        f"{index}. `{row['stage']}`: median {100.0 * float(row['median_fraction_of_total']):.1f}% "
        f"of end-to-end runtime when present; idealized Amdahl limit "
        f"{float(row['amdahl_max_speedup_if_stage_eliminated']):.2f}x."
        for index, row in enumerate(top_runtime, start=1)
    ] or ["No successful measured runs were available."]
    memory_lines = [
        f"{index}. `{row['array_or_allocation']}`: max observed "
        f"{float(row['max_observed_mb']):.2f} MiB; median "
        f"{float(row['median_observed_mb']):.2f} MiB."
        for index, row in enumerate(top_memory, start=1)
    ] or ["No array estimates were available."]
    scaling_lines = [
        f"- `{row['config']}`: frames={row['n_frames']}, median in-child end-to-end="
        f"{float(row['median_total_seconds']):.4f} s, fresh-process="
        f"{float(row['median_fresh_process_seconds']):.4f} s, FPS="
        f"{float(row['frames_per_second']):.1f}, peak RSS="
        f"{float(row['median_peak_rss_mb']):.1f} MiB."
        for row in primary_scaling
        if finite(row["median_total_seconds"]) is not None
        and finite(row["median_fresh_process_seconds"]) is not None
    ]
    method_hotspot_lines: list[str] = []
    hotspot_configs = [
        ("MDCoreAlign default, no trajectory write", "mdcorealign_default_no_write"),
        ("MDCoreAlign long-mode, no trajectory write", "mdcorealign_long_no_write"),
        ("MDCoreAlign long-mode, with trajectory write", "mdcorealign_long_write"),
        ("MDLovoFit external workflow", "mdlovofit"),
        ("Write-only control", "write_only"),
    ]
    for label, config in hotspot_configs:
        hotspots = method_specific_hotspots(
            runs,
            dataset_id="4oelA02_transition_aligned",
            n_frames=10000,
            config=config,
        )
        if not hotspots:
            continue
        summary = "; ".join(
            f"`{stage}` {seconds:.3f} s ({100.0 * fraction:.1f}%)"
            for stage, seconds, fraction in hotspots
        )
        method_hotspot_lines.append(f"- **{label}:** {summary}.")
    failed_summary = ", ".join(
        f"{status}={sum(row['status'] == status for row in failures)}"
        for status in sorted({str(row["status"]) for row in failures})
    ) or "none"
    return "\n".join(
        [
            "# Phase 1 Profiling Decision Report",
            "",
            f"Generated: {timestamp()}",
            "",
            "## Scope and Data Semantics",
            "",
            "This report profiles the frozen current implementation. No FastScan, ANN/Faiss, "
            "or scientific algorithm changes were made. The 10,000-frame 4oelA02 input is a "
            "transition-aligned synthetic long-trajectory scaling diagnostic, not a physically "
            "continuous MD trajectory. Real single-replicate mdCATH inputs available locally "
            "contain approximately 500 frames.",
            "",
            "## Five Largest Runtime Bottlenecks",
            "",
            *stage_lines,
            "",
            "These rankings aggregate heterogeneous output contracts; the method/config-specific "
            "rows in `stage_timing.tsv` are the source of truth. Native MDLovoFit/THESEUS fitting "
            "and native output writing cannot be split inside their subprocesses.",
            "",
            "### Method-specific 10,000-frame hotspots",
            "",
            *method_hotspot_lines,
            "",
            "`unattributed_algorithm_overhead` is measured wall time inside the algorithm call "
            "that is not exposed by the existing callback, including result-object construction, "
            "final refits, and Python orchestration. It is a profiling boundary limitation, not "
            "an assumed operation.",
            "",
            "## Five Largest Memory Consumers",
            "",
            *memory_lines,
            "",
            "Peak RSS includes the profiled process tree. Python `tracemalloc` excludes most NumPy "
            "native buffers, so RSS and explicit array sizes are primary.",
            "",
            "## Scaling",
            "",
            *scaling_lines,
            "",
            "The in-child end-to-end timer excludes Python interpreter/import startup; "
            "`median_fresh_process_seconds` in `method_scaling.tsv` includes those costs. "
            "Speedup fields use the in-child end-to-end timer so fixed process startup does not "
            "dominate short trajectories.",
            "",
            "Frame-scaled costs are trajectory decode, per-frame iterative fitting in default mode, "
            "batch fitting/streaming RMSF in long-mode, transform materialisation, and PDB writing. "
            "Residue-scaled costs include coordinate arrays, residual/weight vectors, covariance "
            "construction, and PDB text volume. No full NxN residue-distance matrix is used in the "
            "default or uniform long-mode paths.",
            "",
            "## I/O-Bound Costs",
            "",
            "NPZ decode and multi-model PDB writing are directly measured controls. `/storage` is "
            "an ext4 filesystem on rotational RAID5, so writing large PDB trajectories can be "
            "storage- and text-format-bound. External tools also require PDB encode/decode overhead.",
            "",
            "## CPU/GPU Assessment",
            "",
            "No current GPU execution path was found. CUDA hardware and Nsight tools are installed, "
            "but there are no CuPy/PyTorch/JAX/CUDA kernels in the active MDCoreAlign packages. "
            "A CPU/GPU crossover, transfer-bound/launch-bound/compute-bound classification, and "
            "GPU memory values are therefore unavailable rather than inferred.",
            "",
            "## Reference Selection and Frame Search",
            "",
            "Reference selection is an O(1) array view and is not a meaningful hotspot. Uniform "
            "long-mode representative selection is also negligible relative to iterative weight "
            "estimation and fitting. Descriptor/ANN paths were not profiled or changed in this phase.",
            "",
            "## Recommended Optimization Order",
            "",
            "1. Reduce or stream full aligned-coordinate materialisation where the output contract "
            "does not require an aligned trajectory.",
            "2. Reduce default-mode per-frame Python/object overhead only after preserving the exact "
            "default scientific output contract.",
            "3. Separate robust RMSF from the ordinary-control alignment so users do not pay for "
            "supplementary control outputs unless requested.",
            "4. Stream or replace CA-only multi-model PDB output for large apply workflows while "
            "keeping external-tool conversion timing separate.",
            "5. Consider a GPU path only after a CPU batch/chunk contract exists; current evidence "
            "cannot justify GPU-first work.",
            "",
            "## Attractive but Currently Irrelevant Optimizations",
            "",
            "- ANN/Faiss is not justified by Phase 1 because current uniform representative selection "
            "is not a leading measured cost and descriptor-guided selection is explicitly out of scope.",
            "- GPU kernel tuning is impossible without a current GPU implementation.",
            "- Optimizing reference-frame lookup would target an O(1), near-zero stage.",
            "- Recompressing cached NPZ files would optimize a benchmark cache rather than normal CLI "
            "trajectory decode and is not a scientific-path improvement.",
            "",
            "## Failures, Timeouts and Unavailable Tools",
            "",
            f"Recorded non-success rows: {len(failures)} ({failed_summary}).",
            "The machine-readable details are in `failures_and_timeouts.tsv`. `py-spy` was not "
            "installed, so no flamegraph was fabricated. Nsight was marked not applicable because "
            "there is no active GPU path.",
            "The cProfile control also enabled `tracemalloc`, which substantially perturbed wall "
            "time. Its function ranking is diagnostic only and is not used for benchmark speedups.",
            "",
        ]
    )


def copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text())


def build_long_mode_extract() -> str:
    return """# Current Long-Mode Extract

Source: `src/mdsuperpose/long_mode.py`.

Call path:

`solve_long_trajectory_alignment` -> uniform representative-frame selection ->
`solve_trajectory_robust_alignment` on the representative subset -> mean frozen
soft weights -> chunked `weighted_kabsch_batch` for all frames -> residual
accumulation -> robust and ordinary `StreamingRMSFAccumulator`.

The mode stores per-frame rotations, translations, RMSD and weighted RMSD.
Aligned coordinates are optional (`store_aligned=False` by default), but each
chunk creates robust and ordinary aligned arrays. Representative-frame fitting
still builds the default result-object graph for the selected subset.

The implementation is approximate and remains separate from default mode.
"""


def build_gpu_extract() -> str:
    return """# Current GPU Path Extract

Static search command:

```bash
rg -n --glob '*.py' '\\b(cupy|torch|jax|cuda|gpu|GPU|device_to_host|host_to_device)\\b' src scripts tests
```

No active GPU compute path was found in `src/mdcorealign`, `src/mdsuperpose`, or
`src/mdstablecore`. The system has an NVIDIA Quadro P5000 and CUDA/Nsight tools,
but hardware/tool availability is not an implementation. GPU transfer, compute,
memory and crossover fields are therefore unavailable in Phase 1.
"""


def write_real_manifest(path: Path) -> None:
    rows = [
        {
            "dataset_id": "4a18O00_real_single_replicate",
            "path": (
                "results/mdcath_benchmark/long_continuous_trajectory/"
                "continuous_inputs/4a18O00__T320__R1_ca_coords.npz"
            ),
            "n_frames": 501,
            "n_residues": 134,
            "semantics": "real_continuous_single_replicate_md",
            "eligible_real_10k": 0,
            "notes": "320 K replicate 1; local cache.",
        },
        {
            "dataset_id": "4cgyA01_real_single_replicate",
            "path": (
                "results/mdcath_benchmark/long_continuous_trajectory/"
                "continuous_inputs/4cgyA01__T320__R1_ca_coords.npz"
            ),
            "n_frames": 501,
            "n_residues": 176,
            "semantics": "real_continuous_single_replicate_md",
            "eligible_real_10k": 0,
            "notes": "320 K replicate 1; local cache.",
        },
        {
            "dataset_id": "4cu9A00_real_single_replicate",
            "path": (
                "results/mdcath_benchmark/long_continuous_trajectory/"
                "continuous_inputs/4cu9A00__T320__R1_ca_coords.npz"
            ),
            "n_frames": 501,
            "n_residues": 173,
            "semantics": "real_continuous_single_replicate_md",
            "eligible_real_10k": 0,
            "notes": "320 K replicate 1; local cache.",
        },
        {
            "dataset_id": "3a2pA00_real_single_replicate",
            "path": (
                "results/mdcath_benchmark/long_continuous_trajectory/"
                "continuous_inputs/3a2pA00__T320__R1_ca_coords.npz"
            ),
            "n_frames": 500,
            "n_residues": 483,
            "semantics": "real_continuous_single_replicate_md",
            "eligible_real_10k": 0,
            "notes": "Large-residue control; 320 K replicate 1.",
        },
        {
            "dataset_id": "4oelA02_transition_aligned",
            "path": (
                "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
                "runs/4oelA02/10000/transition_aligned_coords.npz"
            ),
            "n_frames": 10000,
            "n_residues": 153,
            "semantics": "transition_aligned_synthetic_long_trajectory",
            "eligible_real_10k": 0,
            "notes": "Computational scaling diagnostic only; no biological continuity claim.",
        },
    ]
    write_tsv(path, rows)


def build_patch(paths: list[Path]) -> str:
    """Build a unified add-file patch when no valid Git HEAD exists."""

    chunks: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        lines = path.read_text(errors="replace").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                [],
                lines,
                fromfile="/dev/null",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def build_handoff(
    root: Path,
    profiler: list[dict[str, object]],
) -> Path:
    """Create the compact, credential-free Phase 1 handoff bundle."""

    handoff = root / "handoff_to_chatgpt"
    handoff.mkdir(parents=True, exist_ok=True)
    mapping = [
        (
            PROJECT_ROOT / "docs/scaling_optimization_v1/00_repository_inventory.md",
            handoff / "repository_inventory.md",
        ),
        (
            PROJECT_ROOT / "docs/scaling_optimization_v1/01_current_architecture_map.md",
            handoff / "architecture_map.md",
        ),
        (
            PROJECT_ROOT / "docs/scaling_optimization_v1/02_profiling_decision_report.md",
            handoff / "profiling_decision_report.md",
        ),
        (root / "profiling/stage_timing.tsv", handoff / "stage_timing.tsv"),
        (root / "profiling/memory_scaling.tsv", handoff / "memory_scaling.tsv"),
        (root / "baseline/smoke_benchmark.tsv", handoff / "baseline_summary.tsv"),
    ]
    for source, destination in mapping:
        if source.exists():
            copy_text(source, destination)
    write_tsv(handoff / "profiler_top_functions.tsv", profiler)
    (handoff / "current_long_mode_extract.md").write_text(build_long_mode_extract())
    (handoff / "current_gpu_path_extract.md").write_text(build_gpu_extract())
    write_real_manifest(handoff / "real_long_trajectory_manifest.tsv")

    changed_files = [
        PROJECT_ROOT / "scripts/scaling_optimization/profile_support.py",
        PROJECT_ROOT / "scripts/scaling_optimization/capture_phase0_baseline.py",
        PROJECT_ROOT / "scripts/scaling_optimization/run_profile_matrix.py",
        PROJECT_ROOT / "scripts/scaling_optimization/aggregate_profile_results.py",
        PROJECT_ROOT / "docs/scaling_optimization_v1/00_repository_inventory.md",
        PROJECT_ROOT / "docs/scaling_optimization_v1/01_current_architecture_map.md",
        PROJECT_ROOT / "docs/scaling_optimization_v1/02_profiling_decision_report.md",
        PROJECT_ROOT / "tests/scaling_optimization/test_profile_support.py",
    ]
    (handoff / "git_diff.patch").write_text(build_patch(changed_files))
    readme = f"""# MDCoreAlign Phase 0/1 Performance Handoff

Generated: {timestamp()}

This compact bundle contains repository inventory, architecture mapping, raw
stage/memory summaries, top cProfile functions and the measured decision report.
It excludes trajectories, caches, credentials, private keys and binary evidence.

Reproduce from repository root:

```bash
python scripts/scaling_optimization/capture_phase0_baseline.py
PYTHONPATH=src:. python scripts/scaling_optimization/run_profile_matrix.py \\
  --profile-stages --frame-counts 12,100,500,1000,5000,10000 \\
  --repeats 3 --warmup --include-secondary
PYTHONPATH=src:. python scripts/scaling_optimization/aggregate_profile_results.py
```

The 10k input is transition-aligned synthetic scaling data, not real continuous
MD. No active GPU path exists in the frozen baseline. Phase 1 stops before
algorithm optimisation.
"""
    (handoff / "README.md").write_text(readme)

    archive = root / f"handoff_to_chatgpt_{datetime.now():%Y%m%d}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(handoff, arcname="handoff_to_chatgpt")
    return archive


def verify_integrity(root: Path) -> dict[str, object]:
    """Re-hash frozen evidence and compare it with the Phase 0 table."""

    path = root / "baseline/key_output_hashes.tsv"
    rows = list(csv.DictReader(path.open(), delimiter="\t")) if path.exists() else []
    mismatches = []
    import hashlib

    for row in rows:
        target = PROJECT_ROOT / row["path"]
        if not target.exists():
            mismatches.append({"path": row["path"], "reason": "missing"})
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            mismatches.append(
                {
                    "path": row["path"],
                    "reason": "hash_changed",
                    "expected": row["sha256"],
                    "observed": digest,
                }
            )
    payload = {
        "checked_at": timestamp(),
        "hash_count": len(rows),
        "mismatches": mismatches,
        "completed_evidence_unchanged": not mismatches,
        "status": "pass" if rows and not mismatches else "fail",
    }
    (root / "baseline/post_profile_integrity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    runs = load_runs(args.root)
    stages = stage_rows(runs)
    memory = memory_rows(runs)
    scaling = scaling_rows(runs)
    failures = failure_rows(runs)
    bottlenecks = bottleneck_summary(runs)
    consumers = memory_consumer_summary(runs)
    profiler = profiler_rows(args.root)

    profiling = args.root / "profiling"
    write_tsv(profiling / "stage_timing.tsv", stages)
    write_tsv(profiling / "memory_scaling.tsv", memory)
    write_tsv(profiling / "method_scaling.tsv", scaling)
    write_tsv(profiling / "failures_and_timeouts.tsv", failures)
    write_tsv(profiling / "runtime_bottleneck_ranking.tsv", bottlenecks)
    write_tsv(profiling / "memory_consumer_ranking.tsv", consumers)
    write_tsv(profiling / "profiler_top_functions.tsv", profiler)

    report = build_decision_report(runs, bottlenecks, consumers, scaling)
    report_path = PROJECT_ROOT / "docs/scaling_optimization_v1/02_profiling_decision_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    integrity = verify_integrity(args.root)
    archive = build_handoff(args.root, profiler)
    summary = {
        "generated_at": timestamp(),
        "raw_run_count": len(runs),
        "measured_success_count": len(measured_runs(runs)),
        "failure_or_unavailable_count": len(failures),
        "top_runtime_bottlenecks": bottlenecks[:5],
        "top_memory_consumers": consumers[:5],
        "baseline_integrity": integrity,
        "handoff_archive": archive.relative_to(PROJECT_ROOT).as_posix(),
        "phase_stop": "Phase 1 complete; no Phase 2+ algorithm work performed.",
    }
    (args.root / "profiling/aggregation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(report_path.relative_to(PROJECT_ROOT))
    print(archive.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
