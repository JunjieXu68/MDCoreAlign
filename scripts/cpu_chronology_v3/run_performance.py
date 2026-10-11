#!/usr/bin/env python
"""Run the locked compact one-thread v3 performance study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose.chronology_adaptive import AdaptiveRiskRule  # noqa: E402
from mdsuperpose.chronology_study import run_candidate, run_exact  # noqa: E402
from scripts.cpu_chronology_v3.run_quality_validation import PeakRSS  # noqa: E402
from scripts.cpu_chronology_v3.run_self_variability import load_case  # noqa: E402


OUT = ROOT / "results/cpu_chronology_v3"
PYTHON = Path(sys.executable).resolve()
METHODS = (
    "exact_sequential_f64_compact",
    "throughput_compact_f64",
    "D_block_64_v3",
    "C_correction_2_v3",
    "adaptive_C1_C2_D64_v3",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CHECKPOINT_ROOT = Path(tempfile.gettempdir()) / "mdcorealign_cpu_chronology_v3_performance"


def checkpoint_path(path: Path) -> Path:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_ROOT / path.name


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, row: dict[str, object]) -> None:
    path = checkpoint_path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def rule() -> AdaptiveRiskRule:
    payload = json.loads((OUT / "04_candidates/adaptive_rule.json").read_text())
    return AdaptiveRiskRule(
        **{f"c1_{key}": float(value) for key, value in payload["c1"].items()},
        **{f"c2_{key}": float(value) for key, value in payload["c2"].items()},
    )


def run_method(coords: np.ndarray, chains: np.ndarray, method: str):
    if method == METHODS[0]:
        return run_exact(coords, chains)
    return run_candidate(
        coords,
        chains,
        method,
        adaptive_rule=rule() if method == "adaptive_C1_C2_D64_v3" else None,
    )


def completion_keys(path: Path) -> set[tuple[str, str, int, str]]:
    checkpoint = checkpoint_path(path)
    if checkpoint.exists():
        path = checkpoint
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {
        (row["case_id"], row["method"], int(row["repeat"]), row["timing_kind"])
        for row in read_tsv(path)
    }


def commit_performance_tables() -> None:
    for name in (
        "fresh_long_repeats.tsv",
        "fresh_mdcath_128_repeats.tsv",
        "stage_decomposition_v3.tsv",
        "memory_v3.tsv",
    ):
        checkpoint = CHECKPOINT_ROOT / name
        if checkpoint.exists():
            write_tsv(OUT / "07_performance" / name, read_tsv(checkpoint))


def benchmark_panel(panel_path: Path, panel_name: str, repeats_path: Path) -> None:
    rows = read_tsv(panel_path)
    completed = completion_keys(repeats_path)
    stage_path = OUT / "07_performance/stage_decomposition_v3.tsv"
    memory_path = OUT / "07_performance/memory_v3.tsv"
    started = time.perf_counter()
    for case_index, row in enumerate(rows):
        coords, chains = load_case(row)
        method_order = list(METHODS[case_index % len(METHODS):] + METHODS[:case_index % len(METHODS)])
        for order_position, method in enumerate(method_order):
            pending = [repeat for repeat in range(1, 6) if (row["case_id"], method, repeat, "loaded_in_process") not in completed]
            if pending:
                run_method(coords, chains, method)
            for repeat in pending:
                with PeakRSS() as monitor:
                    result = run_method(coords, chains, method)
                base = {
                    "case_id": row["case_id"],
                    "family_id": row["family_id"],
                    "panel": panel_name,
                    "n_frames": int(row["n_frames"]),
                    "n_residues": int(row["n_residues"]),
                    "method": method,
                    "repeat": repeat,
                    "timing_kind": "loaded_in_process",
                    "loaded_runtime_seconds": result.runtime_seconds,
                    "fresh_process_wall_seconds": "",
                    "method_order_position": order_position,
                    "warmups": 1,
                    "cpu_threads": 1,
                    "cpu_affinity": ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
                    "cpu_affinity": ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
                    "ordinary_rmsf": False,
                    "aligned_coordinates": False,
                    "accepted_level": result.metadata.get("accepted_level", "fixed"),
                }
                append_tsv(repeats_path, base)
                append_tsv(
                    memory_path,
                    {
                        "case_id": row["case_id"],
                        "panel": panel_name,
                        "n_frames": int(row["n_frames"]),
                        "n_residues": int(row["n_residues"]),
                        "method": method,
                        "repeat": repeat,
                        "peak_rss_mib": monitor.peak,
                    },
                )
                for stage_row in result.stage_timing.rows(
                    case_id=row["case_id"], method=method, repeat=repeat
                ):
                    append_tsv(
                        stage_path,
                        {
                            **stage_row,
                            "panel": panel_name,
                            "n_frames": int(row["n_frames"]),
                            "n_residues": int(row["n_residues"]),
                        },
                    )
                print(
                    f"[V6 {panel_name}] {case_index+1}/{len(rows)} {row['case_id']} "
                    f"{method} r{repeat}={result.runtime_seconds:.4f}s "
                    f"wall={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        if panel_name == "fresh_long":
            for method in METHODS:
                key = (row["case_id"], method, 0, "fresh_process")
                if key in completed:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                    output = Path(handle.name)
                command = [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "single",
                    "--panel-path",
                    str(panel_path),
                    "--case-id",
                    row["case_id"],
                    "--method",
                    method,
                    "--json-output",
                    str(output),
                ]
                process_start = time.perf_counter()
                subprocess.run(command, cwd=ROOT, env={**os.environ, **THREAD_ENV}, check=True)
                process_wall = time.perf_counter() - process_start
                payload = json.loads(output.read_text())
                output.unlink(missing_ok=True)
                append_tsv(
                    repeats_path,
                    {
                        "case_id": row["case_id"],
                        "family_id": row["family_id"],
                        "panel": panel_name,
                        "n_frames": int(row["n_frames"]),
                        "n_residues": int(row["n_residues"]),
                        "method": method,
                        "repeat": 0,
                        "timing_kind": "fresh_process",
                        "loaded_runtime_seconds": payload["loaded_runtime_seconds"],
                        "fresh_process_wall_seconds": process_wall,
                        "method_order_position": "",
                        "warmups": 0,
                        "cpu_threads": 1,
                        "cpu_affinity": ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
                        "cpu_affinity": ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
                        "ordinary_rmsf": False,
                        "aligned_coordinates": False,
                        "accepted_level": payload["accepted_level"],
                    },
                )
        del coords, chains
    commit_performance_tables()


def single(panel_path: Path, case_id: str, method: str, output: Path) -> None:
    row = next(row for row in read_tsv(panel_path) if row["case_id"] == case_id)
    coords, chains = load_case(row)
    result = run_method(coords, chains, method)
    output.write_text(
        json.dumps(
            {
                "loaded_runtime_seconds": result.runtime_seconds,
                "accepted_level": result.metadata.get("accepted_level", "fixed"),
            }
        )
    )


def summarize() -> None:
    long_rows = [row for row in read_tsv(OUT / "07_performance/fresh_long_repeats.tsv") if row["timing_kind"] == "loaded_in_process"]
    summary: list[dict[str, object]] = []
    for method in METHODS:
        subset = [row for row in long_rows if row["method"] == method]
        per_case = []
        for case_id in sorted({row["case_id"] for row in subset}):
            values = [float(row["loaded_runtime_seconds"]) for row in subset if row["case_id"] == case_id]
            per_case.append(float(np.median(values)))
        exact = []
        for case_id in sorted({row["case_id"] for row in long_rows}):
            values = [float(row["loaded_runtime_seconds"]) for row in long_rows if row["case_id"] == case_id and row["method"] == METHODS[0]]
            exact.append(float(np.median(values)))
        speedups = np.asarray(exact) / np.asarray(per_case)
        summary.append(
            {
                "method": method,
                "n_trajectories": len(per_case),
                "median_loaded_runtime_seconds": float(np.median(per_case)),
                "median_speedup_vs_exact": float(np.median(speedups)),
                "minimum_speedup_vs_exact": float(np.min(speedups)),
                "maximum_speedup_vs_exact": float(np.max(speedups)),
            }
        )
    write_tsv(OUT / "07_performance/fresh_long_summary.tsv", summary)
    write_tsv(
        OUT / "07_performance/frozen_mdlovofit_direct_comparison.tsv",
        [
            {
                "comparison_status": "not_available_under_identical_contract",
                "reason": "Frozen stock MDLovoFit no-output timings include multi-model PDB parsing and emit native stdout, whereas v3 uses already-decoded coordinates and returns transforms, RMSD and robust RMSF. Multiplying unmatched speedups is prohibited.",
                "mdlovofit_rerun": False,
            }
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("long", "mdcath", "single", "summarize"))
    parser.add_argument("--panel-path", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "single":
        if None in (args.panel_path, args.case_id, args.method, args.json_output):
            raise ValueError("single requires panel path, case id, method and JSON output")
        single(args.panel_path, args.case_id, args.method, args.json_output)
    elif args.phase == "long":
        benchmark_panel(
            OUT / "05_fresh_panels/fresh_long_panel.tsv",
            "fresh_long",
            OUT / "07_performance/fresh_long_repeats.tsv",
        )
    elif args.phase == "mdcath":
        benchmark_panel(
            OUT / "05_fresh_panels/fresh_mdcath_performance_128.tsv",
            "fresh_mdcath_128",
            OUT / "07_performance/fresh_mdcath_128_repeats.tsv",
        )
    else:
        summarize()


if __name__ == "__main__":
    main()
