#!/usr/bin/env python
"""Run the locked CPU thread-scaling subset in isolated processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.cpu_fast_path_v2.run_fast_path_study import (  # noqa: E402
    OUT,
    _run_named,
    load_case,
    read_tsv,
    write_tsv,
)


SYSTEMS = ("ATLAS_1u55_A_R1", "LOCAL_NhaA", "ATLAS_3nci_A_R1")
METHODS = (
    "exact_sequential_f64_compact",
    "B_shared_f64",
    "C_correction_1",
)
THREADS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--method")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def worker(case_id: str, method: str, repeats: int) -> None:
    panel = read_tsv(OUT / "02_real_panel/full_panel.tsv")
    row = next(item for item in panel if item["case_id"] == case_id)
    coords, chains = load_case(row)
    _run_named(coords, chains, method)
    runtimes = []
    for _ in range(repeats):
        result = _run_named(coords, chains, method)
        runtimes.append(result.runtime_seconds)
    print(json.dumps({
        "case_id": case_id,
        "n_frames": int(coords.shape[0]),
        "n_residues": int(coords.shape[1]),
        "method": method,
        "runtimes": runtimes,
    }))


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.case_id is None or args.method is None:
            raise ValueError("worker requires --case-id and --method")
        worker(args.case_id, args.method, args.repeats)
        return
    rows: list[dict[str, object]] = []
    python = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    for case_id in SYSTEMS:
        for method in METHODS:
            for threads in THREADS:
                environment = dict(os.environ)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    environment[name] = str(threads)
                start = time.perf_counter()
                completed = subprocess.run(
                    [
                        str(python),
                        str(script),
                        "--worker",
                        "--case-id",
                        case_id,
                        "--method",
                        method,
                        "--repeats",
                        str(args.repeats),
                    ],
                    cwd=ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                wall = time.perf_counter() - start
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                for repeat, runtime in enumerate(payload["runtimes"], start=1):
                    rows.append({
                        "case_id": case_id,
                        "n_frames": payload["n_frames"],
                        "n_residues": payload["n_residues"],
                        "method": method,
                        "cpu_threads": threads,
                        "repeat": repeat,
                        "loaded_runtime_seconds": runtime,
                        "fresh_process_wall_seconds_for_warmup_plus_repeats": wall,
                        "python": str(python.relative_to(python.anchor)),
                        "cpu_affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
                    })
    write_tsv(OUT / "06_performance/thread_scaling.tsv", rows)


if __name__ == "__main__":
    main()
