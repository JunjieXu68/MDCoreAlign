#!/usr/bin/env python3
"""Measure fresh-process import and first-call latency for one method."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


def _trajectory(n_frames: int, n_residues: int) -> np.ndarray:
    rng = np.random.default_rng(907)
    reference = np.cumsum(rng.normal(size=(n_residues, 3)), axis=0)
    coords = np.repeat(reference[None], n_frames, axis=0)
    coords += rng.normal(scale=0.03, size=coords.shape)
    start = n_residues // 3
    stop = min(n_residues, start + max(8, n_residues // 5))
    phase = np.linspace(0.0, 8.0 * np.pi, n_frames)
    coords[:, start:stop, 0] += np.sin(phase)[:, None]
    return coords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        choices=[
            "CPU_exact",
            "CPU_throughput",
            "GPU_throughput",
            "CPU_W5",
            "GPU_W5",
        ],
    )
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--residues", type=int, default=150)
    args = parser.parse_args()
    coords = _trajectory(args.frames, args.residues)
    chains = np.asarray(["A"] * args.residues)
    import_start = time.perf_counter()
    if args.method == "CPU_exact":
        from mdcorealign.contracts import solve_exact_contract

        function = lambda: solve_exact_contract(
            coords,
            chain_ids=chains,
            representative_count=min(2000, args.frames),
            chunk_size=2000,
        )
    elif args.method == "CPU_throughput":
        from mdcorealign.experimental import solve_fast_global_alignment

        function = lambda: solve_fast_global_alignment(
            coords,
            chain_ids=chains,
            representative_count=min(256, args.frames),
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=False,
        )
    elif args.method == "GPU_throughput":
        from mdcorealign.gpu_backend import solve_fast_global_gpu

        function = lambda: solve_fast_global_gpu(
            coords,
            chain_ids=chains,
            representative_count=min(256, args.frames),
            update_rounds=4,
            chunk_size=2000,
            precision="float32",
            rotation_solver="horn",
        )
    elif args.method == "CPU_W5":
        from mdsuperpose.window_kinematics import scan_local_kinematics

        function = lambda: scan_local_kinematics(
            coords,
            chain_ids=chains,
            evaluate_blocks=False,
        )
    else:
        from mdcorealign.gpu_backend import scan_w5_gpu

        function = lambda: scan_w5_gpu(
            coords,
            chain_ids=chains,
            precision="float32",
        )
    import_seconds = time.perf_counter() - import_start
    call_start = time.perf_counter()
    result = function()
    first_call_seconds = time.perf_counter() - call_start
    payload = {
        "method": args.method,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "frames": args.frames,
        "residues": args.residues,
        "import_seconds": import_seconds,
        "first_call_seconds": first_call_seconds,
        "fresh_process_total_after_input_seconds": (
            import_seconds + first_call_seconds
        ),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "result_available": result is not None,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
