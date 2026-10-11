#!/usr/bin/env python3
"""Run the registered production throughput C1 contract in one subprocess."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mdcorealign.benchmark_minimal import solve_minimal_from_pdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--residues", type=int, required=True)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_json is not None:
        from threadpoolctl import threadpool_info

        args.audit_json.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "threadpools": threadpool_info(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    result = solve_minimal_from_pdb(
        args.input,
        mode="throughput",
        backend="cpu",
        expected_frames=args.frames,
        expected_residues=args.residues,
    )
    weights = np.asarray(result.frozen_weights, dtype=np.float64)
    labels = np.asarray(result.core_labels, dtype=np.uint8)
    output = {
        "status": "success",
        "mode": result.mode,
        "backend": result.backend,
        "mean_unified_rmsd_A": float(np.mean(result.rmsd)),
        "mean_weighted_rmsd_A": float(np.mean(result.weighted_rmsd)),
        "core_size": int(labels.sum()),
        "core_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "representative_count": result.metadata["representative_count"],
        "update_rounds": result.metadata["update_rounds"],
        "aligned_coordinates_materialised": False,
        "rmsf_computed": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
