#!/usr/bin/env python3
"""Prepare centered external inputs and time raw-source input components."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import numpy as np

from mdsuperpose.mdlovofit import write_multimodel_pdb
from scripts.atlas_large_long_v1.common import (
    OUT,
    P7,
    PREFIXES,
    REPEATS,
    ROOT,
    WARMUPS,
    prefix_label,
    read_tsv,
    sha256,
    write_json,
    write_tsv,
)


INPUT_DIR = OUT / "a3_external/centered_inputs"
RAW_PREP = OUT / "a3_raw_source/raw_source_input_component_repeats.tsv"
RAW_CHECKPOINTS = OUT / "a3_raw_source/checkpoints"


def _decode(system: str, frames: int) -> tuple[np.ndarray, float]:
    import MDAnalysis as mda

    source = P7 / "atlas_readiness_inputs" / system
    pdb = source / f"{system}.pdb"
    xtc = source / f"{system}_prod_R1_fit.xtc"
    started = time.perf_counter()
    universe = mda.Universe(str(pdb), str(xtc), convert_units=True)
    ca = universe.select_atoms("protein and name CA")
    values = np.empty((frames, len(ca), 3), dtype=np.float64)
    for index, _timestep in enumerate(universe.trajectory[:frames]):
        values[index] = ca.positions
    return values, time.perf_counter() - started


def _center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=1, keepdims=True)


def prepare() -> None:
    manifest = read_tsv(OUT / "atlas_manifest.tsv")
    rows = []
    for system_index, row in enumerate(manifest, start=1):
        system = row["system"]
        for frames in PREFIXES:
            output = INPUT_DIR / system / f"{prefix_label(frames)}_centered_CA.pdb"
            if output.exists():
                rows.append(
                    {
                        "system": system,
                        "prefix_label": prefix_label(frames),
                        "n_frames": frames,
                        "n_residues": row["n_residues"],
                        "status": "success",
                        "centered_input_path": str(output.relative_to(ROOT)),
                        "centered_input_size_bytes": output.stat().st_size,
                        "centered_input_sha256": sha256(output),
                        "reused_existing": True,
                    }
                )
                continue
            started = time.perf_counter()
            values, decode_seconds = _decode(system, frames)
            center_start = time.perf_counter()
            centered = _center(values)
            center_seconds = time.perf_counter() - center_start
            write_start = time.perf_counter()
            write_multimodel_pdb(centered, output)
            write_seconds = time.perf_counter() - write_start
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label(frames),
                    "n_frames": frames,
                    "n_residues": values.shape[1],
                    "status": "success",
                    "centered_input_path": str(output.relative_to(ROOT)),
                    "centered_input_size_bytes": output.stat().st_size,
                    "centered_input_sha256": sha256(output),
                    "decode_seconds": decode_seconds,
                    "per_frame_center_seconds": center_seconds,
                    "PDB_write_seconds": write_seconds,
                    "total_prepare_seconds": time.perf_counter() - started,
                    "maximum_abs_centered_coordinate_A": float(np.max(np.abs(centered))),
                    "centering_convention": "subtract_each_frame_CA_COG",
                    "reused_existing": False,
                }
            )
            print(f"[A3 external input] {system} {frames}", flush=True)
            del values, centered
    write_tsv(OUT / "a3_external/centered_input_manifest.tsv", rows)


def measure_raw_source() -> None:
    manifest = read_tsv(OUT / "atlas_manifest.tsv")
    rows = []
    temporary = OUT / "a3_raw_source/temporary_centered_input.pdb"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        system = row["system"]
        for frames in PREFIXES:
            checkpoint = RAW_CHECKPOINTS / f"{system}__{prefix_label(frames)}.tsv"
            if checkpoint.exists():
                rows.extend(read_tsv(checkpoint))
                print(f"[A3 raw input] skip {system} {frames}", flush=True)
                continue
            prefix_rows = []
            for index in range(WARMUPS + REPEATS):
                total_start = time.perf_counter()
                values, decode_seconds = _decode(system, frames)
                center_start = time.perf_counter()
                centered = _center(values)
                center_seconds = time.perf_counter() - center_start
                write_start = time.perf_counter()
                write_multimodel_pdb(centered, temporary)
                write_seconds = time.perf_counter() - write_start
                prefix_rows.append(
                    {
                        "system": system,
                        "prefix_label": prefix_label(frames),
                        "n_frames": frames,
                        "n_residues": values.shape[1],
                        "repeat_index": index - WARMUPS,
                        "is_warmup": index < WARMUPS,
                        "status": "success",
                        "raw_PDB_XTC_decode_to_CA_seconds": decode_seconds,
                        "per_frame_center_seconds": center_seconds,
                        "centered_CA_PDB_write_seconds": write_seconds,
                        "internal_raw_source_component_seconds": decode_seconds,
                        "external_raw_source_component_seconds": decode_seconds + center_seconds + write_seconds,
                        "direct_component_wall_seconds": time.perf_counter() - total_start,
                        "temporary_PDB_size_bytes": temporary.stat().st_size,
                        "peak_RSS_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                        "raw_contract_composition": "measured_input_component_plus_corresponding_loaded_method_repeat",
                    }
                )
                temporary.unlink(missing_ok=True)
                del values, centered
            write_tsv(checkpoint, prefix_rows)
            rows.extend(prefix_rows)
            write_tsv(RAW_PREP, rows)
            print(f"[A3 raw input] {system} {frames}", flush=True)
    write_tsv(RAW_PREP, rows)
    write_json(
        OUT / "a3_raw_source/raw_source_input_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "warmups": WARMUPS,
            "measured_repeats": REPEATS,
            "composition_disclosed": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "measure-raw-source"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        measure_raw_source()


if __name__ == "__main__":
    main()
