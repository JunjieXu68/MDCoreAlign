"""Shared paths and lightweight table helpers for stream_align_v1."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/stream_align_v1"
ATLAS = ROOT / "results/atlas_large_long_v1"
SYNTHETIC = ROOT / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"

ATLAS_SYSTEMS = (
    "1jnr_C",
    "1r6w_A",
    "1tg7_A",
    "1u55_A",
    "3nci_A",
    "5vz0_A",
    "6qfk_A",
    "6sup_A",
)
REPRESENTATIVE_COUNTS = (64, 128, 256, 512)
LANDMARK_COUNTS = (16, 32, 64, 128)
LANDMARK_METHODS = (
    "lowest_deformation_only",
    "lowest_deformation_sequence_spacing",
    "lowest_deformation_3d_farthest",
    "chain_balanced_spatial",
)
KEYFRAME_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
REFINEMENT_ROUNDS = (0, 1, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def atlas_manifest() -> dict[str, dict[str, str]]:
    return {row["system"]: row for row in read_tsv(ATLAS / "atlas_manifest.tsv")}


def load_atlas(system: str):
    import numpy as np

    row = atlas_manifest()[system]
    coords = np.load(ROOT / row["coordinate_cache_path"], mmap_mode="r")
    with np.load(ROOT / row["residue_metadata_path"], allow_pickle=False) as archive:
        chains = np.asarray(archive["chain_ids"])
    cache = np.load(
        ATLAS / f"a3_loaded_analysis/result_cache/{system}__f10001.npz",
        allow_pickle=False,
    )
    return coords, chains, cache, row


def frozen_gpu_runtime(system: str, n_frames: int = 10001) -> float | None:
    rows = read_tsv(ATLAS / "atlas_matched_prefix_results.tsv")
    matches = [
        row
        for row in rows
        if row["system"] == system
        and int(row["n_frames"]) == n_frames
        and row["contract"] == "A_loaded_analysis"
        and row["method"] == "GPU_throughput_float32_Horn"
        and row["status"] == "success"
    ]
    return float(matches[0]["median_seconds"]) if matches else None
