"""Shared paths and deterministic helpers for ATLAS large/long v1."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/atlas_large_long_v1"
P7 = ROOT / "results/post_b3_optimisation_v1/p7_timing_atlas"
LOCKED = (
    ROOT
    / "results/database_scale_benchmark_v2/b5_real_long"
    / "b5_atlas_acquisition_candidates_locked.tsv"
)
GPU_PYTHON = ROOT / ".venv-gpu-cu12/bin/python"
MDA_PYTHON = Path("/home/jjxu/miniconda3/envs/mdlovofit/bin/python")
MDLOVOFIT = Path("/usr/local/bin/mdlovofit")
THESEUS = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")
PREFIXES = (250, 500, 1000, 2000, 5000, 10001)
WARMUPS = 1
REPEATS = 5
THREAD_ENV = {
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        fields.extend(field for field in row if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_label(frames: int) -> str:
    return f"f{int(frames):05d}"
