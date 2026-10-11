"""Shared paths and deterministic helpers for the post-B3 work."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "results/database_scale_benchmark_v2"
OUT = ROOT / "results/post_b3_optimisation_v1"
B3 = SOURCE_ROOT / "b3_full_mdcath"
B2 = SOURCE_ROOT / "b2_contracts"
MDL = SOURCE_ROOT / "mdlovofit_comparison"
GPU_PYTHON = ROOT / ".venv-gpu-cu12/bin/python"
THREAD_ENV = {
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one TSV into dictionaries."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_tsv(path: Path):
    """Yield dictionaries from one TSV without materializing it."""

    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write a non-empty union-schema TSV."""

    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    """Atomically write text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(row: dict[str, Any], field: str) -> float:
    """Parse a finite-compatible numeric field or return NaN."""

    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def truth(value: Any) -> bool:
    """Parse a serialized boolean."""

    return str(value).strip().lower() == "true"


def domain_split(domain_id: str) -> str:
    """Return a deterministic 80/20 development/validation domain split."""

    value = int(hashlib.sha256(domain_id.encode()).hexdigest()[:8], 16)
    return "validation" if value % 5 == 0 else "development"


def finite(value: float) -> bool:
    """Return whether a numeric value is finite."""

    return math.isfinite(value)
