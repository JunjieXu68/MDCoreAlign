"""Shared constants and serialization helpers for benchmark v2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/database_scale_benchmark_v2"
B0_ROOT = RESULT_ROOT / "b0_freeze"
B1_ROOT = RESULT_ROOT / "b1_tail_adjudication"
B2_ROOT = RESULT_ROOT / "b2_contracts"
B3_ROOT = RESULT_ROOT / "b3_full_mdcath"
M_ROOT = RESULT_ROOT / "mdlovofit_comparison"

GPU_PYTHON = ROOT / ".venv-gpu-cu12/bin/python"
THREAD_ENV = {
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}

TAIL_DOMAINS = (
    "1fzrA00",
    "1kx5A00",
    "1sr2A00",
    "1vq8Q00",
    "1vs5T00",
    "1vw4K00",
    "2mk5A00",
    "3jamY00",
    "3qq5A02",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Atomically write a union-schema tab-separated table."""

    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    """Atomically write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_manifest_hash(rows: Iterable[dict[str, object]]) -> str:
    """Hash sorted path/digest pairs without rereading source files."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(
            f"{row['path']}\t{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """Return a repository-relative path where possible."""

    try:
        return path.resolve(strict=False).relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
