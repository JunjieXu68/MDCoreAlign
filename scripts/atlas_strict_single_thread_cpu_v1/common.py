"""Shared contracts and deterministic I/O for the strict ATLAS study."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/atlas_strict_single_thread_cpu_v1"
SOURCE_MANIFEST = ROOT / "results/atlas_large_long_v1/a3_external/centered_input_manifest.tsv"
ATLAS_MANIFEST = ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv"
HISTORICAL_TIMING = (
    ROOT
    / "results/no_materialisation_benchmark_v1/02_atlas_stock_no_output"
    / "atlas_no_output_repeats.tsv"
)
STOCK_MDLOVOFIT = Path("/usr/local/bin/mdlovofit")
PYTHON = Path(os.environ.get("MDCOREALIGN_ATLAS_PYTHON", "/home/jjxu/miniconda3/bin/python"))
FRACTION = 0.60
METHOD_MDCA = "MDCoreAlign_throughput_CPU"
METHOD_MDLOVOFIT = "stock_MDLovoFit"
METHODS = (METHOD_MDCA, METHOD_MDLOVOFIT)
PREFIXES = ("f00250", "f00500", "f01000", "f02000", "f05000", "f10001")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
THREAD_ENV_HASH = hashlib.sha256(
    json.dumps(THREAD_ENV, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Atomically write a non-empty tab-separated table."""

    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    if fields is None:
        fields = []
        for row in values:
            fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic JSON."""

    write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setting_id(row: dict[str, str]) -> str:
    """Return the stable setting identifier."""

    return f"{row['system_id']}__{row['prefix_id']}"


def absolute_input(row: dict[str, str]) -> Path:
    """Resolve a repository-relative benchmark input."""

    path = Path(row["input_path"])
    return path if path.is_absolute() else ROOT / path


def thread_environment() -> dict[str, str]:
    """Return a child environment with every numerical pool limited to one."""

    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    return environment


def command_for(method: str, row: dict[str, str], logical_cpu: int) -> list[str]:
    """Build the registered no-materialisation subprocess command."""

    source = absolute_input(row)
    prefix = ["taskset", "-c", str(logical_cpu)]
    if method == METHOD_MDCA:
        return prefix + [
            str(PYTHON),
            "-m",
            "scripts.atlas_strict_single_thread_cpu_v1.run_mdcorealign",
            "--input",
            str(source),
            "--frames",
            str(row["n_frames"]),
            "--residues",
            str(row["n_ca"]),
        ]
    if method == METHOD_MDLOVOFIT:
        return prefix + [str(STOCK_MDLOVOFIT), "-f", f"{FRACTION:.6f}", str(source)]
    raise ValueError(f"unsupported method: {method}")


def warm_page_cache(path: Path) -> int:
    """Sequentially read an input outside the timing boundary."""

    count = 0
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += len(block)
    return count


def stable_hash_int(text: str) -> int:
    """Return a deterministic integer derived from text."""

    return int(hashlib.sha256(text.encode()).hexdigest(), 16)
