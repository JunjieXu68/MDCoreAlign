"""Shared helpers for pre-manuscript analysis scripts.

These helpers intentionally avoid heavy dependencies so that the analysis
package remains easy to run in a fresh checkout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260609


def read_table(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV/TSV file into a list of string dictionaries."""

    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str], *, delimiter: str = "\t") -> None:
    """Write rows with a stable schema, even when the table is empty."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def to_float(value: object, default: float | None = None) -> float | None:
    """Convert a table value to float, returning default for blanks."""

    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int | None = None) -> int | None:
    """Convert a table value to int, returning default for blanks."""

    number = to_float(value)
    if number is None:
        return default
    return int(round(number))


def fmt(value: object, digits: int = 6) -> object:
    """Format floats compactly while preserving blanks and strings."""

    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.{digits}g}"
    return value


def mean(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean or None for empty inputs."""

    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(statistics.fmean(items))


def median(values: Iterable[float]) -> float | None:
    """Return the median or None for empty inputs."""

    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(statistics.median(items))


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int | None:
    """Return data row count for CSV/TSV files, or None for other files."""

    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle, delimiter=delimiter)))


def column_count(path: Path) -> int | None:
    """Return column count for CSV/TSV files, or None for other files."""

    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return len(reader.fieldnames or [])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    """Compute Spearman correlation without requiring SciPy."""

    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if x_array.size < 2 or x_array.size != y_array.size:
        return None
    x_rank = average_ranks(x_array)
    y_rank = average_ranks(y_array)
    x_centered = x_rank - float(np.mean(x_rank))
    y_centered = y_rank - float(np.mean(y_rank))
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def bootstrap_ci(values: Iterable[float], *, seed: int = BOOTSTRAP_SEED, n_bootstrap: int = 2000) -> tuple[float, float] | tuple[None, None]:
    """Return a percentile bootstrap 95% CI for the mean."""

    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if array.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, array.size), replace=True)
    means = np.mean(samples, axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_stat_ci(
    x: Iterable[float],
    y: Iterable[float],
    stat_func,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = 2000,
) -> tuple[float, float] | tuple[None, None]:
    """Return a paired bootstrap CI for a statistic computed on x/y arrays."""

    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if x_array.size == 0 or x_array.size != y_array.size:
        return None, None
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, x_array.size, size=x_array.size)
        value = stat_func(x_array[indices], y_array[indices])
        if value is not None and np.isfinite(value):
            stats.append(float(value))
    if not stats:
        return None, None
    low, high = np.percentile(np.asarray(stats, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def require_files(paths: Iterable[Path]) -> list[str]:
    """Return warning messages for missing files."""

    warnings: list[str] = []
    for path in paths:
        if not path.exists():
            warnings.append(f"missing required file: {path}")
    return warnings

