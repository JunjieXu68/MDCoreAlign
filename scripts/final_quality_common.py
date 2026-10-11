"""Shared helpers for final quality-upgrade scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
for path in (ROOT, SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FINAL_ROOT = ROOT / "results" / "final_quality_upgrade"


def timestamp() -> str:
    """Return a local ISO timestamp."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Return SHA256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str], *, delimiter: str = "\t") -> None:
    """Write a deterministic delimited table with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_table(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV/TSV table into string dictionaries."""

    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _jsonable(value: object) -> object:
    """Convert common non-JSON objects."""

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a required dependency
        np = None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> dict[str, object]:
    """Run a command and capture stdout/stderr."""

    start = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(command, cwd=cwd, timeout=timeout, capture_output=True, text=True)
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "started_at": start.astimezone().isoformat(timespec="seconds"),
            "finished_at": timestamp(),
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "started_at": start.astimezone().isoformat(timespec="seconds"),
            "finished_at": timestamp(),
        }


def environment_snapshot() -> dict[str, object]:
    """Return compact environment metadata for manifests."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(ROOT),
    }


def git_snapshot() -> dict[str, object]:
    """Return git metadata when available."""

    head = run_command(["git", "rev-parse", "HEAD"], timeout=10)
    top = run_command(["git", "rev-parse", "--show-toplevel"], timeout=10)
    return {
        "commit": head["stdout"].strip() if head["returncode"] == 0 else None,
        "repository_toplevel": top["stdout"].strip() if top["returncode"] == 0 else None,
        "available": head["returncode"] == 0,
    }


def write_manifest(
    output_dir: Path,
    *,
    script: str,
    inputs: list[Path] | None = None,
    outputs: list[Path] | None = None,
    commands: list[dict[str, object]] | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Write a final-quality manifest with checksums."""

    inputs = inputs or []
    outputs = outputs or []
    payload: dict[str, object] = {
        "script": script,
        "generated_at": timestamp(),
        "environment": environment_snapshot(),
        "git": git_snapshot(),
        "inputs": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): (
                sha256_file(path) if path.exists() and path.is_file() else None
            )
            for path in inputs
        },
        "outputs": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): (
                sha256_file(path) if path.exists() and path.is_file() else None
            )
            for path in outputs
        },
        "commands": commands or [],
    }
    if extra:
        payload.update(extra)
    write_json(output_dir / "manifest.json", payload)


def to_float(value: object, default: float | None = None) -> float | None:
    """Convert a value to float with blank handling."""

    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def row_count(path: Path) -> int:
    """Return data-row count for CSV/TSV files."""

    if not path.exists():
        return 0
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle, delimiter=delimiter)))


def file_nonempty(path: Path) -> bool:
    """Return whether a path exists and is non-empty."""

    return path.exists() and path.stat().st_size > 0


def has_local_absolute_path(text: str) -> bool:
    """Detect local paths that should not appear in public docs."""

    forbidden = ["/storage/", "/home/", "MDAnalysis_data"]
    return any(marker in text for marker in forbidden)


def ensure_dir(path: Path) -> Path:
    """Create and return a directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path

