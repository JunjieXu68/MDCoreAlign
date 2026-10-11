"""Shared helpers for final submission-upgrade scripts."""

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

FINAL_SUBMISSION_ROOT = ROOT / "results" / "final_submission_upgrade"


def timestamp() -> str:
    """Return a local ISO timestamp."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    """Create and return a directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Return SHA256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert common objects to JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str], *, delimiter: str = "\t") -> None:
    """Write a deterministic delimited table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_table(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV/TSV table."""

    if not path.exists():
        return []
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int = 180) -> dict[str, object]:
    """Run a command and capture compact stdout/stderr."""

    started_at = timestamp()
    try:
        completed = subprocess.run(command, cwd=cwd, timeout=timeout, capture_output=True, text=True)
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "started_at": started_at,
            "finished_at": timestamp(),
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "started_at": started_at,
            "finished_at": timestamp(),
        }


def git_snapshot() -> dict[str, object]:
    """Return lightweight git metadata."""

    commit = run_command(["git", "rev-parse", "HEAD"], timeout=10)
    toplevel = run_command(["git", "rev-parse", "--show-toplevel"], timeout=10)
    remote = run_command(["git", "remote", "get-url", "origin"], timeout=10)
    return {
        "available": commit["returncode"] == 0,
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "repository_toplevel": toplevel["stdout"].strip() if toplevel["returncode"] == 0 else None,
        "origin_remote": remote["stdout"].strip() if remote["returncode"] == 0 else None,
    }


def environment_snapshot() -> dict[str, object]:
    """Return compact environment metadata."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(ROOT),
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
    """Write a manifest with checksums and environment metadata."""

    inputs = inputs or []
    outputs = outputs or []
    payload: dict[str, object] = {
        "script": script,
        "generated_at": timestamp(),
        "environment": environment_snapshot(),
        "git": git_snapshot(),
        "inputs": {
            str(path.relative_to(ROOT) if path.exists() and path.is_relative_to(ROOT) else path): (
                sha256_file(path) if path.exists() and path.is_file() else None
            )
            for path in inputs
        },
        "outputs": {
            str(path.relative_to(ROOT) if path.exists() and path.is_relative_to(ROOT) else path): (
                sha256_file(path) if path.exists() and path.is_file() else None
            )
            for path in outputs
        },
        "commands": commands or [],
    }
    if extra:
        payload.update(extra)
    write_json(output_dir / "manifest.json", payload)


def file_nonempty(path: Path) -> bool:
    """Return whether file exists and is non-empty."""

    return path.exists() and path.is_file() and path.stat().st_size > 0


def row_count(path: Path) -> int:
    """Return data-row count for CSV/TSV."""

    return len(read_table(path))


def has_local_absolute_path(text: str) -> bool:
    """Detect local absolute paths that should not appear in public docs."""

    return any(marker in text for marker in ["/storage/", "/home/", "MDAnalysis_data"])


def env_value(name: str, default: str = "pending") -> str:
    """Return stripped environment variable value or default."""

    value = os.environ.get(name, "").strip()
    return value if value else default


def managed_section(text: str, marker: str, body: str) -> str:
    """Insert or replace a managed Markdown section."""

    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}\n"
    if begin in text and end in text:
        prefix = text.split(begin, 1)[0]
        suffix = text.split(end, 1)[1]
        return prefix + block + suffix.lstrip("\n")
    return text.rstrip() + "\n\n" + block


def safe_status(value: str) -> str:
    """Normalize pending-like release metadata values."""

    return value if value and value != "pending" else "pending"
