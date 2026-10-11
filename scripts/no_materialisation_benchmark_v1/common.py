"""Shared deterministic helpers for no-materialisation benchmark v1."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from mdsuperpose.mdlovofit import parse_mdlovofit_stdout


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/no_materialisation_benchmark_v1"
STOCK = Path("/usr/local/bin/mdlovofit")
TIMED = OUT / "04_instrumented_kernel/bin/mdlovofit_timed"
PYTHON = ROOT / ".venv-gpu-cu12/bin/python"
FRACTION = 0.60
THREAD_ENV = {
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}
SUBDIRS = (
    "00_freeze",
    "01_stock_verification",
    "02_atlas_stock_no_output",
    "03_mdcath_stock_no_output",
    "04_instrumented_kernel",
    "05_optional_database_deployment",
    "06_analysis",
    "07_validation",
    "handoff_to_chatgpt",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        fields.extend(key for key in row if key not in fields)
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stock_command(input_path: Path, output_path: Path | None = None) -> list[str]:
    command = [str(STOCK), "-f", f"{FRACTION:.6f}"]
    if output_path is not None:
        command.extend(("-t", str(output_path)))
    command.append(str(input_path))
    return command


def timed_command(input_path: Path) -> list[str]:
    return [str(TIMED), "-f", f"{FRACTION:.6f}", str(input_path)]


def run_external(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        parsed = parse_mdlovofit_stdout(completed.stdout)
        return {
            "status": "success" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "wall_seconds": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "native_least_mobile_RMSD_A": parsed.average_rmsd_least_mobile,
            "native_most_mobile_RMSD_A": parsed.average_rmsd_most_mobile,
            "native_all_atom_RMSD_A": parsed.average_rmsd_all_atoms,
            "native_fraction": parsed.fraction_of_best_aligned_atoms,
            "native_total_atoms": parsed.total_number_of_atoms,
            "native_selection_atoms": parsed.number_of_atoms_of_selection,
            "frame_rows": parsed.frame_rmsd_row_count,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "return_code": "",
            "wall_seconds": time.perf_counter() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def parse_timing(stderr: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in stderr.splitlines():
        if not line.startswith("NM_TIMING "):
            continue
        _, name, value = line.split(maxsplit=2)
        result[name] = float(value)
    return result
