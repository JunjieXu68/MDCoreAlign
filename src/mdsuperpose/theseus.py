"""Wrapper utilities for the real external THESEUS executable.

This module mirrors the MDLovoFit wrapper style: it resolves an external
executable, prepares PDB inputs, executes the program, captures outputs, and
adapts the resulting files into structured Python records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from .mdlovofit import read_multimodel_pdb_coords, write_multimodel_pdb
from .utils import ensure_xyz_frames


@dataclass(slots=True)
class TheseusParsedOutput:
    """Best-effort parse of selected THESEUS stdout metrics."""

    classical_pairwise_rmsd: float | None
    least_squares_sigma: float | None
    maximum_likelihood_rmsd: float | None
    maximum_likelihood_sigma: float | None
    marginal_log_likelihood: float | None
    aic: float | None
    bic: float | None
    total_rounds: int | None
    converged_fractional_precision: float | None
    median_structure_index: int | None
    n_total: int | None
    n_atoms: int | None
    n_structures: int | None


@dataclass(slots=True)
class TheseusVarianceRecord:
    """Per-residue variance statistics reported by THESEUS."""

    atom_index: int
    residue_name: str
    residue_sequence: int
    variance: float
    std_dev: float
    rmsd: float
    is_core: bool


@dataclass(slots=True)
class TheseusRunResult:
    """Captured result of one external THESEUS execution."""

    executable_path: Path
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    total_runtime_seconds: float
    subprocess_runtime_seconds: float
    input_write_seconds: float
    output_read_seconds: float
    input_paths: tuple[Path, ...]
    output_root: str
    output_directory: Path
    aligned_pdb_path: Path
    average_pdb_path: Path
    variances_path: Path
    residuals_path: Path
    transforms_path: Path
    aligned_coords: np.ndarray | None
    parsed_output: TheseusParsedOutput
    variance_records: tuple[TheseusVarianceRecord, ...]
    output_core_atom_count: int | None
    output_core_atom_fraction: float | None


def _read_executable_from_config(config_path: Path) -> Path | None:
    """Read an optional THESEUS executable path from a JSON config file."""

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a JSON object")

    direct = payload.get("theseus_executable")
    if isinstance(direct, str) and direct.strip():
        return Path(direct)

    executables = payload.get("executables")
    if isinstance(executables, dict):
        nested = executables.get("theseus")
        if isinstance(nested, str) and nested.strip():
            return Path(nested)
    return None


def resolve_theseus_executable(
    executable: str | Path | None = None,
    *,
    env_var: str = "THESEUS_EXECUTABLE",
    config_path: str | Path | None = None,
) -> Path:
    """Resolve the THESEUS executable from CLI, env, config, or PATH."""

    candidates: list[Path] = []
    if executable is not None:
        candidates.append(Path(executable))

    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))

    if config_path is not None:
        config_candidate = _read_executable_from_config(Path(config_path))
        if config_candidate is not None:
            candidates.append(config_candidate)

    which_candidate = shutil.which("theseus")
    if which_candidate:
        candidates.append(Path(which_candidate))

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.exists() and os.access(resolved, os.X_OK):
            return resolved

    raise FileNotFoundError(
        "Could not find the THESEUS executable. Set --theseus-executable, "
        f"the {env_var} environment variable, a JSON config file, or ensure "
        "'theseus' is available on PATH."
    )


def build_theseus_command(
    executable_path: str | Path,
    *,
    output_root: str,
    input_paths: list[str | Path] | tuple[str | Path, ...],
) -> list[str]:
    """Build the external THESEUS command line."""

    if not output_root.strip():
        raise ValueError("output_root must not be empty")
    if not input_paths:
        raise ValueError("at least one input path is required")

    return [
        str(Path(executable_path)),
        "-r",
        output_root,
        *[str(Path(path)) for path in input_paths],
    ]


def parse_theseus_stdout(stdout: str) -> TheseusParsedOutput:
    """Parse selected THESEUS metrics without guessing unavailable fields."""

    float_patterns = {
        "classical_pairwise_rmsd": r"Classical LS pairwise <RMSD>\s+([0-9.eE+-]+)",
        "least_squares_sigma": r"Least-squares <sigma>\s+([0-9.eE+-]+)",
        "maximum_likelihood_rmsd": r"Maximum Likelihood <RMSD>\s+([0-9.eE+-]+)",
        "maximum_likelihood_sigma": r"Maximum Likelihood <sigma>\s+([0-9.eE+-]+)",
        "marginal_log_likelihood": r"Marginal Log Likelihood\s+([0-9.eE+-]+)",
        "aic": r"AIC\s+([0-9.eE+-]+)",
        "bic": r"BIC\s+([0-9.eE+-]+)",
        "converged_fractional_precision": r"Converged to a fractional precision of\s+([0-9.eE+-]+)",
    }
    int_patterns = {
        "total_rounds": r"Total rounds\s*=\s*([0-9]+)",
        "median_structure_index": r"Median structure\s*=\s*#([0-9]+)",
    }
    parsed: dict[str, float | int | None] = {}
    for key, pattern in float_patterns.items():
        match = re.search(pattern, stdout)
        parsed[key] = float(match.group(1)) if match else None
    for key, pattern in int_patterns.items():
        match = re.search(pattern, stdout)
        parsed[key] = int(match.group(1)) if match else None

    count_match = re.search(
        r"N\(total\)\s*=\s*([0-9]+),\s*N\(atoms\)\s*=\s*([0-9]+),\s*N\(structures\)\s*=\s*([0-9]+)",
        stdout,
    )
    if count_match:
        parsed["n_total"] = int(count_match.group(1))
        parsed["n_atoms"] = int(count_match.group(2))
        parsed["n_structures"] = int(count_match.group(3))
    else:
        parsed["n_total"] = None
        parsed["n_atoms"] = None
        parsed["n_structures"] = None

    return TheseusParsedOutput(**parsed)


def parse_theseus_variances(path: str | Path) -> tuple[TheseusVarianceRecord, ...]:
    """Parse THESEUS per-residue variance statistics."""

    records: list[TheseusVarianceRecord] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 7 or tokens[0] != "RES":
                continue
            records.append(
                TheseusVarianceRecord(
                    atom_index=int(tokens[1]),
                    residue_name=tokens[2],
                    residue_sequence=int(tokens[3]),
                    variance=float(tokens[4]),
                    std_dev=float(tokens[5]),
                    rmsd=float(tokens[6]),
                    is_core=(len(tokens) >= 8 and tokens[7].upper() == "CORE"),
                )
            )
    return tuple(records)


def _build_theseus_environment(executable_path: Path) -> dict[str, str]:
    """Build a subprocess environment that prefers the executable's conda lib dir."""

    env = dict(os.environ)
    candidate_lib = executable_path.parent.parent / "lib"
    if candidate_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{candidate_lib}{os.pathsep}{existing}" if existing else str(candidate_lib)
        )
    return env


def run_theseus_on_files(
    input_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    output_root: str,
    output_directory: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    env_var: str = "THESEUS_EXECUTABLE",
    timeout_seconds: float | None = None,
) -> TheseusRunResult:
    """Execute the real external THESEUS binary on one or more PDB inputs."""

    executable_path = resolve_theseus_executable(
        executable,
        env_var=env_var,
        config_path=config_path,
    )
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_input_paths = [Path(path).resolve() for path in input_paths]
    command = build_theseus_command(
        executable_path,
        output_root=output_root,
        input_paths=resolved_input_paths,
    )
    env = _build_theseus_environment(executable_path)

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=output_directory,
        env=env,
        timeout=timeout_seconds,
        check=False,
    )
    total_runtime = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"THESEUS failed with return code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    aligned_pdb_path = output_directory / f"{output_root}_sup.pdb"
    average_pdb_path = output_directory / f"{output_root}_ave.pdb"
    variances_path = output_directory / f"{output_root}_variances.txt"
    residuals_path = output_directory / f"{output_root}_residuals.txt"
    transforms_path = output_directory / f"{output_root}_transf.txt"
    expected_paths = [
        aligned_pdb_path,
        average_pdb_path,
        variances_path,
        residuals_path,
        transforms_path,
    ]
    for path in expected_paths:
        if not path.exists():
            raise FileNotFoundError(f"THESEUS did not create the expected output file: {path}")

    return TheseusRunResult(
        executable_path=executable_path,
        command=tuple(command),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        total_runtime_seconds=total_runtime,
        subprocess_runtime_seconds=total_runtime,
        input_write_seconds=0.0,
        output_read_seconds=0.0,
        input_paths=tuple(resolved_input_paths),
        output_root=output_root,
        output_directory=output_directory,
        aligned_pdb_path=aligned_pdb_path,
        average_pdb_path=average_pdb_path,
        variances_path=variances_path,
        residuals_path=residuals_path,
        transforms_path=transforms_path,
        aligned_coords=None,
        parsed_output=parse_theseus_stdout(completed.stdout),
        variance_records=parse_theseus_variances(variances_path),
        output_core_atom_count=None,
        output_core_atom_fraction=None,
    )


def run_theseus_on_coordinates(
    coords: np.ndarray,
    *,
    workdir: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    env_var: str = "THESEUS_EXECUTABLE",
    timeout_seconds: float | None = None,
    input_name: str = "trajectory_input.pdb",
    output_root: str = "theseus_run",
) -> TheseusRunResult:
    """Write coordinates, run THESEUS, and read aligned output back."""

    coords = ensure_xyz_frames(coords)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / input_name

    total_start = time.perf_counter()

    write_start = time.perf_counter()
    write_multimodel_pdb(coords, input_path)
    input_write_seconds = time.perf_counter() - write_start

    run = run_theseus_on_files(
        [input_path],
        output_root=output_root,
        output_directory=workdir,
        executable=executable,
        config_path=config_path,
        env_var=env_var,
        timeout_seconds=timeout_seconds,
    )

    read_start = time.perf_counter()
    aligned_coords = read_multimodel_pdb_coords(run.aligned_pdb_path, n_atoms=coords.shape[1])
    variance_records = parse_theseus_variances(run.variances_path)
    output_read_seconds = time.perf_counter() - read_start

    output_core_atom_count = sum(int(record.is_core) for record in variance_records) or None
    output_core_atom_fraction = (
        float(output_core_atom_count / coords.shape[1]) if output_core_atom_count is not None else None
    )

    total_runtime = time.perf_counter() - total_start
    return TheseusRunResult(
        executable_path=run.executable_path,
        command=run.command,
        return_code=run.return_code,
        stdout=run.stdout,
        stderr=run.stderr,
        total_runtime_seconds=total_runtime,
        subprocess_runtime_seconds=run.subprocess_runtime_seconds,
        input_write_seconds=input_write_seconds,
        output_read_seconds=output_read_seconds,
        input_paths=run.input_paths,
        output_root=run.output_root,
        output_directory=run.output_directory,
        aligned_pdb_path=run.aligned_pdb_path,
        average_pdb_path=run.average_pdb_path,
        variances_path=run.variances_path,
        residuals_path=run.residuals_path,
        transforms_path=run.transforms_path,
        aligned_coords=aligned_coords,
        parsed_output=run.parsed_output,
        variance_records=variance_records,
        output_core_atom_count=output_core_atom_count,
        output_core_atom_fraction=output_core_atom_fraction,
    )
