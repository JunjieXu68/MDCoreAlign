"""Wrapper utilities for the real external MDLovoFit executable.

This module never modifies MDLovoFit itself. It only resolves the executable,
prepares inputs, executes the program, captures outputs, and adapts the result
into a Python-friendly format.
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

from .pdb_writer import write_multimodel_ca_pdb_streaming
from .utils import ensure_xyz_frames


@dataclass(slots=True)
class MdlovofitParsedOutput:
    """Best-effort parse of selected MDLovoFit stdout metrics."""

    fraction_of_best_aligned_atoms: float | None
    total_number_of_atoms: int | None
    number_of_atoms_of_selection: int | None
    total_number_of_atoms_found: int | None
    different_assignment_for_each_frame: str | None
    number_of_trials_for_each_alignment: int | None
    frame_rmsd_row_count: int | None
    average_rmsd_least_mobile: float | None
    average_rmsd_most_mobile: float | None
    average_rmsd_all_atoms: float | None


@dataclass(slots=True)
class MdlovofitRunResult:
    """Captured result of one external MDLovoFit execution."""

    executable_path: Path
    command: tuple[str, ...]
    requested_fraction: float
    return_code: int
    stdout: str
    stderr: str
    total_runtime_seconds: float
    subprocess_runtime_seconds: float
    input_write_seconds: float
    output_read_seconds: float
    input_paths: tuple[Path, ...]
    output_path: Path
    aligned_coords: np.ndarray | None
    parsed_output: MdlovofitParsedOutput
    output_subset_atom_count: int | None
    output_subset_atom_fraction: float | None
    output_subset_mask_binary: bool | None
    output_subset_mask_constant_across_frames: bool | None


def _read_executable_from_config(config_path: Path) -> Path | None:
    """Read an optional MDLovoFit executable path from a JSON config file."""

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a JSON object")

    direct = payload.get("mdlovofit_executable")
    if isinstance(direct, str) and direct.strip():
        return Path(direct)

    executables = payload.get("executables")
    if isinstance(executables, dict):
        nested = executables.get("mdlovofit")
        if isinstance(nested, str) and nested.strip():
            return Path(nested)
    return None


def resolve_mdlovofit_executable(
    executable: str | Path | None = None,
    *,
    env_var: str = "MDLOVOFIT_EXECUTABLE",
    config_path: str | Path | None = None,
) -> Path:
    """Resolve the MDLovoFit executable from CLI, env, config, or PATH."""

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

    which_candidate = shutil.which("mdlovofit")
    if which_candidate:
        candidates.append(Path(which_candidate))

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.exists() and os.access(resolved, os.X_OK):
            return resolved

    raise FileNotFoundError(
        "Could not find the MDLovoFit executable. Set --mdlovofit-executable, "
        f"the {env_var} environment variable, a JSON config file, or ensure "
        "'mdlovofit' is available on PATH."
    )


def build_mdlovofit_command(
    executable_path: str | Path,
    *,
    fraction: float,
    output_path: str | Path,
    input_paths: list[str | Path] | tuple[str | Path, ...],
) -> list[str]:
    """Build the external MDLovoFit command line."""

    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not input_paths:
        raise ValueError("at least one input path is required")

    return [
        str(Path(executable_path)),
        "-f",
        f"{fraction:.6f}",
        "-t",
        str(Path(output_path)),
        *[str(Path(path)) for path in input_paths],
    ]


def write_multimodel_pdb(coords: np.ndarray, path: str | Path) -> None:
    """Write one trajectory array to a multi-model CA-only PDB file."""

    write_multimodel_ca_pdb_streaming(ensure_xyz_frames(coords), path)


def read_multimodel_pdb_coords(path: str | Path, n_atoms: int) -> np.ndarray:
    """Read aligned multi-model PDB coordinates produced by MDLovoFit."""

    coords, _ = read_multimodel_pdb_with_occupancy(path, n_atoms)
    return coords


def read_multimodel_pdb_with_occupancy(path: str | Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Read aligned multi-model PDB coordinates and occupancy values."""

    path = Path(path)
    frames: list[list[list[float]]] = []
    occupancies: list[list[float]] = []
    current: list[list[float]] = []
    current_occupancies: list[float] = []

    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            current.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            current_occupancies.append(float(line[54:60]))
            if len(current) == n_atoms:
                frames.append(current)
                occupancies.append(current_occupancies)
                current = []
                current_occupancies = []

    if current or current_occupancies:
        raise ValueError("aligned PDB ended with an incomplete frame")
    if not frames:
        raise ValueError(f"no aligned frames could be parsed from {path}")
    return np.asarray(frames, dtype=float), np.asarray(occupancies, dtype=float)


def parse_mdlovofit_stdout(stdout: str) -> MdlovofitParsedOutput:
    """Parse selected MDLovoFit metrics without guessing missing fields."""

    float_patterns = {
        "fraction_of_best_aligned_atoms": r"Fraction of best aligned atoms to consider:\s+([0-9.]+)",
        "average_rmsd_least_mobile": r"Average RMSD of least mobile atoms:\s+([0-9.]+)",
        "average_rmsd_most_mobile": r"Average RMSD of most mobile atoms:\s+([0-9.]+)",
        "average_rmsd_all_atoms": r"Average RMSD of all atoms:\s+([0-9.]+)",
    }
    int_patterns = {
        "total_number_of_atoms": r"Total number of atoms:\s+([0-9]+)",
        "number_of_atoms_of_selection": r"Number of atoms of selection:\s+([0-9]+)",
        "total_number_of_atoms_found": r"Total number of atoms found:\s+([0-9]+)",
        "number_of_trials_for_each_alignment": r"Number of trials for each alignment:\s+([0-9]+)",
    }
    string_patterns = {
        "different_assignment_for_each_frame": r"Different assignment for each frame:\s+(\S+)",
    }
    parsed: dict[str, float | int | str | None] = {}
    for key, pattern in float_patterns.items():
        match = re.search(pattern, stdout)
        parsed[key] = float(match.group(1)) if match else None
    for key, pattern in int_patterns.items():
        match = re.search(pattern, stdout)
        parsed[key] = int(match.group(1)) if match else None
    for key, pattern in string_patterns.items():
        match = re.search(pattern, stdout)
        parsed[key] = match.group(1) if match else None

    frame_row_count = 0
    for line in stdout.splitlines():
        if re.match(r"^\s*\d+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s*$", line):
            frame_row_count += 1
    parsed["frame_rmsd_row_count"] = frame_row_count or None
    return MdlovofitParsedOutput(**parsed)


def run_mdlovofit_on_files(
    input_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    fraction: float,
    output_path: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    env_var: str = "MDLOVOFIT_EXECUTABLE",
    timeout_seconds: float | None = None,
) -> MdlovofitRunResult:
    """Execute the real external MDLovoFit binary on one or more PDB inputs."""

    executable_path = resolve_mdlovofit_executable(
        executable,
        env_var=env_var,
        config_path=config_path,
    )
    output_path = Path(output_path)
    command = build_mdlovofit_command(
        executable_path,
        fraction=fraction,
        output_path=output_path,
        input_paths=input_paths,
    )

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    total_runtime = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"MDLovoFit failed with return code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_path.exists():
        raise FileNotFoundError(f"MDLovoFit did not create the expected output file: {output_path}")

    return MdlovofitRunResult(
        executable_path=executable_path,
        command=tuple(command),
        requested_fraction=float(fraction),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        total_runtime_seconds=total_runtime,
        subprocess_runtime_seconds=total_runtime,
        input_write_seconds=0.0,
        output_read_seconds=0.0,
        input_paths=tuple(Path(path) for path in input_paths),
        output_path=output_path,
        aligned_coords=None,
        parsed_output=parse_mdlovofit_stdout(completed.stdout),
        output_subset_atom_count=None,
        output_subset_atom_fraction=None,
        output_subset_mask_binary=None,
        output_subset_mask_constant_across_frames=None,
    )


def run_mdlovofit_on_coordinates(
    coords: np.ndarray,
    *,
    fraction: float,
    workdir: str | Path,
    executable: str | Path | None = None,
    config_path: str | Path | None = None,
    env_var: str = "MDLOVOFIT_EXECUTABLE",
    timeout_seconds: float | None = None,
    input_name: str = "trajectory_input.pdb",
    output_name: str = "trajectory_aligned.pdb",
) -> MdlovofitRunResult:
    """Write coordinates, run MDLovoFit, and read aligned output back."""

    coords = ensure_xyz_frames(coords)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / input_name
    output_path = workdir / output_name

    total_start = time.perf_counter()

    write_start = time.perf_counter()
    write_multimodel_pdb(coords, input_path)
    input_write_seconds = time.perf_counter() - write_start

    run = run_mdlovofit_on_files(
        [input_path],
        fraction=fraction,
        output_path=output_path,
        executable=executable,
        config_path=config_path,
        env_var=env_var,
        timeout_seconds=timeout_seconds,
    )

    read_start = time.perf_counter()
    aligned_coords, occupancies = read_multimodel_pdb_with_occupancy(output_path, n_atoms=coords.shape[1])
    output_read_seconds = time.perf_counter() - read_start

    output_subset_mask_binary = bool(np.all(np.logical_or(np.isclose(occupancies, 0.0), np.isclose(occupancies, 1.0))))
    output_subset_mask_constant = bool(np.all(np.isclose(occupancies, occupancies[0])))
    if output_subset_mask_binary and output_subset_mask_constant:
        output_subset_atom_count = int(np.sum(occupancies[0] > 0.5))
        output_subset_atom_fraction = float(output_subset_atom_count / coords.shape[1])
    else:
        output_subset_atom_count = None
        output_subset_atom_fraction = None

    total_runtime = time.perf_counter() - total_start
    return MdlovofitRunResult(
        executable_path=run.executable_path,
        command=run.command,
        requested_fraction=run.requested_fraction,
        return_code=run.return_code,
        stdout=run.stdout,
        stderr=run.stderr,
        total_runtime_seconds=total_runtime,
        subprocess_runtime_seconds=run.subprocess_runtime_seconds,
        input_write_seconds=input_write_seconds,
        output_read_seconds=output_read_seconds,
        input_paths=run.input_paths,
        output_path=run.output_path,
        aligned_coords=aligned_coords,
        parsed_output=run.parsed_output,
        output_subset_atom_count=output_subset_atom_count,
        output_subset_atom_fraction=output_subset_atom_fraction,
        output_subset_mask_binary=output_subset_mask_binary,
        output_subset_mask_constant_across_frames=output_subset_mask_constant,
    )
