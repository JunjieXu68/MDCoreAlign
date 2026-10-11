"""Benchmark the MVP against the real mdlovofit binary."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demo_synthetic import build_synthetic_trajectory
from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.quality_metrics import jaccard_similarity
from mdstablecore.trajectory_io import make_trajectory_data

MDLOVOFIT_BIN = Path("/usr/local/bin/mdlovofit")


def write_multimodel_pdb(coords: np.ndarray, path: Path) -> None:
    """Write a single-atom-per-residue trajectory as a multi-model PDB."""

    with path.open("w") as handle:
        serial = 1
        for model_index, frame in enumerate(coords, start=1):
            handle.write(f"MODEL     {model_index}\n")
            for residue_index, xyz in enumerate(frame, start=1):
                handle.write(
                    f"ATOM  {serial:5d}  CA  ALA A{residue_index:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
                )
                serial += 1
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def read_aligned_pdb(path: Path, n_atoms: int) -> np.ndarray:
    """Read mdlovofit output coordinates grouped by atom count."""

    frames: list[list[list[float]]] = []
    current: list[list[float]] = []

    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            current.append([x, y, z])
            if len(current) == n_atoms:
                frames.append(current)
                current = []

    if current:
        raise ValueError("Aligned PDB ended with an incomplete frame")

    return np.asarray(frames, dtype=float)


def residue_rmsf(coords: np.ndarray) -> np.ndarray:
    """Compute per-residue RMSF from aligned residue coordinates."""

    mean_positions = coords.mean(axis=0)
    squared = np.sum((coords - mean_positions[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=0))


def lowest_rmsf_mask(aligned_coords: np.ndarray, fraction: float) -> np.ndarray:
    """Define a proxy stable core from the lowest-RMSF residues."""

    rmsf = residue_rmsf(aligned_coords)
    n_selected = max(1, int(round(fraction * rmsf.shape[0])))
    selected = np.argsort(rmsf)[:n_selected]
    mask = np.zeros(rmsf.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def parse_average_rmsd(stdout: str) -> dict[str, float]:
    """Parse average RMSD values from mdlovofit stdout."""

    patterns = {
        "rmsd_low_mean": r"Average RMSD of least mobile atoms:\s+([0-9.]+)",
        "rmsd_high_mean": r"Average RMSD of most mobile atoms:\s+([0-9.]+)",
        "rmsd_all_mean": r"Average RMSD of all atoms:\s+([0-9.]+)",
    }
    parsed: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if not match:
            raise ValueError(f"Could not parse {key} from mdlovofit output")
        parsed[key] = float(match.group(1))
    return parsed


def run_real_mdlovofit(
    coords: np.ndarray,
    fraction: float,
    workdir: Path,
    binary_path: Path = MDLOVOFIT_BIN,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Run the installed mdlovofit binary and return aligned coordinates and metrics."""

    input_pdb = workdir / f"input_f{fraction:.2f}.pdb"
    output_pdb = workdir / f"aligned_f{fraction:.2f}.pdb"
    write_multimodel_pdb(coords, input_pdb)

    start = time.perf_counter()
    result = subprocess.run(
        [str(binary_path), "-f", f"{fraction:.2f}", "-t", str(output_pdb), str(input_pdb)],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    aligned = read_aligned_pdb(output_pdb, n_atoms=coords.shape[1])
    averages = parse_average_rmsd(result.stdout)
    return elapsed, aligned, averages


def main() -> None:
    """Benchmark the prototype against the real mdlovofit binary."""

    if not MDLOVOFIT_BIN.exists():
        raise FileNotFoundError(f"mdlovofit binary not found at {MDLOVOFIT_BIN}")

    coords, atom_to_residue, expected_core = build_synthetic_trajectory(n_frames=250, n_residues=30, atoms_per_residue=1)
    data = make_trajectory_data(coords, atom_to_residue)

    start = time.perf_counter()
    pipeline_result = run_stable_core_pipeline(data)
    pipeline_time = time.perf_counter() - start
    pipeline_jaccard = jaccard_similarity(pipeline_result.core_mask, expected_core)

    print("Pipeline result")
    print(
        f"time={pipeline_time:.3f}s "
        f"core_size={int(pipeline_result.core_mask.sum())} "
        f"jaccard={pipeline_jaccard:.3f}"
    )
    print(f"core residues={np.flatnonzero(pipeline_result.core_mask)}")
    print()

    workdir = ROOT / "_tmp_mdlovofit_binary"
    workdir.mkdir(exist_ok=True)

    print("Real mdlovofit benchmark")
    print("Note: core quality below uses a proxy core defined by the lowest-RMSF residues")
    for fraction in (0.5, 0.6, 0.7, 0.8):
        elapsed, aligned_coords, averages = run_real_mdlovofit(coords, fraction=fraction, workdir=workdir)
        proxy_core = lowest_rmsf_mask(aligned_coords, fraction=fraction)
        jaccard = jaccard_similarity(proxy_core, expected_core)
        print(
            f"fraction={fraction:.2f} "
            f"time={elapsed:.3f}s "
            f"proxy_core_size={int(proxy_core.sum())} "
            f"proxy_jaccard={jaccard:.3f} "
            f"rmsd_low={averages['rmsd_low_mean']:.3f} "
            f"rmsd_all={averages['rmsd_all_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
