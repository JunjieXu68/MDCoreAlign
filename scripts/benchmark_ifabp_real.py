"""Benchmark the current pipeline and mdlovofit on named real datasets."""

from __future__ import annotations

import argparse
import csv
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

from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.alignment import align_trajectory
from mdstablecore.quality_metrics import jaccard_similarity
from mdstablecore.trajectory_io import load_mdanalysis_trajectory

MDLOVOFIT_BIN = Path("/usr/local/bin/mdlovofit")
RESULTS_DIR = ROOT / "results"

DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "summary_name": "ifabp_benchmark_summary.csv",
        "workdir_name": "_tmp_ifabp_cli",
    },
    "adk_equilibrium": {
        "label": "Adk_equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "summary_name": "adk_equilibrium_benchmark_summary.csv",
        "workdir_name": "_tmp_adk_equilibrium_cli",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="ifabp",
        help="Named real dataset to benchmark",
    )
    return parser.parse_args()


def write_pdb(coords: np.ndarray, path: Path) -> None:
    """Write a CA trajectory as a multi-model PDB."""

    with path.open("w") as handle:
        for model_index, frame in enumerate(coords, start=1):
            serial = 1
            handle.write(f"MODEL     {model_index}\n")
            for residue_index, xyz in enumerate(frame, start=1):
                handle.write(
                    f"ATOM  {serial:5d}  CA  ALA A{residue_index:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
                )
                serial += 1
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def read_aligned(path: Path, n_atoms: int) -> np.ndarray:
    """Read mdlovofit output coordinates grouped by atom count."""

    frames = []
    current = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if len(current) == n_atoms:
            frames.append(current)
            current = []
    if not frames:
        raise ValueError(f"No aligned frames could be parsed from {path}")
    return np.asarray(frames, dtype=float)


def rmsf_mask(aligned: np.ndarray, fraction: float) -> np.ndarray:
    """Define a proxy stable core from the lowest-RMSF residues."""

    mean_positions = aligned.mean(axis=0)
    rmsf = np.sqrt(np.mean(np.sum((aligned - mean_positions[None, :, :]) ** 2, axis=2), axis=0))
    n_selected = max(1, int(round(fraction * rmsf.shape[0])))
    selected = np.argsort(rmsf)[:n_selected]
    mask = np.zeros(rmsf.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def parse_rmsd(stdout: str) -> tuple[float, float]:
    """Parse mean low-fraction and all-atom RMSD from mdlovofit stdout."""

    low = float(re.search(r"Average RMSD of least mobile atoms:\s+([0-9.]+)", stdout).group(1))
    all_atoms = float(re.search(r"Average RMSD of all atoms:\s+([0-9.]+)", stdout).group(1))
    return low, all_atoms


def format_mask(mask: np.ndarray, residue_ids: list[str]) -> str:
    """Format a residue mask into a comma-separated label string."""

    return ",".join([residue_id for residue_id, keep in zip(residue_ids, mask) if keep])


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the benchmark summary rows to CSV."""

    fieldnames = [
        "method",
        "fraction",
        "time_s",
        "core_size",
        "confidence_mean",
        "core_rmsd",
        "all_rmsd",
        "overlap_to_pipeline",
        "core_residues",
        "rmsd_max",
        "reference_mode",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_rmsd_series(
    coords: np.ndarray,
    reference: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-frame RMSD to a fixed reference."""

    if atom_mask is not None:
        coords = coords[:, atom_mask, :]
        reference = reference[atom_mask]
    squared = np.sum((coords - reference[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=1))


def mean_series_rmsd(series: np.ndarray) -> float:
    """Compute the mean RMSD of a per-frame series."""

    return float(np.mean(series))


def max_series_rmsd(series: np.ndarray) -> float:
    """Compute the max RMSD of a per-frame series."""

    return float(np.max(series))


def write_timeseries_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a long-format RMSD time-series CSV."""

    fieldnames = [
        "frame_index",
        "method",
        "fraction",
        "rmsd",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_baseline_timeseries(
    data,
    result,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build article-style RMSD baselines to the first frame."""

    first_reference = data.coords[0]
    core_atom_mask = result.core_mask[data.atom_to_residue]

    raw_series = frame_rmsd_series(data.coords, first_reference)
    standard_aligned = align_trajectory(data.coords, first_reference)
    standard_series = frame_rmsd_series(standard_aligned, first_reference)
    pipeline_corefit = align_trajectory(
        data.coords,
        first_reference,
        atom_mask=core_atom_mask if np.any(core_atom_mask) else None,
    )
    pipeline_all_series = frame_rmsd_series(pipeline_corefit, first_reference)
    pipeline_core_series = frame_rmsd_series(
        pipeline_corefit,
        first_reference,
        atom_mask=core_atom_mask if np.any(core_atom_mask) else None,
    )

    summary_rows = [
        {
            "method": "raw_to_first",
            "fraction": "",
            "time_s": "",
            "core_size": "",
            "confidence_mean": "",
            "core_rmsd": "",
            "all_rmsd": f"{mean_series_rmsd(raw_series):.6f}",
            "overlap_to_pipeline": "",
            "core_residues": "",
            "rmsd_max": f"{max_series_rmsd(raw_series):.6f}",
            "reference_mode": "first_frame_no_alignment",
        },
        {
            "method": "standard_all_ca_to_first",
            "fraction": "",
            "time_s": "",
            "core_size": "",
            "confidence_mean": "",
            "core_rmsd": "",
            "all_rmsd": f"{mean_series_rmsd(standard_series):.6f}",
            "overlap_to_pipeline": "",
            "core_residues": "",
            "rmsd_max": f"{max_series_rmsd(standard_series):.6f}",
            "reference_mode": "first_frame_all_ca_alignment",
        },
        {
            "method": "pipeline_corefit_to_first",
            "fraction": "",
            "time_s": "",
            "core_size": str(int(result.core_mask.sum())),
            "confidence_mean": f"{result.quality.confidence_mean:.6f}",
            "core_rmsd": f"{mean_series_rmsd(pipeline_core_series):.6f}",
            "all_rmsd": f"{mean_series_rmsd(pipeline_all_series):.6f}",
            "overlap_to_pipeline": "1.000000",
            "core_residues": "",
            "rmsd_max": f"{max_series_rmsd(pipeline_all_series):.6f}",
            "reference_mode": "first_frame_pipeline_core_alignment",
        },
    ]

    timeseries_rows: list[dict[str, str]] = []
    series_map = {
        "raw_to_first": ("", raw_series),
        "standard_all_ca_to_first": ("", standard_series),
        "pipeline_corefit_all_to_first": ("", pipeline_all_series),
        "pipeline_corefit_core_to_first": ("", pipeline_core_series),
    }
    for method, (fraction, series) in series_map.items():
        for frame_index, rmsd in enumerate(series):
            timeseries_rows.append(
                {
                    "frame_index": str(frame_index),
                    "method": method,
                    "fraction": fraction,
                    "rmsd": f"{float(rmsd):.6f}",
                }
            )

    return summary_rows, timeseries_rows


def main() -> None:
    """Run the selected real-dataset benchmark."""

    args = parse_args()
    dataset = DATASETS[args.dataset]
    label = str(dataset["label"])
    topology = Path(dataset["topology"])
    trajectory = Path(dataset["trajectory"])
    selection = str(dataset["selection"])
    summary_name = str(dataset["summary_name"])
    workdir = ROOT / str(dataset["workdir_name"])

    workdir.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_mdanalysis_trajectory(
        topology,
        trajectory,
        selection=selection,
        stride=1,
    )
    summary_rows: list[dict[str, str]] = []

    start = time.perf_counter()
    result = run_stable_core_pipeline(data)
    pipeline_time = time.perf_counter() - start
    pipeline_core = format_mask(result.core_mask, data.residue_ids)

    print(f"Pipeline result ({label})")
    print(f"frames={data.n_frames} residues={data.n_residues}")
    print(
        f"time={pipeline_time:.3f}s "
        f"core_size={int(result.core_mask.sum())} "
        f"conf_mean={result.quality.confidence_mean:.3f} "
        f"core_rmsd={result.quality.core_rmsd_mean:.3f} "
        f"all_rmsd={result.quality.all_atom_rmsd_mean:.3f}"
    )
    print("pipeline_core=" + pipeline_core)
    print()
    summary_rows.append(
        {
            "method": "pipeline",
            "fraction": "",
            "time_s": f"{pipeline_time:.6f}",
            "core_size": str(int(result.core_mask.sum())),
            "confidence_mean": f"{result.quality.confidence_mean:.6f}",
            "core_rmsd": f"{result.quality.core_rmsd_mean:.6f}",
            "all_rmsd": f"{result.quality.all_atom_rmsd_mean:.6f}",
            "overlap_to_pipeline": "1.000000",
            "core_residues": pipeline_core,
            "rmsd_max": "",
            "reference_mode": "pipeline_internal_reference",
        }
    )

    baseline_rows, timeseries_rows = build_baseline_timeseries(data, result)
    summary_rows.extend(baseline_rows)

    print(f"Real mdlovofit benchmark ({label})")
    print("Note: overlap uses a proxy core defined by the lowest-RMSF CA residues after mdlovofit alignment")
    for fraction in (0.5, 0.6, 0.7, 0.8):
        input_pdb = workdir / f"{args.dataset}_f{fraction:.2f}_input.pdb"
        output_pdb = workdir / f"{args.dataset}_f{fraction:.2f}_aligned.pdb"
        write_pdb(data.coords, input_pdb)

        start = time.perf_counter()
        completed = subprocess.run(
            [str(MDLOVOFIT_BIN), "-f", f"{fraction:.2f}", "-t", str(output_pdb), str(input_pdb)],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.perf_counter() - start

        aligned = read_aligned(output_pdb, n_atoms=data.coords.shape[1])
        mask = rmsf_mask(aligned, fraction)
        overlap = jaccard_similarity(mask, result.core_mask)
        low_rmsd, all_rmsd = parse_rmsd(completed.stdout)
        proxy_core = format_mask(mask, data.residue_ids)

        print(
            f"fraction={fraction:.2f} "
            f"time={elapsed:.3f}s "
            f"proxy_core_size={int(mask.sum())} "
            f"overlap_to_pipeline={overlap:.3f} "
            f"rmsd_low={low_rmsd:.3f} "
            f"rmsd_all={all_rmsd:.3f}"
        )
        print("proxy_core=" + proxy_core)
        summary_rows.append(
            {
                "method": "mdlovofit_proxy",
                "fraction": f"{fraction:.2f}",
                "time_s": f"{elapsed:.6f}",
                "core_size": str(int(mask.sum())),
                "confidence_mean": "",
                "core_rmsd": f"{low_rmsd:.6f}",
                "all_rmsd": f"{all_rmsd:.6f}",
                "overlap_to_pipeline": f"{overlap:.6f}",
                "core_residues": proxy_core,
                "rmsd_max": "",
                "reference_mode": "mdlovofit_binary_reference",
            }
        )

    summary_path = RESULTS_DIR / summary_name
    write_summary_csv(summary_path, summary_rows)
    timeseries_path = RESULTS_DIR / summary_name.replace("_benchmark_summary.csv", "_rmsd_timeseries.csv")
    write_timeseries_csv(timeseries_path, timeseries_rows)
    print()
    print(f"summary_csv={summary_path}")
    print(f"timeseries_csv={timeseries_path}")


if __name__ == "__main__":
    main()
