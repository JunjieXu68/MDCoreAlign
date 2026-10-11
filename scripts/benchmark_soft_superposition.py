"""Benchmark the stage-1 soft superposition prototype on small datasets."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdstablecore.alignment import align_trajectory
from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.quality_metrics import jaccard_similarity, mean_rmsd
from mdstablecore.soft_superposition import soft_superpose_trajectory
from mdstablecore.trajectory_io import load_mdanalysis_trajectory, make_trajectory_data
from mdstablecore.types import PipelineConfig
from scripts.demo_synthetic import build_synthetic_trajectory

RESULTS_DIR = ROOT / "results"

DATASETS: dict[str, dict[str, object]] = {
    "synthetic": {
        "label": "Synthetic",
    },
    "ifabp": {
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "summary_name": "ifabp_soft_superposition_summary.csv",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ifabp")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--core-fraction", type=float, default=None)
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, str, Path | None]:
    """Load the selected benchmark dataset."""

    dataset = DATASETS[args.dataset]
    label = str(dataset["label"])
    summary_path = None if args.dataset == "synthetic" else RESULTS_DIR / str(dataset["summary_name"])

    if args.dataset == "synthetic":
        coords, _, expected_core = build_synthetic_trajectory(
            n_frames=args.max_frames or 250,
            atoms_per_residue=1,
        )
        return coords, expected_core, label, summary_path

    data = load_mdanalysis_trajectory(
        topology_path=Path(dataset["topology"]),
        trajectory_path=Path(dataset["trajectory"]),
        selection=str(dataset["selection"]),
        stride=args.stride,
        max_frames=args.max_frames,
    )
    return data.coords, None, label, summary_path


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the benchmark summary rows."""

    fieldnames = [
        "dataset",
        "method",
        "time_s",
        "all_rmsd",
        "self_core_rmsd",
        "core_fraction",
        "mean_iterations",
        "overlap_to_reference_core",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_method(
    dataset: str,
    method: str,
    elapsed: float,
    aligned_coords: np.ndarray,
    reference: np.ndarray,
    hard_core_mask: np.ndarray | None = None,
    iterations: np.ndarray | None = None,
    reference_core: np.ndarray | None = None,
) -> dict[str, str]:
    """Build one summary row for a benchmarked method."""

    atom_mask = hard_core_mask if hard_core_mask is not None and np.any(hard_core_mask) else None
    overlap = ""
    if hard_core_mask is not None and reference_core is not None:
        overlap = f"{jaccard_similarity(hard_core_mask, reference_core):.6f}"

    return {
        "dataset": dataset,
        "method": method,
        "time_s": f"{elapsed:.6f}",
        "all_rmsd": f"{mean_rmsd(aligned_coords, reference):.6f}",
        "self_core_rmsd": f"{mean_rmsd(aligned_coords, reference, atom_mask=atom_mask):.6f}" if atom_mask is not None else "",
        "core_fraction": f"{float(np.mean(hard_core_mask)):.6f}" if hard_core_mask is not None else "",
        "mean_iterations": f"{float(np.mean(iterations)):.6f}" if iterations is not None else "",
        "overlap_to_reference_core": overlap,
    }


def main() -> None:
    """Run the selected benchmark."""

    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    coords, expected_core, label, summary_path = load_dataset(args)
    reference = coords[0]

    reference_core = expected_core
    core_fraction = args.core_fraction
    if args.dataset == "ifabp":
        atom_to_residue = np.arange(coords.shape[1], dtype=int)
        data = make_trajectory_data(coords, atom_to_residue)
        baseline = run_stable_core_pipeline(data, config=PipelineConfig(algorithm="fastcore"))
        reference_core = baseline.core_mask
        if core_fraction is None:
            core_fraction = float(np.mean(baseline.core_mask))
    elif core_fraction is None and expected_core is not None:
        core_fraction = float(np.mean(expected_core))

    if core_fraction is None:
        core_fraction = 0.55

    rows: list[dict[str, str]] = []
    print(f"Soft superposition benchmark ({label})")
    print(f"frames={coords.shape[0]} atoms={coords.shape[1]} reference=first_frame core_fraction={core_fraction:.3f}")
    if reference_core is not None:
        print(f"reference_core_size={int(reference_core.sum())}")
    print()

    start = time.perf_counter()
    all_atom_aligned = align_trajectory(coords, reference)
    all_atom_time = time.perf_counter() - start
    rows.append(
        summarize_method(
            dataset=args.dataset,
            method="all_ca_alignment",
            elapsed=all_atom_time,
            aligned_coords=all_atom_aligned,
            reference=reference,
            reference_core=reference_core,
        )
    )
    print("All-CA alignment")
    print(f"time={all_atom_time:.3f}s all_rmsd={mean_rmsd(all_atom_aligned, reference):.3f}")
    print()

    start = time.perf_counter()
    cold = soft_superpose_trajectory(
        coords,
        reference=reference,
        warm_start=False,
        core_fraction=core_fraction,
        max_iterations=args.max_iterations,
    )
    cold_time = time.perf_counter() - start
    rows.append(
        summarize_method(
            dataset=args.dataset,
            method="soft_cold_start",
            elapsed=cold_time,
            aligned_coords=cold.aligned_coords,
            reference=reference,
            hard_core_mask=cold.hard_core_mask,
            iterations=cold.iterations_per_frame,
            reference_core=reference_core,
        )
    )
    print("Soft superposition (cold start)")
    print(
        f"time={cold_time:.3f}s "
        f"all_rmsd={mean_rmsd(cold.aligned_coords, reference):.3f} "
        f"self_core_rmsd={mean_rmsd(cold.aligned_coords, reference, atom_mask=cold.hard_core_mask):.3f} "
        f"core_size={int(cold.hard_core_mask.sum())} "
        f"mean_iterations={float(np.mean(cold.iterations_per_frame)):.2f}"
    )
    if reference_core is not None:
        print(f"overlap_to_reference_core={jaccard_similarity(cold.hard_core_mask, reference_core):.3f}")
    print()

    start = time.perf_counter()
    warm = soft_superpose_trajectory(
        coords,
        reference=reference,
        warm_start=True,
        core_fraction=core_fraction,
        max_iterations=args.max_iterations,
    )
    warm_time = time.perf_counter() - start
    rows.append(
        summarize_method(
            dataset=args.dataset,
            method="soft_warm_start",
            elapsed=warm_time,
            aligned_coords=warm.aligned_coords,
            reference=reference,
            hard_core_mask=warm.hard_core_mask,
            iterations=warm.iterations_per_frame,
            reference_core=reference_core,
        )
    )
    print("Soft superposition (warm start)")
    print(
        f"time={warm_time:.3f}s "
        f"all_rmsd={mean_rmsd(warm.aligned_coords, reference):.3f} "
        f"self_core_rmsd={mean_rmsd(warm.aligned_coords, reference, atom_mask=warm.hard_core_mask):.3f} "
        f"core_size={int(warm.hard_core_mask.sum())} "
        f"mean_iterations={float(np.mean(warm.iterations_per_frame)):.2f}"
    )
    if reference_core is not None:
        print(f"overlap_to_reference_core={jaccard_similarity(warm.hard_core_mask, reference_core):.3f}")
    print()

    if summary_path is not None:
        write_summary_csv(summary_path, rows)
        print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
