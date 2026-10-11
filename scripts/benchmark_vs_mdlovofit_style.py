"""Benchmark the MVP against a MDLovoFit-style fraction baseline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from demo_synthetic import build_synthetic_trajectory
from mdstablecore.baseline import run_mdlovofit_style
from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.quality_metrics import jaccard_similarity
from mdstablecore.trajectory_io import make_trajectory_data


def main() -> None:
    """Benchmark the pipeline against a MDLovoFit-style approximation."""

    coords, atom_to_residue, expected_core = build_synthetic_trajectory(n_frames=250)
    data = make_trajectory_data(coords, atom_to_residue)

    start = time.perf_counter()
    pipeline_result = run_stable_core_pipeline(data)
    pipeline_time = time.perf_counter() - start
    pipeline_jaccard = jaccard_similarity(pipeline_result.core_mask, expected_core)

    print("Pipeline result")
    print(f"time={pipeline_time:.3f}s core_size={int(pipeline_result.core_mask.sum())} jaccard={pipeline_jaccard:.3f}")
    print(f"core residues={np.flatnonzero(pipeline_result.core_mask)}")
    print()
    print("MDLovoFit-style sweep")

    for fraction in (0.5, 0.6, 0.7, 0.8):
        start = time.perf_counter()
        rerun = run_mdlovofit_style(data, fraction=fraction)
        elapsed = time.perf_counter() - start
        jaccard = jaccard_similarity(rerun.selected_residue_mask, expected_core)
        print(
            f"fraction={rerun.fraction:.2f} "
            f"time={elapsed:.3f}s "
            f"core_size={int(rerun.selected_residue_mask.sum())} "
            f"jaccard={jaccard:.3f} "
            f"rmsd_low={rerun.rmsd_low_mean:.3f} "
            f"rmsd_all={rerun.rmsd_all_mean:.3f}"
        )


if __name__ == "__main__":
    main()
