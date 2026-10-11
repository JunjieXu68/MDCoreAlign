"""Tiny smoke benchmark for the scaffold package."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import Trajectory, identity_transform, mean_rmsd, top_fraction_core_mask, uniform_weights


def main() -> None:
    """Run a tiny benchmark that exercises the project scaffold."""

    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=float,
    )
    start = time.perf_counter()
    trajectory = Trajectory(coords=coords)
    transform = identity_transform()
    weights = uniform_weights(trajectory.n_atoms)
    core_mask = top_fraction_core_mask(weights, fraction=0.5)
    elapsed = time.perf_counter() - start

    print("Smoke benchmark")
    print(f"frames={trajectory.n_frames} atoms={trajectory.n_atoms}")
    print(f"identity_rmsd={mean_rmsd(trajectory.coords, trajectory.coords[0]):.6f}")
    print(f"core_size={int(core_mask.sum())}")
    print(f"translation_norm={np.linalg.norm(transform.translation):.6f}")
    print(f"time_s={elapsed:.6f}")


if __name__ == "__main__":
    main()
