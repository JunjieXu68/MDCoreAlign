"""Synthetic outlier demo for the first iterative robust alignment prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import solve_iterative_robust_alignment, solve_rigid_alignment


def subset_rmsd(aligned_coords: np.ndarray, reference: np.ndarray, atom_mask: np.ndarray) -> float:
    """Compute RMSD on a selected atom subset."""

    squared = np.sum((aligned_coords[atom_mask] - reference[atom_mask]) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def main() -> None:
    """Compare unweighted and robust alignment on a toy outlier example."""

    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    theta = 0.45
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.array([1.5, -0.8, 0.4], dtype=float)
    y_clean = x @ rotation.T + translation
    y_noisy = y_clean.copy()
    y_noisy[6] += np.array([7.0, -6.0, 5.0], dtype=float)
    y_noisy[7] += np.array([-8.0, 5.0, -4.0], dtype=float)
    inlier_mask = np.array([True, True, True, True, True, True, False, False], dtype=bool)

    ordinary = solve_rigid_alignment(x, y_noisy)
    robust = solve_iterative_robust_alignment(
        x,
        y_noisy,
        rule="tukey",
        cutoff=3.0,
        max_iterations=25,
        tolerance=1e-6,
    )

    print("Ordinary rigid alignment")
    print(f"all_rmsd_to_noisy_target={ordinary.rmsd:.3f}")
    print(f"inlier_rmsd_to_clean_target={subset_rmsd(ordinary.aligned_coords, y_clean, inlier_mask):.3f}")
    print()

    print("Iterative robust alignment")
    print(f"all_rmsd_to_noisy_target={robust.rmsd:.3f}")
    print(f"weighted_rmsd={robust.weighted_rmsd:.3f}")
    print(f"inlier_rmsd_to_clean_target={subset_rmsd(robust.aligned_coords, y_clean, inlier_mask):.3f}")
    print(f"iterations={len(robust.history)} converged={robust.converged}")
    print("final_weights=" + ",".join(f"{value:.3f}" for value in robust.weights))
    print()

    print("History")
    print("iter  weighted_rmsd  weight_change  transform_change")
    for entry in robust.history:
        print(
            f"{entry.iteration:4d}  "
            f"{entry.weighted_rmsd:13.6f}  "
            f"{entry.weight_change:13.6f}  "
            f"{entry.transform_change:16.6f}"
        )


if __name__ == "__main__":
    main()
