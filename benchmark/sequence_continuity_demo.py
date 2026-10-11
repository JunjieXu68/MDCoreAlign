"""Toy comparison of soft weights with and without sequence continuity regularization."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    benchmark_soft_weight_robust_alignment,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
)


def build_fragmented_core_toy() -> tuple[np.ndarray, np.ndarray]:
    """Build a toy paired-structure problem with isolated sequence holes."""

    reference = np.column_stack(
        [
            np.linspace(0.0, 11.0, 12),
            np.sin(np.linspace(0.0, 2.2, 12)),
            np.cos(np.linspace(0.0, 1.8, 12)),
        ]
    )
    theta = 0.12
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.array([0.3, -0.2, 0.1], dtype=float)
    mobile = reference @ rotation.T + translation

    # Add isolated disruptions inside an otherwise stable sequence segment.
    mobile[[4, 7]] += np.array([0.0, 0.9, 0.0], dtype=float)
    mobile[[0, 11]] += np.array([0.0, 1.3, 0.0], dtype=float)
    return reference, mobile


def main() -> None:
    """Run a toy sequence-regularization comparison and save per-atom weights."""

    output_dir = ROOT / "results" / "mdsuperpose_benchmarks_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, mobile = build_fragmented_core_toy()

    no_reg = solve_iterative_robust_alignment(
        reference,
        mobile,
        rule="huber",
        delta=0.35,
        max_iterations=20,
        tolerance=1e-6,
    )
    with_reg = solve_iterative_robust_alignment(
        reference,
        mobile,
        rule="huber",
        delta=0.35,
        max_iterations=20,
        tolerance=1e-6,
        sequence_continuity_strength=0.6,
        sequence_continuity_passes=2,
    )

    no_reg_core = select_core_by_target_fraction(no_reg.weights, fraction=0.5)
    with_reg_core = select_core_by_target_fraction(with_reg.weights, fraction=0.5)
    rows = []
    for atom_index in range(reference.shape[0]):
        rows.append(
            {
                "atom_index": atom_index,
                "weight_no_sequence_reg": float(no_reg.weights[atom_index]),
                "weight_with_sequence_reg": float(with_reg.weights[atom_index]),
                "core_no_sequence_reg": int(no_reg_core.core_mask[atom_index]),
                "core_with_sequence_reg": int(with_reg_core.core_mask[atom_index]),
            }
        )

    csv_path = output_dir / "sequence_continuity_demo.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Toy sequence-continuity comparison")
    print(f"without_reg_core={no_reg_core.core_indices.tolist()}")
    print(f"with_reg_core={with_reg_core.core_indices.tolist()}")
    print(f"without_reg_weighted_rmsd={no_reg.weighted_rmsd:.6f}")
    print(f"with_reg_weighted_rmsd={with_reg.weighted_rmsd:.6f}")
    print(f"csv_path={csv_path}")


if __name__ == "__main__":
    main()
