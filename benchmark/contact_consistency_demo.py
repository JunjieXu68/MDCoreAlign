"""Toy comparison of soft weights with and without contact-consistency regularization."""

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
    build_contact_graph,
    contact_graph_disagreement,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
)


def build_contact_toy() -> tuple[np.ndarray, np.ndarray, object]:
    """Build a toy structure with sequence-distant contacting residue pairs."""

    strand_a = np.column_stack([np.array([0.0, 2.0, 4.0, 6.0]), np.zeros(4), np.zeros(4)])
    strand_b = np.column_stack([np.array([0.0, 2.0, 4.0, 6.0]), np.full(4, 2.2), np.zeros(4)])
    reference = np.vstack([strand_a, strand_b])

    theta = 0.18
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.array([0.4, -0.3, 0.2], dtype=float)
    mobile = reference @ rotation.T + translation

    # Disturb two residues on the second strand so contacting pairs disagree.
    mobile[[5, 6]] += np.array([0.0, 0.8, 0.0], dtype=float)
    contact_graph = build_contact_graph(
        reference,
        cutoff=2.5,
        min_sequence_separation=4,
        max_neighbors=1,
    )
    return reference, mobile, contact_graph


def main() -> None:
    """Run a toy contact-regularization comparison and save per-atom weights."""

    output_dir = ROOT / "results" / "mdsuperpose_benchmarks_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, mobile, contact_graph = build_contact_toy()

    no_reg = solve_iterative_robust_alignment(
        reference,
        mobile,
        rule="huber",
        delta=0.30,
        max_iterations=20,
        tolerance=1e-6,
    )
    with_reg = solve_iterative_robust_alignment(
        reference,
        mobile,
        rule="huber",
        delta=0.30,
        max_iterations=20,
        tolerance=1e-6,
        contact_graph=contact_graph,
        contact_consistency_strength=0.5,
        contact_consistency_passes=2,
    )

    no_reg_core = select_core_by_target_fraction(no_reg.weights, fraction=0.5)
    with_reg_core = select_core_by_target_fraction(with_reg.weights, fraction=0.5)
    rows = []
    for atom_index in range(reference.shape[0]):
        rows.append(
            {
                "atom_index": atom_index,
                "weight_no_contact_reg": float(no_reg.weights[atom_index]),
                "weight_with_contact_reg": float(with_reg.weights[atom_index]),
                "core_no_contact_reg": int(no_reg_core.core_mask[atom_index]),
                "core_with_contact_reg": int(with_reg_core.core_mask[atom_index]),
            }
        )

    csv_path = output_dir / "contact_consistency_demo.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Toy contact-consistency comparison")
    print(f"contact_edge_count={contact_graph.edge_count}")
    print(f"without_reg_core={no_reg_core.core_indices.tolist()}")
    print(f"with_reg_core={with_reg_core.core_indices.tolist()}")
    print(f"without_reg_contact_disagreement={contact_graph_disagreement(no_reg.weights, contact_graph):.6f}")
    print(f"with_reg_contact_disagreement={contact_graph_disagreement(with_reg.weights, contact_graph):.6f}")
    print(f"without_reg_weights={np.round(no_reg.weights, 3).tolist()}")
    print(f"with_reg_weights={np.round(with_reg.weights, 3).tolist()}")
    print(f"csv_path={csv_path}")


if __name__ == "__main__":
    main()
