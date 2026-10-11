"""Export structure-coloring artifacts for stable-core visualization."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import MDAnalysis as mda

from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.trajectory_io import load_mdanalysis_trajectory

DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "analysis_selection": "protein and name CA",
        "render_selection": "protein",
    },
    "adk_equilibrium": {
        "label": "Adk_equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "analysis_selection": "protein and name CA",
        "render_selection": "protein",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="ifabp",
        help="Named dataset to visualize",
    )
    return parser.parse_args()


def residue_table(result, residue_ids: list[str]) -> list[dict[str, object]]:
    """Convert a pipeline result into per-residue records."""

    rows: list[dict[str, object]] = []
    for index, residue_id in enumerate(residue_ids):
        stability = float(result.residue_scores.combined[index])
        mobility = float(np.clip(1.0 - stability, 0.0, 1.0))
        rows.append(
            {
                "residue_id": residue_id,
                "residue_index": index + 1,
                "stability_score": stability,
                "mobility_score": mobility,
                "confidence": float(result.confidence[index]),
                "is_core": int(bool(result.core_mask[index])),
                "is_uncertain": int(bool(result.uncertain_mask[index])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write per-residue visualization metadata."""

    fieldnames = [
        "residue_id",
        "residue_index",
        "stability_score",
        "mobility_score",
        "confidence",
        "is_core",
        "is_uncertain",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assign_bfactors(atom_group, residue_rows: list[dict[str, object]], field: str) -> None:
    """Assign a residue-level value to every atom through tempfactors."""

    row_lookup = {str(row["residue_id"]): row for row in residue_rows}
    b_factors = np.zeros(atom_group.n_atoms, dtype=float)
    for atom_index, residue in enumerate(atom_group.residues):
        residue_id = f"{residue.resname}{residue.resid}"
        row = row_lookup.get(residue_id)
        value = 0.0 if row is None else float(row[field]) * 100.0
        mask = atom_group.resindices == residue.resindex
        b_factors[mask] = value
    atom_group.atoms.tempfactors = b_factors


def write_pymol_script(path: Path, pdb_path: Path, residue_rows: list[dict[str, object]]) -> None:
    """Write a PyMOL script that colors the structure by mobility."""

    core_resids = [
        str(int("".join(ch for ch in str(row["residue_id"]) if ch.isdigit())))
        for row in residue_rows
        if int(row["is_core"]) == 1
    ]
    uncertain_resids = [
        str(int("".join(ch for ch in str(row["residue_id"]) if ch.isdigit())))
        for row in residue_rows
        if int(row["is_uncertain"]) == 1
    ]
    core_selection = "+".join(core_resids) if core_resids else "none"
    uncertain_selection = "+".join(uncertain_resids) if uncertain_resids else "none"

    path.write_text(
        "\n".join(
            [
                f"load {pdb_path}",
                "hide everything",
                "show cartoon, all",
                "spectrum b, blue_white_red, all, minimum=0, maximum=100",
                "bg_color white",
                f"select fastcore_core, resi {core_selection}",
                "show sticks, fastcore_core and name CA",
                "color marine, fastcore_core",
                f"select fastcore_uncertain, resi {uncertain_selection}",
                "color yellow, fastcore_uncertain",
                "set cartoon_transparency, 0.10",
                "orient",
            ]
        )
        + "\n"
    )


def plot_profile(path: Path, residue_rows: list[dict[str, object]], label: str) -> None:
    """Plot residue-wise mobility and confidence profiles."""

    x = [int(row["residue_index"]) for row in residue_rows]
    mobility = [float(row["mobility_score"]) for row in residue_rows]
    confidence = [float(row["confidence"]) for row in residue_rows]
    core = [bool(row["is_core"]) for row in residue_rows]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, mobility, color="#c23b22", linewidth=1.5, label="mobility")
    ax.plot(x, confidence, color="#355c7d", linewidth=1.2, label="confidence")
    for xi, is_core in zip(x, core):
        if is_core:
            ax.axvspan(xi - 0.5, xi + 0.5, color="#8ecae6", alpha=0.15)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{label}: mobility and confidence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_structure_projection(path: Path, reference_positions: np.ndarray, residue_rows: list[dict[str, object]], label: str) -> None:
    """Plot a 2D PCA-like projection of the CA trace colored by mobility."""

    centered = reference_positions - reference_positions.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    mobility = np.asarray([float(row["mobility_score"]) for row in residue_rows], dtype=float)
    core = np.asarray([bool(row["is_core"]) for row in residue_rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(projected[:, 0], projected[:, 1], color="#cccccc", linewidth=1.0, zorder=1)
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=mobility,
        cmap="coolwarm",
        s=np.where(core, 42.0, 20.0),
        edgecolors="black",
        linewidths=np.where(core, 0.4, 0.0),
        zorder=2,
    )
    ax.set_title(f"{label}: structure projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="mobility")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Export structure visualization artifacts for one dataset."""

    args = parse_args()
    dataset = DATASETS[args.dataset]
    label = str(dataset["label"])
    topology = Path(dataset["topology"])
    trajectory = Path(dataset["trajectory"])
    analysis_selection = str(dataset["analysis_selection"])
    render_selection = str(dataset["render_selection"])

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    prefix = results_dir / args.dataset

    data = load_mdanalysis_trajectory(
        topology,
        trajectory,
        selection=analysis_selection,
        stride=1,
    )
    result = run_stable_core_pipeline(data)
    rows = residue_table(result, data.residue_ids)

    csv_path = prefix.with_name(f"{prefix.name}_structure_scores.csv")
    write_csv(csv_path, rows)

    universe = mda.Universe(str(topology), str(trajectory))
    if not hasattr(universe.atoms, "tempfactors"):
        universe.add_TopologyAttr("tempfactors")
    universe.trajectory[result.refined_reference_index]
    protein = universe.select_atoms(render_selection)
    assign_bfactors(protein, rows, field="mobility_score")

    pdb_path = prefix.with_name(f"{prefix.name}_mobility_bfactor.pdb")
    protein.write(str(pdb_path))

    pml_path = prefix.with_name(f"{prefix.name}_mobility_view.pml")
    write_pymol_script(pml_path, pdb_path, rows)

    profile_path = prefix.with_name(f"{prefix.name}_mobility_profile.png")
    plot_profile(profile_path, rows, label=label)

    reference_positions = data.coords[result.refined_reference_index]
    structure_map_path = prefix.with_name(f"{prefix.name}_structure_projection.png")
    plot_structure_projection(structure_map_path, reference_positions, rows, label=label)

    print(f"structure_scores_csv={csv_path}")
    print(f"mobility_bfactor_pdb={pdb_path}")
    print(f"pymol_script={pml_path}")
    print(f"mobility_profile_png={profile_path}")
    print(f"structure_projection_png={structure_map_path}")


if __name__ == "__main__":
    main()
