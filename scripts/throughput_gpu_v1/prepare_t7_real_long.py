#!/usr/bin/env python3
"""Extract CA arrays from local single-file continuous trajectory sources."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "throughput_gpu_v1" / "t7_real_long"
CACHE = OUTPUT / "input_cache"
SYSTEMS = (
    {
        "system": "YiiP_90ns",
        "topology": Path(
            "/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"
        ),
        "trajectory": Path(
            "/home/jjxu/MDAnalysis_data/yiip_equilibrium/"
            "YiiP_system_90ns_center.xtc"
        ),
    },
    {
        "system": "NhaA",
        "topology": Path(
            "/home/jjxu/MDAnalysis_data/nhaa_equilibrium/"
            "NhaA_non_water.gro"
        ),
        "trajectory": Path(
            "/home/jjxu/MDAnalysis_data/nhaa_equilibrium/"
            "NhaA_non_water.xtc"
        ),
    },
    {
        "system": "ADK",
        "topology": Path(
            "/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"
        ),
        "trajectory": Path(
            "/home/jjxu/MDAnalysis_data/adk_equilibrium/"
            "1ake_007-nowater-core-dt240ps.dcd"
        ),
    },
)


def _chain_labels(atoms) -> np.ndarray:
    chain_ids = [
        str(getattr(atom, "chainID", "") or "").strip()
        for atom in atoms
    ]
    segids = [
        str(getattr(atom, "segid", "") or "").strip()
        for atom in atoms
    ]
    unique_chains = {value for value in chain_ids if value}
    unique_segids = {value for value in segids if value}
    if len(unique_chains) > 1:
        labels = chain_ids
    elif len(unique_segids) > 1:
        labels = segids
    else:
        labels = []
        chain_number = 0
        previous_resid = None
        for atom in atoms:
            resid = int(atom.resid)
            if previous_resid is not None and resid <= previous_resid:
                chain_number += 1
            labels.append(f"chain_{chain_number + 1}")
            previous_resid = resid
    return np.asarray(labels, dtype="U16")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in SYSTEMS:
        if not config["topology"].exists() or not config["trajectory"].exists():
            rows.append(
                {
                    "system": config["system"],
                    "status": "missing_local_input",
                    "topology_path": str(config["topology"]),
                    "trajectory_path": str(config["trajectory"]),
                    "selection": "protein and name CA",
                    "n_frames": "",
                    "n_residues": "",
                    "n_chains": "",
                    "coordinate_unit": "angstrom",
                    "single_source_trajectory_file": True,
                    "stitched": False,
                    "cache_path": "",
                    "source_trajectory_bytes": "",
                    "notes": "Local source file is unavailable.",
                }
            )
            continue
        universe = mda.Universe(
            str(config["topology"]),
            str(config["trajectory"]),
        )
        selection = universe.select_atoms("protein and name CA")
        if selection.n_atoms < 3:
            raise ValueError(
                f"{config['system']} has fewer than three selected CA atoms"
            )
        coords = np.empty(
            (len(universe.trajectory), selection.n_atoms, 3),
            dtype=np.float32,
        )
        for frame_index, _ in enumerate(universe.trajectory):
            coords[frame_index] = selection.positions
        chains = _chain_labels(selection)
        residue_ids = np.asarray(
            [str(residue.resid) for residue in selection.residues],
            dtype="U16",
        )
        residue_names = np.asarray(
            [str(residue.resname) for residue in selection.residues],
            dtype="U8",
        )
        if residue_ids.size != selection.n_atoms:
            residue_ids = np.asarray(
                [str(atom.resid) for atom in selection],
                dtype="U16",
            )
            residue_names = np.asarray(
                [str(atom.resname) for atom in selection],
                dtype="U8",
            )
        cache_path = CACHE / f"{config['system']}_ca_coords.npz"
        np.savez_compressed(
            cache_path,
            coords=coords,
            chain_ids=chains,
            residue_ids=residue_ids,
            residue_names=residue_names,
        )
        rows.append(
            {
                "system": config["system"],
                "status": "ready",
                "topology_path": str(config["topology"]),
                "trajectory_path": str(config["trajectory"]),
                "selection": "protein and name CA",
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "n_chains": len(set(chains.tolist())),
                "coordinate_unit": "angstrom",
                "single_source_trajectory_file": True,
                "stitched": False,
                "cache_path": str(cache_path.relative_to(ROOT)),
                "source_trajectory_bytes": config["trajectory"].stat().st_size,
                "notes": (
                    "Analyzed as a single time-ordered trajectory file. "
                    "No replicate stitching was performed."
                ),
            }
        )
    with (OUTPUT / "real_long_input_manifest.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "input_preparation.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "python": __import__("sys").executable,
                "MDAnalysis_version": mda.__version__,
                "cache_is_benchmark_input_not_timed_IO": True,
                "systems": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
