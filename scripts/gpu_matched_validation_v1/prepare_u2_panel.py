#!/usr/bin/env python3
"""Prepare a non-stitched real continuous trajectory panel for U1-U6."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "gpu_matched_validation_v1" / "u2_real_panel"
CACHE = OUTPUT / "input_cache"
ALL_DOMAIN = (
    ROOT
    / "results/mdcath_benchmark/all_domain_canonical/aggregate/"
    "all_domain_canonical_per_domain_summary.tsv"
)
T7 = ROOT / "results" / "throughput_gpu_v1" / "t7_real_long"
MD_DOMAINS = (
    ("4a18O00", 1),
    ("4a18O00", 2),
    ("4cgyA01", 1),
    ("4cgyA01", 2),
    ("4cu9A00", 1),
    ("3a2pA00", 1),
    ("3a2pA00", 2),
    ("3h7lB02", 1),
    ("4g6hA00", 1),
    ("3danA00", 1),
    ("3wa6A00", 1),
)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _all_domain_rows() -> dict[str, dict[str, str]]:
    with ALL_DOMAIN.open(newline="") as handle:
        return {
            row["domain_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("domain_id")
        }


def _h5_path(row: dict[str, str]) -> Path:
    match = re.search(r"h5_file=([^;]+)", row.get("notes", ""))
    if match is None:
        raise ValueError(f"missing h5_file provenance for {row['domain_id']}")
    return Path(match.group(1))


def _local_panel() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (T7 / "real_long_input_manifest.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    **row,
                    "trajectory_id": row["system"],
                    "source_type": "local_single_file_trajectory",
                    "temperature_K": "not_recorded_here",
                    "replicate": "single_file",
                    "continuity_class": "single_time_ordered_trajectory_file",
                    "biological_validation_role": "primary_real_continuous",
                    "benchmark_size_class": (
                        "long" if int(row["n_frames"]) >= 1000 else "short"
                    ),
                }
            )
    return rows


def _prepare_ifabp() -> dict[str, object]:
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError(
            "Run panel preparation with the MDAnalysis-enabled interpreter."
        ) from exc

    topology = Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf")
    trajectory = Path(
        "/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"
    )
    universe = mda.Universe(str(topology), str(trajectory))
    atoms = universe.select_atoms("protein and name CA")
    coords = np.empty(
        (len(universe.trajectory), atoms.n_atoms, 3),
        dtype=np.float32,
    )
    for frame_index, _ in enumerate(universe.trajectory):
        coords[frame_index] = atoms.positions
    chains = np.asarray(
        [
            str(atom.segid or atom.chainID or "A").strip() or "A"
            for atom in atoms
        ],
        dtype="U16",
    )
    residue_ids = np.asarray([str(atom.resid) for atom in atoms], dtype="U16")
    residue_names = np.asarray(
        [str(atom.resname) for atom in atoms],
        dtype="U8",
    )
    cache = CACHE / "IFABP_ca_coords.npz"
    np.savez_compressed(
        cache,
        coords=coords,
        chain_ids=chains,
        residue_ids=residue_ids,
        residue_names=residue_names,
    )
    return {
        "system": "I-FABP",
        "trajectory_id": "I-FABP_single_DCD",
        "status": "ready",
        "topology_path": str(topology),
        "trajectory_path": str(trajectory),
        "selection": "protein and name CA",
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "n_chains": len(set(chains.tolist())),
        "coordinate_unit": "angstrom",
        "single_source_trajectory_file": True,
        "stitched": False,
        "cache_path": cache.relative_to(ROOT).as_posix(),
        "source_trajectory_bytes": trajectory.stat().st_size,
        "source_type": "local_single_file_trajectory",
        "temperature_K": "not_recorded_here",
        "replicate": "single_file",
        "continuity_class": "single_time_ordered_trajectory_file",
        "biological_validation_role": "primary_real_continuous",
        "benchmark_size_class": "short",
        "notes": (
            "Single DCD analyzed in stored frame order; no replicate stitching."
        ),
    }


def _prepare_mdcath(
    domain_id: str,
    replicate: int,
    source: dict[str, str],
) -> dict[str, object]:
    from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
        extract_ca_trajectory,
    )

    h5_path = _h5_path(source)
    coords, _, residue_metadata, metadata = extract_ca_trajectory(
        h5_path,
        coordinate_path=f"/{domain_id}/320/{replicate - 1}/coords",
        min_ca_atoms=30,
        min_frames=20,
    )
    chains = np.asarray(
        [str(item.get("chain", "0")) for item in residue_metadata],
        dtype="U16",
    )
    residue_ids = np.asarray(
        [str(item.get("residue_id", index + 1)) for index, item in enumerate(residue_metadata)],
        dtype="U16",
    )
    residue_names = np.asarray(
        [str(item.get("resname", "UNK")) for item in residue_metadata],
        dtype="U8",
    )
    trajectory_id = f"{domain_id}__T320__R{replicate}"
    cache = CACHE / f"{trajectory_id}_ca_coords.npz"
    np.savez_compressed(
        cache,
        coords=np.asarray(coords, dtype=np.float32),
        chain_ids=chains,
        residue_ids=residue_ids,
        residue_names=residue_names,
    )
    metadata_path = CACHE / f"{trajectory_id}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return {
        "system": trajectory_id,
        "trajectory_id": trajectory_id,
        "status": "ready",
        "topology_path": f"{h5_path}::{domain_id}/pdb",
        "trajectory_path": (
            f"{h5_path}::/{domain_id}/320/{replicate - 1}/coords"
        ),
        "selection": "CA atoms from embedded PDB metadata",
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "n_chains": len(set(chains.tolist())),
        "coordinate_unit": "angstrom",
        "single_source_trajectory_file": True,
        "stitched": False,
        "cache_path": cache.relative_to(ROOT).as_posix(),
        "source_trajectory_bytes": h5_path.stat().st_size,
        "source_type": "mdCATH_single_320K_replicate",
        "temperature_K": 320,
        "replicate": replicate,
        "continuity_class": "single_replicate_stored_frame_order",
        "biological_validation_role": "supporting_real_continuous",
        "benchmark_size_class": "short",
        "cath_class": source.get("cath_class", ""),
        "notes": (
            "One physical mdCATH replicate; no cross-replicate stitching. "
            "This approximately 500-frame case expands system diversity but "
            "is not described as an ultra-long trajectory."
        ),
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = _local_panel()
    rows.append(_prepare_ifabp())
    sources = _all_domain_rows()
    for domain_id, replicate in MD_DOMAINS:
        rows.append(_prepare_mdcath(domain_id, replicate, sources[domain_id]))
    _write_tsv(OUTPUT / "real_continuous_panel.tsv", rows)

    counts = {
        "n_trajectories": len(rows),
        "n_system_families": len(
            {
                str(row["system"]).split("__", 1)[0]
                for row in rows
            }
        ),
        "n_local_single_file": sum(
            row["source_type"] == "local_single_file_trajectory"
            for row in rows
        ),
        "n_mdcath_single_replicate": sum(
            row["source_type"] == "mdCATH_single_320K_replicate"
            for row in rows
        ),
        "n_long_ge_1000_frames": sum(
            int(row["n_frames"]) >= 1000 for row in rows
        ),
        "minimum_residues": min(int(row["n_residues"]) for row in rows),
        "maximum_residues": max(int(row["n_residues"]) for row in rows),
        "minimum_frames": min(int(row["n_frames"]) for row in rows),
        "maximum_frames": max(int(row["n_frames"]) for row in rows),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "CA only with explicit topology/PDB metadata",
        "all_inputs_unstitched": all(
            str(row["stitched"]).strip().lower() in {"false", "0"}
            for row in rows
        ),
        "counts": counts,
        "coverage_gaps": {
            "800_to_1500_CA": (
                "Unavailable in current local CA panel; largest is 752 CA."
            ),
            "2000_plus_CA": (
                "Unavailable in current local CA panel; not fabricated."
            ),
        },
        "rows": rows,
    }
    (OUTPUT / "panel_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT / "real_continuous_panel_report.md").write_text(
        "\n".join(
            [
                "# U2 real continuous trajectory panel",
                "",
                f"- Trajectories: `{counts['n_trajectories']}`",
                f"- System families: `{counts['n_system_families']}`",
                f"- Frame range: `{counts['minimum_frames']}`–`{counts['maximum_frames']}`",
                f"- CA residue range: `{counts['minimum_residues']}`–`{counts['maximum_residues']}`",
                f"- Trajectories with at least 1,000 frames: `{counts['n_long_ge_1000_frames']}`",
                "- Every entry is a single stored trajectory or one mdCATH replicate.",
                "- No stitched trajectory is used as biological validation.",
                "- The mdCATH entries are continuous replicate segments but are "
                "shorter than the three multi-thousand-frame local trajectories.",
                "- Current local data do not provide 800–1,500 or 2,000+ CA "
                "systems. Those requested size strata are documented as gaps.",
                "- PBC/nojump provenance remains limited to source metadata and "
                "is not inferred here.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
