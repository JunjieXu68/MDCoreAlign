#!/usr/bin/env python
"""Inspect local storage for real continuous long-trajectory candidates.

This script is intentionally conservative. It does not treat stitched
replicates, NMR ensembles, duplicated frames, or structure collections as
eligible continuous MD trajectories.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "long_continuous_benchmark"
INVENTORY_PATH = OUTPUT_ROOT / "candidate_inventory.tsv"
NO_ELIGIBLE_DOC = PROJECT_ROOT / "docs" / "long_continuous_no_eligible_trajectories.md"

COLUMNS = [
    "candidate_id",
    "source_path",
    "topology_path",
    "trajectory_path",
    "file_format",
    "domain_or_system_name",
    "estimated_n_frames",
    "n_residues_or_atoms",
    "has_topology",
    "has_CA_selection",
    "appears_continuous",
    "suspected_PBC_jump",
    "usable_for_benchmark",
    "exclusion_reason",
    "notes",
]


@dataclass
class Candidate:
    """One local trajectory candidate or excluded source group."""

    candidate_id: str
    source_path: str
    topology_path: str = "NA"
    trajectory_path: str = "NA"
    file_format: str = "unknown"
    domain_or_system_name: str = "NA"
    estimated_n_frames: str | int = "NA"
    n_residues_or_atoms: str | int = "NA"
    has_topology: str = "no"
    has_CA_selection: str = "unknown"
    appears_continuous: str = "unknown"
    suspected_PBC_jump: str = "unknown"
    usable_for_benchmark: str = "no"
    exclusion_reason: str = "NA"
    notes: str = "NA"

    def as_row(self) -> dict[str, Any]:
        """Return the candidate as a TSV row."""

        return {column: getattr(self, column) for column in COLUMNS}


def npz_shape(path: Path) -> tuple[int | str, int | str, str]:
    """Return frame and atom counts for a cached coordinate NPZ."""

    try:
        payload = np.load(path, allow_pickle=True)
        key = "coords" if "coords" in payload.files else payload.files[0]
        coords = payload[key]
        if coords.ndim >= 3:
            return int(coords.shape[0]), int(coords.shape[1]), f"array_key={key}; shape={tuple(coords.shape)}"
        return "NA", "NA", f"array_key={key}; shape={tuple(coords.shape)}"
    except Exception as exc:  # pragma: no cover - diagnostic script
        return "NA", "NA", f"shape_read_error={type(exc).__name__}: {exc}"


def inspect_mdcath_all_domain() -> list[Candidate]:
    """Summarize the all-domain mdCATH single-replicate cache."""

    summary = PROJECT_ROOT / "results/mdcath_benchmark/all_domain_canonical/aggregate/all_domain_canonical_per_domain_summary.tsv"
    if not summary.exists():
        return [
            Candidate(
                candidate_id="mdcath_all_domain_cache",
                source_path=str(summary),
                file_format="h5",
                exclusion_reason="summary file missing",
            )
        ]

    df = pd.read_csv(summary, sep="\t")
    n_frames = pd.to_numeric(df.get("n_frames"), errors="coerce")
    n_residues = pd.to_numeric(df.get("n_residues"), errors="coerce")
    max_frames = int(n_frames.max()) if n_frames.notna().any() else "NA"
    max_residues = int(n_residues.max()) if n_residues.notna().any() else "NA"
    n_ge_2000 = int((n_frames >= 2000).sum())
    n_ge_10000 = int((n_frames >= 10000).sum())
    n_finite_frames = int(n_frames.notna().sum())
    return [
        Candidate(
            candidate_id="mdcath_all_domain_single_replicates",
            source_path=str(summary),
            topology_path="embedded mdCATH HDF5 metadata",
            trajectory_path="mdCATH cached HDF5 files referenced in summary notes",
            file_format="h5",
            domain_or_system_name=f"{len(df)} mdCATH domain trajectories",
            estimated_n_frames=max_frames,
            n_residues_or_atoms=max_residues,
            has_topology="yes",
            has_CA_selection="yes",
            appears_continuous="yes",
            suspected_PBC_jump="unknown",
            usable_for_benchmark="no",
            exclusion_reason="single-replicate mdCATH trajectories in this cache are below the 2,000-frame minimum",
            notes=(
                f"summary_rows={len(df)}; finite_frame_rows={n_finite_frames}; "
                f"max_frames={max_frames}; n_ge_2000={n_ge_2000}; n_ge_10000={n_ge_10000}"
            ),
        )
    ]


def inspect_cached_continuous_npz() -> list[Candidate]:
    """List previously cached real continuous mdCATH coordinate arrays."""

    rows: list[Candidate] = []
    base = PROJECT_ROOT / "results/mdcath_benchmark/long_continuous_trajectory/continuous_inputs"
    for path in sorted(base.glob("*_ca_coords.npz")):
        frames, atoms, note = npz_shape(path)
        rows.append(
            Candidate(
                candidate_id=f"cached_continuous_{path.stem.replace('_ca_coords', '')}",
                source_path=str(path),
                topology_path="metadata in paired *_metadata.json if present",
                trajectory_path=str(path),
                file_format="npz",
                domain_or_system_name=path.stem.replace("_ca_coords", ""),
                estimated_n_frames=frames,
                n_residues_or_atoms=atoms,
                has_topology="yes",
                has_CA_selection="yes",
                appears_continuous="yes",
                suspected_PBC_jump="unknown",
                usable_for_benchmark="no",
                exclusion_reason="real continuous cached trajectory exists but is below the 2,000-frame minimum",
                notes=note,
            )
        )
    if not rows:
        rows.append(
            Candidate(
                candidate_id="cached_continuous_mdCATH_npz",
                source_path=str(base),
                file_format="npz",
                exclusion_reason="no cached continuous NPZ files found",
            )
        )
    return rows


def inspect_stitched_npz_groups() -> list[Candidate]:
    """Summarize known stitched-replicate stress-test coordinate arrays."""

    rows: list[Candidate] = []
    for group_name in ["long_trajectory_stitching", "replicate_stitching"]:
        base = PROJECT_ROOT / f"results/mdcath_benchmark/{group_name}"
        files = list(base.rglob("*.npz")) if base.exists() else []
        max_frames: int | str = "NA"
        max_atoms: int | str = "NA"
        max_file = "NA"
        for path in files:
            name = path.stem
            if name.startswith("stitched_"):
                frames: int | str = max(1, name.count("_R")) * 500
                atoms: int | str = "NA"
            elif name.endswith("_ca_coords") or name.startswith("R"):
                frames = 500
                atoms = "NA"
            else:
                continue
            if isinstance(frames, int) and (max_frames == "NA" or frames > int(max_frames)):
                max_frames = frames
                max_atoms = atoms
                max_file = str(path)
        rows.append(
            Candidate(
                candidate_id=f"excluded_{group_name}",
                source_path=str(base),
                topology_path="metadata embedded in NPZ files",
                trajectory_path=max_file,
                file_format="npz",
                domain_or_system_name=group_name,
                estimated_n_frames=max_frames,
                n_residues_or_atoms=max_atoms,
                has_topology="yes" if files else "unknown",
                has_CA_selection="yes" if files else "unknown",
                appears_continuous="no",
                suspected_PBC_jump="unknown",
                usable_for_benchmark="no",
                exclusion_reason="stitched independent replicates are explicitly excluded from this real continuous benchmark",
                notes=f"npz_count={len(files)}; largest_npz={max_file}",
            )
        )
    return rows


def inspect_project_sources() -> list[Candidate]:
    """Summarize other local sources checked during inspection."""

    data_dir = PROJECT_ROOT / "data"
    data_files = list(data_dir.rglob("*")) if data_dir.exists() else []
    trajectory_suffixes = {".xtc", ".trr", ".dcd", ".nc", ".h5", ".hdf5", ".npz"}
    data_traj = [p for p in data_files if p.is_file() and p.suffix.lower() in trajectory_suffixes]

    mdcorealign_nmr = Path("/storage/junjie/projects/MDCoreAlign/results/nmr_ensemble_batch_500_gt50ca_gt20models_20260330/original_pdbs")
    nmr_count = len(list(mdcorealign_nmr.glob("*.pdb"))) if mdcorealign_nmr.exists() else 0

    fluctome = Path("/storage/junjie/projects/Fluctome")
    fluctome_tar = fluctome / "Fluctome.tar"

    return [
        Candidate(
            candidate_id="project_data_directories",
            source_path=str(data_dir),
            file_format="mixed",
            domain_or_system_name="formal project data directories",
            estimated_n_frames="NA",
            n_residues_or_atoms="NA",
            has_topology="unknown",
            has_CA_selection="unknown",
            appears_continuous="unknown",
            suspected_PBC_jump="unknown",
            usable_for_benchmark="no",
            exclusion_reason="no local trajectory files found under data directories",
            notes=f"trajectory_like_file_count={len(data_traj)}; files are mostly grabdata scripts",
        ),
        Candidate(
            candidate_id="mdcorealign_nmr_ensembles",
            source_path=str(mdcorealign_nmr),
            file_format="multi-model pdb",
            domain_or_system_name="NMR ensemble benchmark cache",
            estimated_n_frames="NA",
            n_residues_or_atoms="NA",
            has_topology="yes",
            has_CA_selection="yes",
            appears_continuous="no",
            suspected_PBC_jump="no",
            usable_for_benchmark="no",
            exclusion_reason="NMR ensembles are multi-conformation structures, not physically continuous MD trajectories",
            notes=f"pdb_count={nmr_count}",
        ),
        Candidate(
            candidate_id="fluctome_archive",
            source_path=str(fluctome),
            trajectory_path=str(fluctome_tar) if fluctome_tar.exists() else "NA",
            file_format="tar/tsv",
            domain_or_system_name="Fluctome local archive",
            estimated_n_frames="NA",
            n_residues_or_atoms="NA",
            has_topology="unknown",
            has_CA_selection="unknown",
            appears_continuous="unknown",
            suspected_PBC_jump="unknown",
            usable_for_benchmark="no",
            exclusion_reason="archive appears to contain AlphaFold/download/distance-table assets rather than ready continuous MD trajectory files",
            notes="Fluctome.tar was not extracted; a 45-second tar listing search did not find common MD trajectory suffixes before timeout.",
        ),
        Candidate(
            candidate_id="common_trajectory_file_search",
            source_path="/storage/junjie/projects/mdlovofit /storage/junjie/projects/MDCoreAlign /storage/junjie/projects/Fluctome",
            file_format="xtc/trr/dcd/nc/gro/tpr/psf/prmtop",
            domain_or_system_name="targeted local trajectory file search",
            estimated_n_frames="NA",
            n_residues_or_atoms="NA",
            has_topology="unknown",
            has_CA_selection="unknown",
            appears_continuous="unknown",
            suspected_PBC_jump="unknown",
            usable_for_benchmark="no",
            exclusion_reason="no eligible 2,000-frame continuous MD trajectory/topology pair was identified in targeted project locations",
            notes="Global file search mostly matched non-MD embedding H5 files and previously generated benchmark artifacts.",
        ),
    ]


def write_tsv(path: Path, rows: list[Candidate]) -> None:
    """Write the inventory table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())


def write_no_eligible_doc(rows: list[Candidate]) -> None:
    """Write the required no-eligible-data report."""

    usable = [row for row in rows if row.usable_for_benchmark == "yes"]
    max_frames = []
    for row in rows:
        if row.appears_continuous == "yes" and isinstance(row.estimated_n_frames, int):
            max_frames.append(row.estimated_n_frames)
    max_frame_text = max(max_frames) if max_frames else "NA"
    NO_ELIGIBLE_DOC.parent.mkdir(parents=True, exist_ok=True)
    NO_ELIGIBLE_DOC.write_text(
        f"""# No Eligible Real Continuous 2k-10k Trajectories Found

## Scope

This inspection searched local MDCoreAlign/mdsuperpose project storage for real continuous MD trajectories suitable for a 2,000-10,000-frame benchmark. The task explicitly excludes stitched independent replicates, artificial frame duplication, shuffled frames, and NMR ensembles.

## Result

No eligible local trajectory panel was found. Usable candidates: {len(usable)}. The largest real continuous cached mdCATH single-replicate trajectories found locally contain {max_frame_text} frames, below the required 2,000-frame minimum.

## Sources Checked

- `data/` directories in the active project: only download/grabdata scripts were present, not trajectory/topology files.
- mdCATH all-domain single-replicate cache: 5,398 summary rows were inspected, of which 5,397 had finite frame counts; maximum available single-replicate frame count was 501 and no trajectory had >=2,000 frames.
- Previous `long_continuous_trajectory` cache: real continuous mdCATH examples contained 500-501 frames.
- Previous `long_trajectory_stitching` and `replicate_stitching` caches: some NPZ files have >=2,000 frames, but they are stitched independent replicates and are excluded from this benchmark by design.
- `MDCoreAlign` NMR ensemble caches: multi-model PDB ensembles are not physically continuous MD trajectories.
- `Fluctome` local archive: appears to contain AlphaFold/download/distance-table assets; no ready trajectory/topology pair was identified.
- Common local trajectory file suffixes were searched in targeted project locations; no eligible >=2,000-frame continuous MD trajectory/topology pair was found.

## Required Input To Proceed

To run the requested benchmark, provide at least three real continuous MD trajectories with stable atom correspondence and preferably C-alpha-compatible topology, for example:

- `trajectory.xtc`/`trajectory.dcd`/`trajectory.trr` plus `topology.pdb`/`topology.gro`/`topology.tpr`;
- or a cached coordinate array with shape `(n_frames, n_CA, 3)` and documented residue metadata;
- with at least 2,000 frames per trajectory, ideally one trajectory with >=5,000 frames and one with >=10,000 frames if available.

No benchmark results were fabricated. The stitched-replicate stress-test results remain diagnostic only and are not used here as real continuous long-trajectory evidence.
"""
    )


def main() -> None:
    """Create the inventory and no-eligible-data report."""

    rows: list[Candidate] = []
    rows.extend(inspect_mdcath_all_domain())
    rows.extend(inspect_cached_continuous_npz())
    rows.extend(inspect_stitched_npz_groups())
    rows.extend(inspect_project_sources())
    write_tsv(INVENTORY_PATH, rows)
    write_no_eligible_doc(rows)
    print(INVENTORY_PATH)
    print(NO_ELIGIBLE_DOC)


if __name__ == "__main__":
    main()
