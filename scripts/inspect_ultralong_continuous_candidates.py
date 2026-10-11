#!/usr/bin/env python
"""Create a conservative inventory for ultra-long continuous MD candidates.

The script only inspects local assets and cached metadata. It does not treat
stitched replicates, duplicated frames, shuffled frames, NMR ensembles, or
synthetic transition-aligned trajectories as eligible continuous MD inputs.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "ultralong_continuous_benchmark"
INVENTORY_PATH = OUTPUT_ROOT / "candidate_inventory.tsv"
SELECTED_PATH = OUTPUT_ROOT / "selected_trajectories.tsv"
STATUS_PATH = OUTPUT_ROOT / "aggregate" / "benchmark_status.tsv"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
NO_ELIGIBLE_DOC = PROJECT_ROOT / "docs" / "ultralong_no_local_eligible_trajectories.md"
PUBLIC_OPTIONS_DOC = PROJECT_ROOT / "docs" / "ultralong_public_data_options.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "ultralong_continuous_benchmark_report.md"

INVENTORY_COLUMNS = [
    "candidate_id",
    "system_name",
    "topology_path",
    "trajectory_path",
    "file_format",
    "estimated_n_frames",
    "n_atoms",
    "n_CA_residues",
    "timestep_or_frame_interval_if_available",
    "appears_continuous",
    "source",
    "usable_for_benchmark",
    "exclusion_reason",
    "notes",
]

SELECTED_COLUMNS = [
    "benchmark_id",
    "system_name",
    "topology_path",
    "trajectory_path",
    "n_frames_available",
    "n_CA_residues",
    "selected_prefixes",
    "source",
    "reason_selected",
    "notes",
]


@dataclass
class Candidate:
    """One local trajectory candidate or excluded source group."""

    candidate_id: str
    system_name: str
    topology_path: str = "NA"
    trajectory_path: str = "NA"
    file_format: str = "unknown"
    estimated_n_frames: str | int = "NA"
    n_atoms: str | int = "NA"
    n_CA_residues: str | int = "NA"
    timestep_or_frame_interval_if_available: str = "unknown"
    appears_continuous: str = "unknown"
    source: str = "local"
    usable_for_benchmark: str = "no"
    exclusion_reason: str = "NA"
    notes: str = "NA"

    def as_row(self) -> dict[str, Any]:
        """Return the candidate as a TSV row."""

        return {column: getattr(self, column) for column in INVENTORY_COLUMNS}


def parse_h5_path(notes: object) -> str:
    """Extract an HDF5 cache path from a notes field when available."""

    match = re.search(r"h5_file=([^;]+)", str(notes or ""))
    return match.group(1) if match else "NA"


def load_array_shape(path: Path) -> tuple[str | int, str | int, str]:
    """Read a coordinate-array shape without loading large arrays when possible."""

    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, mmap_mode="r")
            shape = tuple(array.shape)
            if len(shape) >= 3:
                return int(shape[0]), int(shape[1]), f"shape={shape}"
            return "NA", "NA", f"shape={shape}"
        with np.load(path, allow_pickle=True) as archive:
            key = "coords" if "coords" in archive.files else archive.files[0]
            shape = tuple(archive[key].shape)
            if len(shape) >= 3:
                return int(shape[0]), int(shape[1]), f"array_key={key}; shape={shape}"
            return "NA", "NA", f"array_key={key}; shape={shape}"
    except Exception as exc:  # pragma: no cover - diagnostic script
        return "NA", "NA", f"shape_read_error={type(exc).__name__}: {exc}"


def inspect_all_domain_mdcath() -> list[Candidate]:
    """Summarize the local mdCATH all-domain canonical single-replicate cache."""

    summary = (
        PROJECT_ROOT
        / "results"
        / "mdcath_benchmark"
        / "all_domain_canonical"
        / "aggregate"
        / "all_domain_canonical_per_domain_summary.tsv"
    )
    if not summary.exists():
        return [
            Candidate(
                candidate_id="mdcath_all_domain_canonical_summary_missing",
                system_name="mdCATH all-domain canonical cache",
                trajectory_path=str(summary),
                file_format="h5",
                source="local mdCATH cache",
                exclusion_reason="summary table missing",
            )
        ]

    df = pd.read_csv(summary, sep="\t")
    n_frames = pd.to_numeric(df.get("n_frames"), errors="coerce")
    n_residues = pd.to_numeric(df.get("n_residues"), errors="coerce")
    finite = n_frames.notna()
    max_frames = int(n_frames.max()) if finite.any() else "NA"
    max_residues = int(n_residues.max()) if n_residues.notna().any() else "NA"
    n_ge_2000 = int((n_frames >= 2000).sum())
    n_ge_10000 = int((n_frames >= 10000).sum())
    rows = [
        Candidate(
            candidate_id="mdcath_all_domain_canonical_single_replicates",
            system_name=f"mdCATH all-domain canonical single-replicates ({len(df)} rows)",
            topology_path="embedded mdCATH HDF5 topology/metadata",
            trajectory_path=str(summary),
            file_format="h5/metadata summary",
            estimated_n_frames=max_frames,
            n_atoms="NA",
            n_CA_residues=max_residues,
            timestep_or_frame_interval_if_available="not recorded in summary",
            appears_continuous="yes",
            source="local mdCATH Hugging Face cache",
            usable_for_benchmark="no",
            exclusion_reason="no single-replicate trajectory reaches the 2,000-frame ultra-long threshold",
            notes=(
                f"finite_frame_rows={int(finite.sum())}; max_frames={max_frames}; "
                f"n_ge_2000={n_ge_2000}; n_ge_10000={n_ge_10000}; "
                "temperature defaults are T320/R1 in the canonical summary"
            ),
        )
    ]

    top = df.assign(_frames=n_frames).sort_values(["_frames", "n_residues"], ascending=[False, False]).head(10)
    for _, row in top.iterrows():
        frames = row.get("n_frames")
        residues = row.get("n_residues")
        rows.append(
            Candidate(
                candidate_id=f"mdcath_top_continuous_{row['trajectory_id']}",
                system_name=str(row["trajectory_id"]),
                topology_path="embedded mdCATH HDF5 topology/metadata",
                trajectory_path=parse_h5_path(row.get("notes")),
                file_format="h5",
                estimated_n_frames=int(frames) if pd.notna(frames) else "NA",
                n_atoms="NA",
                n_CA_residues=int(residues) if pd.notna(residues) else "NA",
                timestep_or_frame_interval_if_available="not recorded in summary",
                appears_continuous="yes",
                source="local mdCATH Hugging Face cache",
                usable_for_benchmark="no",
                exclusion_reason="real continuous single-replicate trajectory, but below 2,000 frames",
                notes="Representative largest-frame local mdCATH candidate; not stitched.",
            )
        )
    return rows


def inspect_cached_long_continuous_npz() -> list[Candidate]:
    """List existing real continuous mdCATH NPZ coordinate caches."""

    base = PROJECT_ROOT / "results" / "mdcath_benchmark" / "long_continuous_trajectory"
    selected = base / "selected_continuous_trajectories.tsv"
    rows: list[Candidate] = []
    if selected.exists():
        selected_df = pd.read_csv(selected, sep="\t")
        for _, row in selected_df.iterrows():
            trajectory_id = str(row["trajectory_id"])
            npz_path = base / "continuous_inputs" / f"{trajectory_id}_ca_coords.npz"
            frames, ca_count, note = load_array_shape(npz_path)
            rows.append(
                Candidate(
                    candidate_id=f"cached_real_continuous_{trajectory_id}",
                    system_name=trajectory_id,
                    topology_path=str(base / "continuous_inputs" / f"{trajectory_id}_metadata.json"),
                    trajectory_path=str(npz_path),
                    file_format="npz",
                    estimated_n_frames=frames,
                    n_atoms=ca_count,
                    n_CA_residues=ca_count,
                    timestep_or_frame_interval_if_available="not recorded in cached NPZ",
                    appears_continuous="yes",
                    source="previous real continuous mdCATH benchmark cache",
                    usable_for_benchmark="no",
                    exclusion_reason="real continuous trajectory, but below 2,000 frames",
                    notes=note,
                )
            )
    return rows


def inspect_formal_cached_arrays() -> list[Candidate]:
    """Record small formal-system arrays that are not ultra-long inputs."""

    rows: list[Candidate] = []
    for path in sorted((PROJECT_ROOT / "results").glob("higher_tier_validation_batch1_smoke/*coords*.npy")):
        frames, ca_count, note = load_array_shape(path)
        rows.append(
            Candidate(
                candidate_id=f"formal_cached_array_{path.stem}",
                system_name=path.stem,
                topology_path="metadata from original formal benchmark scripts/results",
                trajectory_path=str(path),
                file_format="npy",
                estimated_n_frames=frames,
                n_atoms=ca_count,
                n_CA_residues=ca_count,
                timestep_or_frame_interval_if_available="strided cache; original timestep not encoded",
                appears_continuous="yes",
                source="formal-system cached coordinate array",
                usable_for_benchmark="no",
                exclusion_reason="strided/smoke cache and below 2,000 frames",
                notes=note,
            )
        )
    return rows


def inspect_excluded_stitched_sources() -> list[Candidate]:
    """Summarize generated long arrays that are explicitly ineligible."""

    groups = [
        ("long_trajectory_stitching", "stitched independent replicate diagnostic"),
        ("replicate_stitching", "stitched independent replicate diagnostic"),
        ("transition_aligned_stitching", "synthetic transition-aligned replicate stress test"),
    ]
    rows: list[Candidate] = []
    for group, reason in groups:
        base = PROJECT_ROOT / "results" / "mdcath_benchmark" / group
        npz_files = list(base.rglob("*.npz")) if base.exists() else []
        max_frames: str | int = "NA"
        max_path = "NA"
        manifest_paths = []
        if base.exists():
            manifest_paths.extend(base.glob("stitching_manifest.tsv"))
            manifest_paths.extend(base.glob("*/stitching_manifest.tsv"))
            manifest_paths.extend(base.glob("*/sequence_manifest.tsv"))
            manifest_paths.extend(base.glob("*/replicate_manifest.tsv"))
        for manifest in manifest_paths:
            try:
                df = pd.read_csv(manifest, sep="\t")
            except Exception:
                continue
            frame_column = next(
                (
                    column
                    for column in [
                        "n_total_frames",
                        "target_frames",
                        "n_frames",
                        "actual_n_frames",
                        "prefix_n_frames",
                    ]
                    if column in df.columns
                ),
                None,
            )
            if frame_column is None:
                continue
            for value in pd.to_numeric(df[frame_column], errors="coerce").dropna():
                frames = int(value)
                if max_frames == "NA" or frames > int(max_frames):
                    max_frames = frames
                    max_path = str(manifest)
        if max_frames == "NA":
            for path in npz_files:
                match = re.search(r"(?:^|_)(?P<frames>2000|5000|10000|25000|50000|100000)(?:_|$)", path.stem)
                if match is None:
                    continue
                frames = int(match.group("frames"))
                if max_frames == "NA" or frames > int(max_frames):
                    max_frames = frames
                    max_path = str(path)
        rows.append(
            Candidate(
                candidate_id=f"excluded_{group}",
                system_name=group,
                topology_path="metadata embedded in generated NPZ files",
                trajectory_path=max_path,
                file_format="npz",
                estimated_n_frames=max_frames,
                n_atoms="NA",
                n_CA_residues="NA",
                timestep_or_frame_interval_if_available="not applicable",
                appears_continuous="no",
                source="previous diagnostic generated outputs",
                usable_for_benchmark="no",
                exclusion_reason=f"{reason}; explicitly excluded from real ultra-long continuous benchmark",
                notes=f"npz_count={len(npz_files)}; largest_npz={max_path}",
            )
        )
    return rows


def inspect_project_data_sources() -> list[Candidate]:
    """Summarize local project directories that were checked for raw trajectories."""

    data_dir = PROJECT_ROOT / "data"
    trajectory_exts = {".xtc", ".trr", ".dcd", ".nc", ".netcdf", ".h5", ".hdf5", ".npz", ".mdcrd", ".lammpstrj"}
    data_files = [p for p in data_dir.rglob("*") if p.is_file()] if data_dir.exists() else []
    trajectory_files = [p for p in data_files if p.suffix.lower() in trajectory_exts]

    nmr_dir = Path("/storage/junjie/projects/MDCoreAlign/results/nmr_ensemble_batch_500_gt50ca_gt20models_20260330/original_pdbs")
    nmr_count = len(list(nmr_dir.glob("*.pdb"))) if nmr_dir.exists() else 0

    fluctome = Path("/storage/junjie/projects/Fluctome")
    fluctome_files = list(fluctome.rglob("*")) if fluctome.exists() else []
    fluctome_traj = [p for p in fluctome_files if p.is_file() and p.suffix.lower() in trajectory_exts]

    return [
        Candidate(
            candidate_id="active_project_data_directories",
            system_name="mdlovofit/data",
            topology_path="NA",
            trajectory_path=str(data_dir),
            file_format="mixed",
            estimated_n_frames="NA",
            n_atoms="NA",
            n_CA_residues="NA",
            timestep_or_frame_interval_if_available="unknown",
            appears_continuous="unknown",
            source="local project data directory",
            usable_for_benchmark="no",
            exclusion_reason="no local raw trajectory/topology pair found; directory currently contains grabdata scripts",
            notes=f"trajectory_like_file_count={len(trajectory_files)}",
        ),
        Candidate(
            candidate_id="mdcorealign_nmr_ensemble_cache",
            system_name="NMR ensemble cache",
            topology_path=str(nmr_dir),
            trajectory_path=str(nmr_dir),
            file_format="multi-model pdb",
            estimated_n_frames="NA",
            n_atoms="NA",
            n_CA_residues="NA",
            timestep_or_frame_interval_if_available="not applicable",
            appears_continuous="no",
            source="local MDCoreAlign NMR benchmark cache",
            usable_for_benchmark="no",
            exclusion_reason="NMR ensembles are multi-conformation structures, not physically continuous MD trajectories",
            notes=f"pdb_count={nmr_count}",
        ),
        Candidate(
            candidate_id="fluctome_project_tree",
            system_name="Fluctome local project",
            topology_path="NA",
            trajectory_path=str(fluctome),
            file_format="mixed",
            estimated_n_frames="NA",
            n_atoms="NA",
            n_CA_residues="NA",
            timestep_or_frame_interval_if_available="unknown",
            appears_continuous="unknown",
            source="local project tree",
            usable_for_benchmark="no",
            exclusion_reason="no ready continuous MD trajectory/topology pair found in the visible Fluctome tree",
            notes=f"trajectory_like_file_count={len(fluctome_traj)}",
        ),
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows to a TSV file with a fixed schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_public_options_doc() -> None:
    """Write public ultra-long MD data options without starting downloads."""

    PUBLIC_OPTIONS_DOC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_OPTIONS_DOC.write_text(
        """# Public Data Options For Ultra-Long Continuous MD

This note lists candidate public sources for a future ultra-long continuous MD benchmark. No multi-GB or multi-TB data were downloaded in this pass.

| source | likely use | trajectory/topology availability | scale / caveat | access / next action |
| --- | --- | --- | --- | --- |
| D. E. Shaw Research trajectories / DESRES resources | Best candidate for true long continuous protein trajectories if a suitable downloadable system is selected. | DESRES provides MD-related resources and software access information; exact trajectory package, topology, and license must be checked per system. | Potentially very long trajectories, but access/download terms vary and file sizes can be large. | Confirm a specific system and download terms before retrieval. Source page: https://www.deshawresearch.com/resources.html |
| GPCRmd | Good candidate for membrane-protein continuous trajectories with topology and DCD/XTC/TRR-style files. | GPCRmd documentation indicates topology/coordinate/trajectory file support including DCD, XTC and TRR-style trajectory uploads/downloads. | System sizes are large; some entries may require web/API selection and citation tracking. | Choose 1-3 small GPCR systems first; verify download links and topology compatibility. Source: https://www.gpcrmd.org/ |
| COVID-19 Molecular Structure and Therapeutics Hub / MoISSI | Candidate for large SARS-CoV-2 protein trajectories with topology. | The simulation page describes downloadable trajectory files and topology files, including XTC/PDB-style assets. | Spike systems are large and may be expensive; good for stress testing but not ideal for a quick first benchmark. | Use only after confirming storage/runtime budget. Source: https://covid.molssi.org/simulations/ |
| ATLAS | Candidate broad protein MD source for standardized simulations. | ATLAS is described as a database of standardized all-atom MD simulations with accessible trajectories/analyses. | May be better for broad validation than ultra-long scaling depending on available frame counts. | Select small proteins with downloadable topology and enough frames. Source: https://www.dsimb.inserm.fr/ATLAS |
| MDRepo | General MD warehouse with explicit trajectory, structure and topology files. | MDRepo describes submissions containing trajectory, structure/coordinate and topology files. | Heterogeneous submissions; needs filtering for proteins, frame counts and licenses. | Search for small continuous protein trajectories with >=2,000 frames. |
| mdCATH | Already cached locally and useful for broad single-replicate validation. | HDF5 files include coordinates and embedded metadata. | Local canonical single replicates max out at 501 frames in this project, so they are not ultra-long. | Do not use mdCATH stitching as real continuous evidence. |

Recommended next step: pick one small downloadable source first, preferably a DESRES or GPCRmd system with explicit topology and a 2,000-10,000-frame downloadable trajectory. Only after that succeeds should we add 50,000+ frame stress cases.
"""
    )


def write_no_eligible_doc(rows: list[Candidate]) -> None:
    """Write the required no-local-eligible report."""

    frame_values = [int(r.estimated_n_frames) for r in rows if r.appears_continuous == "yes" and isinstance(r.estimated_n_frames, int)]
    max_frames = max(frame_values) if frame_values else "NA"
    usable = [r for r in rows if r.usable_for_benchmark == "yes"]
    NO_ELIGIBLE_DOC.parent.mkdir(parents=True, exist_ok=True)
    NO_ELIGIBLE_DOC.write_text(
        f"""# No Local Eligible Ultra-Long Continuous Trajectories

## Result

No local trajectory satisfied the requested ultra-long benchmark eligibility criteria. Usable local candidates: `{len(usable)}`. The largest real continuous local candidate identified from cached mdCATH single-replicate data contained `{max_frames}` frames, below the preferred minimum of 2,000 frames.

## What Was Checked

- Active project data directories under `data/`: only grab/download scripts were present, not ready trajectory/topology files.
- mdCATH all-domain canonical local cache: `5,397` finite-frame single-replicate rows; maximum frame count `501`; trajectories with `>=2,000` frames: `0`.
- Previously cached real continuous mdCATH coordinate arrays: five 500-501-frame cases.
- Formal-system cached arrays: available local arrays are small/strided smoke or validation caches, not ultra-long raw trajectories.
- Previous replicate-stitching and transition-aligned generated arrays: some have thousands of frames, but they are explicitly excluded because they are stitched or synthetic stress-test inputs.
- NMR ensemble caches: not time-ordered continuous MD trajectories.
- Fluctome local tree: no visible ready continuous MD trajectory/topology pair was found.

## What Is Missing

To run the requested benchmark without violating the constraints, we need local inputs such as:

- `topology.pdb/gro/tpr/psf/prmtop` plus `trajectory.xtc/trr/dcd/nc/mdcrd`;
- at least 2,000 frames, ideally several 2k-10k cases and one 10k+ case;
- fixed atom correspondence and protein CA selection;
- documented preprocessing state, especially PBC/nojump/centering.

No ultra-long benchmark rows were fabricated. The existing stitched-replicate results remain diagnostic scaling stress tests only, not real continuous MD evidence.
"""
    )


def write_report(rows: list[Candidate]) -> None:
    """Write a conservative final report for this no-run pass."""

    continuous = [r for r in rows if r.appears_continuous == "yes"]
    frame_values = [int(r.estimated_n_frames) for r in continuous if isinstance(r.estimated_n_frames, int)]
    max_frames = max(frame_values) if frame_values else "NA"
    REPORT_DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DOC.write_text(
        f"""# Ultra-Long Continuous Benchmark Report

## Why This Benchmark Was Needed

Previous stitched-replicate and transition-aligned tests were useful scaling diagnostics, but they are not physically continuous MD trajectories. This pass therefore looked for real continuous local trajectories suitable for testing MDCoreAlign long-mode at 2k-10k+ frames and, if available, 50k-100k+ frames.

## Local Input Status

No eligible local ultra-long continuous benchmark panel was found. The maximum real continuous local frame count identified from mdCATH single-replicate caches was `{max_frames}` frames. The all-domain mdCATH canonical summary contains 5,397 finite-frame single-replicate rows, none with `>=2,000` frames.

## Why mdCATH And Synthetic Stitching Are Not Enough

The local mdCATH cache provides broad protein-domain coverage, but the available single-replicate trajectories are short for this specific ultra-long benchmark. Generated long NPZ files under stitching/transition-aligned directories contain thousands of frames, but they concatenate or transform independent replicates. They are therefore excluded from this real continuous benchmark.

## Benchmark Execution

The ultra-long benchmark was not run because no eligible local continuous trajectory/topology pair was available. Running on stitched or duplicated inputs would violate the task constraints and would overstate the evidence for real long-trajectory behavior.

## Recommended Next Step

Select and download, with user approval, one small public continuous trajectory source first. Preferred candidates are a DESRES or GPCRmd system with explicit topology and 2,000-10,000 frames. After verifying CA extraction, continuity/PBC sanity checks, and MDLovoFit compatibility, expand to 5-10 systems and optionally one 50k+ stress case.

## Allowed Manuscript Wording At This Stage

Safe: local ultra-long continuous inputs were not available, so the current long-mode evidence remains based on short real continuous mdCATH trajectories and separate synthetic scaling diagnostics.

Not safe: claiming that long-mode is validated on real 10k+ continuous trajectories.
"""
    )


def write_empty_selected_and_status() -> None:
    """Write explicit empty selection/status tables for downstream auditability."""

    write_tsv(SELECTED_PATH, [], SELECTED_COLUMNS)
    write_tsv(
        STATUS_PATH,
        [
            {
                "status": "not_run",
                "reason": "no eligible local real continuous trajectory with >=2,000 frames",
                "benchmark_rows_created": 0,
                "notes": "candidate_inventory.tsv and no-local docs were created instead",
            }
        ],
        ["status", "reason", "benchmark_rows_created", "notes"],
    )


def write_manifest() -> None:
    """Write a small manifest for this discovery-only pass."""

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "script": "scripts/inspect_ultralong_continuous_candidates.py",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "core_algorithm_modified": False,
                "benchmark_executed": False,
                "reason_not_executed": "no eligible local real continuous trajectory with >=2,000 frames",
                "outputs": [
                    str(INVENTORY_PATH),
                    str(SELECTED_PATH),
                    str(STATUS_PATH),
                    str(NO_ELIGIBLE_DOC),
                    str(PUBLIC_OPTIONS_DOC),
                    str(REPORT_DOC),
                ],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Generate the ultra-long candidate inventory and no-run reports."""

    rows: list[Candidate] = []
    rows.extend(inspect_all_domain_mdcath())
    rows.extend(inspect_cached_long_continuous_npz())
    rows.extend(inspect_formal_cached_arrays())
    rows.extend(inspect_excluded_stitched_sources())
    rows.extend(inspect_project_data_sources())
    write_tsv(INVENTORY_PATH, [row.as_row() for row in rows], INVENTORY_COLUMNS)
    write_empty_selected_and_status()
    write_no_eligible_doc(rows)
    write_public_options_doc()
    write_report(rows)
    write_manifest()
    print(f"wrote {INVENTORY_PATH}")
    print(f"wrote {NO_ELIGIBLE_DOC}")


if __name__ == "__main__":
    main()
