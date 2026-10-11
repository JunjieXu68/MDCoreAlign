"""Audit MD preprocessing metadata for formal trajectory systems."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src", SCRIPT_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.formal_flexibility_profiles import DATASETS  # noqa: E402
from md_trajectory_focus_common import (  # noqa: E402
    FORMAL_SYSTEMS,
    MD_FOCUS_ROOT,
    common_manifest,
    load_formal_system,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "preprocessing_manifest"

FIELDS = [
    "system",
    "trajectory_file",
    "topology_file",
    "n_frames_raw",
    "n_frames_used",
    "time_range_used",
    "frame_stride_used",
    "selection",
    "n_selected_atoms",
    "n_selected_residues",
    "pbc_unwrap_or_nojump_status",
    "protein_centering_status",
    "chain_continuity_check",
    "solvent_ion_removed",
    "equilibration_discarded",
    "reference_strategy",
    "chain_mapping_notes",
    "residue_metadata_source",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def raw_frame_metadata(key: str) -> tuple[int | str, str]:
    """Return raw frame count and used time range when MDAnalysis exposes it."""

    try:
        import MDAnalysis as mda

        config = DATASETS[key]
        universe = mda.Universe(str(config["topology"]), str(config["trajectory"]))
        raw = len(universe.trajectory)
        stride = int(config["stride"])
        max_frames = int(config["max_frames"])
        used_times = []
        for frame_index, _ in enumerate(universe.trajectory[::stride]):
            used_times.append(float(getattr(universe.trajectory.ts, "time", frame_index)))
            if frame_index + 1 >= max_frames:
                break
        if used_times:
            return raw, f"{used_times[0]:.6g}-{used_times[-1]:.6g} ps_or_reader_units"
        return raw, ""
    except Exception as exc:
        return "", f"unavailable: {type(exc).__name__}: {exc}"


def chain_continuity(system) -> str:
    """Check that residue IDs are nondecreasing within each chain when numeric."""

    by_chain: dict[str, list[int]] = {}
    for row in system.metadata_rows:
        residue_id = "".join(ch for ch in str(row["residue_id"]) if ch.isdigit())
        if not residue_id:
            continue
        by_chain.setdefault(str(row["chain"]), []).append(int(residue_id))
    if not by_chain:
        return "not_checkable_no_numeric_residue_ids"
    for values in by_chain.values():
        if any(b < a for a, b in zip(values, values[1:], strict=False)):
            return "warning_nonmonotonic_residue_ids"
    return "pass_numeric_residue_ids_nondecreasing_within_chain"


def run_analysis(systems: list[str]) -> list[dict[str, object]]:
    """Build preprocessing manifest rows."""

    rows = []
    for key in systems:
        system = load_formal_system(key)
        config = DATASETS[key]
        raw_count, time_range = raw_frame_metadata(key)
        rows.append(
            {
                "system": system.label,
                "trajectory_file": str(config["trajectory"]),
                "topology_file": str(config["topology"]),
                "n_frames_raw": raw_count,
                "n_frames_used": system.trajectory.n_frames,
                "time_range_used": time_range,
                "frame_stride_used": config["stride"],
                "selection": config["selection"],
                "n_selected_atoms": system.trajectory.n_atoms,
                "n_selected_residues": len(system.metadata_rows),
                "pbc_unwrap_or_nojump_status": "not_verified_from_manifest_input_used_as_provided",
                "protein_centering_status": "not_verified_from_manifest_input_used_as_provided",
                "chain_continuity_check": chain_continuity(system),
                "solvent_ion_removed": "yes_by_selection_protein_CA",
                "equilibration_discarded": "not_verified_formal_inputs_treated_as_preprocessed",
                "reference_strategy": "first_frame",
                "chain_mapping_notes": chain_notes(system),
                "residue_metadata_source": system.metadata.get("residue_metadata_source", ""),
                "notes": "Residue metadata are loaded from topology using the exact atom selection; PBC/centering status requires upstream trajectory provenance.",
            }
        )
        print(f"[preprocessing] {system.label}")
    return rows


def chain_notes(system) -> str:
    """Summarize chain mapping."""

    chains = [str(row["chain"]) for row in system.metadata_rows]
    unique = list(dict.fromkeys(chains))
    counts = {chain: chains.count(chain) for chain in unique}
    return f"chains={counts}"


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    pbc_unverified = sum("not_verified" in row["pbc_unwrap_or_nojump_status"] for row in rows)
    continuity_warnings = [row["system"] for row in rows if str(row["chain_continuity_check"]).startswith("warning")]
    lines = [
        "# MD Preprocessing Manifest Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Were PBC and chain-continuity issues checked? Chain continuity was checked from residue metadata; PBC/nojump status is not verified from these local manifests in `{pbc_unverified}` systems.",
        "2. What selection was used? All formal systems use `protein and name CA`.",
        "3. Were solvent/ions excluded? Yes, by the `protein and name CA` atom selection.",
        "4. Was equilibration discarded or were pre-equilibrated trajectories used? Not independently verified here; formal input trajectories are treated as already preprocessed.",
        "5. Are residue metadata derived from topology rather than array index? Yes, metadata are loaded through MDAnalysis topology using the exact selection.",
        "",
        f"- Chain continuity warnings: `{', '.join(continuity_warnings) if continuity_warnings else 'none'}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run preprocessing audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "md_preprocessing_manifest.tsv", rows, FIELDS)
    (args.output_dir / "md_preprocessing_manifest_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/audit_md_preprocessing_manifest.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "md_preprocessing_manifest.tsv",
            args.output_dir / "md_preprocessing_manifest_summary.md",
        ],
    )
    print(f"Wrote preprocessing manifest to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
