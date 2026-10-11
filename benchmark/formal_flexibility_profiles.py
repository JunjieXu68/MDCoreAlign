"""Generate formal RMSF and flexible-segment profiles for four promoted systems."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import string
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (  # noqa: E402
    Trajectory,
    build_extended_flexibility_rows,
    build_flexibility_profile,
    build_rmsf_long_rows,
    summarize_flexible_segments,
    summarize_flexibility_profile,
    write_flexibility_json,
    write_table,
)
from mdstablecore.trajectory_io import load_mdanalysis_trajectory  # noqa: E402


DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "output_stem": "ifabp",
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
    "adk_equilibrium": {
        "output_stem": "adk",
        "label": "ADK equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
    "nhaa_equilibrium": {
        "output_stem": "nhaa",
        "label": "NhaA equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "stride": 25,
        "max_frames": 201,
    },
    "yiip_equilibrium_short": {
        "output_stem": "yiip_short",
        "label": "YiiP equilibrium short",
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
        "stride": 5,
        "max_frames": 181,
    },
}

SEGMENT_FIELDS = [
    "system",
    "segment_id",
    "chain",
    "start_residue",
    "end_residue",
    "length",
    "mean_robust_RMSF",
    "max_robust_RMSF",
    "mean_ordinary_RMSF",
    "mean_delta_RMSF",
    "mean_mobility_score",
    "mean_stability_weight",
    "overlaps_soft_mobile_region",
    "overlaps_terminal_region",
    "annotation",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["ifabp", "adk_equilibrium", "nhaa_equilibrium", "yiip_equilibrium_short"],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "flexibility")
    parser.add_argument("--core-fraction", type=float, default=0.60)
    parser.add_argument("--rule", choices=("gaussian", "huber", "tukey"), default="huber")
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=4.685)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--reference-mode", choices=("first_frame", "previous_frame"), default="first_frame")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--flexible-percentile-threshold", type=float, default=0.80)
    parser.add_argument("--min-segment-length", type=int, default=3)
    parser.add_argument("--merge-gap", type=int, default=1)
    parser.add_argument("--terminal-window", type=int, default=5)
    parser.add_argument(
        "--mobility-score-mode",
        choices=("soft_weight", "robust_rmsf"),
        default="soft_weight",
        help="soft_weight uses 1 - soft_stability_weight; robust_rmsf uses scaled robust RMSF.",
    )
    return parser.parse_args()


def load_dataset_with_metadata(name: str) -> tuple[Trajectory, tuple[str, ...], list[dict[str, object]], dict[str, object]]:
    """Load one formal dataset and residue metadata from the same selection."""

    config = DATASETS[name]
    topology = Path(config["topology"])
    trajectory = Path(config["trajectory"])
    selection = str(config["selection"])
    stride = int(config["stride"])
    max_frames = int(config["max_frames"])

    data = load_mdanalysis_trajectory(
        topology_path=topology,
        trajectory_path=trajectory,
        selection=selection,
        stride=stride,
        max_frames=max_frames,
    )
    residue_metadata = load_mdanalysis_residue_metadata(topology, trajectory, selection)
    if len(residue_metadata) != data.coords.shape[1]:
        raise ValueError(
            f"metadata length mismatch for {name}: "
            f"{len(residue_metadata)} labels vs {data.coords.shape[1]} coordinates"
        )
    residue_labels = tuple(row["residue_label"] for row in residue_metadata)
    metadata = {
        "dataset_name": name,
        "label": config["label"],
        "topology_path": str(topology),
        "trajectory_path": str(trajectory),
        "selection": selection,
        "stride": stride,
        "max_frames": max_frames,
        "n_frames": int(data.coords.shape[0]),
        "n_residues": int(data.coords.shape[1]),
        "residue_metadata_source": "MDAnalysis topology using the exact benchmark atom selection",
    }
    return Trajectory(coords=data.coords), residue_labels, residue_metadata, metadata


def load_mdanalysis_residue_metadata(
    topology_path: Path,
    trajectory_path: Path,
    selection: str,
) -> list[dict[str, object]]:
    """Read chain, residue ID, and residue name metadata for selected atoms."""

    import MDAnalysis as mda

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe = mda.Universe(str(topology_path), str(trajectory_path))
    atoms = universe.select_atoms(selection)
    if atoms.n_atoms == 0:
        raise ValueError(f"Selection returned no atoms: {selection}")

    chain_values = infer_chain_values(atoms)
    rows: list[dict[str, object]] = []
    for atom_index, atom in enumerate(atoms):
        resname = str(getattr(atom, "resname", "UNK") or "UNK").upper()[:3]
        resid = str(getattr(atom, "resid", atom_index + 1))
        icode = _safe_atom_attr(atom, "icode", "")
        residue_id = f"{resid}{icode}".strip()
        chain = chain_values[atom_index]
        rows.append(
            {
                "atom_index": atom_index + 1,
                "chain": chain,
                "residue_id": residue_id,
                "resname": resname,
                "atom_name": str(getattr(atom, "name", "CA") or "CA"),
                "segid": str(_safe_atom_attr(atom, "segid", "")),
                "chainID": str(_safe_atom_attr(atom, "chainID", "")),
                "residue_label": f"{chain}:{resname}{residue_id}",
            }
        )
    return rows


def infer_chain_values(atoms) -> list[str]:
    """Infer chain IDs while handling PSF/GRO files without chain metadata."""

    chain_ids = [_safe_atom_attr(atom, "chainID", "") for atom in atoms]
    chain_ids = [str(value).strip() for value in chain_ids if str(value).strip()]
    if len(set(chain_ids)) > 1:
        return [str(_safe_atom_attr(atom, "chainID", "A")).strip() or "A" for atom in atoms]

    segids = [str(_safe_atom_attr(atom, "segid", "")).strip() for atom in atoms]
    unique_segids = [segid for segid in dict.fromkeys(segids) if segid]
    if len(unique_segids) > 1:
        return _chain_values_from_segids(segids)

    residue_keys = [
        (str(getattr(atom, "resname", "UNK")), str(getattr(atom, "resid", index + 1)))
        for index, atom in enumerate(atoms)
    ]
    if len(set(residue_keys)) < len(residue_keys):
        counts: dict[tuple[str, str], int] = {}
        chains: list[str] = []
        alphabet = string.ascii_uppercase
        for key in residue_keys:
            occurrence = counts.get(key, 0)
            chains.append(alphabet[min(occurrence, len(alphabet) - 1)])
            counts[key] = occurrence + 1
        return chains
    return ["A"] * len(atoms)


def _chain_values_from_segids(segids: list[str]) -> list[str]:
    """Map distinct segment IDs to stable one-character chain IDs."""

    unique_segids = [segid for segid in dict.fromkeys(segids) if segid]
    suffixes = [segid[-1].upper() for segid in unique_segids]
    if len(set(suffixes)) == len(unique_segids) and all(suffix.isalnum() for suffix in suffixes):
        mapping = dict(zip(unique_segids, suffixes, strict=True))
    else:
        mapping = {
            segid: string.ascii_uppercase[index % len(string.ascii_uppercase)]
            for index, segid in enumerate(unique_segids)
        }
    return [mapping.get(segid, "A") for segid in segids]


def _safe_atom_attr(atom, attr: str, default: object) -> object:
    """Return an atom attribute while tolerating missing topology fields."""

    try:
        return getattr(atom, attr)
    except Exception:
        return default


def _write_table_with_header(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str],
    delimiter: str,
) -> None:
    """Write a table with a header even when there are no rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def _robust_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Build robust alignment keyword arguments from CLI parameters."""

    params: dict[str, object] = {
        "rule": args.rule,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    if args.rule == "huber":
        params["delta"] = args.delta
    elif args.rule == "gaussian":
        params["sigma"] = args.sigma
    else:
        params["c"] = args.c
    return params


def main() -> None:
    """Generate formal RMSF/flexible-segment outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robust_parameters = _robust_parameters(args)

    combined_summary_rows: list[dict[str, object]] = []
    combined_long_rows: list[dict[str, object]] = []
    metadata_payload: dict[str, object] = {
        "method_version": "0.1.0",
        "generated_at_epoch_seconds": time.time(),
        "analysis_type": "formal_rmsf_flexible_segment_profiles",
        "parameters": {
            "core_fraction": args.core_fraction,
            "reference_mode": args.reference_mode,
            "reference_frame": 0 if args.reference_mode == "first_frame" else "previous_frame",
            "warm_start": args.warm_start,
            "flexible_percentile_threshold": args.flexible_percentile_threshold,
            "min_segment_length": args.min_segment_length,
            "merge_gap": args.merge_gap,
            "terminal_window": args.terminal_window,
            "mobility_score_mode": args.mobility_score_mode,
            **robust_parameters,
        },
        "datasets": [],
    }

    for dataset_name in args.datasets:
        trajectory, residue_labels, residue_metadata, dataset_metadata = load_dataset_with_metadata(dataset_name)
        output_stem = str(DATASETS[dataset_name]["output_stem"])
        system_label = str(DATASETS[dataset_name]["label"])

        start = time.perf_counter()
        profile = build_flexibility_profile(
            trajectory,
            residue_labels=residue_labels,
            reference_mode=args.reference_mode,
            warm_start=args.warm_start,
            flexible_percentile_cutoff=args.flexible_percentile_threshold,
            min_segment_length=args.min_segment_length,
            merge_gap=args.merge_gap,
            mobility_score_mode=args.mobility_score_mode,
            **robust_parameters,
        )
        runtime = time.perf_counter() - start

        profile_rows = build_extended_flexibility_rows(
            profile,
            core_fraction=args.core_fraction,
            terminal_window=args.terminal_window,
        )
        segment_rows = summarize_flexible_segments(
            profile,
            system=system_label,
            core_fraction=args.core_fraction,
            terminal_window=args.terminal_window,
        )
        summary_row = summarize_flexibility_profile(
            profile,
            system=system_label,
            core_fraction=args.core_fraction,
        )
        summary_row.update(
            {
                "dataset_name": dataset_name,
                "runtime_seconds": runtime,
                "core_fraction": args.core_fraction,
                "flexible_percentile_threshold": args.flexible_percentile_threshold,
                "min_segment_length": args.min_segment_length,
                "merge_gap": args.merge_gap,
                "mobility_score_mode": args.mobility_score_mode,
            }
        )
        long_rows = build_rmsf_long_rows(profile, system=system_label, core_fraction=args.core_fraction)

        write_table(args.output_dir / f"{output_stem}_flexibility_profile.csv", profile_rows, delimiter=",")
        _write_table_with_header(
            args.output_dir / f"{output_stem}_flexible_segments.tsv",
            segment_rows,
            fieldnames=SEGMENT_FIELDS,
            delimiter="\t",
        )
        write_table(args.output_dir / f"{output_stem}_residue_metadata.tsv", residue_metadata, delimiter="\t")

        system_json = {
            "dataset": dataset_metadata,
            "parameters": metadata_payload["parameters"],
            "runtime_seconds": runtime,
            "n_flexible_segments": summary_row["n_flexible_segments"],
            "output_files": {
                "flexibility_profile": str(args.output_dir / f"{output_stem}_flexibility_profile.csv"),
                "flexible_segments": str(args.output_dir / f"{output_stem}_flexible_segments.tsv"),
                "residue_metadata": str(args.output_dir / f"{output_stem}_residue_metadata.tsv"),
            },
        }
        write_flexibility_json(args.output_dir / f"{output_stem}_summary.json", system_json)

        combined_summary_rows.append(summary_row)
        combined_long_rows.extend(long_rows)
        metadata_payload["datasets"].append({**dataset_metadata, "runtime_seconds": runtime})

        print(
            f"{dataset_name}: residues={profile.n_residues}, "
            f"segments={summary_row['n_flexible_segments']}, runtime={runtime:.3f}s"
        )

    write_table(args.output_dir / "flexibility_summary_four_systems.tsv", combined_summary_rows, delimiter="\t")
    write_table(args.output_dir / "rmsf_profiles_long.tsv", combined_long_rows, delimiter="\t")
    write_flexibility_json(args.output_dir / "summary.json", metadata_payload)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
