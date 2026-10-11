"""Run a minimal external biological validation on selected real systems."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (
    DomainRange,
    Trajectory,
    assign_domains_from_ranges,
    build_residue_labels,
    enrichment_ratio,
    infer_sequence_blocks,
    overlap_fraction,
    pearson_correlation,
    read_multimodel_pdb_with_occupancy,
    residue_contact_density,
    residue_interblock_contact_density,
    run_mdlovofit_on_coordinates,
    select_core_by_target_fraction,
    solve_trajectory_robust_alignment,
    spearman_correlation,
    top_fraction_mask,
    write_benchmark_csv,
    write_benchmark_json,
)


NAMED_DATASETS = {
    "ifabp": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "proxy_pdb": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water_0.pdb"),
        "external_signal_name": "local_starting_structure_bfactor",
    },
    "adk_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "external_signal_name": "canonical_domain_assignment",
    },
    "nhaa_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "external_signal_name": "first_frame_contact_density",
    },
    "yiip_equilibrium_short": {
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
        "external_signal_name": "first_frame_inter_subunit_contact_density",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse arguments for external biological validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-dataset", choices=sorted(NAMED_DATASETS), required=True)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "external_biological_validation")
    parser.add_argument("--workdir", type=Path, default=Path("/dev/shm/mdca_external_biology"))
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def load_ca_trajectory_with_metadata(
    topology_path: Path,
    trajectory_path: Path,
    *,
    selection: str,
    stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load CA coordinates together with residue metadata."""

    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path), str(trajectory_path))
    atoms = universe.select_atoms(selection)
    if atoms.n_atoms == 0:
        raise ValueError(f"Selection returned no atoms: {selection}")

    frames: list[np.ndarray] = []
    for frame_index, _ in enumerate(universe.trajectory[::stride]):
        frames.append(atoms.positions.astype(float).copy())
        if max_frames is not None and (frame_index + 1) >= max_frames:
            break

    coords = np.asarray(frames, dtype=float)
    residue_numbers = atoms.residues.resids.astype(int).copy()
    residue_names = atoms.residues.resnames.astype(str)
    residue_segids = atoms.residues.segids.astype(str)
    unique_segids = {str(segid) for segid in residue_segids.tolist()}
    if len(unique_segids) > 1:
        residue_labels = [
            f"{segid}:{resname}{resid}"
            for segid, resname, resid in zip(residue_segids, residue_names, residue_numbers, strict=True)
        ]
    else:
        residue_labels = build_residue_labels(residue_names, residue_numbers)
    return coords, residue_numbers, residue_names, residue_labels, residue_segids


def load_ifabp_bfactor_proxy(proxy_pdb: Path) -> tuple[np.ndarray, list[str]]:
    """Load a residue-level CA B-factor proxy from the distributed starting PDB."""

    import MDAnalysis as mda

    universe = mda.Universe(str(proxy_pdb))
    atoms = universe.select_atoms("protein and name CA")
    if atoms.n_atoms == 0:
        raise ValueError(f"Selection returned no CA atoms from {proxy_pdb}")
    residue_numbers = atoms.residues.resids.astype(int)
    residue_names = atoms.residues.resnames.astype(str)
    residue_labels = build_residue_labels(residue_names, residue_numbers)
    bfactors = atoms.tempfactors.astype(float).copy()
    if not np.isfinite(bfactors).all():
        raise ValueError("B-factor proxy must be finite")
    return bfactors, residue_labels


def build_nhaa_contact_density_proxy(
    coords: np.ndarray,
    residue_numbers: np.ndarray,
) -> np.ndarray:
    """Build a conservative first-frame residue packing proxy for NhaA."""

    block_ids = infer_sequence_blocks(residue_numbers)
    return residue_contact_density(
        coords[0],
        residue_numbers=residue_numbers,
        block_ids=block_ids,
        cutoff=10.0,
        min_sequence_separation=2,
    )


def build_yiip_inter_subunit_contact_proxy(
    coords: np.ndarray,
    residue_numbers: np.ndarray,
) -> np.ndarray:
    """Build a conservative first-frame inter-subunit contact-density proxy for YiiP."""

    block_ids = infer_sequence_blocks(residue_numbers)
    return residue_interblock_contact_density(
        coords[0],
        block_ids=block_ids,
        cutoff=10.0,
    )


def solve_soft_profile(
    coords: np.ndarray,
    *,
    method_name: str,
    core_fraction: float,
    sequence_continuity_strength: float = 0.0,
    sequence_continuity_passes: int = 1,
    contact_consistency_strength: float = 0.0,
    contact_consistency_passes: int = 1,
) -> dict[str, object]:
    """Run one soft trajectory solve and summarize residue-wise outputs."""

    trajectory_result = solve_trajectory_robust_alignment(
        Trajectory(coords=coords),
        reference_mode="first_frame",
        warm_start=False,
        record_history=False,
        rule="huber",
        delta=0.25,
        max_iterations=20,
        tolerance=1e-6,
        sequence_continuity_strength=sequence_continuity_strength,
        sequence_continuity_passes=sequence_continuity_passes,
        contact_consistency_strength=contact_consistency_strength,
        contact_consistency_passes=contact_consistency_passes,
    )
    weights = np.mean(
        np.asarray([frame.result.weights for frame in trajectory_result.frame_results], dtype=float),
        axis=0,
    )
    residuals = np.mean(
        np.asarray([frame.result.residual_distances for frame in trajectory_result.frame_results], dtype=float),
        axis=0,
    )
    core = select_core_by_target_fraction(weights, core_fraction)
    return {
        "method_name": method_name,
        "trajectory_result": trajectory_result,
        "stability_score": weights,
        "mobility_score": np.clip(1.0 - weights, 0.0, 1.0),
        "residual": residuals,
        "core_mask": core.core_mask,
        "core_fraction": core.core_fraction,
    }


def solve_external_mdlovofit_profile(
    coords: np.ndarray,
    *,
    core_fraction: float,
    workdir: Path,
    executable: Path | None,
    config_path: Path | None,
) -> dict[str, object]:
    """Run external MDLovoFit and recover the occupancy-derived subset mask."""

    run = run_mdlovofit_on_coordinates(
        coords,
        fraction=core_fraction,
        workdir=workdir,
        executable=executable,
        config_path=config_path,
    )
    _, occupancies = read_multimodel_pdb_with_occupancy(run.output_path, n_atoms=coords.shape[1])
    if not bool(run.output_subset_mask_binary) or not bool(run.output_subset_mask_constant_across_frames):
        raise RuntimeError("MDLovoFit output occupancy mask was not binary and constant across frames")
    core_mask = occupancies[0] > 0.5
    return {
        "method_name": "external_mdlovofit",
        "run": run,
        "core_mask": core_mask,
        "core_fraction": float(np.mean(core_mask)),
    }


def summarize_ifabp_against_bfactor(
    *,
    residue_labels: list[str],
    bfactor_proxy: np.ndarray,
    profile: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build residue-level and summary outputs for the I-FABP B-factor proxy."""

    core_mask = np.asarray(profile["core_mask"], dtype=bool)
    low_b_mask = top_fraction_mask(bfactor_proxy, 0.2, highest=False)
    high_b_mask = top_fraction_mask(bfactor_proxy, 0.2, highest=True)

    residue_rows: list[dict[str, object]] = []
    for index, residue_label in enumerate(residue_labels):
        row = {
            "dataset_name": "ifabp",
            "method_name": str(profile["method_name"]),
            "residue_index": index + 1,
            "residue_id": residue_label,
            "bfactor_proxy": float(bfactor_proxy[index]),
            "hard_core_label": int(core_mask[index]),
        }
        if "stability_score" in profile:
            stability = np.asarray(profile["stability_score"], dtype=float)
            mobility = np.asarray(profile["mobility_score"], dtype=float)
            residual = np.asarray(profile["residual"], dtype=float)
            row.update(
                {
                    "stability_score": float(stability[index]),
                    "mobility_score": float(mobility[index]),
                    "residual": float(residual[index]),
                }
            )
        residue_rows.append(row)

    summary = {
        "dataset_name": "ifabp",
        "external_signal_name": "local_starting_structure_bfactor",
        "method_name": str(profile["method_name"]),
        "n_residues": len(residue_labels),
        "core_fraction": float(profile["core_fraction"]),
        "selected_mean_bfactor": float(np.mean(bfactor_proxy[core_mask])),
        "nonselected_mean_bfactor": float(np.mean(bfactor_proxy[~core_mask])),
        "low_bfactor_core_overlap_fraction": overlap_fraction(core_mask, low_b_mask),
        "low_bfactor_core_enrichment": enrichment_ratio(core_mask, low_b_mask),
        "high_bfactor_core_enrichment": enrichment_ratio(core_mask, high_b_mask),
    }
    if "mobility_score" in profile:
        mobility = np.asarray(profile["mobility_score"], dtype=float)
        summary.update(
            {
                "spearman_mobility_vs_bfactor": spearman_correlation(mobility, bfactor_proxy),
                "pearson_mobility_vs_bfactor": pearson_correlation(mobility, bfactor_proxy),
                "top20_mobility_high_bfactor_overlap": overlap_fraction(
                    top_fraction_mask(mobility, 0.2, highest=True),
                    high_b_mask,
                ),
            }
        )
    return residue_rows, summary


def summarize_adk_by_domain(
    *,
    residue_labels: list[str],
    residue_numbers: np.ndarray,
    profile: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build residue-level and summary outputs for the ADK domain validation."""

    domain_ranges = [
        DomainRange("CORE", 1, 29),
        DomainRange("NMP", 30, 67),
        DomainRange("CORE", 68, 117),
        DomainRange("LID", 118, 160),
        DomainRange("CORE", 161, 214),
    ]
    domain_labels = assign_domains_from_ranges(residue_numbers, domain_ranges)
    core_mask = np.asarray(profile["core_mask"], dtype=bool)
    core_domain_mask = domain_labels == "CORE"
    mobile_domain_mask = (domain_labels == "LID") | (domain_labels == "NMP")

    residue_rows: list[dict[str, object]] = []
    for index, residue_label in enumerate(residue_labels):
        row = {
            "dataset_name": "adk_equilibrium",
            "method_name": str(profile["method_name"]),
            "residue_index": index + 1,
            "residue_id": residue_label,
            "residue_number": int(residue_numbers[index]),
            "domain_label": str(domain_labels[index]),
            "hard_core_label": int(core_mask[index]),
        }
        if "stability_score" in profile:
            stability = np.asarray(profile["stability_score"], dtype=float)
            mobility = np.asarray(profile["mobility_score"], dtype=float)
            residual = np.asarray(profile["residual"], dtype=float)
            row.update(
                {
                    "stability_score": float(stability[index]),
                    "mobility_score": float(mobility[index]),
                    "residual": float(residual[index]),
                }
            )
        residue_rows.append(row)

    summary = {
        "dataset_name": "adk_equilibrium",
        "external_signal_name": "canonical_domain_assignment",
        "method_name": str(profile["method_name"]),
        "n_residues": len(residue_labels),
        "core_fraction": float(profile["core_fraction"]),
        "selected_core_domain_fraction": float(np.mean(core_domain_mask[core_mask])),
        "selected_mobile_domain_fraction": float(np.mean(mobile_domain_mask[core_mask])),
        "core_domain_enrichment": enrichment_ratio(core_mask, core_domain_mask),
        "mobile_domain_enrichment": enrichment_ratio(core_mask, mobile_domain_mask),
    }
    if "stability_score" in profile:
        stability = np.asarray(profile["stability_score"], dtype=float)
        mobility = np.asarray(profile["mobility_score"], dtype=float)
        summary.update(
            {
                "core_domain_mean_stability": float(np.mean(stability[core_domain_mask])),
                "nmp_domain_mean_stability": float(np.mean(stability[domain_labels == "NMP"])),
                "lid_domain_mean_stability": float(np.mean(stability[domain_labels == "LID"])),
                "core_domain_mean_mobility": float(np.mean(mobility[core_domain_mask])),
                "nmp_domain_mean_mobility": float(np.mean(mobility[domain_labels == "NMP"])),
                "lid_domain_mean_mobility": float(np.mean(mobility[domain_labels == "LID"])),
                "top20_mobility_mobile_domain_overlap": overlap_fraction(
                    top_fraction_mask(mobility, 0.2, highest=True),
                    mobile_domain_mask,
                ),
            }
        )
    return residue_rows, summary


def summarize_nhaa_against_contact_density(
    *,
    residue_labels: list[str],
    residue_numbers: np.ndarray,
    contact_density_proxy: np.ndarray,
    profile: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build residue-level and summary outputs for the NhaA packing proxy."""

    core_mask = np.asarray(profile["core_mask"], dtype=bool)
    high_density_mask = top_fraction_mask(contact_density_proxy, 0.2, highest=True)
    low_density_mask = top_fraction_mask(contact_density_proxy, 0.2, highest=False)

    residue_rows: list[dict[str, object]] = []
    for index, residue_label in enumerate(residue_labels):
        row = {
            "dataset_name": "nhaa_equilibrium",
            "method_name": str(profile["method_name"]),
            "residue_index": index + 1,
            "residue_id": residue_label,
            "residue_number": int(residue_numbers[index]),
            "contact_density_proxy": float(contact_density_proxy[index]),
            "hard_core_label": int(core_mask[index]),
        }
        if "stability_score" in profile:
            stability = np.asarray(profile["stability_score"], dtype=float)
            mobility = np.asarray(profile["mobility_score"], dtype=float)
            residual = np.asarray(profile["residual"], dtype=float)
            row.update(
                {
                    "stability_score": float(stability[index]),
                    "mobility_score": float(mobility[index]),
                    "residual": float(residual[index]),
                }
            )
        residue_rows.append(row)

    summary = {
        "dataset_name": "nhaa_equilibrium",
        "external_signal_name": "first_frame_contact_density",
        "method_name": str(profile["method_name"]),
        "n_residues": len(residue_labels),
        "core_fraction": float(profile["core_fraction"]),
        "selected_mean_contact_density": float(np.mean(contact_density_proxy[core_mask])),
        "nonselected_mean_contact_density": float(np.mean(contact_density_proxy[~core_mask])),
        "high_contact_core_overlap_fraction": overlap_fraction(core_mask, high_density_mask),
        "high_contact_core_enrichment": enrichment_ratio(core_mask, high_density_mask),
        "low_contact_core_enrichment": enrichment_ratio(core_mask, low_density_mask),
    }
    if "stability_score" in profile:
        stability = np.asarray(profile["stability_score"], dtype=float)
        mobility = np.asarray(profile["mobility_score"], dtype=float)
        summary.update(
            {
                "spearman_stability_vs_contact_density": spearman_correlation(stability, contact_density_proxy),
                "pearson_stability_vs_contact_density": pearson_correlation(stability, contact_density_proxy),
                "spearman_mobility_vs_contact_density": spearman_correlation(mobility, contact_density_proxy),
                "top20_stability_high_contact_overlap": overlap_fraction(
                    top_fraction_mask(stability, 0.2, highest=True),
                    high_density_mask,
                ),
            }
        )
    return residue_rows, summary


def summarize_yiip_against_interface_contact_density(
    *,
    residue_labels: list[str],
    residue_numbers: np.ndarray,
    interface_contact_proxy: np.ndarray,
    profile: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build residue-level and summary outputs for the YiiP interface proxy."""

    core_mask = np.asarray(profile["core_mask"], dtype=bool)
    high_interface_mask = top_fraction_mask(interface_contact_proxy, 0.2, highest=True)
    low_interface_mask = top_fraction_mask(interface_contact_proxy, 0.2, highest=False)

    residue_rows: list[dict[str, object]] = []
    for index, residue_label in enumerate(residue_labels):
        row = {
            "dataset_name": "yiip_equilibrium_short",
            "method_name": str(profile["method_name"]),
            "residue_index": index + 1,
            "residue_id": residue_label,
            "residue_number": int(residue_numbers[index]),
            "inter_subunit_contact_density_proxy": float(interface_contact_proxy[index]),
            "hard_core_label": int(core_mask[index]),
        }
        if "stability_score" in profile:
            stability = np.asarray(profile["stability_score"], dtype=float)
            mobility = np.asarray(profile["mobility_score"], dtype=float)
            residual = np.asarray(profile["residual"], dtype=float)
            row.update(
                {
                    "stability_score": float(stability[index]),
                    "mobility_score": float(mobility[index]),
                    "residual": float(residual[index]),
                }
            )
        residue_rows.append(row)

    summary = {
        "dataset_name": "yiip_equilibrium_short",
        "external_signal_name": "first_frame_inter_subunit_contact_density",
        "method_name": str(profile["method_name"]),
        "n_residues": len(residue_labels),
        "core_fraction": float(profile["core_fraction"]),
        "selected_mean_inter_subunit_contact_density": float(np.mean(interface_contact_proxy[core_mask])),
        "nonselected_mean_inter_subunit_contact_density": float(np.mean(interface_contact_proxy[~core_mask])),
        "high_interface_core_overlap_fraction": overlap_fraction(core_mask, high_interface_mask),
        "high_interface_core_enrichment": enrichment_ratio(core_mask, high_interface_mask),
        "low_interface_core_enrichment": enrichment_ratio(core_mask, low_interface_mask),
    }
    if "stability_score" in profile:
        stability = np.asarray(profile["stability_score"], dtype=float)
        mobility = np.asarray(profile["mobility_score"], dtype=float)
        summary.update(
            {
                "spearman_stability_vs_inter_subunit_contact_density": spearman_correlation(
                    stability, interface_contact_proxy
                ),
                "pearson_stability_vs_inter_subunit_contact_density": pearson_correlation(
                    stability, interface_contact_proxy
                ),
                "spearman_mobility_vs_inter_subunit_contact_density": spearman_correlation(
                    mobility, interface_contact_proxy
                ),
                "top20_stability_high_interface_overlap": overlap_fraction(
                    top_fraction_mask(stability, 0.2, highest=True),
                    high_interface_mask,
                ),
            }
        )
    return residue_rows, summary


def write_note(path: Path, *, dataset_name: str, summaries: list[dict[str, object]], caveat: str) -> None:
    """Write one concise markdown note for external biological validation."""

    lines = [f"# External Biological Validation: {dataset_name}", ""]
    lines.append("## Summary")
    for summary in summaries:
        method_name = str(summary["method_name"])
        if dataset_name == "ifabp":
            lines.append(
                f"- `{method_name}`: mean B-factor core/non-core = "
                f"{summary['selected_mean_bfactor']:.3f} / {summary['nonselected_mean_bfactor']:.3f}, "
                f"low-B enrichment = {summary['low_bfactor_core_enrichment']:.3f}"
                + (
                    f", Spearman(mobility,B) = {summary['spearman_mobility_vs_bfactor']:.3f}"
                    if "spearman_mobility_vs_bfactor" in summary
                    else ""
                )
            )
        elif dataset_name == "nhaa_equilibrium":
            lines.append(
                f"- `{method_name}`: mean contact density core/non-core = "
                f"{summary['selected_mean_contact_density']:.3f} / {summary['nonselected_mean_contact_density']:.3f}, "
                f"high-contact enrichment = {summary['high_contact_core_enrichment']:.3f}"
                + (
                    f", Spearman(stability,contact-density) = "
                    f"{summary['spearman_stability_vs_contact_density']:.3f}"
                    if "spearman_stability_vs_contact_density" in summary
                    else ""
                )
            )
        elif dataset_name == "yiip_equilibrium_short":
            lines.append(
                f"- `{method_name}`: mean inter-subunit contact density core/non-core = "
                f"{summary['selected_mean_inter_subunit_contact_density']:.3f} / "
                f"{summary['nonselected_mean_inter_subunit_contact_density']:.3f}, "
                f"high-interface enrichment = {summary['high_interface_core_enrichment']:.3f}"
                + (
                    f", Spearman(stability,inter-subunit-contact) = "
                    f"{summary['spearman_stability_vs_inter_subunit_contact_density']:.3f}"
                    if "spearman_stability_vs_inter_subunit_contact_density" in summary
                    else ""
                )
            )
        else:
            lines.append(
                f"- `{method_name}`: selected CORE fraction = "
                f"{summary['selected_core_domain_fraction']:.3f}, "
                f"CORE enrichment = {summary['core_domain_enrichment']:.3f}"
                + (
                    f", mean mobility CORE/NMP/LID = "
                    f"{summary['core_domain_mean_mobility']:.3f} / "
                    f"{summary['nmp_domain_mean_mobility']:.3f} / "
                    f"{summary['lid_domain_mean_mobility']:.3f}"
                    if "core_domain_mean_mobility" in summary
                    else ""
                )
            )
    lines.append("")
    lines.append("## Caveat")
    lines.append(f"- {caveat}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run a minimal external biological validation for one named dataset."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    dataset = NAMED_DATASETS[args.named_dataset]
    coords, residue_numbers, _, residue_labels, _ = load_ca_trajectory_with_metadata(
        dataset["topology"],
        dataset["trajectory"],
        selection=str(dataset["selection"]),
        stride=args.stride,
        max_frames=args.max_frames,
    )

    soft_profiles = [
        solve_soft_profile(
            coords,
            method_name="soft_weight_no_regularization",
            core_fraction=args.fraction,
        ),
        solve_soft_profile(
            coords,
            method_name="soft_weight_seq_contact_regularized",
            core_fraction=args.fraction,
            sequence_continuity_strength=0.6,
            sequence_continuity_passes=2,
            contact_consistency_strength=0.5,
            contact_consistency_passes=2,
        ),
    ]
    external_profile = solve_external_mdlovofit_profile(
        coords,
        core_fraction=args.fraction,
        workdir=args.workdir / args.named_dataset,
        executable=args.mdlovofit_executable,
        config_path=args.mdlovofit_config,
    )
    profiles = soft_profiles + [external_profile]

    residue_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    if args.named_dataset == "ifabp":
        bfactor_proxy, proxy_labels = load_ifabp_bfactor_proxy(dataset["proxy_pdb"])
        if proxy_labels != residue_labels:
            raise ValueError("I-FABP B-factor proxy labels did not match trajectory residue labels")
        for profile in profiles:
            rows, summary = summarize_ifabp_against_bfactor(
                residue_labels=residue_labels,
                bfactor_proxy=bfactor_proxy,
                profile=profile,
            )
            residue_rows.extend(rows)
            summary_rows.append(summary)
        caveat = (
            "The I-FABP external signal here is the CA B-factor vector preserved in the local distributed "
            "starting-structure PDB. It is external to the alignment method, but its provenance has not yet "
            "been independently audited against a separate deposited experimental reference."
        )
    elif args.named_dataset == "nhaa_equilibrium":
        contact_density_proxy = build_nhaa_contact_density_proxy(coords, residue_numbers)
        for profile in profiles:
            rows, summary = summarize_nhaa_against_contact_density(
                residue_labels=residue_labels,
                residue_numbers=residue_numbers,
                contact_density_proxy=contact_density_proxy,
                profile=profile,
            )
            residue_rows.extend(rows)
            summary_rows.append(summary)
        caveat = (
            "The NhaA external signal here is a conservative first-frame CA contact-density proxy derived from "
            "the input structure itself. It is external to the alignment output and biologically interpretable "
            "as local packing, but it is not a direct experimental mobility measurement."
        )
    elif args.named_dataset == "yiip_equilibrium_short":
        interface_contact_proxy = build_yiip_inter_subunit_contact_proxy(coords, residue_numbers)
        for profile in profiles:
            rows, summary = summarize_yiip_against_interface_contact_density(
                residue_labels=residue_labels,
                residue_numbers=residue_numbers,
                interface_contact_proxy=interface_contact_proxy,
                profile=profile,
            )
            residue_rows.extend(rows)
            summary_rows.append(summary)
        caveat = (
            "The YiiP external signal here is a conservative first-frame CA inter-subunit contact-density proxy "
            "derived from the dimeric input structure. It is biologically interpretable as interface packing, "
            "but it is not a direct experimental mobility measurement."
        )
    else:
        for profile in profiles:
            rows, summary = summarize_adk_by_domain(
                residue_labels=residue_labels,
                residue_numbers=residue_numbers,
                profile=profile,
            )
            residue_rows.extend(rows)
            summary_rows.append(summary)
        caveat = (
            "The ADK external signal here is the canonical CORE/NMP/LID domain decomposition used in the "
            "adenylate kinase conformational-dynamics literature; this validation checks domain-level "
            "consistency rather than residue-specific experimental truth."
        )

    prefix = args.output_dir / args.named_dataset
    residue_csv = Path(f"{prefix}_external_biology_residues.csv")
    summary_csv = Path(f"{prefix}_external_biology_summary.csv")
    payload_json = Path(f"{prefix}_external_biology_payload.json")
    note_md = Path(f"{prefix}_external_biology_note.md")
    write_benchmark_csv(residue_csv, residue_rows)
    write_benchmark_csv(summary_csv, summary_rows)
    write_benchmark_json(
        payload_json,
        {
            "dataset_name": args.named_dataset,
            "fraction": args.fraction,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "residue_rows": residue_rows,
            "summary_rows": summary_rows,
        },
    )
    write_note(note_md, dataset_name=args.named_dataset, summaries=summary_rows, caveat=caveat)

    print("External biological validation complete")
    print(f"residue_csv={residue_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"payload_json={payload_json}")
    print(f"note_md={note_md}")


if __name__ == "__main__":
    main()
