#!/usr/bin/env python
"""Freeze compact legacy outputs for the Phase 2B regression panel."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.flexibility import (
    build_flexibility_profile,
    mean_soft_weights_from_result,
)
from mdsuperpose.long_mode import (
    build_long_flexibility_profile,
    solve_long_trajectory_alignment,
)
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v2/regression"
REFERENCE_ROOT = RESULT_ROOT / "reference_outputs"
LONG_PANEL_ROOT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
)
ROBUST_KWARGS: dict[str, object] = {
    "rule": "huber",
    "delta": 1.0,
    "max_iterations": 20,
    "tolerance": 1e-6,
    "objective_tolerance": 1e-5,
    "objective_patience": 2,
    "sequence_continuity_strength": 0.05,
    "contact_consistency_strength": 0.0,
    "record_history": False,
}
FORMAL_SYSTEMS = {
    "formal_ifabp": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
    },
    "formal_adk": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
    },
    "formal_nhaa": {
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
    },
    "formal_yiip_short": {
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
    },
}
REAL_REPLICATES = (
    "4a18O00__T320__R1_ca_coords.npz",
    "4cgyA01__T320__R1_ca_coords.npz",
    "4cu9A00__T320__R1_ca_coords.npz",
    "3h7lB02__T320__R1_ca_coords.npz",
    "3a2pA00__T320__R1_ca_coords.npz",
)


@dataclass(slots=True)
class RegressionCase:
    """One compact development-panel coordinate case."""

    case_id: str
    category: str
    source: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray | None]]
    run_default: bool = True
    notes: str = ""


def repo_path(path: Path) -> str:
    """Use repository-relative paths and mark external local inputs explicitly."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external:{path.resolve()}"


def load_npz(path: Path, *, frames: int | None = None, uniform: bool = False) -> tuple[np.ndarray, None]:
    """Load a coordinate NPZ, optionally selecting a deterministic frame subset."""

    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=float)
    if frames is not None and coords.shape[0] > frames:
        if uniform:
            indices = np.linspace(0, coords.shape[0] - 1, frames, dtype=int)
            coords = coords[indices]
        else:
            coords = coords[:frames]
    return coords, None


def load_formal(config: dict[str, object], max_frames: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Load uniformly distributed CA frames and chain metadata with MDAnalysis."""

    import MDAnalysis as mda
    from MDAnalysis.exceptions import NoDataError

    universe = mda.Universe(str(config["topology"]), str(config["trajectory"]))
    atoms = universe.select_atoms(str(config["selection"]))
    if atoms.n_atoms < 3:
        raise ValueError("formal CA selection contains fewer than three atoms")
    frame_indices = np.unique(
        np.linspace(0, len(universe.trajectory) - 1, min(max_frames, len(universe.trajectory)), dtype=int)
    )
    coords = np.empty((frame_indices.shape[0], atoms.n_atoms, 3), dtype=float)
    for output_index, frame_index in enumerate(frame_indices):
        universe.trajectory[int(frame_index)]
        coords[output_index] = atoms.positions
    try:
        chain = np.asarray(atoms.chainIDs, dtype=str)
    except (AttributeError, NoDataError):
        chain = np.asarray(atoms.segids, dtype=str)
    return coords, chain


def rigid_toy(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic rigid and hinge-motion regression toys."""

    rng = np.random.default_rng(42)
    reference = rng.normal(size=(24, 3))
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    frames = []
    for frame_index in range(20):
        if kind == "translation":
            frame = reference + np.asarray([0.3, -0.2, 0.1]) * frame_index
        elif kind == "rotation":
            fraction = frame_index / 19.0
            theta = angle * fraction
            local_rotation = np.asarray(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            frame = reference @ local_rotation.T + np.asarray([1.0, -2.0, 0.5])
        elif kind == "hinge":
            frame = reference.copy()
            fraction = frame_index / 19.0
            moving = frame[12:] - frame[12]
            theta = angle * fraction
            local_rotation = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )
            frame[12:] = moving @ local_rotation.T + frame[12]
            frame = frame @ rotation.T + np.asarray([2.0, -1.0, 0.3])
        else:
            raise ValueError(f"unknown toy kind: {kind}")
        frames.append(frame)
    return np.asarray(frames), np.asarray(["A"] * reference.shape[0])


def build_cases() -> list[RegressionCase]:
    """Construct the required development panel from locally available inputs."""

    cases: list[RegressionCase] = []
    for kind in ("translation", "rotation", "hinge"):
        cases.append(
            RegressionCase(
                case_id=f"toy_{kind}",
                category="toy",
                source=f"generated:seed42:{kind}",
                loader=lambda kind=kind: rigid_toy(kind),
            )
        )

    for case_id, config in FORMAL_SYSTEMS.items():
        topology = Path(config["topology"])
        trajectory = Path(config["trajectory"])
        cases.append(
            RegressionCase(
                case_id=case_id,
                category="formal_md",
                source=f"{repo_path(topology)};{repo_path(trajectory)}",
                loader=lambda config=config: load_formal(config),
                notes="Uniformly selected development frames; not a replacement for formal evidence.",
            )
        )

    selected_path = LONG_PANEL_ROOT / "selected_domains.tsv"
    with selected_path.open(newline="") as handle:
        selected_rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in selected_rows:
        domain_id = row["domain_id"]
        path = LONG_PANEL_ROOT / "runs" / domain_id / "10000/transition_aligned_coords.npz"
        cases.append(
            RegressionCase(
                case_id=f"mdcath_{domain_id}",
                category=f"mdcath_class_{row['cath_class']}_{row['selection_bucket']}",
                source=repo_path(path),
                loader=lambda path=path: load_npz(path, frames=100, uniform=True),
                notes="100-frame deterministic development subset of synthetic scaling data.",
            )
        )

    real_root = (
        PROJECT_ROOT
        / "results/mdcath_benchmark/long_continuous_trajectory/continuous_inputs"
    )
    for filename in REAL_REPLICATES:
        path = real_root / filename
        if path.is_file():
            cases.append(
                RegressionCase(
                    case_id=f"real_{path.stem.replace('__T320__R1_ca_coords', '')}",
                    category="real_500_frame_replicate",
                    source=repo_path(path),
                    loader=lambda path=path: load_npz(path),
                )
            )

    repeated_path = LONG_PANEL_ROOT / "runs/1nd4A02/10000/transition_aligned_coords.npz"
    cases.append(
        RegressionCase(
            case_id="repeated_segment_1nd4A02_1500",
            category="transition_aligned_repeated_segment",
            source=repo_path(repeated_path),
            loader=lambda: load_npz(repeated_path, frames=1500),
            notes="A/A-prime/A-double-prime-style transition-aligned diagnostic; not continuous MD.",
        )
    )
    scaling_path = LONG_PANEL_ROOT / "runs/4oelA02/10000/transition_aligned_coords.npz"
    cases.append(
        RegressionCase(
            case_id="synthetic_scaling_4oelA02_10000",
            category="transition_aligned_synthetic_10k",
            source=repo_path(scaling_path),
            loader=lambda: load_npz(scaling_path),
            notes="Primary computational scaling case; no biological continuity claim.",
        )
    )
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def long_arrays(coords: np.ndarray, chain: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run and compact the frozen legacy 2,000-representative long-mode."""

    start = time.perf_counter()
    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=min(2000, coords.shape[0]),
        frame_selection="uniform",
        random_seed=42,
        chunk_size=min(2000, coords.shape[0]),
        store_aligned=False,
        **ROBUST_KWARGS,
    )
    runtime = time.perf_counter() - start
    labels = (
        [f"{value}:{index + 1}:UNK" for index, value in enumerate(chain)]
        if chain is not None
        else None
    )
    profile = build_long_flexibility_profile(
        result,
        residue_labels=labels,
        min_segment_length=3,
        merge_gap=1,
    )
    core = select_core_by_target_fraction(result.frozen_weights, 0.6)
    arrays = {
        "legacy_long_selected_indices": result.selected_frame_indices,
        "legacy_long_frozen_weights": result.frozen_weights,
        "legacy_long_core_labels": core.core_mask.astype(np.uint8),
        "legacy_long_rotations": result.rotations,
        "legacy_long_translations": result.translations,
        "legacy_long_rmsd": result.rmsd,
        "legacy_long_weighted_rmsd": result.weighted_rmsd,
        "legacy_long_robust_rmsf": result.robust_rmsf,
        "legacy_long_ordinary_rmsf": result.ordinary_rmsf,
        "legacy_long_flexible_labels": (profile.flexible_segment_id > 0).astype(np.uint8),
    }
    metadata = {
        "legacy_long_runtime_seconds": runtime,
        "legacy_long_peak_memory_status": "not_measured_in_golden_freeze",
        "legacy_long_average_iterations": result.average_iterations_per_frame,
        "legacy_long_converged_fraction": result.converged_fraction,
    }
    return arrays, metadata


def default_arrays(coords: np.ndarray, chain: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run and compact the frozen default trajectory implementation."""

    start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        coords,
        reference_mode="first_frame",
        warm_start=True,
        **ROBUST_KWARGS,
    )
    runtime = time.perf_counter() - start
    labels = (
        [f"{value}:{index + 1}:UNK" for index, value in enumerate(chain)]
        if chain is not None
        else None
    )
    profile = build_flexibility_profile(
        coords,
        residue_labels=labels,
        trajectory_result=result,
        min_segment_length=3,
        merge_gap=1,
    )
    weights = np.asarray([frame.result.weights for frame in result.frame_results], dtype=float)
    mean_weights = np.mean(weights, axis=0)
    core = select_core_by_target_fraction(mean_weights, 0.6)
    arrays = {
        "legacy_default_weights": weights,
        "legacy_default_mean_weights": mean_weights,
        "legacy_default_core_labels": core.core_mask.astype(np.uint8),
        "legacy_default_rotations": np.asarray(
            [frame.result.rotation for frame in result.frame_results], dtype=float
        ),
        "legacy_default_translations": np.asarray(
            [frame.result.translation for frame in result.frame_results], dtype=float
        ),
        "legacy_default_rmsd": np.asarray(
            [frame.result.rmsd for frame in result.frame_results], dtype=float
        ),
        "legacy_default_weighted_rmsd": np.asarray(
            [frame.result.weighted_rmsd for frame in result.frame_results], dtype=float
        ),
        "legacy_default_robust_rmsf": profile.robust_aligned_rmsf,
        "legacy_default_ordinary_rmsf": profile.ordinary_aligned_rmsf,
        "legacy_default_flexible_labels": (profile.flexible_segment_id > 0).astype(np.uint8),
    }
    metadata = {
        "legacy_default_runtime_seconds": runtime,
        "legacy_default_average_iterations": result.average_iterations_per_frame,
        "legacy_default_converged_fraction": result.converged_fraction,
    }
    return arrays, metadata


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Run only selected case IDs.")
    args = parser.parse_args()
    requested = set(args.case or [])

    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for case in build_cases():
        if requested and case.case_id not in requested:
            continue
        row: dict[str, object] = {
            "case_id": case.case_id,
            "category": case.category,
            "source": case.source,
            "status": "pending",
            "notes": case.notes,
        }
        print(f"[golden] {case.case_id}", flush=True)
        try:
            coords, chain = case.loader()
            if coords.ndim != 3 or coords.shape[2] != 3:
                raise ValueError(f"invalid coordinate shape {coords.shape}")
            long_output, long_metadata = long_arrays(coords, chain)
            default_output, default_metadata = default_arrays(coords, chain)
            arrays = {**long_output, **default_output}
            output = REFERENCE_ROOT / f"{case.case_id}.npz"
            np.savez_compressed(output, **arrays)
            metadata_path = REFERENCE_ROOT / f"{case.case_id}.json"
            metadata = {
                "case_id": case.case_id,
                "category": case.category,
                "source": case.source,
                "n_frames": int(coords.shape[0]),
                "n_residues": int(coords.shape[1]),
                "dtype": str(coords.dtype),
                "robust_kwargs": ROBUST_KWARGS,
                "representative_cap": min(2000, coords.shape[0]),
                "frame_selection": "uniform",
                "seed": 42,
                "chain_metadata_available": chain is not None,
                "mdlovofit_reference": (
                    "not rerun; external comparison remains frozen in Phase 0/1 "
                    "because Phase 2B gates internal numerical parity"
                ),
                **long_metadata,
                **default_metadata,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            row.update(
                {
                    "status": "success",
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "reference_npz": repo_path(output),
                    "reference_json": repo_path(metadata_path),
                    "reference_sha256": sha256(output),
                    "legacy_long_runtime_seconds": long_metadata["legacy_long_runtime_seconds"],
                    "legacy_default_runtime_seconds": default_metadata["legacy_default_runtime_seconds"],
                    "mdlovofit_status": "existing_phase1_or_official_evidence_not_rerun",
                }
            )
            summary.append(
                {
                    "case_id": case.case_id,
                    "check": "golden_reference_created",
                    "status": "pass",
                    "details": (
                        f"long/default compact outputs; frames={coords.shape[0]}; "
                        f"residues={coords.shape[1]}"
                    ),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            summary.append(
                {
                    "case_id": case.case_id,
                    "check": "golden_reference_created",
                    "status": "fail",
                    "details": f"{type(exc).__name__}: {exc}",
                }
            )
        manifest.append(row)
        write_tsv(RESULT_ROOT / "golden_manifest.tsv", manifest)
        write_tsv(RESULT_ROOT / "regression_summary.tsv", summary)

    failures = sum(row["status"] != "success" for row in manifest)
    print(f"cases={len(manifest)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
