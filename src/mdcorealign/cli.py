"""Command line interface for MDCoreAlign."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from mdsuperpose import (
    Trajectory,
    build_flexibility_profile,
    build_long_flexibility_profile,
    mean_soft_weights_from_result,
    select_core_by_target_fraction,
    solve_long_trajectory_alignment,
    solve_trajectory_robust_alignment,
    summarize_flexibility_profile,
    summarize_flexible_segments,
    write_flexibility_profile_csv,
)
from mdsuperpose.alignment import apply_transform
from mdsuperpose.coordinate_scale import audit_coordinate_scale
from mdsuperpose.diagnostics import solve_trajectory_with_optional_diagnostics
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.flexibility import (
    FlexibilityProfile,
    descending_ranks,
    detect_flexible_segments,
    load_ca_multimodel_pdb_trajectory,
    percentile_scores,
)
from mdsuperpose.trajectory import (
    load_trajectory_from_array_file,
    load_trajectory_from_multimodel_pdb,
)
from mdsuperpose.window_kinematics import (
    scan_local_kinematics,
    scan_motion_domains,
)

from . import __version__
from .contracts import solve_contract
from .coordinate_export import (
    ArrayCoordinateSource,
    MDAnalysisCoordinateSource,
    _mda_atom_records,
    atom_identity,
    build_coordinate_writer,
    file_sha256,
    load_residue_annotations,
    map_annotations_to_atoms,
    parse_frame_indices,
    pdb_compatibility_issue,
    resolve_export_contract,
    rows_from_records,
    stream_apply_transforms,
    synthetic_ca_records,
)
from .experimental import (
    MOTION_GRAPH_WARNING,
    local_deformation_rows,
    motion_graph_rows,
    solve_diagnostic_rich_alignment,
    solve_fast_global_alignment,
)
from .input_validation import (
    PROTEIN_RESNAMES,
    PROTEIN_SELECTION_POLICY,
    classify_input,
    select_mdanalysis_atoms,
    support_format_rows,
)
from .output_schema import (
    FRAME_METRIC_COLUMNS,
    RESIDUE_PROFILE_COLUMNS,
    SCHEMA_VERSION,
    canonical_contract_metadata,
    canonical_timing,
    schema_metadata,
)
from .result_provenance import (
    build_source_manifest,
    load_source_manifest,
    resolve_source_files,
    sequence_fingerprint,
    write_source_manifest,
)

DEFAULT_CORE_FRACTION = 0.6
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_DELTA = 0.2
DEFAULT_RULE = "huber"
DEFAULT_FIT_SELECTION = "protein and name CA"

CANONICAL_PUBLIC_WORKFLOWS = {
    "throughput": "throughput",
    "exact": "exact",
    "local-deformation": "local-deformation",
    "apply": "apply",
}
"""Stable workflow names; legacy modes remain distinct unless documented."""

LEGACY_MODE_COMPATIBILITY = {
    "default": "legacy_per_frame_robust_alignment",
    "long": "legacy_representative_weight_long_mode",
    "long-fast": "legacy_reduced_representative_long_mode",
    "alignment-only": "legacy_alignment_output_alias",
    "fast-global": "experimental_throughput_with_extra_outputs",
    "diagnostic-rich": "experimental_throughput_plus_W5",
}
"""Documentation map only; it intentionally does not alter dispatch behaviour."""

_ALL_ALIGN_MODES = tuple(CANONICAL_PUBLIC_WORKFLOWS)[:2] + tuple(
    LEGACY_MODE_COMPATIBILITY
)
_ALL_ALIGN_BACKENDS = ("cpu", "gpu", "auto")


def _parse_align_mode(value: str) -> str:
    """Accept stable and legacy modes while keeping legacy names out of basic help."""

    if value not in _ALL_ALIGN_MODES:
        allowed = ", ".join(_ALL_ALIGN_MODES)
        raise argparse.ArgumentTypeError(f"unknown mode '{value}'; choose one of: {allowed}")
    return value


def _parse_backend(value: str) -> str:
    """Accept CPU, GPU and the experimental auto backend."""

    if value not in _ALL_ALIGN_BACKENDS:
        allowed = ", ".join(_ALL_ALIGN_BACKENDS)
        raise argparse.ArgumentTypeError(
            f"unknown backend '{value}'; choose one of: {allowed}"
        )
    return value


class _AdvancedHelpAction(argparse.Action):
    """Print the compatibility and research controls hidden from basic help."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print(
            "Advanced and legacy align options\n"
            "\n"
            "Legacy modes (accepted for compatibility): default, long, long-fast,\n"
            "alignment-only, fast-global, diagnostic-rich. These are not recommended\n"
            "as first-line product workflows.\n"
            "\n"
            "Advanced controls include --contract, --gpu-precision,\n"
            "--gpu-rotation-solver, --reference-mode, --warm-start, robust-weight\n"
            "parameters, representative-frame budgets, chunking, and experimental\n"
            "balanced-lite and motion-graph controls. See\n"
            "docs/LEGACY_AND_ADVANCED.md and\n"
            "docs/cli_reference.md before changing them.\n"
        )
        parser.exit()


def _write_csv(path: Path, rows: list[dict[str, object]], *, delimiter: str = ",") -> None:
    """Write dictionaries as CSV/TSV with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_table_with_fields(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
    *,
    delimiter: str = "\t",
) -> None:
    """Write a table with an explicit header, even when there are no rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one versioned public JSON record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    versioned = dict(payload)
    versioned.setdefault("schema_version", SCHEMA_VERSION)
    versioned.setdefault("software_version", __version__)
    path.write_text(json.dumps(_jsonable(versioned), indent=2, sort_keys=True) + "\n")


def _jsonable(value: object) -> object:
    """Convert NumPy scalar values into JSON-friendly objects."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class PdbAtomRecord:
    """Lightweight atom metadata from the first model of a PDB trajectory."""

    record_name: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resid: str
    icode: str
    segid: str
    element: str
    occupancy: float = 1.0
    b_factor: float = 0.0

    @property
    def residue_key(self) -> str:
        """Return a stable chain/residue key for reports."""

        chain = self.chain or self.segid or "_"
        return f"{chain}:{self.resname}{self.resid}{self.icode}".strip()


@dataclass(frozen=True)
class PreparedInput:
    """Loaded trajectory plus fit/apply selections for one CLI command."""

    fit_trajectory: Trajectory
    fit_labels: tuple[str, ...]
    apply_coords: np.ndarray
    apply_records: tuple[PdbAtomRecord, ...] | None
    apply_labels: tuple[str, ...]
    input_format: str
    fit_selection: str
    apply_selection: str
    warnings: list[str]
    chain_report_rows: list[dict[str, object]]
    unit_provenance: dict[str, object]
    default_export_selection: str
    n_apply_available_atoms: int
    source_is_ca_only: bool
    source_atom_identities: tuple[str, ...] = ()
    fit_atom_identities: tuple[str, ...] = ()
    source_metadata_kind: str = "topology"

    @property
    def n_fit_residues(self) -> int:
        """Return approximate number of unique fit residues."""

        return len(set(self.fit_labels))

    @property
    def canonical_default_export_selection(self) -> str:
        """Return the user-facing default export selection."""

        if self.source_is_ca_only:
            return "all_available_atoms"
        return self.default_export_selection

    @property
    def n_exportable_atoms(self) -> int:
        """Return the number of source atoms available to normal export."""

        return int(self.n_apply_available_atoms)


CHAIN_FIELDS = [
    "original_chain",
    "original_segid",
    "normalized_chain",
    "n_residues",
    "resid_min",
    "resid_max",
    "duplicate_resids",
    "missing_chain_id",
    "notes",
]

AUDIT_FIELDS = [
    "system",
    "input",
    "topology_readable",
    "trajectory_readable",
    "n_frames",
    "n_atoms",
    "fit_selection",
    "n_fit_atoms",
    "n_fit_residues",
    "apply_selection",
    "n_apply_atoms",
    "chain_segid_summary",
    "duplicate_residue_labels",
    "missing_chain_ids",
    "nan_coordinates",
    "large_frame_to_frame_displacement_A",
    "center_of_geometry_drift_A",
    "box_vector_availability",
    "suspected_PBC_imaging_issue",
    "estimated_memory_class",
    "warnings",
    "input_readiness_status",
    "input_readiness_reason",
    "input_readiness_recommended_action",
    "recommended_action",
]


def _timestamp() -> str:
    """Return a local ISO timestamp."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_selection(selection: str | None) -> str:
    """Normalize a small supported selection expression."""

    return (selection or DEFAULT_FIT_SELECTION).strip().lower()


def _is_supported_pdb_selection(selection: str) -> bool:
    """Return whether the simple PDB selector supports this expression."""

    selection = _normalize_selection(selection)
    return selection in {
        "all",
        "protein",
        "name ca",
        "protein and name ca",
        "backbone",
        "protein and backbone",
    }


def _select_records(records: tuple[PdbAtomRecord, ...], selection: str) -> np.ndarray:
    """Select PDB atoms with the documented conservative expression subset."""

    selection = _normalize_selection(selection)
    if not _is_supported_pdb_selection(selection):
        raise ValueError(
            f"unsupported selection '{selection}'. Supported PDB selections are: "
            "'protein and name CA', 'protein and backbone', 'protein', and 'all'."
        )
    mask = np.zeros(len(records), dtype=bool)
    for index, record in enumerate(records):
        is_protein = record.resname.upper() in PROTEIN_RESNAMES
        atom_name = record.atom_name.upper()
        if selection == "all":
            mask[index] = True
        elif selection == "protein":
            mask[index] = is_protein
        elif selection in {"name ca", "protein and name ca"}:
            mask[index] = (is_protein or selection == "name ca") and atom_name == "CA"
        elif selection in {"backbone", "protein and backbone"}:
            mask[index] = is_protein and atom_name in {"N", "CA", "C", "O"}
    return mask


def _parse_pdb_atom_line(line: str) -> tuple[PdbAtomRecord, list[float]]:
    """Parse one ATOM/HETATM PDB line into metadata and coordinates."""

    record_name = line[0:6].strip()
    serial_text = line[6:11].strip()
    serial = int(serial_text) if serial_text.isdigit() else 0
    atom_name = line[12:16].strip()
    altloc = line[16].strip()
    resname = line[17:20].strip() or "UNK"
    chain = line[21].strip()
    resid = line[22:26].strip() or "0"
    icode = line[26].strip()
    segid = line[72:76].strip() if len(line) >= 76 else ""
    element = line[76:78].strip() if len(line) >= 78 else ""
    occupancy_text = line[54:60].strip() if len(line) >= 60 else ""
    b_factor_text = line[60:66].strip() if len(line) >= 66 else ""
    occupancy = float(occupancy_text) if occupancy_text else 1.0
    b_factor = float(b_factor_text) if b_factor_text else 0.0
    coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return (
        PdbAtomRecord(
            record_name=record_name,
            serial=serial,
            atom_name=atom_name,
            altloc=altloc,
            resname=resname,
            chain=chain,
            resid=resid,
            icode=icode,
            segid=segid,
            element=element,
            occupancy=occupancy,
            b_factor=b_factor,
        ),
        coords,
    )


def _load_pdb_models(path: Path) -> tuple[np.ndarray, tuple[PdbAtomRecord, ...]]:
    """Load ATOM/HETATM models from a simple PDB trajectory."""

    frames: list[list[list[float]]] = []
    records_by_frame: list[list[PdbAtomRecord]] = []
    current_coords: list[list[float]] = []
    current_records: list[PdbAtomRecord] = []

    def flush_frame() -> None:
        if current_coords:
            frames.append(list(current_coords))
            records_by_frame.append(list(current_records))
            current_coords.clear()
            current_records.clear()

    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                record, coords = _parse_pdb_atom_line(line)
                if record.altloc not in {"", "A"}:
                    continue
                current_records.append(record)
                current_coords.append(coords)
            elif line.startswith("ENDMDL"):
                flush_frame()
    flush_frame()
    if not frames:
        raise ValueError(f"no ATOM/HETATM coordinate frames could be parsed from {path}")
    n_atoms = len(frames[0])
    reference_signature = [
        (record.record_name, record.atom_name, record.resname, record.chain, record.resid, record.icode)
        for record in records_by_frame[0]
    ]
    for frame_index, records in enumerate(records_by_frame[1:], start=1):
        if len(records) != n_atoms:
            raise ValueError(
                "topology/trajectory atom counts mismatch across PDB models; "
                f"frame 0 has {n_atoms} atoms but frame {frame_index} has {len(records)}"
            )
        signature = [
            (record.record_name, record.atom_name, record.resname, record.chain, record.resid, record.icode)
            for record in records
        ]
        if signature != reference_signature:
            raise ValueError(
                "topology/trajectory atom identity changed across PDB models; "
                "MDCoreAlign requires consistent atom order and residue mapping"
            )
    coords = np.asarray(frames, dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("input coordinates contain NaN or infinite values; clean the trajectory before alignment")
    return coords, tuple(records_by_frame[0])


def _records_to_labels(records: tuple[PdbAtomRecord, ...]) -> tuple[str, ...]:
    """Convert atom records into residue labels used by flexibility outputs."""

    return tuple(record.residue_key for record in records)


def _chain_mapping_rows(records: tuple[PdbAtomRecord, ...] | None) -> list[dict[str, object]]:
    """Build chain mapping report rows."""

    if records is None:
        return [
            {
                "original_chain": "",
                "original_segid": "",
                "normalized_chain": "A",
                "n_residues": "",
                "resid_min": "",
                "resid_max": "",
                "duplicate_resids": "",
                "missing_chain_id": "",
                "notes": "array input has no chain metadata; default chain A is assumed",
            }
        ]
    groups: dict[tuple[str, str], list[PdbAtomRecord]] = {}
    for record in records:
        groups.setdefault((record.chain, record.segid), []).append(record)
    rows: list[dict[str, object]] = []
    for (chain, segid), group in sorted(groups.items(), key=lambda item: (item[0][0] or "_", item[0][1] or "_")):
        residue_keys = [record.residue_key for record in group]
        unique_residues = sorted(set(residue_keys))
        duplicate_count = len(residue_keys) - len(unique_residues)
        numeric_resids = []
        for record in group:
            try:
                numeric_resids.append(int(record.resid))
            except ValueError:
                pass
        rows.append(
            {
                "original_chain": chain,
                "original_segid": segid,
                "normalized_chain": chain or "_",
                "n_residues": len(unique_residues),
                "resid_min": min(numeric_resids) if numeric_resids else "",
                "resid_max": max(numeric_resids) if numeric_resids else "",
                "duplicate_resids": duplicate_count,
                "missing_chain_id": int(chain == ""),
                "notes": "duplicate_resids counts atom-level repeats within residue; expected for non-CA apply selections"
                if duplicate_count
                else "",
            }
        )
    return rows


def _chain_mapping_rows_from_labels(labels: tuple[str, ...]) -> list[dict[str, object]]:
    """Build a conservative chain report from residue labels."""

    groups: dict[str, list[str]] = {}
    for label in labels:
        chain = label.split(":", 1)[0] if ":" in label else "_"
        groups.setdefault(chain or "_", []).append(label)
    rows: list[dict[str, object]] = []
    for chain, group in sorted(groups.items()):
        numeric_resids: list[int] = []
        for label in group:
            digits = "".join(character for character in label.split(":", 1)[-1] if character.isdigit())
            if digits:
                numeric_resids.append(int(digits))
        rows.append(
            {
                "original_chain": "" if chain == "_" else chain,
                "original_segid": "",
                "normalized_chain": chain,
                "n_residues": len(set(group)),
                "resid_min": min(numeric_resids) if numeric_resids else "",
                "resid_max": max(numeric_resids) if numeric_resids else "",
                "duplicate_resids": len(group) - len(set(group)),
                "missing_chain_id": int(chain == "_"),
                "notes": "metadata inferred from trajectory backend labels",
            }
        )
    return rows


def _write_chain_mapping_report(path: Path, rows: list[dict[str, object]]) -> None:
    """Write chain mapping report."""

    _write_table_with_fields(path, rows, CHAIN_FIELDS)


def _write_run_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_files: dict[str, object],
    warnings: list[str],
    prepared: PreparedInput | None = None,
) -> None:
    """Write run metadata for reproducibility."""

    external_tools = {
        "MDLovoFit": shutil.which(os.environ.get("MDLOVOFIT_EXECUTABLE", "mdlovofit")) is not None,
        "THESEUS": shutil.which(os.environ.get("THESEUS_EXECUTABLE", "theseus")) is not None,
    }
    input_files = {"input": str(input_path)}
    topology = getattr(args, "topology", None)
    trajectory = getattr(args, "trajectory", None)
    if topology:
        input_files["topology"] = str(topology)
    if trajectory:
        input_files["trajectory"] = str(trajectory)
    format_overrides = {
        "input_format": getattr(args, "input_format", None),
        "topology_format": getattr(args, "topology_format", None),
        "trajectory_format": getattr(args, "trajectory_format", None),
    }
    mode = getattr(args, "mode", None)
    contract = getattr(args, "contract", None)
    if mode in {"exact", "throughput"}:
        contract = mode
    if getattr(args, "command", None) == "align" and contract in {
        "exact",
        "throughput",
    }:
        contract_fields = canonical_contract_metadata(
            contract,
            result_mode=str(mode),
        )
    elif getattr(args, "command", None) == "apply":
        contract_fields = {
            "method": "coordinate materialization",
            "mode": "apply",
            "execution_contract": "coordinate_materialization",
            "experimental_mode": False,
        }
    else:
        contract_fields = {}
    representative_count_k = None
    update_rounds_m = None
    if getattr(args, "command", None) == "align":
        representative_count_k = int(
            getattr(
                args,
                "exact_representative_count"
                if contract == "exact"
                else "fast_representative_count",
            )
        )
        if contract != "exact":
            update_rounds_m = int(args.fast_update_rounds)
    _write_json(
        path,
        {
            **contract_fields,
            "package_version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "timestamp": _timestamp(),
            "input_files": input_files,
            "format_overrides": format_overrides,
            "unit_provenance": (
                prepared.unit_provenance if prepared is not None else None
            ),
            "output_files": output_files,
            "external_tools_detected": external_tools,
            "random_seed_if_any": None,
            "representative_count_K": representative_count_k,
            "update_rounds_M": update_rounds_m,
            "requested_parameters": getattr(args, "_requested_parameters", {}),
            "effective_parameters": getattr(args, "_effective_parameters", {}),
            "warnings": warnings,
        },
    )


def _input_path_from_args(args: argparse.Namespace) -> Path:
    """Return the primary coordinate input path from legacy or topology/trajectory CLI options."""

    input_value = getattr(args, "input", None)
    topology_value = getattr(args, "topology", None)
    trajectory_value = getattr(args, "trajectory", None)
    if input_value:
        return Path(input_value)
    if trajectory_value:
        return Path(trajectory_value)
    if topology_value:
        return Path(topology_value)
    raise ValueError(
        "No input coordinates were provided. Use --input for array/PDB examples, "
        "or provide --topology and --trajectory for topology+trajectory workflows."
    )


def _infer_input_format(path: Path, requested: str, *, topology_format: str | None = None, trajectory_format: str | None = None) -> str:
    """Resolve input format from CLI value and file suffix."""

    # A topology override describes a different file and cannot choose how the
    # primary coordinate trajectory is decoded (e.g. PDB topology + XTC frames).
    override = trajectory_format
    if override:
        normalized = override.lower()
        if normalized in {"pdb", "ca-pdb", "array"}:
            return normalized
        if normalized in {"npy", "npz"}:
            return "array"
        return override.upper()
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return "array"
    if suffix in {".pdb", ".ent"}:
        return "ca-pdb"
    if suffix in {".gro", ".xtc", ".dcd", ".trr", ".nc", ".netcdf", ".mdcrd", ".trj", ".tng", ".h5md", ".gsd", ".lammpstrj", ".dump", ".dms", ".mol2", ".xyz"}:
        return suffix.lstrip(".").upper()
    raise ValueError("Format detection failed. Check file extension or use --topology-format / --trajectory-format.")


def _load_trajectory(
    path: str | Path,
    *,
    input_format: str = "auto",
    array_key: str | None = None,
    chain_id: str | None = None,
) -> tuple[Trajectory, tuple[str, ...]]:
    """Load a supported trajectory and residue labels."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    resolved_format = _infer_input_format(path, input_format)
    if resolved_format == "array":
        trajectory = load_trajectory_from_array_file(path, key=array_key)
        if not np.isfinite(trajectory.coords).all():
            raise ValueError("input coordinates contain NaN or infinite values; clean the trajectory before alignment")
        labels = tuple(f"A:UNK{index + 1}" for index in range(trajectory.n_atoms))
        return trajectory, labels
    if resolved_format == "pdb":
        trajectory = load_trajectory_from_multimodel_pdb(path)
        labels = tuple(f"A:UNK{index + 1}" for index in range(trajectory.n_atoms))
        return trajectory, labels
    if resolved_format == "ca-pdb":
        trajectory, labels = load_ca_multimodel_pdb_trajectory(path, chain_id=chain_id)
        return trajectory, labels
    raise ValueError("input-format must be one of: auto, array, pdb, ca-pdb")


def _load_mdanalysis_prepared_input(args: argparse.Namespace, resolved_format: str, warnings: list[str]) -> PreparedInput:
    """Load topology+trajectory input through the standard MDAnalysis adapter."""

    topology_path = Path(args.topology) if args.topology else None
    trajectory_path = Path(args.trajectory) if args.trajectory else None
    if topology_path is None:
        raise ValueError("A reliable topology is required for residue-level stable-core and RMSF analysis.")
    if trajectory_path is None:
        trajectory_path = topology_path
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - installation integrity boundary
        raise ValueError(
            "The standard MDCoreAlign installation includes MDAnalysis. "
            "This environment appears incomplete. Reinstall the package with "
            "`python -m pip install --force-reinstall <wheel-path>`."
        ) from exc

    try:
        kwargs = {}
        if getattr(args, "topology_format", None):
            kwargs["topology_format"] = args.topology_format
        if getattr(args, "trajectory_format", None):
            kwargs["format"] = args.trajectory_format
        universe = mda.Universe(str(topology_path), str(trajectory_path), **kwargs)
    except Exception as exc:  # pragma: no cover - backend-specific
        raise ValueError(
            "Could not read topology+trajectory input through MDAnalysis. "
            "Check file compatibility, atom counts, and format overrides."
        ) from exc
    fit_atoms = select_mdanalysis_atoms(universe, args.fit_selection, chain=getattr(args, "chain", None))
    apply_atoms = select_mdanalysis_atoms(universe, args.apply_selection or args.fit_selection)
    protein_atoms = select_mdanalysis_atoms(universe, "protein")
    if fit_atoms.n_atoms == 0:
        raise ValueError(
            f"fit-selection '{args.fit_selection}' returned zero atoms. "
            "RNA/DNA-only or non-protein systems are not formally validated; define an explicit experimental selection if appropriate."
        )
    if apply_atoms.n_atoms == 0:
        raise ValueError(f"apply-selection '{args.apply_selection}' returned zero atoms.")
    fit_frames: list[np.ndarray] = []
    apply_frames: list[np.ndarray] = []
    for _ in universe.trajectory:
        fit_frames.append(fit_atoms.positions.astype(float).copy())
        apply_frames.append(apply_atoms.positions.astype(float).copy())
    fit_coords = np.asarray(fit_frames, dtype=float)
    apply_coords = np.asarray(apply_frames, dtype=float)
    if not np.isfinite(fit_coords).all() or not np.isfinite(apply_coords).all():
        raise ValueError("input coordinates contain NaN or infinite values; clean the trajectory before alignment")

    def labels_from_atoms(atoms) -> tuple[str, ...]:
        labels: list[str] = []
        for atom in atoms:
            chain = getattr(atom, "chainID", "") or getattr(atom, "segid", "") or "_"
            resname = getattr(atom, "resname", "UNK") or "UNK"
            resid = getattr(atom, "resid", len(labels) + 1)
            icode = getattr(atom, "icode", "") or ""
            labels.append(f"{chain}:{resname}{resid}{icode}")
        return tuple(labels)

    fit_labels = labels_from_atoms(fit_atoms)
    apply_labels = labels_from_atoms(apply_atoms)
    if np.any(~np.isin(np.arange(apply_atoms.n_atoms), np.arange(fit_atoms.n_atoms))):
        warnings.append(
            "apply-selection may include atoms outside fit-selection; fitted transforms are applied to them, "
            "but they are not used for stability weights or RMSF by default"
        )
    warnings.append(
        "topology+trajectory input was read through MDAnalysis; verify residue metadata and atom order in input_audit.tsv"
    )
    from MDAnalysis.core._get_readers import get_parser_for

    topology_format_used = get_parser_for(str(topology_path), format=kwargs.get("topology_format")).format
    if not isinstance(topology_format_used, str):
        topology_format_used = topology_format_used[0]
    trajectory_format_used = universe.trajectory.format
    if not isinstance(trajectory_format_used, str):
        trajectory_format_used = trajectory_format_used[0]
    return PreparedInput(
        fit_trajectory=Trajectory(coords=fit_coords),
        fit_labels=fit_labels,
        apply_coords=apply_coords,
        apply_records=_mda_atom_records(apply_atoms),
        apply_labels=apply_labels,
        input_format=resolved_format,
        fit_selection=args.fit_selection,
        apply_selection=args.apply_selection or args.fit_selection,
        warnings=warnings,
        chain_report_rows=_chain_mapping_rows_from_labels(apply_labels),
        unit_provenance={
            "source_unit": "backend_managed",
            "conversion_factor_to_angstrom": 1.0,
            "canonical_internal_unit": "angstrom",
            "unit_resolution": (
                "MDAnalysis coordinate readers expose positions in Angstrom; "
                "no geometry-based unit inference was used"
            ),
            "topology_format_used": topology_format_used,
            "trajectory_format_used": trajectory_format_used,
        },
        default_export_selection="protein",
        n_apply_available_atoms=int(protein_atoms.n_atoms),
        source_is_ca_only=bool(
            protein_atoms.n_atoms == universe.atoms.n_atoms
            and protein_atoms.n_atoms > 0
            and all(str(name).upper() == "CA" for name in protein_atoms.names)
        ),
        source_atom_identities=tuple(atom_identity(r) for r in _mda_atom_records(universe.atoms)),
        fit_atom_identities=tuple(atom_identity(r) for r in _mda_atom_records(fit_atoms)),
    )


def _prepare_input(args: argparse.Namespace) -> PreparedInput:
    """Load an input file and apply validated fit and apply selections."""

    if getattr(args, "input", None) and (getattr(args, "topology", None) or getattr(args, "trajectory", None)):
        raise ValueError("use --input or --topology/--trajectory, not both input conventions")
    input_path = _input_path_from_args(args)
    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if getattr(args, "trajectory", None) and not getattr(args, "topology", None):
        if input_path.suffix.lower() not in {".pdb", ".ent", ".npy", ".npz"}:
            raise ValueError("A reliable topology is required for residue-level stable-core and RMSF analysis.")
    resolved_format = _infer_input_format(
        input_path,
        args.input_format,
        topology_format=getattr(args, "topology_format", None),
        trajectory_format=(getattr(args, "trajectory_format", None) or (
            getattr(args, "topology_format", None)
            if getattr(args, "topology", None) and not getattr(args, "trajectory", None)
            else None
        )),
    )
    fit_selection = getattr(args, "fit_selection", DEFAULT_FIT_SELECTION)
    apply_selection = getattr(args, "apply_selection", None) or fit_selection
    warnings: list[str] = []
    requested_source_unit = getattr(args, "source_unit", "auto")
    if resolved_format != "array" and getattr(args, "array_key", None) is not None:
        raise ValueError("--array-key is only valid for a NumPy .npz archive")
    if resolved_format in {"pdb", "ca-pdb"} and requested_source_unit == "nanometer":
        raise ValueError(
            "PDB coordinate fields are interpreted as Angstrom. Do not pass "
            "--source-unit nanometer for PDB input; convert the file explicitly "
            "and record that preprocessing instead."
        )
    if getattr(args, "topology", None) and (getattr(args, "trajectory", None) or resolved_format not in {"pdb", "ca-pdb", "array"}):
        if requested_source_unit == "nanometer":
            raise ValueError(
                "Topology+trajectory coordinates are unit-normalized by the "
                "MDAnalysis backend. Do not apply a second nanometer conversion."
            )
        return _load_mdanalysis_prepared_input(args, resolved_format, warnings)
    if resolved_format in {"pdb", "ca-pdb"}:
        coords, records = _load_pdb_models(input_path)
        fit_mask = _select_records(records, fit_selection)
        if getattr(args, "chain", None) is not None:
            chain_mask = np.asarray([(record.chain or record.segid) == args.chain for record in records])
            if not np.any(chain_mask & fit_mask):
                raise ValueError(f"chain '{args.chain}' has no atoms in fit-selection '{fit_selection}'")
            fit_mask &= chain_mask
        apply_mask = _select_records(records, apply_selection)
        if not np.any(fit_mask):
            resnames = {record.resname.upper() for record in records}
            nucleic_names = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "ADE", "CYT", "GUA", "THY", "URA"}
            if resnames and resnames.issubset(nucleic_names):
                raise ValueError(
                    f"fit-selection '{fit_selection}' returned zero atoms. "
                    "RNA/DNA-only systems are not formally validated; the protein C-alpha default does not apply. "
                    "Use an explicit experimental fit-selection only if you define a stable atom correspondence."
                )
            raise ValueError(
                f"fit-selection '{fit_selection}' returned zero atoms. "
                "Use --fit-selection 'protein and name CA' for protein CA fitting, or inspect the input with audit-preprocessing."
            )
        if not np.any(apply_mask):
            raise ValueError(
                f"apply-selection '{apply_selection}' returned zero atoms. "
                "Use --apply-selection 'protein', 'all', or the same expression as --fit-selection."
            )
        if np.any(apply_mask & ~fit_mask):
            warnings.append(
                "apply-selection includes atoms outside fit-selection; fitted transforms are applied to them, "
                "but they are not used for stability weights or RMSF by default"
            )
        if "all" in _normalize_selection(apply_selection):
            warnings.append("ligands, lipids, solvent, or HETATM records can be transformed but are not part of stable-core inference by default")
        fit_records = tuple(record for record, keep in zip(records, fit_mask, strict=True) if keep)
        apply_records = tuple(record for record, keep in zip(records, apply_mask, strict=True) if keep)
        protein_mask = _select_records(records, "protein")
        protein_records = tuple(
            record for record, keep in zip(records, protein_mask, strict=True) if keep
        )
        fit_coords = coords[:, fit_mask, :]
        apply_coords = coords[:, apply_mask, :]
        return PreparedInput(
            fit_trajectory=Trajectory(coords=fit_coords),
            fit_labels=_records_to_labels(fit_records),
            apply_coords=apply_coords,
            apply_records=apply_records,
            apply_labels=_records_to_labels(apply_records),
            input_format=resolved_format,
            fit_selection=fit_selection,
            apply_selection=apply_selection,
            warnings=warnings,
            chain_report_rows=_chain_mapping_rows(apply_records),
            unit_provenance={
                "source_unit": "angstrom",
                "conversion_factor_to_angstrom": 1.0,
                "canonical_internal_unit": "angstrom",
                "unit_resolution": "PDB coordinate-field convention",
            },
            default_export_selection="protein",
            n_apply_available_atoms=len(protein_records),
            source_is_ca_only=bool(
                len(protein_records) == len(records)
                and bool(protein_records)
                and all(record.atom_name.upper() == "CA" for record in records)
            ),
            source_atom_identities=tuple(atom_identity(record) for record in records),
            fit_atom_identities=tuple(atom_identity(record) for record in fit_records),
        )

    if getattr(args, "chain", None) is not None:
        raise ValueError("--chain requires real topology/chain metadata; array rows have synthetic identifiers")
    array_selections = {"all", "protein", "name ca", "protein and name ca"}
    if any(_normalize_selection(value) not in array_selections for value in (fit_selection, apply_selection)):
        raise ValueError("array input has no topology for this selection; use all coordinate rows or provide a topology")
    if input_path.suffix.lower() == ".npz":
        with np.load(input_path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("NumPy archive contains no coordinate arrays")
            args.array_key = args.array_key or ("coords" if "coords" in archive.files else archive.files[0])
    elif args.array_key is not None:
        raise ValueError("--array-key is only valid for a NumPy .npz archive")
    trajectory, labels = _load_trajectory(
        input_path,
        input_format=resolved_format,
        array_key=args.array_key,
        chain_id=args.chain,
    )
    warnings.append("array rows use synthetic CA/UNK metadata; no real residue or atom identity was supplied")
    requested_source_unit = getattr(args, "source_unit", "auto")
    conversion_factor = 10.0 if requested_source_unit == "nanometer" else 1.0
    if conversion_factor != 1.0:
        trajectory = Trajectory(coords=trajectory.coords * conversion_factor)
    source_unit = (
        requested_source_unit if requested_source_unit != "auto" else "unknown"
    )
    if requested_source_unit == "auto":
        warnings.append(
            "array coordinate unit is unknown; values were preserved without "
            "geometry-based inference or rescaling"
        )
    return PreparedInput(
        fit_trajectory=trajectory,
        fit_labels=labels,
        apply_coords=trajectory.coords,
        apply_records=None,
        apply_labels=labels,
        input_format=resolved_format,
        fit_selection=fit_selection,
        apply_selection=apply_selection,
        warnings=warnings,
        chain_report_rows=_chain_mapping_rows(None),
        unit_provenance={
            "source_unit": source_unit,
            "conversion_factor_to_angstrom": conversion_factor,
            "canonical_internal_unit": (
                "angstrom" if requested_source_unit != "auto" else "unknown"
            ),
            "unit_resolution": (
                "explicit --source-unit"
                if requested_source_unit != "auto"
                else "unresolved array metadata; coordinates preserved"
            ),
        },
        default_export_selection="all",
        n_apply_available_atoms=trajectory.n_atoms,
        source_is_ca_only=True,
        source_atom_identities=tuple(atom_identity(r) for r in synthetic_ca_records(trajectory.n_atoms)),
        fit_atom_identities=tuple(atom_identity(r) for r in synthetic_ca_records(trajectory.n_atoms)),
        source_metadata_kind="synthetic_array",
    )


def _aligned_apply_coords(result, prepared: PreparedInput) -> np.ndarray:
    """Apply fitted frame transforms to the apply-selection coordinates."""

    rotations = getattr(result, "rotations", None)
    translations = getattr(result, "translations", None)
    if rotations is not None and translations is not None:
        if prepared.apply_coords.shape[0] != rotations.shape[0]:
            raise ValueError("apply-selection trajectory does not match fitted transform frame count")
        return apply_transforms_batch(prepared.apply_coords, rotations, translations)
    if prepared.apply_coords.shape[0] != len(result.frame_results):
        raise ValueError("apply-selection trajectory does not match fit trajectory frame count")
    aligned = np.empty_like(prepared.apply_coords, dtype=float)
    for frame, coords in zip(result.frame_results, prepared.apply_coords, strict=True):
        aligned[frame.frame_index] = apply_transform(coords, frame.result)
    return aligned


def _format_pdb_atom_line(record: PdbAtomRecord, serial: int, xyz: np.ndarray) -> str:
    """Format one atom line while preserving lightweight metadata."""

    chain = (record.chain or " ")[:1]
    resid = int(record.resid) if str(record.resid).strip().lstrip("-").isdigit() else serial
    element = (record.element or record.atom_name[:1] or "C")[:2].rjust(2)
    return (
        f"{record.record_name:<6}{serial:5d} {record.atom_name[:4]:>4}{record.altloc[:1]:1s}"
        f"{record.resname[:3]:>3} {chain:1s}{resid:4d}{record.icode[:1]:1s}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.00:6.2f}{0.00:6.2f}          {element}\n"
    )


def _write_aligned_output(
    coords: np.ndarray,
    path: Path,
    *,
    records: tuple[PdbAtomRecord, ...] | None,
) -> None:
    """Write aligned output using PDB metadata when available."""

    synthetic = records is None
    if synthetic:
        records = synthetic_ca_records(coords.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        if synthetic:
            handle.write("REMARK 950 SYNTHETIC CA/UNK COORDINATE ROWS; NO RESIDUE IDENTITY PROVIDED\n")
        for model_index, frame in enumerate(coords, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            for atom_index, (record, xyz) in enumerate(zip(records, frame, strict=True), start=1):
                handle.write(_format_pdb_atom_line(record, atom_index, xyz))
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def _estimate_memory_class(coords: np.ndarray) -> str:
    """Return a coarse memory class for an input trajectory."""

    megabytes = coords.size * coords.itemsize / (1024 * 1024)
    if megabytes < 10:
        return "small"
    if megabytes < 250:
        return "medium"
    return "large"


def _duplicate_residue_labels(labels: tuple[str, ...]) -> int:
    """Count duplicate residue labels."""

    return len(labels) - len(set(labels))


def _audit_row(args: argparse.Namespace, prepared: PreparedInput) -> dict[str, object]:
    """Build one input-audit row."""

    coords = prepared.fit_trajectory.coords
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=2) if prepared.fit_trajectory.n_frames > 1 else np.zeros((0, prepared.fit_trajectory.n_atoms))
    centers = np.mean(coords, axis=1)
    center_delta = np.linalg.norm(centers - centers[0], axis=1)
    missing_chain_ids = 0
    if prepared.apply_records is not None:
        missing_chain_ids = sum(record.chain == "" for record in prepared.apply_records)
    max_jump = float(np.max(diffs)) if diffs.size else 0.0
    warnings = list(prepared.warnings)
    if max_jump > getattr(args, "jump_threshold", 15.0):
        warnings.append("large frame-to-frame displacement detected; inspect PBC/imaging/nojump preprocessing")
    if missing_chain_ids:
        warnings.append("missing chain IDs detected; chain labels were normalized conservatively")
    if not np.isfinite(coords).all():
        warnings.append("NaN or infinite coordinates detected")
    duplicate_labels = _duplicate_residue_labels(prepared.fit_labels)
    chain_summary = ";".join(
        f"{row['normalized_chain'] or '_'}:{row['n_residues']}"
        for row in prepared.chain_report_rows
    )
    input_path = _input_path_from_args(args)
    readiness = classify_input(
        input_path=input_path if not getattr(args, "trajectory", None) else None,
        topology_path=getattr(args, "topology", None),
        trajectory_path=getattr(args, "trajectory", None),
        topology_format=getattr(args, "topology_format", None),
        trajectory_format=getattr(args, "trajectory_format", None),
        fit_selection=prepared.fit_selection,
        large_jump_detected=bool(max_jump > getattr(args, "jump_threshold", 15.0)),
        missing_chain_ids=missing_chain_ids,
        duplicate_residue_labels=duplicate_labels,
        nan_coordinates=bool(not np.isfinite(coords).all()),
        selection_size=prepared.fit_trajectory.n_atoms,
    )
    return {
        "system": getattr(args, "system_name", input_path.stem),
        "input": str(input_path),
        "topology_readable": True,
        "trajectory_readable": True,
        "n_frames": prepared.fit_trajectory.n_frames,
        "n_atoms": int(prepared.apply_coords.shape[1]),
        "fit_selection": prepared.fit_selection,
        "n_fit_atoms": prepared.fit_trajectory.n_atoms,
        "n_fit_residues": prepared.n_fit_residues,
        "apply_selection": prepared.apply_selection,
        "n_apply_atoms": int(prepared.apply_coords.shape[1]),
        "chain_segid_summary": chain_summary,
        "duplicate_residue_labels": duplicate_labels,
        "missing_chain_ids": missing_chain_ids,
        "nan_coordinates": bool(not np.isfinite(coords).all()),
        "large_frame_to_frame_displacement_A": max_jump,
        "center_of_geometry_drift_A": float(np.max(center_delta)) if center_delta.size else 0.0,
        "box_vector_availability": "not_available_for_core_pdb_array_reader",
        "suspected_PBC_imaging_issue": bool(max_jump > getattr(args, "jump_threshold", 15.0)),
        "estimated_memory_class": _estimate_memory_class(prepared.apply_coords),
        "warnings": "; ".join(warnings),
        "input_readiness_status": readiness.status,
        "input_readiness_reason": readiness.reason,
        "input_readiness_recommended_action": readiness.recommended_action,
        "recommended_action": readiness.recommended_action,
    }


def _write_input_audit_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    prepared: PreparedInput,
    *,
    include_json: bool = True,
) -> dict[str, object]:
    """Write the input audit and chain mapping report."""

    audit_row = _audit_row(args, prepared)
    audit_tsv = output_dir / "input_audit.tsv"
    audit_json = output_dir / "input_audit.json"
    chain_report = output_dir / "chain_mapping_report.tsv"
    _write_table_with_fields(audit_tsv, [audit_row], AUDIT_FIELDS)
    if include_json:
        _write_json(audit_json, audit_row)
    _write_chain_mapping_report(chain_report, prepared.chain_report_rows)
    outputs = {
        "input_audit_tsv": str(audit_tsv),
        "chain_mapping_report_tsv": str(chain_report),
    }
    if include_json:
        outputs["input_audit_json"] = str(audit_json)
    return outputs


FORMAT_FIELDS = [
    "category",
    "format_or_pair",
    "status",
    "requires_topology",
    "requires_trajectory",
    "recommended_for",
    "notes",
]


def _run_formats(args: argparse.Namespace) -> int:
    """Print and optionally write the MDCoreAlign format support policy."""

    rows = support_format_rows()
    if args.out:
        _write_table_with_fields(Path(args.out), rows, FORMAT_FIELDS)
        print(f"Wrote supported formats table: {args.out}")
        return 0
    validated_names = {"PDB + XTC", "GRO + XTC", "multi-model PDB"}
    validated_outputs = {
        "annotated PDB structure",
        "PDBx/mmCIF structure",
        "XTC aligned trajectory",
        "multi-model PDB compatibility",
    }
    sections = [
        (
            "Validated inputs",
            [row for row in rows if row["format_or_pair"] in validated_names],
        ),
        (
            "Validated outputs",
            [row for row in rows if row["format_or_pair"] in validated_outputs],
        ),
        (
            "Standard runtime dependencies",
            [row for row in rows if row["category"] == "standard runtime dependencies"],
        ),
        (
            "Available optional-reader ecosystem and support boundaries",
            [
                row
                for row in rows
                if row["format_or_pair"] not in validated_names
                and row["format_or_pair"] not in validated_outputs
                and row["category"] != "standard runtime dependencies"
            ],
        ),
    ]
    compact_fields = ["format_or_pair", "status", "recommended_for", "notes"]
    for section_name, section_rows in sections:
        print(section_name)
        widths = {
            field: max(len(field), *(len(str(row[field])) for row in section_rows))
            for field in compact_fields
        }
        print("  ".join(field.ljust(widths[field]) for field in compact_fields))
        print("  ".join("-" * widths[field] for field in compact_fields))
        for row in section_rows:
            print("  ".join(str(row[field]).ljust(widths[field]) for field in compact_fields))
        print()
    return 0


def _package_version(distribution: str) -> str:
    """Return an installed distribution version without importing it."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _doctor_report() -> dict[str, object]:
    """Inspect the single-product runtime without changing the environment."""

    cupy_installed = importlib.util.find_spec("cupy") is not None
    cuda_visible = False
    gpu_ready = False
    detected_gpu = "none"
    gpu_detail = "CuPy is not installed"
    if cupy_installed:
        try:
            from .gpu_backend import gpu_environment

            environment = gpu_environment()
            cuda_visible = True
            gpu_ready = True
            detected_gpu = str(environment["device_name"])
            gpu_detail = (
                f"CuPy {environment['cupy_version']}; CUDA runtime "
                f"{environment['cuda_runtime_version']}; compute capability "
                f"{environment['compute_capability']}"
            )
        except Exception as exc:  # pragma: no cover - host CUDA dependent.
            gpu_detail = f"CuPy import/device check failed: {exc}"
    return {
        "mdcorealign_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": _package_version("numpy"),
        "mdanalysis_version": _package_version("MDAnalysis"),
        "gemmi_version": _package_version("gemmi"),
        "cupy_installed": cupy_installed,
        "cuda_visible": cuda_visible,
        "gpu_backend_ready": gpu_ready,
        "detected_gpu": detected_gpu,
        "gpu_detail": gpu_detail,
        "recommended_backend": "gpu" if gpu_ready else "cpu",
    }


def _run_doctor(args: argparse.Namespace) -> int:
    """Report product and GPU readiness without installing dependencies."""

    report = _doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    labels = {
        "mdcorealign_version": "MDCoreAlign version",
        "python_version": "Python version",
        "numpy_version": "NumPy",
        "mdanalysis_version": "MDAnalysis",
        "gemmi_version": "Gemmi",
        "cupy_installed": "CuPy installed",
        "cuda_visible": "CUDA visible",
        "gpu_backend_ready": "GPU backend ready",
        "detected_gpu": "Detected GPU",
        "gpu_detail": "GPU detail",
        "recommended_backend": "Recommended backend",
    }
    for key, label in labels.items():
        value = report[key]
        if isinstance(value, bool):
            value = "yes" if value else "no"
        print(f"{label}: {value}")
    if not report["gpu_backend_ready"]:
        print(
            "GPU setup: the installed MDCoreAlign package already contains "
            "the GPU backend. Install a CuPy build compatible with this "
            "machine's CUDA runtime to enable --backend gpu."
        )
    print("VT02.2 apply interface: mdcorealign apply RESULT_DIR -o OUTPUT")
    print(f"Default fit selection: {DEFAULT_FIT_SELECTION}")
    print("Default export selection: protein when available")
    print("Annotated PDB/mmCIF: yes; XTC annotated topology companion: yes")
    return 0


def _robust_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Collect robust solver settings from parsed CLI arguments."""

    options = {
        "rule": args.rule,
        "delta": args.delta,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "sequence_continuity_strength": args.sequence_lambda,
        "contact_consistency_strength": args.contact_lambda,
    }
    # Legacy Gaussian/Tukey defaults use sigma/cutoff=1; preserve those unless
    # the user explicitly supplies the public scale option.
    if "delta" in getattr(args, "_explicit_options", ()):
        if args.rule == "gaussian":
            options["sigma"] = args.delta
        elif args.rule == "tukey":
            options["cutoff"] = args.delta
    return options


def _prepare_parameter_contract(args: argparse.Namespace) -> dict[str, object]:
    """Resolve aliases and explicit scientific options before reading/fitting input."""

    from mdsuperpose.window_refine import validate_weighting_parameters

    from .contracts import resolve_contract_parameters

    if hasattr(args, "_contract_options"):
        return args._contract_options
    explicit = set(getattr(args, "_explicit_options", ()))
    contract = getattr(args, "contract", None)
    mode = args.mode
    if contract is not None:
        if "mode" in explicit and mode != contract:
            raise ValueError("Explicit --mode and --contract select conflicting methods")
        args.mode = mode = contract
    elif mode in {"exact", "throughput"}:
        contract = mode
    gpu_options = {
        target: getattr(args, source)
        for source, target in (("gpu_precision", "precision"), ("gpu_rotation_solver", "rotation_solver"))
        if source in explicit
    }
    backend = getattr(args, "backend", "cpu")
    gpu_execution = backend == "gpu" and mode != "exact"
    if backend == "auto" and mode != "exact":
        from .gpu_backend import gpu_available

        gpu_execution = gpu_available()
    if gpu_options and not gpu_execution:
        raise ValueError("Explicit --gpu-precision/--gpu-rotation-solver requires GPU execution; CPU uses float64 SVD")
    args._gpu_options = (
        {"precision": args.gpu_precision, "rotation_solver": args.gpu_rotation_solver}
        if gpu_execution else {}
    )
    for option, supported_modes in (
        ("exact_representative_count", {"exact"}),
        ("fast_representative_count", {"throughput", "balanced-lite", "fast-global", "diagnostic-rich"}),
    ):
        if option in explicit and mode not in supported_modes:
            flag = "--" + option.replace("_", "-")
            raise ValueError(f"mode '{mode}' does not support {flag}")
    aliases = {
        "delta": "delta", "rule": "rule", "max_iterations": "max_iterations",
        "tolerance": "tolerance", "sequence_lambda": "sequence_continuity_strength",
        "contact_lambda": "contact_consistency_strength", "warm_start": "warm_start",
        "reference_mode": "reference_mode", "fast_update_rounds": "update_rounds",
    }
    args._requested_parameters = {
        target: getattr(args, source) if source in explicit else "DEFAULT"
        for source, target in aliases.items()
    }
    if contract is None and mode not in {"fast-global", "diagnostic-rich"}:
        if "fast_update_rounds" in explicit:
            raise ValueError(f"mode '{mode}' does not support --fast-update-rounds; use throughput")
        validate_weighting_parameters(args.delta, args.sequence_lambda)
        args._effective_parameters = {
            **_robust_kwargs(args),
            "weight_scale_parameter": {"huber": "delta", "gaussian": "sigma", "tukey": "cutoff"}[args.rule],
            "weight_scale": (
                args.delta if args.rule == "huber" or "delta" in explicit else 1.0
            ),
        }
        args._contract_options = {}
        return {}
    parameter_contract = contract or "throughput"
    options = {
        target: getattr(args, source) for source, target in aliases.items()
        if source in explicit
    }
    options = resolve_contract_parameters(parameter_contract, options)
    args._contract_options = options
    return options


def _weights_rows(result, *, core_fraction: float, residue_labels: tuple[str, ...]) -> list[dict[str, object]]:
    """Build mean soft-weight rows from a trajectory alignment result."""

    weights = mean_soft_weights_from_result(result)
    mean_residual = getattr(result, "mean_residual", None)
    if mean_residual is None:
        residuals = np.mean(np.asarray([frame.result.residual_distances for frame in result.frame_results], dtype=float), axis=0)
    else:
        residuals = np.asarray(mean_residual, dtype=float)
    core = select_core_by_target_fraction(weights, core_fraction)
    rows: list[dict[str, object]] = []
    for index, (label, weight, residual, is_core) in enumerate(
        zip(residue_labels, weights, residuals, core.core_mask, strict=True),
        start=1,
    ):
        rows.append(
            {
                "atom_index": index,
                "residue_label": label,
                "soft_stability_weight": float(weight),
                "mobility_score": float(np.clip(1.0 - weight, 0.0, 1.0)),
                "mean_residual": float(residual),
                "core_label": int(bool(is_core)),
            }
        )
    return rows


def _summary_from_result(
    result,
    *,
    command: str,
    input_path: Path,
    output_dir: Path,
    core_fraction: float,
    prepared: PreparedInput,
    flexible_percentile: float | None = None,
) -> dict[str, object]:
    """Build a compact trajectory-alignment summary."""

    weights = mean_soft_weights_from_result(result)
    n_frames = int(getattr(result, "n_frames", len(getattr(result, "frame_results", []))))
    readiness = classify_input(
        input_path=input_path if str(input_path) else None,
        topology_path=getattr(prepared, "topology_path", None),
        fit_selection=prepared.fit_selection,
        selection_size=prepared.fit_trajectory.n_atoms,
    )
    result_metadata = getattr(result, "metadata", {})
    execution_contract = result_metadata.get("execution_contract")
    result_mode = getattr(result, "mode", "default")
    mode = execution_contract if execution_contract in {"exact", "throughput"} else result_mode
    is_fixed_budget = mode == "throughput"
    is_experimental = mode in {
        "fast-global",
        "diagnostic-rich",
        "balanced-lite",
    }
    confidence = getattr(result, "confidence", None)
    contract_metadata = canonical_contract_metadata(
        execution_contract,
        result_mode=result_mode,
    )
    summary = {
        **schema_metadata(__version__),
        **contract_metadata,
        "command": command,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "fit_selection": prepared.fit_selection,
        "analysis_coordinate_selection": prepared.fit_selection,
        "n_frames": n_frames,
        "n_fit_atoms": prepared.fit_trajectory.n_atoms,
        "n_fit_residues": prepared.n_fit_residues,
        "n_analysis_coordinate_atoms": prepared.fit_trajectory.n_atoms,
        "default_export_selection": prepared.canonical_default_export_selection,
        "n_exportable_atoms": prepared.n_exportable_atoms,
        "source_is_ca_only": prepared.source_is_ca_only,
        "legacy_metadata": {
            "analysis_loaded_selection": prepared.apply_selection,
            "n_analysis_loaded_atoms": int(prepared.apply_coords.shape[1]),
            "former_field_names": {
                "apply_selection": "analysis_loaded_selection",
                "n_apply_atoms": "n_analysis_loaded_atoms",
                "n_atoms": "n_analysis_loaded_atoms",
            },
        },
        "reference_mode": getattr(result, "reference_mode", "first_frame"),
        "warm_start": getattr(result, "warm_start", False),
        "long_mode": bool(mode in {"long", "long-fast"}),
        "selected_weight_fit_frames": int(
            len(getattr(result, "selected_frame_indices", []))
            or getattr(result, "metadata", {}).get("representative_count", 0)
        ),
        "max_weight_fit_frames": getattr(result, "max_weight_fit_frames", None),
        "frame_selection": getattr(result, "selection_strategy", None),
        "chunk_size": getattr(result, "chunk_size", None),
        "end_to_end_seconds": result.total_runtime_seconds,
        "average_runtime_per_frame": result.average_runtime_per_frame,
        "mean_rmsd": result.mean_rmsd,
        "mean_weighted_rmsd": result.mean_weighted_rmsd,
        "weighted_RMSD": result.mean_weighted_rmsd,
        "unified_posthoc_RMSD": result.mean_rmsd,
        "converged_fraction": (
            None if is_fixed_budget else result.converged_fraction
        ),
        "converged": (
            None
            if is_experimental or is_fixed_budget
            else bool(result.converged_fraction == 1.0)
        ),
        "convergence_interpretation": (
            "not_applicable_fixed_budget"
            if is_fixed_budget
            else (
                confidence.convergence_status
                if confidence is not None
                else None
            )
        ),
        "input_readiness_status": readiness.status,
        "input_readiness_reason": readiness.reason,
        "input_readiness_recommended_action": readiness.recommended_action,
        "core_fraction": core_fraction,
        "core_definition": "top_fraction_by_continuous_stability_weight",
        "flexible_percentile": flexible_percentile,
        "effective_core_size": float(np.sum(weights)),
        "effective_core_fraction": float(np.mean(weights)),
        "warnings": prepared.warnings,
        "unit_provenance": prepared.unit_provenance,
        "output_contract": getattr(result, "metadata", {}).get(
            "output_contract", "trajectory_alignment"
        ),
        "external_tools_required": False,
        "backend_requested": result_metadata.get(
            "backend_requested", "cpu"
        ),
        "backend_used": result_metadata.get(
            "backend_used",
            result_metadata.get("backend", "cpu"),
        ),
        "quality_status": "completed" if is_fixed_budget else result_metadata.get("quality_status"),
        "quality_warning": None if is_fixed_budget else result_metadata.get("quality_warning"),
        "quality_audit_performed": False if is_fixed_budget else result_metadata.get("quality_audit_performed"),
        "fixed_budget_estimator": is_fixed_budget,
        "sampled_audit": result_metadata.get("sampled_audit"),
    }
    if is_fixed_budget:
        summary["fixed_update_rounds"] = int(
            getattr(result, "update_rounds", result.average_iterations_per_frame)
        )
    else:
        summary["average_iterations_per_frame"] = result.average_iterations_per_frame
        summary["n_iterations"] = result.average_iterations_per_frame
        summary["core_threshold"] = core_fraction
        summary["total_runtime_seconds"] = result.total_runtime_seconds
        summary["mdcorealign_version"] = __version__
    if is_experimental and confidence is not None:
        summary["fast_global_confidence"] = confidence.to_dict()
    return summary


def _long_mode_weight_frame_limit(args: argparse.Namespace, n_frames: int) -> int:
    """Return the effective representative-frame budget for long modes."""

    requested = int(args.max_weight_fit_frames)
    if args.mode == "long-fast" and requested == 2000:
        requested = 500
    return max(1, min(requested, n_frames))


def _residue_metadata_from_labels(
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse conservative chain/residue metadata without index-only identity."""

    chains: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[str] = []
    positions: list[int] = []
    chain_fallback: dict[str, int] = {}
    for label in labels:
        chain, _, residue = str(label).partition(":")
        chain = chain or "_"
        match = re.match(r"(?P<name>[A-Za-z]+)?(?P<id>-?\d+.*)?$", residue)
        residue_name = (
            match.group("name") if match and match.group("name") else "UNK"
        )
        residue_id = (
            match.group("id") if match and match.group("id") else residue
        )
        numeric = re.search(r"-?\d+", residue_id)
        if numeric is not None:
            position = int(numeric.group())
        else:
            chain_fallback[chain] = chain_fallback.get(chain, 0) + 1
            position = chain_fallback[chain]
        chains.append(chain)
        residue_names.append(residue_name)
        residue_ids.append(residue_id or str(position))
        positions.append(position)
    return (
        np.asarray(residue_ids, dtype=object),
        np.asarray(residue_names, dtype=object),
        np.asarray(chains, dtype=object),
        np.asarray(positions, dtype=np.int64),
    )


def _solve_for_mode(args: argparse.Namespace, prepared: PreparedInput):
    """Dispatch to the default or long-trajectory solver."""

    parameter_options = _prepare_parameter_contract(args)
    backend = getattr(args, "backend", "cpu")
    contract = getattr(args, "contract", None)
    if args.mode in {"exact", "throughput"}:
        contract = args.mode
    if contract is not None:
        if args.reference_mode != "first_frame":
            raise ValueError(
                f"contract '{contract}' currently requires "
                "--reference-mode first_frame"
            )
        if args.mode not in {"default", "exact", "throughput", "balanced-lite"}:
            raise ValueError(
                "--contract is an explicit alternative to --mode; leave "
                "--mode at its default value"
            )
        _, _, chain_ids, _ = _residue_metadata_from_labels(
            prepared.fit_labels
        )
        options: dict[str, object] = {
            **parameter_options,
            "representative_count": (
                args.exact_representative_count
                if contract == "exact"
                else args.fast_representative_count
            ),
            "chunk_size": args.chunk_size,
        }
        if contract in {"exact", "throughput"} and backend in {"cpu", "auto"} and (
            contract == "exact" or backend == "cpu"
        ):
            options["compute_ordinary_rmsf"] = args.ordinary_rmsf
        if contract != "exact":
            options.update(
                {
                    "update_rounds": args.fast_update_rounds,
                    **args._gpu_options,
                }
            )
        if contract == "balanced-lite":
            options.update(
                {
                    "audit_subset_count": args.audit_subset_count,
                    "audit_heldout_count": args.audit_heldout_count,
                    "allow_limited_refinement": (
                        args.allow_limited_refinement
                    ),
                    "refinement_representative_count": (
                        args.refinement_representative_count
                    ),
                }
            )
        if backend == "cpu":
            options.pop("precision", None)
            options.pop("rotation_solver", None)
        execution = solve_contract(
            prepared.fit_trajectory.coords,
            contract=contract,
            backend=backend,
            chain_ids=chain_ids,
            store_aligned=False,
            **options,
        )
        if execution.alignment is None:
            raise ValueError(
                f"contract '{contract}' does not produce aligned coordinates"
            )
        execution.alignment.metadata.update(
            {
                "requested_parameters": args._requested_parameters,
                "execution_contract": contract,
                "backend_requested": backend,
                "backend_used": execution.backend_used,
                "quality_status": execution.quality_status,
                "quality_warning": execution.quality_warning,
                "refinement_performed": execution.refinement_performed,
                "sampled_audit": (
                    None
                    if execution.sampled_audit is None
                    else execution.sampled_audit.to_dict()
                ),
                "contract_timing": execution.timing,
            }
        )
        args._effective_parameters = execution.alignment.metadata.get("effective_parameters", {})
        if contract == "throughput":
            execution.alignment.metadata.update(
                {
                    "quality_status": "completed",
                    "quality_warning": None,
                    "quality_audit_performed": False,
                    "fixed_budget_estimator": True,
                    "convergence_interpretation": (
                        "not_applicable_fixed_budget"
                    ),
                }
            )
        if not args.ordinary_rmsf:
            execution.alignment.ordinary_rmsf = None
            execution.alignment.metadata["ordinary_rmsf_computed"] = False
        return execution.alignment

    if args.mode in {"fast-global", "diagnostic-rich"}:
        if args.reference_mode != "first_frame":
            raise ValueError(
                f"experimental mode '{args.mode}' currently requires "
                "--reference-mode first_frame"
            )
        _, _, chain_ids, _ = _residue_metadata_from_labels(
            prepared.fit_labels
        )
        if args.mode == "fast-global" and backend in {"gpu", "auto"}:
            execution = solve_contract(
                prepared.fit_trajectory.coords,
                contract="throughput",
                backend=backend,
                chain_ids=chain_ids,
                store_aligned=True,
                representative_count=args.fast_representative_count,
                update_rounds=args.fast_update_rounds,
                chunk_size=args.chunk_size,
                **args._gpu_options,
                delta=parameter_options["delta"],
                sequence_continuity_strength=parameter_options["sequence_continuity_strength"],
            )
            execution.alignment.metadata.update(
                {
                    "execution_contract": "throughput",
                    "backend_requested": backend,
                    "backend_used": execution.backend_used,
                    "quality_status": execution.quality_status,
                    "quality_warning": execution.quality_warning,
                }
            )
            if not args.ordinary_rmsf:
                execution.alignment.ordinary_rmsf = None
                execution.alignment.metadata["ordinary_rmsf_computed"] = False
            args._effective_parameters = execution.alignment.metadata.get("effective_parameters", {})
            execution.alignment.metadata["requested_parameters"] = args._requested_parameters
            return execution.alignment
        if args.mode == "diagnostic-rich" and backend != "cpu":
            raise ValueError(
                "GPU diagnostic-rich mode is not combined in one command yet. "
                "Run GPU throughput and GPU local-deformation separately."
            )
        if args.mode == "fast-global":
            result = solve_fast_global_alignment(
                prepared.fit_trajectory.coords,
                chain_ids=chain_ids,
                representative_count=args.fast_representative_count,
                update_rounds=args.fast_update_rounds,
                chunk_size=args.chunk_size,
                store_aligned=True,
                compute_confidence=True,
                compute_ordinary_rmsf=args.ordinary_rmsf,
                delta=parameter_options["delta"],
                sequence_continuity_strength=parameter_options["sequence_continuity_strength"],
            )
            args._effective_parameters = result.metadata.get("effective_parameters", {})
            result.metadata["requested_parameters"] = args._requested_parameters
            return result
        result = solve_diagnostic_rich_alignment(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            representative_count=args.fast_representative_count,
            update_rounds=args.fast_update_rounds,
            chunk_size=args.chunk_size,
            descriptor_chunk_size=args.descriptor_chunk_size,
            block_count=args.local_block_count,
            experimental_motion_graph=args.experimental_motion_graph,
            compute_ordinary_rmsf=args.ordinary_rmsf,
            delta=parameter_options["delta"],
            sequence_continuity_strength=parameter_options["sequence_continuity_strength"],
        )
        args._effective_parameters = result.metadata.get("effective_parameters", {})
        result.metadata["requested_parameters"] = args._requested_parameters
        return result
    if backend == "gpu":
        raise ValueError(
            "GPU execution is currently available only for experimental "
            "throughput/balanced-lite contracts and local-deformation. The "
            "public default and long modes remain CPU paths."
        )
    if args.mode in {"long", "long-fast"}:
        return solve_long_trajectory_alignment(
            prepared.fit_trajectory,
            reference_mode=args.reference_mode,
            mode=args.mode,
            max_weight_fit_frames=_long_mode_weight_frame_limit(args, prepared.fit_trajectory.n_frames),
            frame_selection=args.frame_selection,
            random_seed=args.random_seed,
            chunk_size=args.chunk_size,
            store_aligned=False,
            compute_ordinary_rmsf=args.ordinary_rmsf,
            **_robust_kwargs(args),
        )
    return solve_trajectory_robust_alignment(
        prepared.fit_trajectory,
        reference_mode=args.reference_mode,
        warm_start=args.warm_start,
        **_robust_kwargs(args),
    )


def _result_transforms(result) -> tuple[np.ndarray, np.ndarray]:
    """Return one explicit rigid transform per frame from any alignment mode."""

    rotations = getattr(result, "rotations", None)
    translations = getattr(result, "translations", None)
    if rotations is not None and translations is not None:
        return np.asarray(rotations, dtype=float), np.asarray(translations, dtype=float)
    frames = getattr(result, "frame_results", None)
    if not frames:
        raise ValueError("alignment result does not contain per-frame transforms")
    return (
        np.asarray([frame.result.rotation for frame in frames], dtype=float),
        np.asarray([frame.result.translation for frame in frames], dtype=float),
    )


def _result_frame_metrics(result) -> tuple[np.ndarray, np.ndarray]:
    """Return full-set and weighted RMSD arrays from any alignment mode."""

    rmsd = getattr(result, "rmsd", None)
    weighted = getattr(result, "weighted_rmsd", None)
    if rmsd is not None and weighted is not None:
        return np.asarray(rmsd, dtype=float), np.asarray(weighted, dtype=float)
    frames = getattr(result, "frame_results", None)
    if not frames:
        raise ValueError("alignment result does not contain per-frame RMSD")
    return (
        np.asarray([frame.result.rmsd for frame in frames], dtype=float),
        np.asarray([frame.result.weighted_rmsd for frame in frames], dtype=float),
    )


def _write_stable_alignment_contract(
    output_dir: Path,
    result,
    *,
    diagnostics_dir: Path,
    prepared: PreparedInput,
    args: argparse.Namespace,
    materialized: bool,
) -> dict[str, object]:
    """Write stable scientific outputs shared by exact and throughput modes."""

    rotations, translations = _result_transforms(result)
    rmsd, weighted_rmsd = _result_frame_metrics(result)
    weights = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
    core = select_core_by_target_fraction(weights, args.core_fraction)
    residue_ids, residue_names, chains, _ = _residue_metadata_from_labels(
        prepared.fit_labels
    )
    robust_rmsf = np.asarray(result.robust_rmsf, dtype=float)
    ordinary_value = getattr(result, "ordinary_rmsf", None)
    ordinary_rmsf = (
        None if ordinary_value is None else np.asarray(ordinary_value, dtype=float)
    )
    mean_residual = np.asarray(result.mean_residual, dtype=float)
    mobility = np.clip(1.0 - weights, 0.0, 1.0)
    mobility_ranks = descending_ranks(mobility)
    mobility_percentiles = percentile_scores(mobility)
    segment_ids = detect_flexible_segments(
        mobility,
        percentile_cutoff=0.8,
        min_segment_length=3,
        merge_gap=1,
        chain=chains,
    )

    transforms_path = output_dir / "transforms.npz"
    np.savez_compressed(
        transforms_path,
        rotations=rotations,
        translations=translations,
        n_frames=np.asarray(rotations.shape[0], dtype=np.int64),
        n_fit_atoms=np.asarray(weights.size, dtype=np.int64),
        fit_labels=np.asarray(prepared.fit_labels, dtype=str),
        fit_selection=np.asarray(prepared.fit_selection),
        source_unit=np.asarray(prepared.unit_provenance.get("source_unit", "unknown")),
    )
    frame_metrics_path = output_dir / "frame_metrics.tsv"
    _write_table_with_fields(
        frame_metrics_path,
        [
            {
                "frame_index": index,
                "full_set_RMSD": float(full),
                "weighted_RMSD": float(weighted_value),
            }
            for index, (full, weighted_value) in enumerate(
                zip(rmsd, weighted_rmsd, strict=True)
            )
        ],
        list(FRAME_METRIC_COLUMNS),
    )
    residue_profile_path = output_dir / "residue_profile.tsv"
    residue_rows: list[dict[str, object]] = []
    for index in range(weights.size):
        row: dict[str, object] = {
            "residue_index": index + 1,
            "chain": str(chains[index]),
            "residue_id": str(residue_ids[index]),
            "resname": str(residue_names[index]),
            "stable_weight": float(weights[index]),
            "mobility_score": float(mobility[index]),
            "core_label": int(core.core_mask[index]),
            "robust_RMSF": float(robust_rmsf[index]),
            "mean_residual": float(mean_residual[index]),
            "flexible_segment_id": int(segment_ids[index]),
            "mobility_rank": int(mobility_ranks[index]),
            "mobility_percentile": float(mobility_percentiles[index]),
        }
        if ordinary_rmsf is not None:
            row["ordinary_RMSF"] = float(ordinary_rmsf[index])
        residue_rows.append(row)
    _write_tsv_fields = list(RESIDUE_PROFILE_COLUMNS)
    if ordinary_rmsf is not None:
        _write_tsv_fields.append("ordinary_RMSF")
    _write_table_with_fields(
        residue_profile_path, residue_rows, _write_tsv_fields
    )
    flexible_segments_path = output_dir / "flexible_segments.tsv"
    segment_rows: list[dict[str, object]] = []
    for segment_id in sorted(set(segment_ids) - {0}):
        indices = np.flatnonzero(segment_ids == segment_id)
        segment_rows.append(
            {
                "segment_id": int(segment_id),
                "chain": str(chains[indices[0]]),
                "start_residue_id": str(residue_ids[indices[0]]),
                "end_residue_id": str(residue_ids[indices[-1]]),
                "length": int(indices.size),
                "mean_mobility_score": float(np.mean(mobility[indices])),
                "mean_stable_weight": float(np.mean(weights[indices])),
                "mean_robust_RMSF": float(np.mean(robust_rmsf[indices])),
            }
        )
    _write_table_with_fields(
        flexible_segments_path,
        segment_rows,
        [
            "segment_id",
            "chain",
            "start_residue_id",
            "end_residue_id",
            "length",
            "mean_mobility_score",
            "mean_stable_weight",
            "mean_robust_RMSF",
        ],
    )
    diagnostics_path = diagnostics_dir / "diagnostics.json"
    _write_json(
        diagnostics_path,
        {
            **canonical_contract_metadata(
                getattr(result, "metadata", {}).get("execution_contract"),
                result_mode=getattr(result, "mode", args.mode),
            ),
            "backend_requested": args.backend,
            "backend_used": getattr(result, "metadata", {}).get("backend_used", "cpu"),
            "timing": canonical_timing(getattr(result, "timing", {})),
            "legacy_timing_schema": "internal-v5",
            "legacy_timing": getattr(result, "timing", {}),
            "quality_status": getattr(result, "metadata", {}).get("quality_status"),
            "quality_warning": getattr(result, "metadata", {}).get("quality_warning"),
            "weights_frozen_before_all_frame_fit": True,
            "all_frames_explicitly_fitted": True,
            "aligned_coordinates_materialized": materialized,
            "ordinary_rmsf_computed": ordinary_rmsf is not None,
            "fixed_budget_estimator": bool(
                getattr(result, "metadata", {}).get(
                    "fixed_budget_estimator", False
                )
            ),
            "quality_audit_performed": getattr(result, "metadata", {}).get(
                "quality_audit_performed"
            ),
        },
    )
    return {
        "transforms_npz": str(transforms_path),
        "frame_metrics_tsv": str(frame_metrics_path),
        "residue_profile_tsv": str(residue_profile_path),
        "flexible_segments_tsv": str(flexible_segments_path),
        "diagnostics_json": str(diagnostics_path),
    }


def _write_selected_frame_indices(path: Path, result) -> None:
    """Write long-mode selected frame indices, or an empty table for default mode."""

    indices = getattr(result, "selected_frame_indices", None)
    rows: list[dict[str, object]] = []
    if indices is not None:
        rows = [
            {
                "selection_order": order,
                "frame_index": int(frame_index),
                "selection_strategy": getattr(result, "selection_strategy", ""),
            }
            for order, frame_index in enumerate(np.asarray(indices, dtype=int), start=1)
        ]
    _write_table_with_fields(path, rows, ["selection_order", "frame_index", "selection_strategy"])


def _write_long_mode_artifacts(output_dir: Path, result, *, args: argparse.Namespace) -> dict[str, object]:
    """Write long-mode diagnostics and timing files when applicable."""

    if getattr(result, "mode", "default") not in {"long", "long-fast"}:
        return {}
    selected_path = output_dir / "selected_frame_indices.tsv"
    long_summary_path = output_dir / "long_mode_summary.json"
    timing_path = output_dir / "timing.json"
    _write_selected_frame_indices(selected_path, result)
    _write_json(
        long_summary_path,
        {
            "mode": result.mode,
            "approximate_long_mode": True,
            "n_frames": result.n_frames,
            "n_atoms": result.n_atoms,
            "selected_weight_fit_frames": int(len(result.selected_frame_indices)),
            "max_weight_fit_frames": int(result.max_weight_fit_frames),
            "frame_selection": result.selection_strategy,
            "random_seed": int(args.random_seed),
            "chunk_size": int(result.chunk_size),
            "chunks_processed": int(result.chunks_processed),
            "weights_estimated_from_selected_frames": True,
            "frozen_weights_applied_to_all_frames": True,
            "RMSF_uses_all_frames": True,
            "interpretation_note": (
                "Long mode estimates continuous stability weights from representative frames and applies frozen "
                "weights to the full trajectory. It should be validated against default mode for short or "
                "highly heterogeneous cases."
            ),
        },
    )
    _write_json(timing_path, getattr(result, "timing", {}))
    return {
        "selected_frame_indices_tsv": str(selected_path),
        "long_mode_summary_json": str(long_summary_path),
        "timing_json": str(timing_path),
    }


def _write_experimental_mode_artifacts(
    output_dir: Path,
    result,
    *,
    prepared: PreparedInput,
) -> dict[str, object]:
    """Write explicit fast-global and diagnostic-rich contract artefacts."""

    if getattr(result, "metadata", {}).get("execution_contract") == "throughput":
        return {}
    if getattr(result, "mode", "") not in {
        "fast-global",
        "diagnostic-rich",
        "balanced-lite",
    }:
        return {}
    selected_path = output_dir / "selected_frame_indices.tsv"
    confidence_path = output_dir / "fast_global_confidence.json"
    timing_path = output_dir / "timing.json"
    _write_selected_frame_indices(selected_path, result)
    confidence = getattr(result, "confidence", None)
    confidence_payload = (
        confidence.to_dict()
        if confidence is not None
        else {
            "quality_status": getattr(result, "metadata", {}).get(
                "quality_status"
            ),
            "quality_warning": getattr(result, "metadata", {}).get(
                "quality_warning"
            ),
            "risk_signals": getattr(result, "risk_signals", {}),
            "sampled_audit": getattr(result, "metadata", {}).get(
                "sampled_audit"
            ),
        }
    )
    _write_json(confidence_path, confidence_payload)
    _write_json(timing_path, result.timing)
    outputs: dict[str, object] = {
        "selected_frame_indices_tsv": str(selected_path),
        "fast_global_confidence_json": str(confidence_path),
        "timing_json": str(timing_path),
    }
    if getattr(result, "local_deformation", None) is not None:
        residue_ids, residue_names, chain_ids, positions = (
            _residue_metadata_from_labels(prepared.fit_labels)
        )
        local_path = output_dir / "local_deformation.tsv"
        _write_table_with_fields(
            local_path,
            local_deformation_rows(
                result.local_deformation,
                residue_ids=residue_ids,
                residue_names=residue_names,
                chain_ids=chain_ids,
                residue_positions=positions,
                source_provenance=prepared.unit_provenance,
            ),
            [
                "atom_index",
                "residue_id",
                "resname",
                "chain",
                "residue_position",
                "local_deformation_index",
                "local_rigidity_score",
                "percentile",
                "rank",
                "locally_deforming_segment_id",
                "is_terminal",
                "chain_break_before",
                "temporal_block_reproducibility",
                "confidence",
                "uncertainty",
                "source_unit",
                "conversion_factor_to_angstrom",
                "canonical_internal_unit",
            ],
        )
        outputs["local_deformation_tsv"] = str(local_path)
    if getattr(result, "motion_graph", None) is not None:
        edge_rows, residue_rows = motion_graph_rows(result.motion_graph)
        edges_path = output_dir / "experimental_motion_graph_edges.tsv"
        residues_path = (
            output_dir / "experimental_motion_graph_residue_diagnostics.tsv"
        )
        _write_table_with_fields(edges_path, edge_rows, list(edge_rows[0]))
        _write_table_with_fields(
            residues_path, residue_rows, list(residue_rows[0])
        )
        outputs.update(
            {
                "experimental_motion_graph_edges_tsv": str(edges_path),
                "experimental_motion_graph_residues_tsv": str(residues_path),
            }
        )
    return outputs


def _write_canonical_diagnostics(
    diagnostics_dir: Path,
    result,
) -> dict[str, object]:
    """Write diagnostics shared by canonical exact and throughput workflows."""

    outputs: dict[str, object] = {}
    timing_path = diagnostics_dir / "timing.json"
    raw_timing = getattr(result, "timing", {})
    _write_json(
        timing_path,
        {
            **canonical_contract_metadata(
                getattr(result, "metadata", {}).get("execution_contract"),
                result_mode=getattr(result, "mode", "exact"),
            ),
            "timing": canonical_timing(raw_timing),
            "legacy_timing_schema": "internal-v5",
            "legacy_timing": raw_timing,
        },
    )
    outputs["timing_json"] = str(timing_path)
    indices = getattr(result, "selected_frame_indices", None)
    if indices is not None and np.asarray(indices).size:
        selected_path = diagnostics_dir / "selected_frame_indices.tsv"
        _write_selected_frame_indices(selected_path, result)
        outputs["selected_frame_indices_tsv"] = str(selected_path)
    return outputs


def _relative_output_map(
    output_dir: Path,
    outputs: dict[str, object],
) -> dict[str, object]:
    """Return output paths relative to the result root where possible."""

    relative: dict[str, object] = {}
    for name, value in outputs.items():
        if value is None:
            relative[name] = None
            continue
        path = Path(str(value))
        try:
            relative[name] = str(path.relative_to(output_dir))
        except ValueError:
            relative[name] = str(path)
    return relative


def _write_optional_local_deformation(
    diagnostics_dir: Path,
    prepared: PreparedInput,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Write optional W5 diagnostics without changing the alignment result."""

    if not getattr(args, "local_deformation", False):
        return {}
    residue_ids, residue_names, chain_ids, positions = (
        _residue_metadata_from_labels(prepared.fit_labels)
    )
    if args.backend == "gpu":
        from .gpu_backend import scan_w5_gpu

        local = scan_w5_gpu(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            residue_positions=positions,
            precision=args.gpu_precision,
            frame_chunk_size=args.descriptor_chunk_size,
        )
    else:
        local = scan_local_kinematics(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            residue_positions=positions,
            block_count=args.local_block_count,
            evaluate_blocks=True,
            deforming_percentile=0.8,
            minimum_segment_length=3,
            merge_gap=1,
            descriptor_chunk_size=args.descriptor_chunk_size,
        )
    rows = local_deformation_rows(
        local,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_positions=positions,
        source_provenance=prepared.unit_provenance,
    )
    local_path = diagnostics_dir / "local_deformation.tsv"
    _write_table_with_fields(local_path, rows, list(rows[0]))
    segment_rows: list[dict[str, object]] = []
    for segment_id in sorted(
        set(local.locally_deforming_segments).difference({0})
    ):
        indices = np.flatnonzero(
            local.locally_deforming_segments == segment_id
        )
        if not indices.size:
            continue
        segment_rows.append(
            {
                "segment_id": int(segment_id),
                "chain": str(chain_ids[indices[0]]),
                "start_residue_id": str(residue_ids[indices[0]]),
                "end_residue_id": str(residue_ids[indices[-1]]),
                "length": int(indices.size),
                "mean_local_deformation_index": float(
                    np.mean(local.local_deformation_index[indices])
                ),
                "mean_local_rigidity_score": float(
                    np.mean(local.local_rigidity_score[indices])
                ),
                "mean_confidence": float(np.mean(local.confidence[indices])),
            }
        )
    segments_path = diagnostics_dir / "locally_deforming_segments.tsv"
    _write_table_with_fields(
        segments_path,
        segment_rows,
        [
            "segment_id",
            "chain",
            "start_residue_id",
            "end_residue_id",
            "length",
            "mean_local_deformation_index",
            "mean_local_rigidity_score",
            "mean_confidence",
        ],
    )
    return {
        "local_deformation_tsv": str(local_path),
        "locally_deforming_segments_tsv": str(segments_path),
    }


def _run_alignment(args: argparse.Namespace) -> int:
    """Run robust trajectory alignment."""

    _prepare_parameter_contract(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path_from_args(args)
    prepared = _prepare_input(args)
    source_manifest_path = diagnostics_dir / "source_manifest.json"
    source_manifest = build_source_manifest(
        args,
        prepared,
        software_version=__version__,
    )
    write_source_manifest(source_manifest_path, source_manifest)
    audit_outputs = _write_input_audit_outputs(
        diagnostics_dir,
        args,
        prepared,
        include_json=False,
    )
    compatibility_outputs: dict[str, object] = {}
    if args.legacy_artifacts:
        compatibility_outputs.update(
            _write_input_audit_outputs(
                output_dir,
                args,
                prepared,
                include_json=True,
            )
        )
    if args.dry_run:
        audit_row = _audit_row(args, prepared)
        summary_path = output_dir / "summary.json"
        metadata_path = diagnostics_dir / "run_metadata.json"
        primary_outputs = {"summary_json": str(summary_path)}
        diagnostic_outputs = {
            **audit_outputs,
            "source_manifest_json": str(source_manifest_path),
            "run_metadata_json": str(metadata_path),
        }
        dry_summary = {
            **schema_metadata(__version__),
            "method": "dry-run input audit",
            "command": "align --dry-run",
            "fit_selection": prepared.fit_selection,
            "analysis_coordinate_selection": prepared.fit_selection,
            "n_frames": prepared.fit_trajectory.n_frames,
            "n_fit_atoms": prepared.fit_trajectory.n_atoms,
            "n_fit_residues": prepared.n_fit_residues,
            "n_analysis_coordinate_atoms": prepared.fit_trajectory.n_atoms,
            "default_export_selection": (
                prepared.canonical_default_export_selection
            ),
            "n_exportable_atoms": prepared.n_exportable_atoms,
            "source_is_ca_only": prepared.source_is_ca_only,
            "legacy_metadata": {
                "analysis_loaded_selection": prepared.apply_selection,
                "n_analysis_loaded_atoms": int(prepared.apply_coords.shape[1]),
            },
            "warnings": prepared.warnings,
            "unit_provenance": prepared.unit_provenance,
            "input_readiness_status": audit_row["input_readiness_status"],
            "input_readiness_reason": audit_row["input_readiness_reason"],
            "input_readiness_recommended_action": audit_row["input_readiness_recommended_action"],
            "alignment_run": False,
            "primary_outputs": _relative_output_map(
                output_dir, primary_outputs
            ),
            "diagnostic_outputs": _relative_output_map(
                output_dir, diagnostic_outputs
            ),
            "compatibility_outputs": _relative_output_map(
                output_dir, compatibility_outputs
            ),
        }
        dry_summary["outputs"] = {
            **dry_summary["primary_outputs"],
            **dry_summary["diagnostic_outputs"],
            **dry_summary["compatibility_outputs"],
        }
        _write_json(summary_path, dry_summary)
        _write_run_metadata(
            metadata_path,
            args=args,
            input_path=input_path,
            output_files=dry_summary["outputs"],
            warnings=prepared.warnings,
            prepared=prepared,
        )
        print(f"Dry run completed. Wrote input audit: {audit_outputs['input_audit_tsv']}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote run metadata: {metadata_path}")
        return 0
    result = _solve_for_mode(args, prepared)
    aligned_path = Path(args.out) if getattr(args, "out", None) else output_dir / "aligned_trajectory.pdb"
    materialize = args.materialize
    if materialize is None:
        materialize = args.mode not in {"exact", "throughput"}
    if materialize and aligned_path.suffix.lower() not in {"", ".pdb", ".ent"}:
        raise ValueError(
            "The legacy in-command aligned trajectory writer is PDB. "
            "Use --out aligned.pdb, or convert aligned_trajectory.pdb with a dedicated MD tool."
        )
    summary_path = output_dir / "summary.json"
    metadata_path = diagnostics_dir / "run_metadata.json"
    if materialize:
        aligned = _aligned_apply_coords(result, prepared)
        _write_aligned_output(aligned, aligned_path, records=prepared.apply_records)
    if args.legacy_artifacts:
        weights_path = output_dir / "weights.csv"
        _write_csv(
            weights_path,
            _weights_rows(
                result,
                core_fraction=args.core_fraction,
                residue_labels=prepared.fit_labels,
            ),
        )
        compatibility_outputs["weights_csv"] = str(weights_path)
    contract_outputs: dict[str, object] = {}
    canonical_contract = args.mode in {"exact", "throughput"} or args.contract in {
        "exact",
        "throughput",
        "balanced-lite",
    }
    if canonical_contract:
        contract_outputs = _write_stable_alignment_contract(
            output_dir,
            result,
            diagnostics_dir=diagnostics_dir,
            prepared=prepared,
            args=args,
            materialized=bool(materialize),
        )
    long_outputs = _write_long_mode_artifacts(output_dir, result, args=args)
    experimental_outputs = _write_experimental_mode_artifacts(
        output_dir, result, prepared=prepared
    )
    canonical_diagnostic_outputs = (
        _write_canonical_diagnostics(diagnostics_dir, result)
        if canonical_contract
        else {}
    )
    local_outputs = _write_optional_local_deformation(
        diagnostics_dir,
        prepared,
        args,
    )
    summary = _summary_from_result(
        result,
        command="align",
        input_path=input_path,
        output_dir=output_dir,
        core_fraction=args.core_fraction,
        prepared=prepared,
    )
    primary_outputs: dict[str, object] = {
        "summary_json": str(summary_path),
        **{
            name: value
            for name, value in contract_outputs.items()
            if not str(value).startswith(str(diagnostics_dir))
        },
    }
    if materialize:
        primary_outputs["aligned_output"] = str(aligned_path)
    diagnostic_outputs: dict[str, object] = {
        **audit_outputs,
        "source_manifest_json": str(source_manifest_path),
        "run_metadata_json": str(metadata_path),
        **{
            name: value
            for name, value in contract_outputs.items()
            if str(value).startswith(str(diagnostics_dir))
        },
        **canonical_diagnostic_outputs,
        **local_outputs,
        **long_outputs,
        **experimental_outputs,
    }
    summary["primary_outputs"] = _relative_output_map(
        output_dir, primary_outputs
    )
    summary["diagnostic_outputs"] = _relative_output_map(
        output_dir, diagnostic_outputs
    )
    summary["compatibility_outputs"] = _relative_output_map(
        output_dir, compatibility_outputs
    )
    summary["outputs"] = {
        **summary["primary_outputs"],
        **summary["diagnostic_outputs"],
        **summary["compatibility_outputs"],
    }
    risk_signals = getattr(result, "risk_signals", None)
    summary["diagnostics"] = {
        "risk_signals_available": risk_signals is not None,
        "risk_signals": risk_signals,
    }
    summary["aligned_coordinates_materialized"] = bool(materialize)
    summary["n_default_export_atoms"] = prepared.n_exportable_atoms
    summary["n_materialized_atoms"] = int(prepared.apply_coords.shape[1]) if materialize else 0
    summary["materialized_selection"] = prepared.apply_selection if materialize else None
    summary["n_exportable_atoms_scope"] = (
        "materialized_selection" if materialize else "default_export_selection"
    )
    if materialize:
        # Legacy in-command exports can include ligands or other non-protein
        # atoms. Their summary count describes that actual written selection;
        # source_manifest retains the separate future default export count.
        summary["n_exportable_atoms"] = int(prepared.apply_coords.shape[1])
    summary["weights_frozen_before_all_frame_fit"] = bool(
        args.mode in {"exact", "throughput"} or args.contract is not None
    )
    summary["all_frames_explicitly_fitted"] = True
    _write_json(summary_path, summary)
    _write_run_metadata(
        metadata_path,
        args=args,
        input_path=input_path,
        output_files=summary["outputs"],
        warnings=prepared.warnings,
        prepared=prepared,
    )
    if materialize:
        print(f"Wrote aligned trajectory: {aligned_path}")
    else:
        print("Aligned coordinates were not materialized; use `mdcorealign apply` with transforms.npz.")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote run metadata: {metadata_path}")
    return 0


def _suffix_export_contract(
    output_value: str | None,
    purpose: str | None,
    output_format: str | None,
) -> tuple[str, str, bool]:
    """Infer the current export contract from a suffix and reject contradictions."""

    if output_value is None:
        return resolve_export_contract(purpose, output_format)
    suffix = Path(output_value).suffix.lower()
    suffix_contracts = {
        ".pdb": ("structure", "pdb"),
        ".ent": ("structure", "pdb"),
        ".cif": ("structure", "mmcif"),
        ".mmcif": ("structure", "mmcif"),
        ".xtc": ("trajectory", "xtc"),
    }
    if suffix not in suffix_contracts:
        raise ValueError(
            "output suffix must be one of: .pdb, .cif, .mmcif, .xtc"
        )
    inferred_purpose, inferred_format = suffix_contracts[suffix]
    if output_format is not None and output_format != inferred_format:
        raise ValueError(
            f"--format {output_format} conflicts with output suffix '{suffix}'"
        )
    if purpose is not None:
        if purpose == "compatibility" and inferred_format == "pdb":
            return "compatibility", "pdb", True
        if purpose != inferred_purpose:
            raise ValueError(
                f"--purpose {purpose} conflicts with output suffix '{suffix}'"
            )
    return inferred_purpose, inferred_format, False


def _apply_result_directory(args: argparse.Namespace) -> Path | None:
    """Resolve the positional and legacy result-directory arguments."""

    positional = getattr(args, "result", None)
    legacy = getattr(args, "result_dir", None)
    if positional and legacy and Path(positional).resolve() != Path(legacy).resolve():
        raise ValueError("RESULT_DIR and --result-dir identify different results")
    value = positional or legacy
    return None if value is None else Path(value)


def _restore_read_configuration(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    """Replay the original decoder, rejecting conflicting read-time overrides."""

    configuration = manifest.get("read_configuration")
    if not isinstance(configuration, dict):
        if manifest["schema_version"] != "2.1.0":
            raise ValueError("source manifest lacks read_configuration; rerun align")
        if manifest.get("input_format") == "array":
            raise ValueError(
                "legacy array manifest does not establish the original array key and units; rerun align"
            )
        # Legacy PDB and standard MDAnalysis coordinates have a defined Angstrom
        # convention. Source-file hashes and the saved fit labels still apply.
        configuration = {
            "fit_selection": manifest["fit_selection"],
            "input_format": manifest.get("input_format", "auto"),
            "topology_format": manifest.get("topology_format"),
            "trajectory_format": manifest.get("trajectory_format"),
            "source_unit": "auto",
            "array_key": None,
            "chain": None,
        }
    else:
        required = {
            "reader", "input_format", "topology_format", "trajectory_format",
            "array_key", "chain", "fit_selection", "source_unit", "unit_provenance",
            "source_frame_indices", "protein_selection_policy",
        }
        if not required.issubset(configuration):
            raise ValueError("source manifest read_configuration is incomplete; rerun align")
        if configuration["reader"] not in {"builtin", "MDAnalysis"}:
            raise ValueError("source manifest reader is unsupported; rerun align")
        if configuration["protein_selection_policy"] != PROTEIN_SELECTION_POLICY:
            raise ValueError("saved protein selection policy differs from this reader; rerun align")
        if configuration["source_frame_indices"] != list(range(int(manifest["n_frames"]))):
            raise ValueError("saved source frame ordering cannot be safely replayed; rerun align")
    explicit = set(getattr(args, "_explicit_options", ()))
    for name in ("fit_selection", "input_format", "topology_format", "trajectory_format", "array_key", "source_unit", "chain"):
        saved = configuration[name]
        requested = getattr(args, name, None)
        if name in explicit and requested != saved:
            # 'auto' explicitly requests the registered interpretation. A
            # format name is case insensitive, but chain/key/selection are not.
            same_format = name.endswith("format") and str(requested).upper() == str(saved).upper()
            if not same_format and not (name in {"input_format", "source_unit"} and requested == "auto"):
                raise ValueError(f"--{name.replace('_', '-')} conflicts with the saved read configuration")
        setattr(args, name, saved)


def _is_ordered_atom_subset(subset: tuple[str, ...], complete: tuple[str, ...]) -> bool:
    """Require a one-to-one, source-ordered subset, including repeated identities."""

    remaining = iter(complete)
    return all(any(candidate == identity for candidate in remaining) for identity in subset)


def _run_apply(args: argparse.Namespace) -> int:
    """Apply saved transforms and write annotated coordinate outputs."""

    if args.input and (args.topology or args.trajectory):
        raise ValueError("use --input or --topology/--trajectory, not both input conventions")
    if args.out is not None and args.output is not None:
        raise ValueError("use only one of -o/--output or the legacy --out option")
    output_value = args.output or args.out
    result_dir = _apply_result_directory(args)
    if getattr(args, "result", None) and output_value is None:
        raise ValueError(
            "the simple apply workflow requires -o OUTPUT with .pdb, .cif, .mmcif, or .xtc"
        )
    purpose, output_format, legacy_contract = _suffix_export_contract(
        output_value, args.purpose, args.output_format
    )

    manifest: dict[str, object] | None = None
    source_resolution: dict[str, str] = {}
    if result_dir is not None:
        source_manifest_path = result_dir / "diagnostics" / "source_manifest.json"
        if source_manifest_path.is_file():
            manifest = load_source_manifest(result_dir)
            resolved = resolve_source_files(
                manifest,
                result_dir,
                input_override=args.input,
                topology_override=args.topology,
                trajectory_override=args.trajectory,
            )
            args.input = None if resolved.single_input is None else str(resolved.single_input)
            args.topology = None if resolved.topology is None else str(resolved.topology)
            args.trajectory = None if resolved.trajectory is None else str(resolved.trajectory)
            if manifest.get("input_kind") == "topology_as_trajectory":
                args.trajectory = None
            source_resolution = resolved.resolution
            _restore_read_configuration(args, manifest)
            args.apply_selection = (
                args.apply_selection
                or str(
                    manifest.get(
                        "default_export_selection_expression",
                        manifest["default_export_selection"],
                    )
                )
            )
        elif getattr(args, "result", None):
            load_source_manifest(result_dir)

    if args.fit_selection is None:
        args.fit_selection = DEFAULT_FIT_SELECTION
    input_path = _input_path_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.transforms is not None and result_dir is not None:
        result_transform = result_dir / "transforms.npz"
        if Path(args.transforms).resolve() != result_transform.resolve():
            raise ValueError(
                "--transforms and RESULT_DIR identify different transform archives"
            )
    transforms_value = args.transforms
    if transforms_value is None and result_dir is not None:
        transforms_value = str(result_dir / "transforms.npz")
    if transforms_value is None:
        raise ValueError(
            "saved transforms are required; use RESULT_DIR or --transforms FILE"
        )
    transforms_path = Path(transforms_value)
    if not transforms_path.is_file():
        raise FileNotFoundError(f"transform file does not exist: {transforms_path}")
    with np.load(transforms_path, allow_pickle=False) as archive:
        if not {"rotations", "translations"}.issubset(archive.files):
            raise ValueError("transform archive must contain rotations and translations")
        rotations = np.asarray(archive["rotations"], dtype=float)
        translations = np.asarray(archive["translations"], dtype=float)
        expected_fit_atoms = (
            int(archive["n_fit_atoms"]) if "n_fit_atoms" in archive.files else None
        )
        expected_labels = (
            tuple(str(value) for value in archive["fit_labels"])
            if "fit_labels" in archive.files
            else None
        )

    prepared: PreparedInput | None = None
    resolved_format = _infer_input_format(
        input_path,
        args.input_format,
        topology_format=getattr(args, "topology_format", None),
        trajectory_format=(getattr(args, "trajectory_format", None) or (
            getattr(args, "topology_format", None)
            if getattr(args, "topology", None) and not getattr(args, "trajectory", None)
            else None
        )),
    )
    use_streaming_reader = bool(
        args.topology
        and (args.trajectory or resolved_format not in {"pdb", "ca-pdb", "array"})
    )
    if use_streaming_reader:
        if args.source_unit == "nanometer":
            raise ValueError("MDAnalysis coordinates are already normalized to Angstrom; do not request a second nanometer conversion")
        if args.array_key is not None:
            raise ValueError("--array-key is only valid for a NumPy .npz archive")
        source = MDAnalysisCoordinateSource(
            args.topology,
            args.trajectory or args.topology,
            fit_selection=args.fit_selection,
            apply_selection=args.apply_selection or args.fit_selection,
            topology_format=args.topology_format,
            trajectory_format=args.trajectory_format,
            chain_id=args.chain,
        )
        warnings = [
            "topology+trajectory coordinates were streamed through MDAnalysis; "
            "verify atom mapping and upstream PBC/nojump preprocessing"
        ]
    else:
        prepared = _prepare_input(args)
        source = ArrayCoordinateSource(
            prepared.apply_coords,
            records=prepared.apply_records,
            fit_labels=prepared.fit_labels,
            fit_selection=prepared.fit_selection,
            apply_selection=prepared.apply_selection,
            provenance={
                "reader": "MDCoreAlign built-in reader",
                **prepared.unit_provenance,
            },
        )
        source.source_atom_identities = prepared.source_atom_identities
        source.fit_atom_identities = prepared.fit_atom_identities
        warnings = list(prepared.warnings)

    if rotations.shape != (source.n_frames, 3, 3):
        raise ValueError(
            "transform frame count does not match the input trajectory; use the original frame ordering"
        )
    if translations.shape != (source.n_frames, 3):
        raise ValueError("translations must have shape (n_frames, 3)")
    if expected_fit_atoms is not None and expected_fit_atoms != source.n_fit_atoms:
        raise ValueError(
            "fit-selection size does not match transforms; reuse the saved fit selection"
        )
    if expected_labels is not None and expected_labels != source.fit_labels:
        raise ValueError(
            "fit residue labels/order do not match transforms; use the original topology and selection"
        )
    if manifest is not None:
        if int(manifest["n_frames"]) != source.n_frames:
            raise ValueError("source frame count does not match source_manifest.json")
        expected_exportable_atoms = manifest.get(
            "n_exportable_atoms", manifest.get("n_apply_available_atoms")
        )
        default_selection = str(manifest.get("default_export_selection_expression", manifest["default_export_selection"]))
        using_default_export = _normalize_selection(source.apply_selection) == _normalize_selection(default_selection)
        if using_default_export and int(expected_exportable_atoms) != source.n_atoms:
            raise ValueError(
                "default export atom count does not match source_manifest.json; "
                "check topology identity and export selection"
            )
        observed_fingerprint = sequence_fingerprint(source.fit_labels)
        if observed_fingerprint != manifest["fit_labels_order_fingerprint"]:
            raise ValueError("fit atom order fingerprint does not match source_manifest.json")
        if manifest["schema_version"] == "2.2.0":
            source_identities = tuple(str(value) for value in manifest.get("source_atom_identities", ()))
            if not source_identities or source_identities != source.source_atom_identities:
                raise ValueError("source atom identity/order does not match source_manifest.json")
            if sequence_fingerprint(source_identities) != manifest.get("source_atom_order_fingerprint"):
                raise ValueError("source atom order fingerprint does not match source_manifest.json")
            if sequence_fingerprint(source.fit_atom_identities) != manifest.get("fit_atom_order_fingerprint"):
                raise ValueError("fit atom identity/order does not match source_manifest.json")
            if not _is_ordered_atom_subset(source.apply_atom_identities, source_identities):
                raise ValueError("export selection is not a source-ordered atom subset")
            configuration = manifest["read_configuration"]
            if prepared is not None:
                saved_units = configuration["unit_provenance"]
                for key in ("source_unit", "conversion_factor_to_angstrom", "canonical_internal_unit"):
                    if prepared.unit_provenance.get(key) != saved_units.get(key):
                        raise ValueError("source unit conversion differs from the saved read configuration")
        elif not using_default_export:
            # Old manifests lack a complete ordered atom identity record. They
            # can safely retain their proven default export, but cannot certify
            # an arbitrary new export subset with this schema.
            raise ValueError("legacy manifest cannot verify a different export subset; rerun align")

    frame_indices = parse_frame_indices(
        source.n_frames,
        purpose=purpose,
        frames=args.frames,
        frame_range=args.frame_range,
        stride=args.stride,
        all_selected_frames=args.all_selected_frames,
    )
    default_names = {
        ("structure", "mmcif"): "aligned_structure.cif",
        ("structure", "pdb"): "aligned_structure.pdb",
        ("trajectory", "xtc"): "aligned_trajectory.xtc",
        ("compatibility", "pdb"): "aligned_trajectory.pdb",
    }
    output_path = (
        Path(output_value)
        if output_value
        else output_dir / default_names[(purpose, output_format)]
    )
    allowed_suffixes = {
        "mmcif": {".cif", ".mmcif"},
        "xtc": {".xtc"},
        "pdb": {".pdb", ".ent"},
    }
    if output_path.suffix.lower() not in allowed_suffixes[output_format]:
        suffixes = ", ".join(sorted(allowed_suffixes[output_format]))
        raise ValueError(
            f"format '{output_format}' requires one of these output suffixes: {suffixes}"
        )

    annotated_export = bool(manifest is not None and purpose != "compatibility")
    residue_profile = None
    atom_annotations = None
    if annotated_export:
        if result_dir is None:
            raise ValueError("annotated export requires an alignment result directory")
        residue_profile = load_residue_annotations(result_dir / "residue_profile.tsv")
        profile_labels = tuple(
            f"{row.chain_id}:{row.residue_name}{row.residue_id}{row.insertion_code}"
            for row in residue_profile
        )
        if profile_labels != source.fit_labels:
            raise ValueError("residue profile identity/order does not match the verified fit atoms")
        exported_labels = {record.residue_key for record in source.records}
        if not exported_labels.issubset(set(profile_labels)):
            annotated_export = False
            warnings.append(
                "export includes residues outside the analysed fit selection; occupancy and B-factor "
                "remain source values for this export, with no invented RMSF/core annotations; "
                "residue_profile.tsv remains authoritative for the analysed residues"
            )
        else:
            selected_profile = tuple(row for row, label in zip(residue_profile, profile_labels, strict=True) if label in exported_labels)
            atom_annotations = map_annotations_to_atoms(source.records, selected_profile)
        if bool(manifest.get("source_is_ca_only")):
            warning = (
                "CA-only source detected: export remains CA-only; missing full-atom "
                "coordinates cannot be reconstructed and no atoms were invented"
            )
            warnings.append(warning)
            print(f"Warning: {warning}", file=sys.stderr)

    source_result_summary_path = (
        result_dir / "summary.json" if result_dir is not None else None
    )
    source_result_summary = (
        json.loads(source_result_summary_path.read_text())
        if source_result_summary_path is not None
        and source_result_summary_path.is_file()
        else {}
    )
    source_result_schema_version = source_result_summary.get("schema_version")
    result_hash_path = (
        source_result_summary_path
        if source_result_summary_path is not None
        and source_result_summary_path.is_file()
        else transforms_path
    )
    annotation_mapping = {
        "occupancy": (
            "reporting_core_label_1_core_0_noncore"
            if annotated_export
            else "legacy_source_value"
        ),
        "b_factor": (
            "stable_core_aligned_robust_RMSF_angstrom"
            if annotated_export
            else "legacy_source_value"
        ),
    }
    provenance = {
        **schema_metadata(__version__),
        "export_schema_version": "2.1.0",
        "source_result_schema_version": source_result_schema_version,
        "method": "coordinate materialization",
        "mode": "apply",
        "execution_contract": "coordinate_materialization",
        "experimental_mode": False,
        "software": "MDCoreAlign",
        "purpose": purpose,
        "output_format": output_format,
        "legacy_contract": legacy_contract,
        "visualization_annotations": annotated_export,
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "transforms": str(transforms_path),
        "transforms_sha256": file_sha256(transforms_path),
        "result_sha256": file_sha256(result_hash_path),
        "fit_selection": source.fit_selection,
        "export_selection": source.apply_selection,
        "annotation_mapping": annotation_mapping,
        "source": source.provenance,
        "source_resolution": source_resolution,
        "frame_indices": frame_indices.tolist(),
        "coordinate_unit_internal": "angstrom",
        "coordinate_unit_output": (
            "nanometer_on_disk" if output_format == "xtc" else "angstrom"
        ),
        "source_topology_sha256": (
            file_sha256(args.topology) if args.topology else None
        ),
        "source_trajectory_sha256": (
            file_sha256(args.trajectory)
            if args.trajectory
            else file_sha256(input_path)
        ),
        "occupancy_semantics": annotation_mapping["occupancy"],
        "b_factor_semantics": annotation_mapping["b_factor"],
        "reporting_core_residue_fraction": (
            float(np.mean([row.core_label for row in residue_profile]))
            if residue_profile is not None
            else None
        ),
        "algorithm_rerun": False,
    }
    writer = build_coordinate_writer(
        output_format,
        output_path,
        records=source.records,
        n_selected_frames=int(frame_indices.size),
        allow_large_text_output=args.allow_large_text_output,
        provenance=provenance,
        synthetic_records=source.records_are_synthetic,
        annotations=atom_annotations,
        residue_profile=residue_profile,
    )
    statistics = stream_apply_transforms(
        source,
        rotations,
        translations,
        frame_indices,
        writer,
        chunk_size=args.chunk_size,
        drop_unit_cell=args.drop_unit_cell,
    )

    companion_path: Path | None = None
    adjacent_sidecar: Path | None = None
    if output_format == "xtc" and manifest is not None:
        issue = pdb_compatibility_issue(source.records, atom_annotations, n_models=1)
        companion_format = "pdb" if issue is None else "mmcif"
        companion_suffix = ".pdb" if companion_format == "pdb" else ".cif"
        companion_path = output_path.with_name(
            f"{output_path.stem}.topology{companion_suffix}"
        )
        if issue is not None:
            warnings.append(
                f"XTC topology uses mmCIF because PDB cannot safely represent {issue}"
            )
        companion_writer = build_coordinate_writer(
            companion_format,
            companion_path,
            records=source.records,
            n_selected_frames=1,
            allow_large_text_output=True,
            provenance={**provenance, "purpose": "xtc_topology_companion"},
            synthetic_records=source.records_are_synthetic,
            annotations=atom_annotations,
            residue_profile=residue_profile,
        )
        stream_apply_transforms(
            source,
            rotations,
            translations,
            np.asarray([0], dtype=np.int64),
            companion_writer,
            chunk_size=1,
            drop_unit_cell=args.drop_unit_cell,
        )
        adjacent_sidecar = Path(str(output_path) + ".mdcorealign.json")

    summary_path = output_dir / "apply_summary.json"
    metadata_path = output_dir / "run_metadata.json"
    provenance_path = output_dir / "coordinate_export_provenance.json"
    selected_frames_path = output_dir / "selected_frames.tsv"
    atom_metadata_path = output_dir / "export_atom_metadata.tsv"
    _write_table_with_fields(
        selected_frames_path,
        [
            {"output_order": index + 1, "source_frame_index": int(frame)}
            for index, frame in enumerate(frame_indices)
        ],
        ["output_order", "source_frame_index"],
    )
    atom_rows = rows_from_records(source.records)
    _write_table_with_fields(
        atom_metadata_path,
        atom_rows,
        list(atom_rows[0]) if atom_rows else ["atom_index"],
    )
    if output_format == "xtc":
        unit_cell_policy = (
            "dropped_by_explicit_request"
            if args.drop_unit_cell and statistics.box_frames_seen
            else "rotated_and_preserved"
            if statistics.box_frames_written
            else "not_available_in_input"
        )
        time_policy = "preserved_when_available"
    elif output_format == "mmcif":
        unit_cell_policy = (
            "dropped_by_explicit_request"
            if args.drop_unit_cell and statistics.box_frames_seen
            else "stored_as_per-model_cartesian_box_vectors"
            if statistics.box_frames_written
            else "not_available_in_input"
        )
        time_policy = "stored_in_mdcorealign_frame_category_when_available"
    else:
        unit_cell_policy = (
            "not_represented_in_pdb_and_documented"
            if statistics.box_frames_seen
            else "not_available_in_input"
        )
        time_policy = "not_represented_in_pdb"
    provenance.update(
        {
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "n_exported_frames": statistics.n_frames,
            "n_exported_atoms": statistics.n_atoms,
            "peak_chunk_frames": statistics.peak_chunk_frames,
            "box_frames_seen": statistics.box_frames_seen,
            "box_frames_written": statistics.box_frames_written,
            "unit_cell_policy": unit_cell_policy,
            "time_policy": time_policy,
            "topology_companion": (
                None if companion_path is None else str(companion_path)
            ),
        }
    )
    _write_json(provenance_path, provenance)
    if adjacent_sidecar is not None:
        _write_json(
            adjacent_sidecar,
            {
                "schema_version": "2.1.0",
                "software_version": __version__,
                "source_result_schema_version": source_result_schema_version,
                "source_topology_sha256": provenance["source_topology_sha256"],
                "source_trajectory_sha256": provenance["source_trajectory_sha256"],
                "result_sha256": provenance["result_sha256"],
                "fit_selection": source.fit_selection,
                "export_selection": source.apply_selection,
                "frame_count": statistics.n_frames,
                "atom_count": statistics.n_atoms,
                "coordinate_unit_on_disk": "nanometer",
                "unit_cell_policy": unit_cell_policy,
                "time_policy": time_policy,
                "topology_companion": str(companion_path),
            },
        )
    outputs = {
        "aligned_coordinates": str(output_path),
        "provenance": str(provenance_path),
        "selected_frames": str(selected_frames_path),
        "atom_metadata": str(atom_metadata_path),
    }
    if companion_path is not None:
        outputs["annotated_topology_companion" if annotated_export else "topology_companion"] = str(companion_path)
    if adjacent_sidecar is not None:
        outputs["adjacent_xtc_provenance"] = str(adjacent_sidecar)
    summary = {
        **schema_metadata(__version__),
        "method": "coordinate materialization",
        "mode": "apply",
        "execution_contract": "coordinate_materialization",
        "experimental_mode": False,
        "command": "apply",
        "input": str(input_path),
        "transforms": str(transforms_path),
        "output": str(output_path),
        "purpose": purpose,
        "output_format": output_format,
        "source_result_schema_version": source_result_schema_version,
        "legacy_contract": legacy_contract,
        "visualization_annotations": annotated_export,
        "annotation_mapping": annotation_mapping,
        "n_frames": statistics.n_frames,
        "n_source_frames": source.n_frames,
        "export_selection": source.apply_selection,
        "n_exported_atoms": statistics.n_atoms,
        "n_fit_atoms": source.n_fit_atoms,
        "complete_protein_export": bool(
            manifest is not None
            and manifest.get("default_export_selection") == "protein"
            and not manifest.get("source_is_ca_only")
            and statistics.n_atoms == int(manifest.get("n_exportable_atoms", manifest.get("n_apply_available_atoms", -1)))
            and _normalize_selection(source.apply_selection) == _normalize_selection(str(manifest.get("default_export_selection_expression", "protein")))
        ),
        "source_is_ca_only": bool(
            manifest and manifest.get("source_is_ca_only")
        ),
        "legacy_metadata": {
            "format": output_format,
            "n_apply_atoms": statistics.n_atoms,
            "apply_selection": source.apply_selection,
            "ca_only_source": bool(
                manifest and manifest.get("source_is_ca_only")
            ),
        },
        "selected_frame_indices": frame_indices.tolist(),
        "streaming_transform_application": True,
        "chunk_size": args.chunk_size,
        "peak_chunk_frames": statistics.peak_chunk_frames,
        "coordinate_unit": "angstrom",
        "unit_cell_policy": unit_cell_policy,
        "time_policy": time_policy,
        "topology_required_to_read_output": output_format == "xtc",
        "fit_selection_verified": expected_fit_atoms is not None,
        "fit_label_order_verified": expected_labels is not None,
        "source_manifest_verified": manifest is not None,
        "algorithm_rerun": False,
        "outputs": outputs,
        "warnings": warnings,
    }
    _write_json(summary_path, summary)
    _write_run_metadata(
        metadata_path,
        args=args,
        input_path=input_path,
        output_files={
            "aligned_output": str(output_path),
            "apply_summary_json": str(summary_path),
            "run_metadata_json": str(metadata_path),
            **outputs,
        },
        warnings=warnings,
        prepared=prepared,
    )
    print(f"Applied saved transforms: {output_path}")
    if companion_path is not None:
        print(f"Wrote annotated topology companion: {companion_path}")
    print(f"Wrote apply summary: {summary_path}")
    return 0


def _run_flexibility(args: argparse.Namespace) -> int:
    """Run robust alignment and flexibility profiling."""

    _prepare_parameter_contract(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path_from_args(args)
    prepared = _prepare_input(args)
    audit_outputs = _write_input_audit_outputs(output_dir, args, prepared)
    if args.mode == "alignment-only":
        raise ValueError(
            "mode 'alignment-only' is intended for `mdcorealign align`; use "
            "'default', 'long', 'long-fast', 'fast-global', or "
            "'diagnostic-rich' for flexibility outputs"
        )
    result = _solve_for_mode(args, prepared)
    if args.mode in {"exact", "throughput"} or args.contract is not None:
        residue_ids, residue_names, chains, _ = _residue_metadata_from_labels(
            prepared.fit_labels
        )
        stability = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
        mobility = np.clip(1.0 - stability, 0.0, 1.0)
        profile = FlexibilityProfile(
            residue_id=tuple(str(value) for value in residue_ids),
            resname=tuple(str(value) for value in residue_names),
            chain=tuple(str(value) for value in chains),
            robust_aligned_rmsf=np.asarray(result.robust_rmsf, dtype=float),
            ordinary_aligned_rmsf=(
                None
                if result.ordinary_rmsf is None
                else np.asarray(result.ordinary_rmsf, dtype=float)
            ),
            soft_stability_weight=stability,
            mobility_score=mobility,
            flexible_segment_id=detect_flexible_segments(
                mobility,
                percentile_cutoff=args.flexible_percentile,
                min_segment_length=args.min_segment_length,
                merge_gap=args.merge_gap,
                chain=chains,
            ),
            rank=descending_ranks(mobility),
            percentile=percentile_scores(mobility),
        )
    elif args.mode in {"long", "long-fast"}:
        profile = build_long_flexibility_profile(
            result,
            residue_labels=prepared.fit_labels,
            flexible_percentile_cutoff=args.flexible_percentile,
            min_segment_length=args.min_segment_length,
            merge_gap=args.merge_gap,
        )
    else:
        profile = build_flexibility_profile(
            prepared.fit_trajectory,
            residue_labels=prepared.fit_labels,
            trajectory_result=result,
            reference_mode=args.reference_mode,
            warm_start=args.warm_start,
            flexible_percentile_cutoff=args.flexible_percentile,
            min_segment_length=args.min_segment_length,
            merge_gap=args.merge_gap,
            rule=args.rule,
            delta=args.delta,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            sequence_continuity_strength=args.sequence_lambda,
            contact_consistency_strength=args.contact_lambda,
            compute_ordinary_rmsf=args.ordinary_rmsf,
        )
    aligned = _aligned_apply_coords(result, prepared)
    aligned_path = output_dir / "aligned_trajectory.pdb"
    weights_path = output_dir / "weights.csv"
    profile_path = output_dir / "flexibility_profile.csv"
    segments_path = output_dir / "flexible_segments.tsv"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    _write_aligned_output(aligned, aligned_path, records=prepared.apply_records)
    _write_csv(weights_path, _weights_rows(result, core_fraction=args.core_fraction, residue_labels=prepared.fit_labels))
    write_flexibility_profile_csv(profile, profile_path)
    _write_csv(
        segments_path,
        summarize_flexible_segments(profile, system=args.system_name, core_fraction=args.core_fraction),
        delimiter="\t",
    )
    long_outputs = _write_long_mode_artifacts(output_dir, result, args=args)
    experimental_outputs = _write_experimental_mode_artifacts(
        output_dir, result, prepared=prepared
    )
    summary = _summary_from_result(
        result,
        command="flexibility",
        input_path=input_path,
        output_dir=output_dir,
        core_fraction=args.core_fraction,
        prepared=prepared,
        flexible_percentile=args.flexible_percentile,
    )
    summary.update(
        {
            "system": args.system_name,
            "flexible_percentile": args.flexible_percentile,
            "min_segment_length": args.min_segment_length,
            "merge_gap": args.merge_gap,
            "flexibility_summary": summarize_flexibility_profile(profile, system=args.system_name, core_fraction=args.core_fraction),
            "outputs": {
                "aligned_output": str(aligned_path),
                "weights_csv": str(weights_path),
                "flexibility_profile_csv": str(profile_path),
                "flexible_segments_tsv": str(segments_path),
                "summary_json": str(summary_path),
                "run_metadata_json": str(metadata_path),
                **long_outputs,
                **experimental_outputs,
                **audit_outputs,
            },
            "interpretation_notes": [
                "mobility_score = 1 - soft_stability_weight when the default soft_weight mode is used",
                "robust_RMSF is alignment-frame dependent and should be interpreted with the chosen reference/fit protocol",
                "RMSF/flexible segments are computed on the analysis selection, not automatically on all apply-selection atoms",
                "ligands/lipids/solvent can be transformed by apply-selection but are not part of stable-core inference by default",
            ],
        }
    )
    _write_json(summary_path, summary)
    _write_run_metadata(metadata_path, args=args, input_path=input_path, output_files=summary["outputs"], warnings=prepared.warnings, prepared=prepared)
    print(f"Wrote aligned trajectory: {aligned_path}")
    print(f"Wrote weights: {weights_path}")
    print(f"Wrote flexibility profile: {profile_path}")
    print(f"Wrote flexible segments: {segments_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote run metadata: {metadata_path}")
    return 0


def _run_diagnostics(args: argparse.Namespace) -> int:
    """Run lightweight alignment diagnostics."""

    if args.backend != "cpu" or args.contract is not None:
        raise ValueError(
            "`diagnostics` profiles the exact public CPU workflow. Use "
            "`align --contract throughput|balanced-lite --backend gpu` for "
            "experimental GPU timing and sampled quality metadata."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path_from_args(args)
    prepared = _prepare_input(args)
    audit_outputs = _write_input_audit_outputs(output_dir, args, prepared)
    diagnostic = solve_trajectory_with_optional_diagnostics(
        prepared.fit_trajectory,
        diagnostics=args.trace,
        system=args.system_name,
        run_id=args.run_id,
        reference_mode=args.reference_mode,
        warm_start=args.warm_start,
        **_robust_kwargs(args),
    )
    summary_path = output_dir / "diagnostics_summary.json"
    runtime_path = output_dir / "runtime_profile.json"
    trace_path = output_dir / "trace.tsv"
    metadata_path = output_dir / "run_metadata.json"
    _write_json(
        summary_path,
        {
            "command": "diagnostics",
            "mdcorealign_version": __version__,
            "input": str(input_path),
            "system": args.system_name,
            "run_id": args.run_id,
            "fit_selection": prepared.fit_selection,
            "apply_selection": prepared.apply_selection,
            "analysis_selection": prepared.fit_selection,
            "n_frames": len(diagnostic.result.frame_results),
            "n_fit_atoms": prepared.fit_trajectory.n_atoms,
            "n_fit_residues": prepared.n_fit_residues,
            "n_apply_atoms": int(prepared.apply_coords.shape[1]),
            "mean_rmsd": diagnostic.result.mean_rmsd,
            "mean_weighted_rmsd": diagnostic.result.mean_weighted_rmsd,
            "average_iterations_per_frame": diagnostic.result.average_iterations_per_frame,
            "converged_fraction": diagnostic.result.converged_fraction,
            "trace_written": bool(args.trace),
            "warnings": prepared.warnings,
        },
    )
    _write_json(runtime_path, diagnostic.runtime_profile)
    if args.trace:
        _write_csv(trace_path, diagnostic.trace_rows, delimiter="\t")
    output_files = {
        "diagnostics_summary_json": str(summary_path),
        "runtime_profile_json": str(runtime_path),
        "trace_tsv": str(trace_path) if args.trace else None,
        "run_metadata_json": str(metadata_path),
        **audit_outputs,
    }
    _write_run_metadata(metadata_path, args=args, input_path=input_path, output_files=output_files, warnings=prepared.warnings, prepared=prepared)
    print(f"Wrote diagnostics summary: {summary_path}")
    print(f"Wrote runtime profile: {runtime_path}")
    if args.trace:
        print(f"Wrote trace: {trace_path}")
    print(f"Wrote run metadata: {metadata_path}")
    return 0


def _chain_continuity(labels: Iterable[str]) -> str:
    """Check residue-number monotonicity within simple chain labels."""

    by_chain: dict[str, list[int]] = {}
    for index, label in enumerate(labels, start=1):
        chain = label.split(":", 1)[0] if ":" in label else "A"
        digits = "".join(character for character in label if character.isdigit())
        resid = int(digits) if digits else index
        by_chain.setdefault(chain, []).append(resid)
    for values in by_chain.values():
        if any(next_value < value for value, next_value in zip(values, values[1:], strict=False)):
            return "warning_nonmonotonic_residue_ids"
    return "pass_numeric_residue_ids_nondecreasing_within_chain"


def _run_audit_preprocessing(args: argparse.Namespace) -> int:
    """Run lightweight local trajectory sanity checks."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path_from_args(args)
    prepared = _prepare_input(args)
    outputs = _write_input_audit_outputs(output_dir, args, prepared)
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    row = _audit_row(args, prepared)
    row.update(
        {
            "method": "input audit",
            "command": "audit-preprocessing",
            "analysis_selection": prepared.fit_selection,
            "external_tools_required": False,
        }
    )
    _write_json(summary_path, row)
    # Backward-compatible filename from earlier release-candidate builds.
    _write_table_with_fields(output_dir / "preprocessing_audit.tsv", [_audit_row(args, prepared)], AUDIT_FIELDS)
    output_files = {**outputs, "summary_json": str(summary_path), "run_metadata_json": str(metadata_path)}
    _write_run_metadata(metadata_path, args=args, input_path=input_path, output_files=output_files, warnings=prepared.warnings, prepared=prepared)
    print(f"Wrote input audit: {outputs['input_audit_tsv']}")
    print(f"Wrote input audit JSON: {outputs['input_audit_json']}")
    print(f"Wrote chain mapping report: {outputs['chain_mapping_report_tsv']}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote run metadata: {metadata_path}")
    return 0


def _run_audit_coordinate_scale(args: argparse.Namespace) -> int:
    """Run a read-only chain-aware coordinate-scale integrity audit."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_input(args)
    chains: list[str] = []
    residues: list[str] = []
    for index, label in enumerate(prepared.fit_labels, start=1):
        parts = str(label).split(":")
        chains.append(parts[0] if len(parts) > 1 else "A")
        residue_text = parts[-1]
        digits = "".join(
            character
            for character in residue_text
            if character.isdigit() or character == "-"
        )
        residues.append(digits or str(index))
    summary, frame_rows = audit_coordinate_scale(
        prepared.fit_trajectory.coords,
        chain_ids=np.asarray(chains, dtype=str),
        residue_ids=np.asarray(residues, dtype=str),
    )
    summary_row = {
        "system": args.system_name,
        "input": str(_input_path_from_args(args)),
        "fit_selection": prepared.fit_selection,
        **summary.to_dict(),
    }
    _write_table_with_fields(
        output_dir / "coordinate_scale_audit.tsv",
        [summary_row],
        list(summary_row),
    )
    _write_table_with_fields(
        output_dir / "coordinate_scale_per_frame.tsv",
        frame_rows,
        list(frame_rows[0]),
    )
    _write_json(output_dir / "coordinate_scale_audit.json", summary_row)
    print(
        "Coordinate-scale status: "
        f"{summary.status}; inferred unit: {summary.inferred_internal_unit}"
    )
    print(f"Wrote coordinate-scale audit: {output_dir / 'coordinate_scale_audit.tsv'}")
    return 0


def _run_local_deformation(args: argparse.Namespace) -> int:
    """Run the experimental alignment-free W5 local-deformation analysis."""

    if args.window_size != 5:
        raise ValueError(
            "The reintegrated local-deformation contract is currently locked "
            "to chain-aware window-size 5."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path_from_args(args)
    prepared = _prepare_input(args)
    audit_outputs = _write_input_audit_outputs(output_dir, args, prepared)
    residue_ids, residue_names, chain_ids, positions = (
        _residue_metadata_from_labels(prepared.fit_labels)
    )
    backend = args.backend
    if backend == "auto":
        from .gpu_backend import gpu_available

        backend = "gpu" if gpu_available() else "cpu"
    if backend == "gpu" and args.experimental_motion_graph:
        raise ValueError(
            "The first GPU backend ports W5 local deformation only. Run the "
            "optional soft motion graph with --backend cpu."
        )
    if backend == "gpu":
        from .gpu_backend import scan_w5_gpu

        graph = None
        local = scan_w5_gpu(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            residue_positions=positions,
            precision=args.gpu_precision,
            frame_chunk_size=args.descriptor_chunk_size,
        )
    elif args.experimental_motion_graph:
        graph = scan_motion_domains(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            residue_positions=positions,
            block_count=args.block_count,
            nonlocal_sample_count=args.nonlocal_sample_count,
            evaluate_blocks=True,
            descriptor_chunk_size=args.descriptor_chunk_size,
            deforming_percentile=args.deforming_percentile,
            minimum_segment_length=args.min_segment_length,
            merge_gap=args.merge_gap,
        )
        local = graph.local
    else:
        graph = None
        local = scan_local_kinematics(
            prepared.fit_trajectory.coords,
            chain_ids=chain_ids,
            residue_positions=positions,
            block_count=args.block_count,
            evaluate_blocks=True,
            deforming_percentile=args.deforming_percentile,
            minimum_segment_length=args.min_segment_length,
            merge_gap=args.merge_gap,
            descriptor_chunk_size=args.descriptor_chunk_size,
        )
    rows = local_deformation_rows(
        local,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_positions=positions,
        source_provenance=prepared.unit_provenance,
    )
    local_path = (
        Path(args.output)
        if args.output is not None
        else output_dir / "local_deformation.tsv"
    )
    _write_table_with_fields(local_path, rows, list(rows[0]))
    segment_rows: list[dict[str, object]] = []
    segment_ids = local.locally_deforming_segments
    for segment_id in sorted(set(segment_ids).difference({0})):
        indices = np.flatnonzero(segment_ids == segment_id)
        if indices.size == 0:
            continue
        segment_rows.append(
            {
                "segment_id": int(segment_id),
                "chain": str(chain_ids[indices[0]]),
                "start_residue_id": str(residue_ids[indices[0]]),
                "end_residue_id": str(residue_ids[indices[-1]]),
                "length": int(indices.size),
                "mean_local_deformation_index": float(
                    np.mean(local.local_deformation_index[indices])
                ),
                "mean_local_rigidity_score": float(
                    np.mean(local.local_rigidity_score[indices])
                ),
                "mean_confidence": float(np.mean(local.confidence[indices])),
            }
        )
    segments_path = output_dir / "locally_deforming_segments.tsv"
    _write_table_with_fields(
        segments_path,
        segment_rows,
        [
            "segment_id",
            "chain",
            "start_residue_id",
            "end_residue_id",
            "length",
            "mean_local_deformation_index",
            "mean_local_rigidity_score",
            "mean_confidence",
        ],
    )
    outputs: dict[str, object] = {
        "local_deformation_tsv": str(local_path),
        "locally_deforming_segments_tsv": str(segments_path),
        **audit_outputs,
    }
    if graph is not None:
        edge_rows, residue_rows = motion_graph_rows(graph)
        edges_path = output_dir / "experimental_motion_graph_edges.tsv"
        residues_path = (
            output_dir / "experimental_motion_graph_residue_diagnostics.tsv"
        )
        _write_table_with_fields(edges_path, edge_rows, list(edge_rows[0]))
        _write_table_with_fields(
            residues_path, residue_rows, list(residue_rows[0])
        )
        outputs.update(
            {
                "experimental_motion_graph_edges_tsv": str(edges_path),
                "experimental_motion_graph_residues_tsv": str(residues_path),
            }
        )
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    outputs.update(
        {
            "summary_json": str(summary_path),
            "run_metadata_json": str(metadata_path),
        }
    )
    _write_json(
        summary_path,
        {
            "command": "local-deformation",
            "method": "experimental alignment-free W5 local deformation",
            "status": "experimental",
            "output_contract": "alignment_free_local_deformation",
            "score_is_rmsd": False,
            "score_is_global_mobility": False,
            "hard_biological_domains_claimed": False,
            "window_size": 5,
            "descriptor": local.metadata["descriptor"],
            "n_frames": prepared.fit_trajectory.n_frames,
            "n_residues": prepared.fit_trajectory.n_atoms,
            "n_locally_deforming_segments": len(segment_rows),
            "temporal_block_reproducibility": (
                local.temporal_block_reproducibility
            ),
            "temporal_block_segment_jaccard": (
                local.temporal_block_segment_jaccard
            ),
            "unit_provenance": prepared.unit_provenance,
            "timing": (
                graph.timing if graph is not None else local.timing
            ),
            "motion_graph_enabled": graph is not None,
            "backend_requested": args.backend,
            "backend_used": backend,
            "motion_graph_warning": (
                MOTION_GRAPH_WARNING if graph is not None else None
            ),
            "warnings": prepared.warnings,
            "outputs": outputs,
        },
    )
    _write_run_metadata(
        metadata_path,
        args=args,
        input_path=input_path,
        output_files=outputs,
        warnings=prepared.warnings,
        prepared=prepared,
    )
    print(f"Wrote local-deformation profile: {local_path}")
    print(f"Wrote locally deforming segments: {segments_path}")
    if graph is not None:
        print(
            "Wrote experimental soft motion-graph diagnostics; hard partitions "
            "must not be interpreted as mature biological domains."
        )
    print(f"Wrote summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the parser through the dedicated parser-construction module."""

    from .cli_parser import build_parser as build_cli_parser

    return build_cli_parser()


def run(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process status code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"mdcorealign: error: {exc}", file=sys.stderr)
        print(
            "Suggestion: run the command with --help, confirm atom correspondence, "
            "and use --input-format ca-pdb for CA-only multi-model PDB examples.",
            file=sys.stderr,
        )
        return 2


def main(argv: list[str] | None = None) -> None:
    """Console entry point."""

    raise SystemExit(run(argv))
