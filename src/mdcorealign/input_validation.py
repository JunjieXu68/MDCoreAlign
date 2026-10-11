"""Input-format policy and readiness classification for MDCoreAlign.

The functions in this module are deliberately conservative. They help the CLI
explain what is recommended, experimental, or unsupported without claiming that
every backend-readable file format is formally validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

READY_TO_RUN = "ready_to_run"
RUNNABLE_WITH_WARNINGS = "runnable_with_warnings"
NEEDS_PREPROCESSING = "needs_preprocessing"
NEEDS_CONVERSION = "needs_conversion"
UNSUPPORTED = "unsupported"


TOPOLOGY_FORMATS = {
    ".pdb": "PDB",
    ".ent": "PDB",
    ".gro": "GRO",
    ".psf": "PSF",
    ".prmtop": "PRMTOP/PARM7",
    ".parm7": "PRMTOP/PARM7",
    ".top": "PRMTOP/PARM7",
    ".tpr": "TPR",
}

TRAJECTORY_FORMATS = {
    ".xtc": "XTC",
    ".dcd": "DCD",
    ".trr": "TRR",
    ".nc": "NC/NetCDF",
    ".netcdf": "NC/NetCDF",
    ".pdb": "PDB multi-model",
    ".ent": "PDB multi-model",
    ".mdcrd": "TRJ/MDCRD",
    ".trj": "TRJ/MDCRD",
    ".tng": "TNG",
    ".h5md": "H5MD",
    ".gsd": "GSD",
    ".lammpstrj": "LAMMPS dump",
    ".dump": "LAMMPS dump",
    ".dms": "DMS",
    ".mol2": "MOL2",
    ".xyz": "XYZ",
    ".npy": "NumPy array",
    ".npz": "NumPy archive",
}

RECOMMENDED_TOPOLOGIES = {"PDB", "GRO", "PSF", "PRMTOP/PARM7", "TPR"}
RECOMMENDED_TRAJECTORIES = {"XTC", "DCD", "NC/NetCDF", "PDB multi-model", "TRJ/MDCRD"}
EXPERIMENTAL_TRAJECTORIES = {"TRR", "TNG", "H5MD", "GSD", "LAMMPS dump", "DMS", "MOL2", "XYZ"}

NUCLEIC_RESNAMES = {
    "A",
    "C",
    "G",
    "U",
    "T",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "ADE",
    "CYT",
    "GUA",
    "THY",
    "URA",
}

# One residue-name policy for both the built-in PDB and MDAnalysis adapters.
# Record type alone is not protein identity: MSE may be HETATM and DNA may be ATOM.
PROTEIN_RESNAMES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "ASH GLH CYX CYM HSD HSE HSP HID HIE HIP LYN MSE SEC PYL HYP ACE NME NMA NH2".split()
)
PROTEIN_SELECTION_POLICY = "residue_name_protein_v1"


def select_mdanalysis_atoms(universe, selection: str, *, chain: str | None = None):
    """Apply the same protein/backbone and primary-altloc definition as PDB I/O."""

    from MDAnalysis.exceptions import SelectionError

    protein = "(resname " + " ".join(sorted(PROTEIN_RESNAMES)) + ")"
    expression = re.sub(r"\bbackbone\b", f"({protein} and name N CA C O)", selection, flags=re.I)
    expression = re.sub(r"\bprotein\b", protein, expression, flags=re.I)
    # The built-in documented CA expressions are case insensitive. Preserve
    # arbitrary backend expressions, but normalize this shared shorthand.
    expression = re.sub(r"\bname\s+ca\b", "name CA", expression, flags=re.I)
    try:
        atoms = universe.select_atoms(expression)
    except SelectionError as exc:
        raise ValueError(f"invalid selection '{selection}': {exc}") from exc
    keep = []
    for atom in atoms:
        try:
            altloc = str(atom.altLoc)
        except (AttributeError, ValueError):
            altloc = ""
        try:
            atom_chain = str(atom.chainID or atom.segid or "")
        except (AttributeError, ValueError):
            atom_chain = str(getattr(atom, "segid", "") or "")
        keep.append(altloc in {"", "A"} and (chain is None or atom_chain == chain))
    if chain is not None and not any(keep):
        raise ValueError(f"chain '{chain}' has no atoms in fit-selection '{selection}'")
    return atoms[np.asarray(keep, dtype=bool)]


@dataclass(frozen=True)
class InputReadiness:
    """Machine-readable input readiness decision."""

    status: str
    reason: str
    recommended_action: str
    topology_format: str | None = None
    trajectory_format: str | None = None


def support_format_rows() -> list[dict[str, str]]:
    """Return the public input/output support table used by `mdcorealign formats`."""

    rows = [
        {
            "category": "recommended topology files",
            "format_or_pair": "PDB",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "no",
            "recommended_for": "protein metadata and small examples",
            "notes": "must preserve residue and chain metadata",
        },
        {
            "category": "recommended topology files",
            "format_or_pair": "GRO",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "no",
            "recommended_for": "GROMACS topology-style input",
            "notes": "use with XTC/TRR and valid residue metadata",
        },
        {
            "category": "recommended topology files",
            "format_or_pair": "PSF",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "no",
            "recommended_for": "CHARMM/NAMD DCD workflows",
            "notes": "use with DCD or compatible trajectory",
        },
        {
            "category": "recommended topology files",
            "format_or_pair": "PRMTOP/PARM7",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "no",
            "recommended_for": "AMBER workflows",
            "notes": "use with NC/NetCDF or MDCRD where supported",
        },
        {
            "category": "recommended topology files",
            "format_or_pair": "TPR",
            "status": "expected",
            "requires_topology": "yes",
            "requires_trajectory": "no",
            "recommended_for": "GROMACS workflows when backend support is available",
            "notes": "backend support can vary by environment",
        },
        {
            "category": "recommended topology files",
            "format_or_pair": "multi-model PDB",
            "status": "validated",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "small examples and structural ensembles",
            "notes": "atom order must be identical across models",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "XTC",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "GROMACS protein MD",
            "notes": "requires valid topology and preprocessing",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "DCD",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "CHARMM/NAMD protein MD",
            "notes": "requires PSF/PDB topology metadata",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "TRR",
            "status": "experimental",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "GROMACS workflows needing full precision",
            "notes": "backend-readable but not formally manuscript-validated",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "NC/NetCDF",
            "status": "recommended",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "AMBER protein MD",
            "notes": "requires PRMTOP/PARM7 metadata",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "PDB multi-model",
            "status": "recommended",
            "requires_topology": "no",
            "requires_trajectory": "yes",
            "recommended_for": "small examples and NMR-like ensembles",
            "notes": "not recommended for large production trajectories",
        },
        {
            "category": "recommended trajectory files",
            "format_or_pair": "TRJ/MDCRD",
            "status": "expected",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "AMBER workflows where backend support exists",
            "notes": "confirm atom order and residue metadata",
        },
        {
            "category": "recommended topology+trajectory pairs",
            "format_or_pair": "PDB + XTC",
            "status": "validated",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "portable protein MD input",
            "notes": "formal claims still use CA-level protein fitting",
        },
        {
            "category": "recommended topology+trajectory pairs",
            "format_or_pair": "GRO + XTC",
            "status": "validated",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "GROMACS protein MD",
            "notes": "preprocess nojump/center upstream",
        },
        {
            "category": "recommended topology+trajectory pairs",
            "format_or_pair": "PSF + DCD",
            "status": "expected",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "CHARMM/NAMD protein MD",
            "notes": "available through MDAnalysis; not fixture-validated in IO v1",
        },
        {
            "category": "recommended topology+trajectory pairs",
            "format_or_pair": "PRMTOP/PARM7 + NC/NetCDF",
            "status": "expected",
            "requires_topology": "yes",
            "requires_trajectory": "yes",
            "recommended_for": "AMBER protein MD",
            "notes": "available through MDAnalysis; not fixture-validated in IO v1",
        },
    ]
    for fmt in ["TNG", "H5MD", "GSD", "LAMMPS dump / lammpstrj", "DMS", "MOL2", "XYZ"]:
        rows.append(
            {
                "category": "backend-readable experimental formats",
                "format_or_pair": fmt,
                "status": "experimental",
                "requires_topology": "depends",
                "requires_trajectory": "depends",
                "recommended_for": "expert use only",
                "notes": "may work if topology, atom order and residue metadata are valid; not formally validated",
            }
        )
    for fmt, notes in [
        ("CSV/TSV", "stable public table outputs"),
        ("JSON", "stable public summary/metadata outputs"),
        ("annotated PDB structure", "available in standard installation; complete-protein frame or ensemble with core-label occupancy and robust-RMSF B-factor"),
        ("PDBx/mmCIF structure", "available in standard installation; complete-protein annotated export with a versioned custom residue-profile category through Gemmi"),
        ("XTC aligned trajectory", "available in standard installation; streaming complete-protein trajectory export through MDAnalysis with an automatic annotated topology companion"),
        ("multi-model PDB compatibility", "advanced backward-compatible text export with explicit format limits"),
    ]:
        rows.append(
            {
                "category": "output formats",
                "format_or_pair": fmt,
                "status": "validated" if fmt in {
                    "annotated PDB structure",
                    "PDBx/mmCIF structure",
                    "XTC aligned trajectory",
                    "multi-model PDB compatibility",
                } else "recommended",
                "requires_topology": "no",
                "requires_trajectory": "no",
                "recommended_for": "MDCoreAlign outputs",
                "notes": notes,
            }
        )
    for dependency, notes in [
        ("NumPy", "standard dependency for numerical analysis and compatibility PDB output"),
        ("MDAnalysis", "standard dependency for topology/trajectory reading and XTC output"),
        ("Gemmi", "standard dependency for PDBx/mmCIF output"),
    ]:
        rows.append(
            {
                "category": "standard runtime dependencies",
                "format_or_pair": dependency,
                "status": "documented",
                "requires_topology": "depends",
                "requires_trajectory": "depends",
                "recommended_for": "coordinate I/O",
                "notes": notes,
            }
        )
    for item, notes in [
        ("XVG/DAT/CSV without coordinates", "analysis-only files cannot be aligned"),
        ("trajectory without topology", "residue-level stable-core and RMSF analysis require reliable metadata"),
        ("variable atom-set trajectory", "fixed correspondence is required"),
        ("raw wrapped trajectory without preprocessing", "unwrap/nojump/center upstream before interpretation"),
    ]:
        rows.append(
            {
                "category": "unsupported/not recommended inputs",
                "format_or_pair": item,
                "status": "not recommended",
                "requires_topology": "yes",
                "requires_trajectory": "yes",
                "recommended_for": "none",
                "notes": notes,
            }
        )
    return rows


def infer_format(path: str | Path | None, *, override: str | None = None, role: str = "trajectory") -> str | None:
    """Infer a conservative format label from extension or explicit override."""

    if override:
        return override.upper()
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    mapping = TOPOLOGY_FORMATS if role == "topology" else TRAJECTORY_FORMATS
    return mapping.get(suffix)


def _read_pdb_resnames(path: Path, limit: int = 5000) -> tuple[set[str], int, int]:
    """Return residue names, CA count and atom count from a PDB-like file."""

    resnames: set[str] = set()
    ca_count = 0
    atom_count = 0
    with path.open(errors="ignore") as handle:
        for line_index, line in enumerate(handle):
            if line_index > limit:
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_count += 1
            atom_name = line[12:16].strip().upper()
            resname = line[17:20].strip().upper()
            if resname:
                resnames.add(resname)
            if atom_name == "CA" and resname in PROTEIN_RESNAMES:
                ca_count += 1
    return resnames, ca_count, atom_count


def classify_input(
    *,
    input_path: str | Path | None = None,
    topology_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    topology_format: str | None = None,
    trajectory_format: str | None = None,
    fit_selection: str = "protein and name CA",
    large_jump_detected: bool = False,
    missing_chain_ids: int = 0,
    duplicate_residue_labels: int = 0,
    nan_coordinates: bool = False,
    selection_size: int | None = None,
) -> InputReadiness:
    """Classify whether an input is ready, risky, convertible, or unsupported."""

    topology = Path(topology_path) if topology_path else None
    trajectory = Path(trajectory_path) if trajectory_path else None
    single_input = Path(input_path) if input_path else None

    if topology is not None and str(topology) == "":
        return InputReadiness(UNSUPPORTED, "empty topology path", "Provide a topology file with residue metadata.")
    if trajectory is not None and str(trajectory) == "":
        return InputReadiness(UNSUPPORTED, "empty trajectory path", "Provide a coordinate trajectory file.")

    for role, path in [("topology", topology), ("trajectory", trajectory), ("input", single_input)]:
        if path is not None and not path.exists():
            return InputReadiness(UNSUPPORTED, f"{role} file does not exist: {path}", f"Check the {role} path.")

    inferred_topology = infer_format(topology, override=topology_format, role="topology")
    inferred_trajectory = infer_format(trajectory or single_input, override=trajectory_format, role="trajectory")

    if trajectory is not None and topology is None:
        suffix = trajectory.suffix.lower()
        if suffix not in {".pdb", ".ent", ".npy", ".npz"}:
            return InputReadiness(
                NEEDS_CONVERSION,
                "trajectory provided without reliable topology metadata",
                "A reliable topology is required for residue-level stable-core and RMSF analysis.",
                topology_format=inferred_topology,
                trajectory_format=inferred_trajectory,
            )

    if (topology is not None and inferred_topology is None) or ((trajectory or single_input) is not None and inferred_trajectory is None):
        return InputReadiness(
            NEEDS_CONVERSION,
            "format detection failed",
            "Format detection failed. Check file extension or use --topology-format / --trajectory-format.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )

    pdb_probe = single_input if single_input and single_input.suffix.lower() in {".pdb", ".ent"} else topology
    if pdb_probe is not None and pdb_probe.suffix.lower() in {".pdb", ".ent"} and pdb_probe.exists():
        resnames, ca_count, atom_count = _read_pdb_resnames(pdb_probe)
        if atom_count == 0:
            return InputReadiness(UNSUPPORTED, "no coordinate atoms found", "Provide a coordinate file with ATOM/HETATM records.")
        if ca_count == 0 and "name ca" in fit_selection.lower():
            if resnames and resnames.issubset(NUCLEIC_RESNAMES):
                return InputReadiness(
                    UNSUPPORTED,
                    "RNA/DNA-like input has no protein C-alpha atoms",
                    "RNA/DNA-only systems are not formally validated; use experimental mode only if you define a stable atom selection.",
                    topology_format=inferred_topology,
                    trajectory_format=inferred_trajectory,
                )
            return InputReadiness(
                UNSUPPORTED,
                "default protein C-alpha selection would be empty",
                "Use --fit-selection 'protein and name CA' for proteins, or provide an explicit validated atom selection for experimental non-protein use.",
                topology_format=inferred_topology,
                trajectory_format=inferred_trajectory,
            )

    if selection_size == 0:
        return InputReadiness(
            UNSUPPORTED,
            "selection returned zero atoms",
            "Use --fit-selection 'protein and name CA' for proteins, or define an explicit experimental selection.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )
    if nan_coordinates:
        return InputReadiness(
            NEEDS_PREPROCESSING,
            "coordinates contain NaN or infinite values",
            "Remove invalid frames or regenerate the trajectory before analysis.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )
    if large_jump_detected:
        return InputReadiness(
            NEEDS_PREPROCESSING,
            "large frame-to-frame displacement detected",
            "Run nojump/unwrap/center preprocessing before interpreting RMSF.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )
    if missing_chain_ids or duplicate_residue_labels:
        return InputReadiness(
            RUNNABLE_WITH_WARNINGS,
            "metadata are readable but chain/residue labels need inspection",
            "Inspect chain_mapping_report.tsv before biological interpretation.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )
    if inferred_trajectory in EXPERIMENTAL_TRAJECTORIES:
        return InputReadiness(
            RUNNABLE_WITH_WARNINGS,
            "backend-readable format is experimental",
            "Other MDAnalysis-readable formats may work if topology, atom order and residue metadata are valid, but they are not yet formally validated in the MDCoreAlign manuscript.",
            topology_format=inferred_topology,
            trajectory_format=inferred_trajectory,
        )
    return InputReadiness(
        READY_TO_RUN,
        "input appears compatible with the conservative protein CA workflow",
        "Review the preprocessing audit and interpret the completed analysis with any warnings in mind.",
        topology_format=inferred_topology,
        trajectory_format=inferred_trajectory,
    )
