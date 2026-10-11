"""Purpose-based, bounded-memory coordinate export for saved transforms."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .input_validation import select_mdanalysis_atoms

PURPOSE_DEFAULT_FORMAT = {
    "structure": "mmcif",
    "trajectory": "xtc",
    "compatibility": "pdb",
}

PURPOSE_FORMATS = {
    "structure": frozenset({"mmcif", "pdb"}),
    "trajectory": frozenset({"xtc"}),
    "compatibility": frozenset({"pdb"}),
}

TEXT_OUTPUT_VALUE_LIMIT = 1_000_000


@dataclass(frozen=True)
class ExportAtomRecord:
    """Portable atom metadata required by PDB and PDBx/mmCIF writers."""

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
        """Return the chain-aware residue label used by transform archives."""

        return f"{self.chain or '_'}:{self.resname}{self.resid}{self.icode}".strip()


@dataclass(frozen=True)
class ExportFrameChunk:
    """One bounded coordinate chunk with optional trajectory metadata."""

    frame_indices: np.ndarray
    coordinates_A: np.ndarray
    times_ps: np.ndarray
    steps: np.ndarray
    box_vectors_A: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class ExportStatistics:
    """Summary of one streaming transform-application run."""

    n_frames: int
    n_atoms: int
    peak_chunk_frames: int
    box_frames_seen: int
    box_frames_written: int
    time_values_seen: int
    step_values_seen: int


@dataclass(frozen=True)
class ResidueAnnotation:
    """One authoritative row loaded from ``residue_profile.tsv``."""

    chain_id: str
    residue_id: str
    insertion_code: str
    residue_name: str
    stable_weight: float
    core_label: int
    mobility_score: float
    robust_rmsf_angstrom: float
    mean_residual_angstrom: float
    flexible_segment_id: int


@dataclass(frozen=True)
class AtomAnnotation:
    """Residue-level visualization values propagated to one exported atom."""

    occupancy: float
    b_factor: float
    residue: ResidueAnnotation


def pdb_compatibility_issue(
    records: Sequence[ExportAtomRecord],
    annotations: Sequence[AtomAnnotation] | None = None,
    *,
    n_models: int = 1,
) -> str | None:
    """Return the first metadata reason that requires PDBx/mmCIF instead of PDB."""

    if len(records) > 99_999:
        return "more than 99,999 atoms"
    if n_models > 9_999:
        return "more than 9,999 models"
    for record in records:
        if len(record.chain) > 1:
            return "a multi-character chain identifier"
        resid = str(record.resid).strip()
        if resid.lstrip("-").isdigit() and not (-999 <= int(resid) <= 9999):
            return "a residue number outside the PDB four-column field"
    if annotations is not None:
        for annotation in annotations:
            if not 0.0 <= annotation.occupancy <= 1.0:
                return "an occupancy annotation outside [0, 1]"
            if not 0.0 <= annotation.b_factor < 1000.0:
                return "a robust RMSF outside the PDB B-factor field"
    return None


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_export_contract(
    purpose: str | None,
    output_format: str | None,
) -> tuple[str, str, bool]:
    """Resolve purpose and format while preserving the legacy PDB default."""

    legacy = purpose is None and output_format is None
    if purpose is None:
        if output_format is None or output_format == "pdb":
            purpose = "compatibility"
        elif output_format == "mmcif":
            purpose = "structure"
        elif output_format == "xtc":
            purpose = "trajectory"
        else:
            raise ValueError(f"unsupported coordinate output format: {output_format}")
    if purpose not in PURPOSE_DEFAULT_FORMAT:
        raise ValueError("purpose must be one of: structure, trajectory, compatibility")
    output_format = output_format or PURPOSE_DEFAULT_FORMAT[purpose]
    if output_format not in PURPOSE_FORMATS[purpose]:
        allowed = ", ".join(sorted(PURPOSE_FORMATS[purpose]))
        raise ValueError(
            f"purpose '{purpose}' does not support format '{output_format}'; "
            f"supported format(s): {allowed}"
        )
    return purpose, output_format, legacy


def parse_frame_indices(
    n_frames: int,
    *,
    purpose: str,
    frames: str | None = None,
    frame_range: str | None = None,
    stride: int = 1,
    all_selected_frames: bool = False,
) -> np.ndarray:
    """Resolve deterministic zero-based frame indices for coordinate export."""

    if n_frames < 1:
        raise ValueError("input trajectory must contain at least one frame")
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    if frames and frame_range:
        raise ValueError("use only one of --frames or --frame-range")
    if frames:
        try:
            requested = [int(value.strip()) for value in frames.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("--frames must be a comma-separated list of zero-based integers") from exc
        if not requested:
            raise ValueError("--frames did not contain any frame indices")
        indices = requested
    elif frame_range:
        parts = frame_range.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("--frame-range must use START:STOP or START:STOP:STEP")
        try:
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else n_frames
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        except ValueError as exc:
            raise ValueError("--frame-range values must be integers") from exc
        if step < 1:
            raise ValueError("--frame-range step must be positive")
        indices = list(range(start, stop, step))
    elif purpose == "structure" and not all_selected_frames:
        indices = [0]
    else:
        indices = list(range(n_frames))
    indices = indices[::stride]
    if not indices:
        raise ValueError("frame selection is empty")
    if min(indices) < 0 or max(indices) >= n_frames:
        raise ValueError(f"frame indices must be between 0 and {n_frames - 1}")
    if len(set(indices)) != len(indices):
        raise ValueError("frame selection contains duplicate indices")
    return np.asarray(indices, dtype=np.int64)


def validate_transforms(
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite right-handed rigid transforms for every source frame."""

    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if rotations.shape != (n_frames, 3, 3):
        raise ValueError("rotations must have shape (n_frames, 3, 3) matching the input")
    if translations.shape != (n_frames, 3):
        raise ValueError("translations must have shape (n_frames, 3) matching the input")
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("transforms contain NaN or infinite values")
    identity = np.eye(3, dtype=np.float64)
    orthogonality_error = np.max(
        np.abs(rotations @ np.swapaxes(rotations, 1, 2) - identity[None, :, :])
    )
    determinants = np.linalg.det(rotations)
    if orthogonality_error > 1e-5 or np.max(np.abs(determinants - 1.0)) > 1e-5:
        raise ValueError("rotations must be orthogonal, right-handed matrices")
    return rotations, translations


def _value_or_default(obj: object, name: str, default: object) -> object:
    try:
        value = getattr(obj, name)
    except (AttributeError, RuntimeError, ValueError):
        return default
    return default if value is None else value


def atom_record_from_object(record: object, index: int) -> ExportAtomRecord:
    """Convert CLI or MDAnalysis atom metadata into a writer-neutral record."""

    return ExportAtomRecord(
        record_name=str(_value_or_default(record, "record_name", "ATOM")),
        serial=int(_value_or_default(record, "serial", index + 1)),
        atom_name=str(_value_or_default(record, "atom_name", "CA")),
        altloc=str(_value_or_default(record, "altloc", "")),
        resname=str(_value_or_default(record, "resname", "UNK")),
        chain=str(_value_or_default(record, "chain", "")),
        resid=str(_value_or_default(record, "resid", index + 1)),
        icode=str(_value_or_default(record, "icode", "")),
        segid=str(_value_or_default(record, "segid", "")),
        element=str(_value_or_default(record, "element", "C")),
        occupancy=float(_value_or_default(record, "occupancy", 1.0)),
        b_factor=float(
            _value_or_default(
                record,
                "b_factor",
                _value_or_default(record, "tempfactor", 0.0),
            )
        ),
    )


def atom_identity(record: object) -> str:
    """Serialize source atom identity, including order-sensitive atom metadata."""

    return json.dumps([
        str(getattr(record, field, ""))
        for field in ("record_name", "atom_name", "altloc", "resname", "chain", "resid", "icode", "segid")
    ], separators=(",", ":"))


def synthetic_ca_records(n_atoms: int) -> tuple[ExportAtomRecord, ...]:
    """Label coordinate rows explicitly as synthetic CA/UNK, not real residues."""

    return tuple(
        ExportAtomRecord("ATOM", index + 1, "CA", "", "UNK", "A", str(index + 1), "", "", "C")
        for index in range(n_atoms)
    )


class ArrayCoordinateSource:
    """Expose an in-memory input array through the common chunk source API."""

    def __init__(
        self,
        coordinates_A: np.ndarray,
        *,
        records: Sequence[object] | None,
        fit_labels: Sequence[str],
        fit_selection: str,
        apply_selection: str,
        times_ps: np.ndarray | None = None,
        steps: np.ndarray | None = None,
        box_vectors_A: Sequence[np.ndarray | None] | None = None,
        provenance: dict[str, object] | None = None,
    ) -> None:
        values = np.asarray(coordinates_A)
        if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] < 1:
            raise ValueError("coordinates must have shape (n_frames, n_atoms, 3)")
        if values.shape[1] < 1:
            raise ValueError("coordinates must contain at least one atom")
        if not np.isfinite(values).all():
            raise ValueError("input coordinates contain NaN or infinite values")
        self._coordinates_A = values
        self.n_frames = int(values.shape[0])
        self.n_atoms = int(values.shape[1])
        self.records_are_synthetic = records is None
        self.records = tuple(
            atom_record_from_object(record, index)
            for index, record in enumerate(records)
        ) if records is not None else synthetic_ca_records(self.n_atoms)
        if len(self.records) != self.n_atoms:
            raise ValueError("atom metadata count does not match apply-selection coordinates")
        self.fit_labels = tuple(str(value) for value in fit_labels)
        self.n_fit_atoms = len(self.fit_labels)
        self.fit_selection = fit_selection
        self.apply_selection = apply_selection
        self.provenance = dict(provenance or {})
        self.provenance.setdefault("reader", "MDCoreAlign array/PDB reader")
        self.provenance.setdefault("coordinate_unit", "angstrom")
        self.provenance.setdefault("metadata_kind", "synthetic_array" if self.records_are_synthetic else "topology")
        self.apply_atom_identities = tuple(atom_identity(record) for record in self.records)
        self._times_ps = _optional_vector(times_ps, self.n_frames, "times_ps", dtype=float)
        self._steps = _optional_vector(steps, self.n_frames, "steps", dtype=np.int64)
        if box_vectors_A is None:
            self._boxes = tuple(None for _ in range(self.n_frames))
        else:
            if len(box_vectors_A) != self.n_frames:
                raise ValueError("box-vector metadata must match the number of frames")
            self._boxes = tuple(_validated_box(box) for box in box_vectors_A)

    def iter_chunks(self, indices: np.ndarray, chunk_size: int) -> Iterator[ExportFrameChunk]:
        """Yield selected input frames in bounded chunks."""

        for start in range(0, indices.size, chunk_size):
            selected = indices[start : start + chunk_size]
            yield ExportFrameChunk(
                frame_indices=selected.copy(),
                coordinates_A=np.asarray(self._coordinates_A[selected], dtype=np.float64),
                times_ps=(
                    np.full(selected.size, np.nan, dtype=float)
                    if self._times_ps is None
                    else np.asarray(self._times_ps[selected], dtype=float)
                ),
                steps=(
                    selected.copy()
                    if self._steps is None
                    else np.asarray(self._steps[selected], dtype=np.int64)
                ),
                box_vectors_A=tuple(self._boxes[int(index)] for index in selected),
            )


def _optional_vector(
    values: np.ndarray | None,
    n_frames: int,
    name: str,
    *,
    dtype: type,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=dtype)
    if array.shape != (n_frames,):
        raise ValueError(f"{name} must have shape (n_frames,)")
    return array


def _validated_box(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        return None
    if np.any(np.linalg.norm(values, axis=1) <= 0.0):
        return None
    return values.copy()


def _mda_atom_records(atoms: object) -> tuple[ExportAtomRecord, ...]:
    records: list[ExportAtomRecord] = []
    for index, atom in enumerate(atoms):
        chain = str(
            _value_or_default(atom, "chainID", "")
            or _value_or_default(atom, "segid", "")
            or ""
        )
        records.append(
            ExportAtomRecord(
                record_name=str(_value_or_default(atom, "record_type", "ATOM")),
                serial=int(_value_or_default(atom, "id", index + 1)),
                atom_name=str(_value_or_default(atom, "name", "CA")),
                altloc=str(_value_or_default(atom, "altLoc", "")),
                resname=str(_value_or_default(atom, "resname", "UNK")),
                chain=chain,
                resid=str(_value_or_default(atom, "resid", index + 1)),
                icode=str(_value_or_default(atom, "icode", "")),
                segid=str(_value_or_default(atom, "segid", "")),
                element=str(_value_or_default(atom, "element", "")),
                occupancy=float(_value_or_default(atom, "occupancy", 1.0)),
                b_factor=float(_value_or_default(atom, "tempfactor", 0.0)),
            )
        )
    return tuple(records)


def _profile_residue_parts(value: str, insertion_code: str = "") -> tuple[str, str]:
    """Split a profile residue identifier into auth sequence ID and insertion code."""

    text = str(value).strip()
    if insertion_code:
        return text, insertion_code.strip()
    match = re.fullmatch(r"(-?\d+)([^\d].*)?", text)
    if match is None:
        return text, ""
    return match.group(1), (match.group(2) or "")


def load_residue_annotations(
    path: str | Path,
) -> tuple[ResidueAnnotation, ...]:
    """Load and validate the authoritative residue profile for coordinate export."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"alignment result has no residue profile: {profile_path}"
        )
    with profile_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "chain",
        "residue_id",
        "resname",
        "stable_weight",
        "core_label",
        "mobility_score",
        "robust_RMSF",
        "mean_residual",
        "flexible_segment_id",
    }
    if not rows:
        raise ValueError("residue_profile.tsv is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            "residue_profile.tsv is missing required field(s): "
            + ", ".join(sorted(missing))
        )
    annotations: list[ResidueAnnotation] = []
    for row in rows:
        residue_id, insertion_code = _profile_residue_parts(
            row["residue_id"], row.get("insertion_code", "")
        )
        core_label = int(row["core_label"])
        if core_label not in {0, 1}:
            raise ValueError("core_label must be binary for visualization export")
        annotation = ResidueAnnotation(
            chain_id=str(row["chain"] or "_"),
            residue_id=residue_id,
            insertion_code=insertion_code,
            residue_name=str(row["resname"] or "UNK").upper(),
            stable_weight=float(row["stable_weight"]),
            core_label=core_label,
            mobility_score=float(row["mobility_score"]),
            robust_rmsf_angstrom=float(row["robust_RMSF"]),
            mean_residual_angstrom=float(row["mean_residual"]),
            flexible_segment_id=int(row["flexible_segment_id"] or 0),
        )
        numerical = (
            annotation.stable_weight,
            annotation.mobility_score,
            annotation.robust_rmsf_angstrom,
            annotation.mean_residual_angstrom,
        )
        if not np.isfinite(numerical).all():
            raise ValueError("residue profile contains non-finite annotation values")
        annotations.append(annotation)
    return tuple(annotations)


def _annotation_key(
    chain: str,
    residue_name: str,
    residue_id: str,
    insertion_code: str,
) -> tuple[str, str, str, str]:
    normalized_chain = "" if chain in {"", "_", ".", "?"} else chain
    return (
        normalized_chain,
        residue_name.upper(),
        str(residue_id).strip(),
        str(insertion_code).strip(),
    )


def map_annotations_to_atoms(
    records: Sequence[ExportAtomRecord],
    residue_annotations: Sequence[ResidueAnnotation],
) -> tuple[AtomAnnotation, ...]:
    """Map residue annotations by validated chain/residue identity, never array index."""

    by_key: dict[tuple[str, str, str, str], ResidueAnnotation] = {}
    for annotation in residue_annotations:
        key = _annotation_key(
            annotation.chain_id,
            annotation.residue_name,
            annotation.residue_id,
            annotation.insertion_code,
        )
        if key in by_key:
            raise ValueError(
                "residue annotation mapping is ambiguous for "
                f"{annotation.chain_id}:{annotation.residue_name}{annotation.residue_id}"
            )
        by_key[key] = annotation

    mapped: list[AtomAnnotation] = []
    unmatched: set[str] = set()
    used: set[tuple[str, str, str, str]] = set()
    for record in records:
        candidate_chains = [record.chain]
        if record.segid and record.segid not in candidate_chains:
            candidate_chains.append(record.segid)
        annotation = None
        matched_key = None
        for chain in candidate_chains:
            key = _annotation_key(
                chain,
                record.resname,
                record.resid,
                record.icode,
            )
            if key in by_key:
                annotation = by_key[key]
                matched_key = key
                break
        if annotation is None:
            unmatched.add(record.residue_key)
            continue
        used.add(matched_key)
        mapped.append(
            AtomAnnotation(
                occupancy=float(annotation.core_label),
                b_factor=float(annotation.robust_rmsf_angstrom),
                residue=annotation,
            )
        )
    if unmatched:
        preview = ", ".join(sorted(unmatched)[:10])
        raise ValueError(
            "residue annotation mapping is incomplete for exported protein residues: "
            f"{preview}. Use the original topology or export mmCIF after correcting metadata."
        )
    unused = set(by_key).difference(used)
    if unused:
        preview = ", ".join(
            f"{chain or '_'}:{name}{resid}{icode}"
            for chain, name, resid, icode in sorted(unused)[:10]
        )
        raise ValueError(
            "the source topology does not contain every analysed residue: " + preview
        )
    return tuple(mapped)


class MDAnalysisCoordinateSource:
    """Read selected topology/trajectory frames lazily through MDAnalysis."""

    def __init__(
        self,
        topology: str | Path,
        trajectory: str | Path,
        *,
        fit_selection: str,
        apply_selection: str,
        topology_format: str | None = None,
        trajectory_format: str | None = None,
        chain_id: str | None = None,
    ) -> None:
        try:
            import MDAnalysis as mda
        except ImportError as exc:  # pragma: no cover - installation integrity boundary
            raise ValueError(
                "The standard MDCoreAlign installation includes MDAnalysis. "
                "This environment appears incomplete. Reinstall the package with "
                "`python -m pip install --force-reinstall <wheel-path>`."
            ) from exc
        kwargs: dict[str, str] = {}
        if topology_format:
            kwargs["topology_format"] = topology_format
        if trajectory_format:
            kwargs["format"] = trajectory_format
        try:
            self._universe = mda.Universe(str(topology), str(trajectory), **kwargs)
        except Exception as exc:
            raise ValueError(
                "Could not read topology and trajectory for streaming export; "
                "check format overrides, atom counts, and file integrity."
            ) from exc
        self._fit_atoms = select_mdanalysis_atoms(self._universe, fit_selection, chain=chain_id)
        self._apply_atoms = select_mdanalysis_atoms(self._universe, apply_selection)
        if self._fit_atoms.n_atoms == 0:
            raise ValueError(f"fit-selection '{fit_selection}' returned zero atoms")
        if self._apply_atoms.n_atoms == 0:
            raise ValueError(f"apply-selection '{apply_selection}' returned zero atoms")
        self.records = _mda_atom_records(self._apply_atoms)
        self.records_are_synthetic = False
        fit_records = _mda_atom_records(self._fit_atoms)
        self.source_atom_identities = tuple(atom_identity(r) for r in _mda_atom_records(self._universe.atoms))
        self.fit_atom_identities = tuple(atom_identity(r) for r in fit_records)
        self.apply_atom_identities = tuple(atom_identity(r) for r in self.records)
        self.fit_labels = tuple(record.residue_key for record in fit_records)
        self.n_fit_atoms = int(self._fit_atoms.n_atoms)
        self.n_atoms = int(self._apply_atoms.n_atoms)
        self.n_frames = int(len(self._universe.trajectory))
        self.fit_selection = fit_selection
        self.apply_selection = apply_selection
        self.provenance = {
            "reader": "MDAnalysis",
            "coordinate_unit": "angstrom",
            "topology": str(topology),
            "trajectory": str(trajectory),
            "topology_format_override": topology_format,
            "trajectory_format_override": trajectory_format,
        }

    def iter_chunks(self, indices: np.ndarray, chunk_size: int) -> Iterator[ExportFrameChunk]:
        """Decode only selected frames into bounded Angstrom coordinate chunks."""

        for start in range(0, indices.size, chunk_size):
            selected = indices[start : start + chunk_size]
            coords = np.empty((selected.size, self.n_atoms, 3), dtype=np.float64)
            times = np.full(selected.size, np.nan, dtype=np.float64)
            steps = np.empty(selected.size, dtype=np.int64)
            boxes: list[np.ndarray | None] = []
            for offset, frame_index in enumerate(selected):
                timestep = self._universe.trajectory[int(frame_index)]
                coords[offset] = np.asarray(self._apply_atoms.positions, dtype=np.float64)
                try:
                    times[offset] = float(timestep.time)
                except (AttributeError, ValueError, TypeError):
                    pass
                steps[offset] = int(timestep.data.get("step", timestep.frame))
                boxes.append(_validated_box(getattr(timestep, "triclinic_dimensions", None)))
            if not np.isfinite(coords).all():
                raise ValueError("input coordinates contain NaN or infinite values")
            yield ExportFrameChunk(
                frame_indices=selected.copy(),
                coordinates_A=coords,
                times_ps=times,
                steps=steps,
                box_vectors_A=tuple(boxes),
            )


class CoordinateWriter:
    """Minimal writer protocol implemented by all coordinate exporters."""

    preserves_boxes = False

    def write_chunk(
        self,
        frame_indices: np.ndarray,
        coordinates_A: np.ndarray,
        times_ps: np.ndarray,
        steps: np.ndarray,
        box_vectors_A: Sequence[np.ndarray | None],
    ) -> int:
        """Write one aligned chunk and return the number of boxes preserved."""

        raise NotImplementedError

    def close(self) -> None:
        """Finalize the output."""

        raise NotImplementedError


def _format_pdb_atom_line(
    record: ExportAtomRecord,
    serial: int,
    xyz: np.ndarray,
    *,
    occupancy: float = 1.0,
    b_factor: float = 0.0,
) -> str:
    chain = (record.chain or " ")[:1]
    resid_text = str(record.resid).strip()
    resid = int(resid_text) if resid_text.lstrip("-").isdigit() else serial
    element = (record.element or record.atom_name[:1] or "C")[:2].rjust(2)
    return (
        f"{record.record_name:<6}{serial:5d} {record.atom_name[:4]:>4}{record.altloc[:1]:1s}"
        f"{record.resname[:3]:>3} {chain:1s}{resid:4d}{record.icode[:1]:1s}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {element}\n"
    )


class StreamingPDBWriter(CoordinateWriter):
    """Write the established multi-model PDB representation one chunk at a time."""

    def __init__(
        self,
        path: str | Path,
        records: Sequence[ExportAtomRecord],
        *,
        n_selected_frames: int,
        allow_large_text_output: bool,
        synthetic_records: bool = False,
        annotations: Sequence[AtomAnnotation] | None = None,
        provenance: dict[str, object] | None = None,
    ) -> None:
        self.path = Path(path)
        self.records = tuple(records)
        self.annotations = None if annotations is None else tuple(annotations)
        if self.annotations is not None and len(self.annotations) != len(self.records):
            raise ValueError("atom annotation count does not match exported atom count")
        self._generic_ca_layout = bool(synthetic_records and self.annotations is None)
        self.provenance = dict(provenance or {})
        _validate_text_output_size(
            len(self.records), n_selected_frames, allow_large_text_output, "PDB"
        )
        issue = pdb_compatibility_issue(
            self.records, self.annotations, n_models=n_selected_frames
        )
        if issue is not None:
            if issue == "more than 99,999 atoms":
                raise ValueError("PDB supports at most 99,999 atom serials; use PDBx/mmCIF")
            if issue == "more than 9,999 models":
                raise ValueError("multi-model PDB supports at most 9,999 model numbers; use XTC")
            if "multi-character" in issue:
                raise ValueError("PDB cannot preserve multi-character chain IDs; use PDBx/mmCIF")
            if "residue number" in issue:
                raise ValueError("PDB residue number is outside the 4-column field; use PDBx/mmCIF")
            raise ValueError(f"PDB cannot safely represent {issue}; use PDBx/mmCIF")
        if self._generic_ca_layout and len(self.records) > 9_999:
            raise ValueError("legacy CA-only PDB supports at most 9,999 residues; use PDBx/mmCIF")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", buffering=1024 * 1024)
        if self.annotations is not None:
            core_fraction = float(
                self.provenance.get(
                    "reporting_core_residue_fraction",
                    np.mean([annotation.occupancy for annotation in self.annotations]),
                )
            )
            remarks = [
                "REMARK 950 MDCOREALIGN VISUALIZATION ANNOTATION",
                "REMARK 950 OCCUPANCY = REPORTING CORE LABEL: 1.00 CORE, 0.00 NON-CORE",
                "REMARK 950 B-FACTOR = STABLE-CORE-ALIGNED RMSF IN ANGSTROM",
                "REMARK 950 AUTHORITATIVE VALUES ARE STORED IN residue_profile.tsv",
                "REMARK 950 THESE ARE ANALYSIS ANNOTATIONS, NOT EXPERIMENTAL OCCUPANCIES",
                "REMARK 950 OR CRYSTALLOGRAPHIC TEMPERATURE FACTORS",
                f"REMARK 950 SOFTWARE VERSION = {self.provenance.get('software_version', 'unknown')}",
                f"REMARK 950 EXPORT SCHEMA VERSION = {self.provenance.get('export_schema_version', '2.0.0')}",
                f"REMARK 950 CORE RESIDUE FRACTION = {core_fraction:.6f}",
                f"REMARK 950 FIT SELECTION = {self.provenance.get('fit_selection', 'unknown')}",
                f"REMARK 950 EXPORT SELECTION = {self.provenance.get('export_selection', self.provenance.get('apply_selection', 'unknown'))}",
                f"REMARK 950 SOURCE RESULT SHA256 = {self.provenance.get('result_sha256', 'unknown')}",
            ]
            self._handle.write("\n".join(remarks) + "\n")
        if synthetic_records:
            self._handle.write("REMARK 950 SYNTHETIC CA/UNK COORDINATE ROWS; NO RESIDUE IDENTITY PROVIDED\n")

    def write_chunk(
        self,
        frame_indices: np.ndarray,
        coordinates_A: np.ndarray,
        times_ps: np.ndarray,
        steps: np.ndarray,
        box_vectors_A: Sequence[np.ndarray | None],
    ) -> int:
        for frame_index, frame in zip(frame_indices, coordinates_A, strict=True):
            model_number = int(frame_index) + 1
            if self._generic_ca_layout:
                self._handle.write(f"MODEL     {model_number}\n")
            else:
                self._handle.write(f"MODEL     {model_number:4d}\n")
            for atom_index, (record, xyz) in enumerate(
                zip(self.records, frame, strict=True), start=1
            ):
                if not np.isfinite(xyz).all() or np.any(np.abs(xyz) >= 10_000.0):
                    raise ValueError("PDB 8.3 coordinate fields require finite absolute values < 10000")
                if self._generic_ca_layout:
                    self._handle.write(
                        f"ATOM  {atom_index:5d}  CA  UNK A{atom_index:4d}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        "  1.00  0.00           C\n"
                    )
                else:
                    annotation = (
                        None if self.annotations is None else self.annotations[atom_index - 1]
                    )
                    occupancy = record.occupancy if annotation is None else annotation.occupancy
                    b_factor = record.b_factor if annotation is None else annotation.b_factor
                    if not 0.0 <= occupancy <= 1.0:
                        raise ValueError("PDB occupancy annotation must be between 0 and 1")
                    if not 0.0 <= b_factor < 1000.0:
                        raise ValueError(
                            "PDB B-factor field cannot safely represent this robust RMSF; use PDBx/mmCIF"
                        )
                    self._handle.write(
                        _format_pdb_atom_line(
                            record,
                            atom_index,
                            xyz,
                            occupancy=occupancy,
                            b_factor=b_factor,
                        )
                    )
            self._handle.write("TER\nENDMDL\n")
        return 0

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.write("END\n")
            self._handle.close()


def _validate_text_output_size(
    n_atoms: int,
    n_frames: int,
    allow_large: bool,
    format_name: str,
) -> None:
    if n_atoms * n_frames > TEXT_OUTPUT_VALUE_LIMIT and not allow_large:
        raise ValueError(
            f"{format_name} text output would contain {n_atoms * n_frames:,} atom rows. "
            "Use XTC for full trajectories or pass --allow-large-text-output explicitly."
        )


class GemmiMMCIFWriter(CoordinateWriter):
    """Write metadata-preserving PDBx/mmCIF structure models with Gemmi."""

    preserves_boxes = True

    def __init__(
        self,
        path: str | Path,
        records: Sequence[ExportAtomRecord],
        *,
        n_selected_frames: int,
        allow_large_text_output: bool,
        provenance: dict[str, object],
        annotations: Sequence[AtomAnnotation] | None = None,
        residue_profile: Sequence[ResidueAnnotation] | None = None,
    ) -> None:
        try:
            import gemmi
        except ImportError as exc:  # pragma: no cover - installation integrity boundary
            raise ValueError(
                "The standard MDCoreAlign installation includes Gemmi. "
                "This environment appears incomplete. Reinstall the package with "
                "`python -m pip install --force-reinstall <wheel-path>`."
            ) from exc
        _validate_text_output_size(
            len(records), n_selected_frames, allow_large_text_output, "PDBx/mmCIF"
        )
        self._gemmi = gemmi
        self.path = Path(path)
        self.records = tuple(records)
        self.annotations = None if annotations is None else tuple(annotations)
        if self.annotations is not None and len(self.annotations) != len(self.records):
            raise ValueError("atom annotation count does not match exported atom count")
        self._residue_profile = tuple(residue_profile or ())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._provenance_rows = [
            [
                str(key),
                json.dumps(value, sort_keys=True, separators=(",", ":")),
            ]
            for key, value in sorted(provenance.items())
        ]
        self._frame_rows: list[list[str]] = []
        self._box_rows: list[list[str]] = []
        self._atom_rows: list[list[str]] = []
        self._row_id = 0
        self._model_count = 0

    def write_chunk(
        self,
        frame_indices: np.ndarray,
        coordinates_A: np.ndarray,
        times_ps: np.ndarray,
        steps: np.ndarray,
        box_vectors_A: Sequence[np.ndarray | None],
    ) -> int:
        boxes_written = 0
        for offset, (frame_index, frame) in enumerate(
            zip(frame_indices, coordinates_A, strict=True)
        ):
            self._model_count += 1
            model_number = self._model_count
            time_value = "." if not np.isfinite(times_ps[offset]) else f"{times_ps[offset]:.9g}"
            self._frame_rows.append(
                [str(model_number), str(int(frame_index)), time_value, str(int(steps[offset]))]
            )
            box = box_vectors_A[offset]
            if box is not None:
                for axis, vector in zip(("a", "b", "c"), box, strict=True):
                    self._box_rows.append(
                        [str(model_number), axis, *(f"{value:.9g}" for value in vector)]
                    )
                boxes_written += 1
            for atom_index, (record, xyz) in enumerate(
                zip(self.records, frame, strict=True), start=1
            ):
                self._row_id += 1
                label_seq_id = str(record.resid) if str(record.resid).lstrip("-").isdigit() else str(atom_index)
                chain = record.chain or record.segid or "A"
                element = (record.element or record.atom_name[:1] or "C").strip().upper()
                self._atom_rows.append(
                    [
                        record.record_name or "ATOM", str(self._row_id), element,
                        record.atom_name or "CA", record.altloc or ".",
                        record.resname or "UNK", chain, "1", label_seq_id,
                        record.icode or "?", f"{xyz[0]:.6f}", f"{xyz[1]:.6f}",
                        f"{xyz[2]:.6f}",
                        f"{(record.occupancy if self.annotations is None else self.annotations[atom_index - 1].occupancy):.6f}",
                        f"{(record.b_factor if self.annotations is None else self.annotations[atom_index - 1].b_factor):.6f}",
                        str(record.resid),
                        record.resname or "UNK", record.chain or ".",
                        record.atom_name or "CA", str(model_number),
                    ]
                )
        return boxes_written

    def close(self) -> None:
        document = self._gemmi.cif.Document()
        block = document.add_new_block("mdcorealign_export")
        provenance_loop = block.init_loop(
            "_mdcorealign_export.", ["key", "value"]
        )
        for row in self._provenance_rows:
            provenance_loop.add_row([row[0], self._gemmi.cif.quote(row[1])])
        frame_loop = block.init_loop(
            "_mdcorealign_frame.",
            ["model_number", "source_frame_index", "time_ps", "step"],
        )
        for row in self._frame_rows:
            frame_loop.add_row(row)
        if self._box_rows:
            box_loop = block.init_loop(
                "_mdcorealign_box_vector.",
                ["model_number", "axis", "x_angstrom", "y_angstrom", "z_angstrom"],
            )
            for row in self._box_rows:
                box_loop.add_row(row)
        if self._residue_profile:
            profile_loop = block.init_loop(
                "_mdcorealign_residue_profile.",
                [
                    "schema_version",
                    "chain_id",
                    "residue_id",
                    "insertion_code",
                    "residue_name",
                    "stable_weight",
                    "core_label",
                    "mobility_score",
                    "robust_rmsf_angstrom",
                    "mean_residual_angstrom",
                    "flexible_segment_id",
                ],
            )
            for annotation in self._residue_profile:
                profile_loop.add_row(
                    [
                        "2.0.0",
                        annotation.chain_id or ".",
                        annotation.residue_id,
                        annotation.insertion_code or "?",
                        annotation.residue_name,
                        f"{annotation.stable_weight:.12g}",
                        str(annotation.core_label),
                        f"{annotation.mobility_score:.12g}",
                        f"{annotation.robust_rmsf_angstrom:.12g}",
                        f"{annotation.mean_residual_angstrom:.12g}",
                        str(annotation.flexible_segment_id),
                    ]
                )
            source_loop = block.init_loop(
                "_mdcorealign_source_atom_metadata.",
                ["atom_index", "original_occupancy", "original_b_iso_or_equiv"],
            )
            for index, record in enumerate(self.records, start=1):
                source_loop.add_row(
                    [str(index), f"{record.occupancy:.6g}", f"{record.b_factor:.6g}"]
                )
        atom_loop = block.init_loop(
            "_atom_site.",
            [
                "group_PDB", "id", "type_symbol", "label_atom_id",
                "label_alt_id", "label_comp_id", "label_asym_id",
                "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code",
                "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
                "B_iso_or_equiv", "auth_seq_id", "auth_comp_id",
                "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num",
            ],
        )
        for row in self._atom_rows:
            atom_loop.add_row(row)
        document.write_file(str(self.path))


class MDAnalysisXTCWriter(CoordinateWriter):
    """Write aligned trajectory chunks as XTC using MDAnalysis unit conversion."""

    preserves_boxes = True

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        try:
            from MDAnalysis.lib.formats.libmdaxdr import XTCFile
        except ImportError as exc:  # pragma: no cover - installation integrity boundary
            raise ValueError(
                "The standard MDCoreAlign installation includes MDAnalysis. "
                "This environment appears incomplete. Reinstall the package with "
                "`python -m pip install --force-reinstall <wheel-path>`."
            ) from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n_atoms = int(n_atoms)
        self._writer = XTCFile(str(self.path), "w")

    def write_chunk(
        self,
        frame_indices: np.ndarray,
        coordinates_A: np.ndarray,
        times_ps: np.ndarray,
        steps: np.ndarray,
        box_vectors_A: Sequence[np.ndarray | None],
    ) -> int:
        boxes_written = 0
        for offset, frame in enumerate(coordinates_A):
            coordinates_nm = np.asarray(frame, dtype=np.float32) * np.float32(0.1)
            if coordinates_nm.shape != (self._n_atoms, 3):
                raise ValueError("XTC coordinate frame does not match the writer atom count")
            time_ps = (
                float(times_ps[offset])
                if np.isfinite(times_ps[offset])
                else float(frame_indices[offset])
            )
            box = box_vectors_A[offset]
            if box is None:
                box_nm = np.zeros((3, 3), dtype=np.float32)
            else:
                box_nm = np.asarray(box, dtype=np.float32) * np.float32(0.1)
                boxes_written += 1
            self._writer.write(
                coordinates_nm,
                box_nm,
                int(steps[offset]),
                time_ps,
                1000.0,
            )
        return boxes_written

    def close(self) -> None:
        self._writer.close()


def build_coordinate_writer(
    output_format: str,
    output_path: str | Path,
    *,
    records: Sequence[ExportAtomRecord],
    n_selected_frames: int,
    allow_large_text_output: bool,
    provenance: dict[str, object],
    synthetic_records: bool = False,
    annotations: Sequence[AtomAnnotation] | None = None,
    residue_profile: Sequence[ResidueAnnotation] | None = None,
) -> CoordinateWriter:
    """Build one optional-dependency writer for a resolved export contract."""

    if output_format == "pdb":
        return StreamingPDBWriter(
            output_path,
            records,
            n_selected_frames=n_selected_frames,
            allow_large_text_output=allow_large_text_output,
            synthetic_records=synthetic_records,
            annotations=annotations,
            provenance=provenance,
        )
    if output_format == "mmcif":
        return GemmiMMCIFWriter(
            output_path,
            records,
            n_selected_frames=n_selected_frames,
            allow_large_text_output=allow_large_text_output,
            provenance=provenance,
            annotations=annotations,
            residue_profile=residue_profile,
        )
    if output_format == "xtc":
        return MDAnalysisXTCWriter(output_path, len(records))
    raise ValueError(f"unsupported coordinate output format: {output_format}")


def stream_apply_transforms(
    source: ArrayCoordinateSource | MDAnalysisCoordinateSource,
    rotations: np.ndarray,
    translations: np.ndarray,
    frame_indices: np.ndarray,
    writer: CoordinateWriter,
    *,
    chunk_size: int,
    drop_unit_cell: bool,
) -> ExportStatistics:
    """Apply saved transforms and stream bounded aligned chunks to a writer."""

    if chunk_size < 1:
        raise ValueError("chunk size must be a positive integer")
    rotations, translations = validate_transforms(
        rotations, translations, n_frames=source.n_frames
    )
    n_written = 0
    peak_chunk_frames = 0
    box_frames_seen = 0
    box_frames_written = 0
    time_values_seen = 0
    step_values_seen = 0
    try:
        for chunk in source.iter_chunks(frame_indices, chunk_size):
            selected_rotations = rotations[chunk.frame_indices]
            selected_translations = translations[chunk.frame_indices]
            aligned = (
                chunk.coordinates_A @ np.swapaxes(selected_rotations, 1, 2)
                + selected_translations[:, None, :]
            )
            transformed_boxes: list[np.ndarray | None] = []
            for box, rotation in zip(
                chunk.box_vectors_A, selected_rotations, strict=True
            ):
                if box is None:
                    transformed_boxes.append(None)
                else:
                    box_frames_seen += 1
                    transformed_boxes.append(None if drop_unit_cell else box @ rotation.T)
            box_frames_written += writer.write_chunk(
                chunk.frame_indices,
                aligned,
                chunk.times_ps,
                chunk.steps,
                transformed_boxes,
            )
            n_written += int(chunk.frame_indices.size)
            peak_chunk_frames = max(peak_chunk_frames, int(chunk.frame_indices.size))
            time_values_seen += int(np.count_nonzero(np.isfinite(chunk.times_ps)))
            step_values_seen += int(chunk.steps.size)
    finally:
        writer.close()
    return ExportStatistics(
        n_frames=n_written,
        n_atoms=source.n_atoms,
        peak_chunk_frames=peak_chunk_frames,
        box_frames_seen=box_frames_seen,
        box_frames_written=box_frames_written,
        time_values_seen=time_values_seen,
        step_values_seen=step_values_seen,
    )


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    """Write one deterministic JSON sidecar."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rows_from_records(records: Iterable[ExportAtomRecord]) -> list[dict[str, object]]:
    """Return writer metadata rows for validation and review reports."""

    return [
        {
            "atom_index": index,
            "record_name": record.record_name,
            "atom_name": record.atom_name,
            "resname": record.resname,
            "chain": record.chain,
            "resid": record.resid,
            "icode": record.icode,
            "segid": record.segid,
            "element": record.element,
        }
        for index, record in enumerate(records, start=1)
    ]
