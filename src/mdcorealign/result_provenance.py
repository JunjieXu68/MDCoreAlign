"""Versioned source provenance and safe source recovery for alignment results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

from .input_validation import PROTEIN_SELECTION_POLICY
from .output_schema import SCHEMA_VERSION

SOURCE_MANIFEST_SCHEMA_VERSION = "2.2.0"


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one source file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_fingerprint(values: Iterable[object]) -> str:
    """Hash an ordered sequence without relying on platform-specific repr output."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _path_record(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source file does not exist: {resolved}")
    return {
        "absolute_path": str(resolved),
        "basename_hint": resolved.name,
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def build_source_manifest(
    args: Any,
    prepared: Any,
    *,
    software_version: str,
) -> dict[str, object]:
    """Build the self-sufficient source record written by ``align``."""

    topology = _path_record(getattr(args, "topology", None))
    trajectory = _path_record(getattr(args, "trajectory", None))
    single = _path_record(getattr(args, "input", None))
    if single is not None:
        input_kind = "single_input"
    elif topology is not None and trajectory is not None:
        input_kind = "topology_trajectory"
    elif topology is not None:
        input_kind = "topology_as_trajectory"
        trajectory = dict(topology)
    else:
        raise ValueError("cannot register source provenance without an input file")

    topology_format = prepared.unit_provenance.get("topology_format_used", getattr(args, "topology_format", None))
    trajectory_format = prepared.unit_provenance.get("trajectory_format_used", getattr(args, "trajectory_format", None))
    input_format = prepared.input_format
    fit_labels = tuple(str(value) for value in prepared.fit_labels)
    source_hashes = [
        record["sha256"]
        for record in (single, topology, trajectory)
        if record is not None
    ]
    frame_fingerprint = sequence_fingerprint(
        [prepared.fit_trajectory.n_frames, *source_hashes]
    )
    canonical_export_selection = (
        "all_available_atoms"
        if prepared.source_is_ca_only
        else str(prepared.default_export_selection)
    )
    export_selection_expression = (
        "all"
        if prepared.source_is_ca_only
        else str(prepared.default_export_selection)
    )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_result_schema_version": SCHEMA_VERSION,
        "software_version": software_version,
        "input_kind": input_kind,
        "single_input": single,
        "topology": topology,
        "trajectory": trajectory,
        "single_input_path": None if single is None else single["absolute_path"],
        "topology_path": None if topology is None else topology["absolute_path"],
        "trajectory_path": None if trajectory is None else trajectory["absolute_path"],
        "input_sha256": None if single is None else single["sha256"],
        "topology_sha256": None if topology is None else topology["sha256"],
        "trajectory_sha256": None if trajectory is None else trajectory["sha256"],
        "input_format": input_format,
        "topology_format": topology_format,
        "trajectory_format": trajectory_format,
        "read_configuration": {
            "reader": "MDAnalysis" if prepared.unit_provenance.get("source_unit") == "backend_managed" else "builtin",
            "input_format": input_format,
            "topology_format": topology_format,
            "trajectory_format": trajectory_format,
            "array_key": getattr(args, "array_key", None),
            "chain": getattr(args, "chain", None),
            "fit_selection": str(prepared.fit_selection),
            "source_unit": getattr(args, "source_unit", "auto"),
            "unit_provenance": dict(prepared.unit_provenance),
            "source_frame_indices": list(range(prepared.fit_trajectory.n_frames)),
            "protein_selection_policy": PROTEIN_SELECTION_POLICY,
        },
        "source_metadata_kind": prepared.source_metadata_kind,
        "source_atom_identities": list(prepared.source_atom_identities),
        "source_atom_order_fingerprint": sequence_fingerprint(prepared.source_atom_identities),
        "fit_atom_order_fingerprint": sequence_fingerprint(prepared.fit_atom_identities),
        "n_frames": int(prepared.fit_trajectory.n_frames),
        "n_fit_atoms": int(prepared.fit_trajectory.n_atoms),
        "n_analysis_coordinate_atoms": int(prepared.fit_trajectory.n_atoms),
        "n_exportable_atoms": int(prepared.n_exportable_atoms),
        "fit_selection": str(prepared.fit_selection),
        "analysis_coordinate_selection": str(prepared.fit_selection),
        "default_export_selection": canonical_export_selection,
        "default_export_selection_expression": export_selection_expression,
        "source_is_ca_only": bool(prepared.source_is_ca_only),
        "legacy_metadata": {
            "analysis_loaded_selection": str(prepared.apply_selection),
            "n_analysis_loaded_atoms": int(prepared.apply_coords.shape[1]),
            "former_field_names": {
                "analysis_apply_selection": "analysis_loaded_selection",
                "n_apply_available_atoms": "n_exportable_atoms",
            },
        },
        "fit_labels_order_fingerprint": sequence_fingerprint(fit_labels),
        "fit_labels": list(fit_labels),
        "source_coordinate_unit": prepared.unit_provenance.get(
            "canonical_internal_unit", "unknown"
        ),
        "frame_order_fingerprint": frame_fingerprint,
        "source_files_copied_into_result": False,
    }


def write_source_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    """Write one deterministic source manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_source_manifest(result_dir: str | Path) -> dict[str, object]:
    """Load and minimally validate a source manifest."""

    path = Path(result_dir) / "diagnostics" / "source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"alignment result has no source manifest: {path}. "
            "For a legacy result without a source manifest, supply "
            "--topology/--trajectory or --input explicitly."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") not in {"2.1.0", SOURCE_MANIFEST_SCHEMA_VERSION}:
        raise ValueError("unsupported source manifest version; rerun align with the current version")
    required = {
        "schema_version",
        "input_kind",
        "n_frames",
        "n_fit_atoms",
        "n_analysis_coordinate_atoms",
        "n_exportable_atoms",
        "fit_selection",
        "analysis_coordinate_selection",
        "default_export_selection",
        "default_export_selection_expression",
        "source_is_ca_only",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            "source manifest is incomplete; missing field(s): "
            + ", ".join(sorted(missing))
        )
    return payload


@dataclass(frozen=True)
class ResolvedSourceFiles:
    """Validated source paths recovered for one ``apply`` operation."""

    single_input: Path | None
    topology: Path | None
    trajectory: Path | None
    resolution: dict[str, str]


def _candidate_paths(
    record: dict[str, object],
    *,
    result_dir: Path,
    explicit: str | Path | None,
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if explicit is not None:
        candidates.append(("explicit_override", Path(explicit).expanduser()))
    stored = record.get("absolute_path")
    if stored:
        candidates.append(("stored_absolute_path", Path(str(stored))))
    basename = str(record.get("basename_hint") or "")
    if basename and Path(basename).name == basename:
        candidates.extend(
            [
                ("result_directory", result_dir / basename),
                ("result_parent", result_dir.parent / basename),
            ]
        )
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for origin, candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append((origin, resolved))
    return unique


def _resolve_one(
    role: str,
    record: dict[str, object] | None,
    *,
    result_dir: Path,
    explicit: str | Path | None,
) -> tuple[Path | None, str]:
    if record is None:
        if explicit is None:
            return None, "not_applicable"
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"explicit {role} file does not exist: {candidate}")
        return candidate, "explicit_unregistered"
    expected_hash = str(record.get("sha256") or "")
    mismatches: list[str] = []
    for origin, candidate in _candidate_paths(
        record, result_dir=result_dir, explicit=explicit
    ):
        if not candidate.is_file():
            continue
        observed = file_sha256(candidate)
        if observed == expected_hash:
            return candidate, origin
        mismatches.append(f"{candidate} ({observed})")
        if origin == "explicit_override":
            raise ValueError(
                f"explicit {role} hash does not match the alignment source; "
                f"expected {expected_hash}, observed {observed}"
            )
    detail = ""
    if mismatches:
        detail = " Hash-mismatched candidate(s): " + "; ".join(mismatches)
    raise FileNotFoundError(
        f"could not recover the original {role} file with SHA-256 {expected_hash}."
        f"{detail} Supply moved files explicitly, for example: "
        "mdcorealign apply RESULT_DIR -o aligned.pdb --topology /new/path/system.gro "
        "--trajectory /new/path/trajectory.xtc"
    )


def resolve_source_files(
    manifest: dict[str, object],
    result_dir: str | Path,
    *,
    input_override: str | Path | None = None,
    topology_override: str | Path | None = None,
    trajectory_override: str | Path | None = None,
) -> ResolvedSourceFiles:
    """Resolve original inputs in the registered precedence order and verify hashes."""

    root = Path(result_dir).resolve()
    kind = str(manifest["input_kind"])
    resolution: dict[str, str] = {}
    if kind == "single_input":
        single, resolution["single_input"] = _resolve_one(
            "single input",
            cast(dict[str, object] | None, manifest.get("single_input")),
            result_dir=root,
            explicit=input_override,
        )
        if topology_override or trajectory_override:
            raise ValueError(
                "this result was created from one self-contained input; use --input for a moved source"
            )
        return ResolvedSourceFiles(single, None, None, resolution)

    if input_override:
        raise ValueError(
            "this result was created from topology+trajectory input; use --topology and --trajectory"
        )
    topology, resolution["topology"] = _resolve_one(
        "topology",
        cast(dict[str, object] | None, manifest.get("topology")),
        result_dir=root,
        explicit=topology_override,
    )
    trajectory, resolution["trajectory"] = _resolve_one(
        "trajectory",
        cast(dict[str, object] | None, manifest.get("trajectory")),
        result_dir=root,
        explicit=trajectory_override,
    )
    return ResolvedSourceFiles(None, topology, trajectory, resolution)
