"""Export formal MDCoreAlign CA trajectories as THESEUS multi-model PDB inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submission_critical_common import FORMAL_SYSTEMS, load_formal_system  # noqa: E402


STEMS = {
    "ifabp": "ifabp",
    "adk_equilibrium": "adk",
    "nhaa_equilibrium": "nhaa",
    "yiip_equilibrium_short": "yiip_short",
}


def sha256_file(path: Path) -> str:
    """Return a SHA256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chain(value: object) -> str:
    """Return a single-character chain ID."""

    text = str(value or "A").strip()
    return (text[0] if text else "A").upper()


def _resid(value: object, fallback: int) -> int:
    """Return an integer residue number accepted by PDB format."""

    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return fallback
    return int(digits[-4:])


def write_metadata_pdb(coords: np.ndarray, metadata_rows: list[dict[str, object]], path: Path) -> None:
    """Write a metadata-aware CA-only multi-model PDB file."""

    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_atoms, 3)")
    if coords.shape[1] != len(metadata_rows):
        raise ValueError("metadata length must match number of selected atoms")

    with path.open("w") as handle:
        for model_index, frame in enumerate(coords, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            for atom_index, (xyz, meta) in enumerate(zip(frame, metadata_rows, strict=True), start=1):
                serial = ((atom_index - 1) % 99999) + 1
                resname = str(meta.get("resname", "ALA") or "ALA").upper()[:3]
                chain = _chain(meta.get("chain", "A"))
                resid = _resid(meta.get("residue_id", atom_index), atom_index)
                handle.write(
                    f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resid:4d}    "
                    f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
                    "  1.00  0.00           C\n"
                )
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the THESEUS input manifest."""

    fields = [
        "system",
        "input_pdb",
        "n_frames",
        "n_residues",
        "atom_selection",
        "residue_correspondence_check",
        "input_sha256",
        "source_trajectory",
        "source_topology",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reviewer2_hardening" / "theseus_rerun" / "inputs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "reviewer2_hardening" / "theseus_rerun" / "theseus_input_manifest.tsv",
    )
    return parser.parse_args()


def main() -> int:
    """Export all formal systems."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for key in FORMAL_SYSTEMS:
        system = load_formal_system(key)
        stem = STEMS[key]
        output_path = args.output_dir / f"{stem}_ca_multimodel.pdb"
        write_metadata_pdb(system.trajectory.coords, system.metadata_rows, output_path)
        correspondence_ok = (
            len(system.metadata_rows) == system.trajectory.coords.shape[1]
            and system.trajectory.coords.ndim == 3
            and system.trajectory.coords.shape[0] == int(system.metadata["n_frames"])
        )
        manifest_rows.append(
            {
                "system": system.label,
                "input_pdb": output_path,
                "n_frames": int(system.trajectory.coords.shape[0]),
                "n_residues": int(system.trajectory.coords.shape[1]),
                "atom_selection": system.metadata["selection"],
                "residue_correspondence_check": "pass" if correspondence_ok else "fail",
                "input_sha256": sha256_file(output_path),
                "source_trajectory": system.metadata["trajectory_path"],
                "source_topology": system.metadata["topology_path"],
                "notes": "CA-only multi-model PDB exported from the same formal benchmark selection",
            }
        )
    write_manifest(args.manifest, manifest_rows)
    print(f"Wrote THESEUS inputs to {args.output_dir}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
