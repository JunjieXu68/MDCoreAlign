"""Shared utilities for the scientific-evidence upgrade study.

This module is intentionally outside the installable package. It provides
deterministic file handling and mdCATH input access without changing any
production MDCoreAlign implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "scientific_evidence_upgrade_v1"
MDCATH_SNAPSHOT = (
    REPO_ROOT
    / "results/mdcath_benchmark/cache/datasets--compsciencelab--mdCATH"
    / "snapshots/5e3ed8aec62b689e01751db16275fdcdbc39e47f/data"
)
B3_MANIFEST = (
    REPO_ROOT
    / "results/database_scale_benchmark_v2/b2_contracts"
    / "b3_all_trajectory_manifest_locked.tsv"
)


def relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one file without retaining it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    """Return a deterministic SHA-256 key used for prospective ordering."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table as dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write a deterministic tab-separated table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: object) -> None:
    """Write canonical indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_mdcath_h5(domain_id: str) -> Path:
    """Resolve a locally cached mdCATH HDF5 file for one domain."""

    path = MDCATH_SNAPSHOT / f"mdcath_dataset_{domain_id}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return str(value)


def ca_atom_metadata(h5: h5py.File, domain_id: str) -> dict[str, np.ndarray]:
    """Extract CA indices and residue metadata from mdCATH's embedded PDB."""

    pdb_text = _decode_scalar(h5[domain_id]["pdbProteinAtoms"][()])
    atom_indices: list[int] = []
    residue_ids: list[int] = []
    residue_names: list[str] = []
    chains: list[str] = []
    atom_index = 0
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            atom_indices.append(atom_index)
            residue_names.append(line[17:20].strip() or "UNK")
            chain = line[21:22].strip()
            chains.append(chain or "_")
            try:
                residue_ids.append(int(line[22:26]))
            except ValueError:
                residue_ids.append(len(residue_ids) + 1)
        atom_index += 1
    expected = int(h5[domain_id].attrs["numResidues"])
    if len(atom_indices) != expected:
        raise ValueError(
            f"{domain_id}: embedded PDB contains {len(atom_indices)} CA atoms; "
            f"expected {expected}"
        )
    return {
        "atom_indices": np.asarray(atom_indices, dtype=np.int64),
        "residue_ids": np.asarray(residue_ids, dtype=np.int64),
        "residue_names": np.asarray(residue_names, dtype="U3"),
        "chain_ids": np.asarray(chains, dtype="U4"),
    }


def load_mdcath_ca(
    domain_id: str,
    temperature: int = 320,
    replicate_zero_based: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load CA coordinates plus bundled RMSD/RMSF arrays from local mdCATH."""

    path = resolve_mdcath_h5(domain_id)
    with h5py.File(path, "r") as h5:
        metadata = ca_atom_metadata(h5, domain_id)
        group = h5[domain_id][str(int(temperature))][str(int(replicate_zero_based))]
        coords = np.asarray(
            group["coords"][:, metadata["atom_indices"], :],
            dtype=np.float64,
        )
        bundled_rmsd = np.asarray(group["rmsd"], dtype=np.float64)
        bundled_rmsf = np.asarray(group["rmsf"], dtype=np.float64)
    return coords, metadata, bundled_rmsd, bundled_rmsf


def core_mask_exact_fraction(scores: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Select exactly floor(fraction*N) residues with deterministic tie breaks."""

    values = np.asarray(scores, dtype=np.float64)
    count = max(1, int(np.floor(float(fraction) * values.size)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def rank_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation with average ranks and no SciPy dependency."""

    def ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranked[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranked

    a = ranks(left)
    b = ranks(right)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
