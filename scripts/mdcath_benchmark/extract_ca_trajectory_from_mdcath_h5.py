#!/usr/bin/env python
"""Extract a CA-only trajectory from one mdCATH HDF5 file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    find_atom_name_ca_indices,
    find_coordinate_datasets,
    find_pdb_text,
    import_h5py,
    maybe_convert_units_to_angstrom,
    normalize_coordinate_array,
    parse_domain_id_from_h5_path,
    parse_pdb_ca_metadata,
    read_dataset_text,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-file", type=Path, required=True)
    parser.add_argument("--coordinate-path", default="auto", help="HDF5 coordinate dataset path or auto.")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, help="Optional frame cap after extraction.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--min-ca-atoms", type=int, default=30)
    parser.add_argument("--min-frames", type=int, default=20)
    return parser.parse_args()


def extract_ca_trajectory(
    h5_file: Path,
    *,
    coordinate_path: str = "auto",
    max_frames: int | None = None,
    stride: int = 1,
    min_ca_atoms: int = 30,
    min_frames: int = 20,
    contiguous_full_atom_read: bool = False,
    direct_contiguous_read: bool = False,
) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, object]]:
    """Extract CA coordinates and residue metadata from one HDF5 file."""

    h5py = import_h5py()
    with h5py.File(h5_file, "r") as handle:
        if coordinate_path == "auto" or coordinate_path == "schema_pending":
            candidates = find_coordinate_datasets(handle)
            if not candidates:
                raise ValueError("missing_coordinates: no numeric (..., atoms, 3) coordinate datasets found")
            coordinate_path = str(candidates[0]["path"])
        if coordinate_path not in handle:
            stripped = coordinate_path.lstrip("/")
            if stripped in handle:
                coordinate_path = stripped
            else:
                raise ValueError(f"missing_coordinates: coordinate path not found: {coordinate_path}")
        dataset = handle[coordinate_path]
        attrs = {str(key): dataset.attrs[key] for key in dataset.attrs.keys()}
        if stride <= 0:
            raise ValueError("stride must be positive")

        domain_id = parse_domain_id_from_h5_path(h5_file)
        direct_pdb_path = f"/{domain_id}/pdb"
        if direct_pdb_path in handle:
            pdb_path = direct_pdb_path
            pdb_text = read_dataset_text(handle[direct_pdb_path])
        else:
            pdb_path, pdb_text = find_pdb_text(handle)
        if pdb_text:
            ca_indices, labels, residue_metadata = parse_pdb_ca_metadata(pdb_text)
            metadata_source = f"pdb_text:{pdb_path}"
        else:
            if len(dataset.shape) < 2:
                raise ValueError("missing_atom_metadata: invalid coordinate shape")
            ca_indices, labels, residue_metadata, atom_name_path = find_atom_name_ca_indices(
                handle, int(dataset.shape[-2])
            )
            metadata_source = f"atom_names:{atom_name_path}" if atom_name_path else "unavailable"
            if ca_indices is None:
                raise ValueError("missing_atom_metadata: no PDB text or atom-name CA metadata found")
        if len(dataset.shape) == 3 and dataset.shape[-1] == 3:
            stop = int(dataset.shape[0])
            if max_frames is not None:
                stop = min(stop, int(max_frames) * stride)
            if (
                direct_contiguous_read
                and dataset.chunks is None
                and dataset.compression is None
                and dataset.id.get_offset() is not None
            ):
                # Read verified contiguous bytes directly, then apply the same
                # ordered CA indices and float conversion as the HDF5 path.
                frame_shape = (stop, int(dataset.shape[1]), 3)
                value_count = int(np.prod(frame_shape))
                byte_count = value_count * dataset.dtype.itemsize
                descriptor = os.open(h5_file, os.O_RDONLY)
                try:
                    raw = os.pread(
                        descriptor,
                        byte_count,
                        int(dataset.id.get_offset()),
                    )
                finally:
                    os.close(descriptor)
                if len(raw) != byte_count:
                    raise IOError(
                        "direct contiguous HDF5 read returned "
                        f"{len(raw)}/{byte_count} bytes"
                    )
                full_coords = np.frombuffer(
                    raw,
                    dtype=dataset.dtype,
                    count=value_count,
                ).reshape(frame_shape)
                full_coords = full_coords[::stride]
                coords = np.asarray(full_coords[:, ca_indices, :], dtype=float)
                del full_coords, raw
                read_strategy = "direct_contiguous_bytes_then_CA"
            elif contiguous_full_atom_read and dataset.chunks is None:
                # Contiguous mdCATH datasets are much faster to read once
                # sequentially than through thousands of HDF5 point selections.
                full_coords = np.asarray(dataset[slice(0, stop, stride)])
                coords = np.asarray(full_coords[:, ca_indices, :], dtype=float)
                del full_coords
                read_strategy = "contiguous_full_atom_then_CA"
            else:
                # The default preserves the lower-memory historical behavior.
                coords = np.asarray(
                    dataset[slice(0, stop, stride), ca_indices, :],
                    dtype=float,
                )
                read_strategy = "HDF5_CA_fancy_index"
            original_n_atoms = int(dataset.shape[1])
        else:
            coords = normalize_coordinate_array(dataset[()])
            original_n_atoms = int(coords.shape[1])
            coords = coords[::stride]
            if max_frames is not None:
                coords = coords[:max_frames]
            coords = coords[:, ca_indices, :]
            read_strategy = "normalized_full_dataset_then_CA"
        coords, unit_status = maybe_convert_units_to_angstrom(coords, attrs)

    if ca_indices.shape[0] < min_ca_atoms:
        raise ValueError(f"too_few_ca_atoms: {ca_indices.shape[0]} < {min_ca_atoms}")
    if coords.shape[0] < min_frames:
        raise ValueError(f"too_few_frames: {coords.shape[0]} < {min_frames}")
    if int(np.max(ca_indices)) >= original_n_atoms:
        raise ValueError("ca_selection_failed: CA indices exceed coordinate atom dimension")
    ca_coords = coords
    if not np.isfinite(ca_coords).all():
        raise ValueError("coordinate_nan_or_inf: CA trajectory contains non-finite values")
    metadata = {
        "domain_id": parse_domain_id_from_h5_path(h5_file),
        "h5_file": str(h5_file),
        "coordinate_path": coordinate_path,
        "metadata_source": metadata_source,
        "unit_status": unit_status,
        "n_frames": int(ca_coords.shape[0]),
        "n_ca_atoms": int(ca_coords.shape[1]),
        "original_n_atoms": original_n_atoms,
        "stride": int(stride),
        "max_frames": max_frames,
        "read_strategy": read_strategy,
        "labels": labels,
        "residue_metadata": residue_metadata,
    }
    return ca_coords, labels, residue_metadata, metadata


def main() -> None:
    """Run extraction and write NPZ/JSON outputs."""

    args = parse_args()
    ca_coords, labels, residue_metadata, metadata = extract_ca_trajectory(
        args.h5_file,
        coordinate_path=args.coordinate_path,
        max_frames=args.max_frames,
        stride=args.stride,
        min_ca_atoms=args.min_ca_atoms,
        min_frames=args.min_frames,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, coords=ca_coords, labels=np.asarray(labels, dtype=object))
    write_json(args.metadata_json, metadata)
    print(args.output_npz)


if __name__ == "__main__":
    main()
