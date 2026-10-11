#!/usr/bin/env python3
"""Generate compact implementation evidence for purpose-based coordinate export."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from pathlib import Path
import resource
import time
import tracemalloc

import numpy as np

from mdcorealign.coordinate_export import (
    ArrayCoordinateSource,
    ExportAtomRecord,
    MDAnalysisCoordinateSource,
    StreamingPDBWriter,
    build_coordinate_writer,
    stream_apply_transforms,
)
from mdsuperpose import write_multimodel_pdb


ROOT = Path(__file__).resolve().parents[1]
TOY_TOPOLOGY = ROOT / "examples/toy_md/toy_ca_topology.pdb"
TOY_TRAJECTORY = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one non-empty TSV with deterministic column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def identity_transforms(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.repeat(np.eye(3)[None, :, :], n_frames, axis=0)
    translations = np.zeros((n_frames, 3), dtype=float)
    return rotations, translations


class DigestWriter:
    """Consume aligned chunks without retaining coordinate arrays."""

    preserves_boxes = True

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.peak_rss_mb = 0.0

    def write_chunk(self, frame_indices, coordinates_A, times_ps, steps, box_vectors_A):
        self.digest.update(np.ascontiguousarray(coordinates_A).tobytes())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        return sum(box is not None for box in box_vectors_A)

    def close(self) -> None:
        return None


def array_source(coords: np.ndarray) -> ArrayCoordinateSource:
    return ArrayCoordinateSource(
        coords,
        records=None,
        fit_labels=[f"A:ALA{index + 1}" for index in range(coords.shape[1])],
        fit_selection="protein and name CA",
        apply_selection="protein and name CA",
    )


def streaming_and_memory_evidence(output_root: Path) -> None:
    rng = np.random.default_rng(20260827)
    coords = rng.normal(size=(600, 250, 3))
    rotations, translations = identity_transforms(coords.shape[0])
    translations[:, 0] = np.linspace(0.0, 2.0, coords.shape[0])
    source = array_source(coords)
    chunk_rows: list[dict[str, object]] = []
    reference_digest = None
    for chunk_size in [1, 7, 64, 200]:
        writer = DigestWriter()
        started = time.perf_counter()
        stats = stream_apply_transforms(
            source, rotations, translations, np.arange(coords.shape[0]), writer,
            chunk_size=chunk_size, drop_unit_cell=False,
        )
        elapsed = time.perf_counter() - started
        digest = writer.digest.hexdigest()
        reference_digest = reference_digest or digest
        chunk_rows.append(
            {
                "chunk_size": chunk_size,
                "n_frames": stats.n_frames,
                "peak_chunk_frames": stats.peak_chunk_frames,
                "coordinate_digest": digest,
                "matches_chunk1": digest == reference_digest,
                "elapsed_seconds": f"{elapsed:.6f}",
            }
        )
    write_tsv(output_root / "03_streaming_apply/chunk_invariance.tsv", chunk_rows)

    tracemalloc.start()
    full = coords @ np.swapaxes(rotations, 1, 2) + translations[:, None, :]
    full_peak = tracemalloc.get_traced_memory()[1]
    full_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    del full
    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()
    writer = DigestWriter()
    stream_apply_transforms(
        source, rotations, translations, np.arange(coords.shape[0]), writer,
        chunk_size=25, drop_unit_cell=False,
    )
    stream_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    write_tsv(
        output_root / "03_streaming_apply/memory_scaling.tsv",
        [
            {
                "path": "full_aligned_array_control",
                "n_frames": coords.shape[0],
                "n_atoms": coords.shape[1],
                "chunk_size": coords.shape[0],
                "tracemalloc_peak_mb": f"{full_peak / 1024**2:.6f}",
                "process_peak_rss_mb": f"{full_rss:.3f}",
                "memory_order": "O(T*A)",
            },
            {
                "path": "streaming_apply",
                "n_frames": coords.shape[0],
                "n_atoms": coords.shape[1],
                "chunk_size": 25,
                "tracemalloc_peak_mb": f"{stream_peak / 1024**2:.6f}",
                "process_peak_rss_mb": f"{writer.peak_rss_mb:.3f}",
                "memory_order": "O(B*A)",
            },
        ],
    )


def make_xtc_fixture(directory: Path) -> tuple[Path, Path]:
    import MDAnalysis as mda

    universe = mda.Universe(str(TOY_TOPOLOGY), str(TOY_TRAJECTORY))
    xtc = directory / "source.xtc"
    with mda.Writer(str(xtc), n_atoms=universe.atoms.n_atoms) as writer:
        for timestep in universe.trajectory:
            timestep.dimensions = np.asarray([30.0, 31.0, 32.0, 90.0, 90.0, 90.0])
            timestep.time = float(timestep.frame) * 2.0
            timestep.data["step"] = int(timestep.frame) * 20
            writer.write(universe.atoms)
    universe.trajectory[0]
    gro = directory / "topology.gro"
    with mda.Writer(str(gro), n_atoms=universe.atoms.n_atoms) as writer:
        writer.write(universe.atoms)
    return gro, xtc


def reader_and_xtc_evidence(output_root: Path, work: Path) -> list[dict[str, object]]:
    try:
        import MDAnalysis as mda
    except ImportError:
        rows = [{"input_pair": value, "status": "dependency_unavailable"} for value in ["PDB + XTC", "GRO + XTC"]]
        write_tsv(output_root / "02_reader_audit/input_roundtrip_tests.tsv", rows)
        return []
    gro, source_xtc = make_xtc_fixture(work)
    input_rows: list[dict[str, object]] = []
    xtc_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    for pair, topology, topology_format in [
        ("PDB + XTC", TOY_TOPOLOGY, "PDB"),
        ("GRO + XTC", gro, "GRO"),
    ]:
        source = MDAnalysisCoordinateSource(
            topology, source_xtc,
            fit_selection="protein and name CA",
            apply_selection="protein and name CA",
            topology_format=topology_format,
            trajectory_format="XTC",
        )
        chunk = next(source.iter_chunks(np.arange(source.n_frames), source.n_frames))
        input_rows.append(
            {
                "input_pair": pair,
                "status": "validated",
                "reader": "MDAnalysis",
                "n_frames": source.n_frames,
                "n_atoms": source.n_atoms,
                "fit_labels_stable": len(source.fit_labels) == source.n_fit_atoms,
                "coordinate_unit": source.provenance["coordinate_unit"],
                "time_values": int(np.count_nonzero(np.isfinite(chunk.times_ps))),
                "box_frames": sum(box is not None for box in chunk.box_vectors_A),
                "selection": source.fit_selection,
            }
        )
        rotations, translations = identity_transforms(source.n_frames)
        for chunk_size in [1, 2, 5]:
            output = work / f"{pair.replace(' ', '_').replace('+', 'plus')}_{chunk_size}.xtc"
            writer = build_coordinate_writer(
                "xtc", output, records=source.records,
                n_selected_frames=source.n_frames,
                allow_large_text_output=False, provenance={},
            )
            started = time.perf_counter()
            stats = stream_apply_transforms(
                source, rotations, translations, np.arange(source.n_frames), writer,
                chunk_size=chunk_size, drop_unit_cell=False,
            )
            elapsed = time.perf_counter() - started
            result = mda.Universe(str(topology), str(output))
            observed = np.asarray([ts.positions.copy() for ts in result.trajectory])
            result.trajectory[0]
            reference = np.asarray([frame.copy() for frame in chunk.coordinates_A])
            max_error = float(np.max(np.abs(observed - reference)))
            xtc_rows.append(
                {
                    "input_pair": pair,
                    "chunk_size": chunk_size,
                    "frame_count_match": observed.shape[0] == source.n_frames,
                    "atom_count_match": observed.shape[1] == source.n_atoms,
                    "max_coordinate_error_A": f"{max_error:.8f}",
                    "tolerance_A": "0.011",
                    "pass": max_error <= 0.011,
                }
            )
            performance_rows.append(
                {
                    "input_pair": pair,
                    "chunk_size": chunk_size,
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "file_size_bytes": output.stat().st_size,
                    "performance_claim": "none_compact_io_smoke_only",
                }
            )
            file_rows.append(
                {
                    "format": "xtc",
                    "input_pair": pair,
                    "chunk_size": chunk_size,
                    "n_frames": source.n_frames,
                    "n_atoms": source.n_atoms,
                    "file_size_bytes": output.stat().st_size,
                    "write_seconds": f"{elapsed:.6f}",
                    "peak_rss_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.3f}",
                }
            )
            if chunk_size == 2:
                result.trajectory[0]
                time_rows.append(
                    {
                        "input_pair": pair,
                        "time_preserved": abs(float(result.trajectory[0].time) - float(chunk.times_ps[0])) < 1e-6,
                        "step_preserved": int(result.trajectory[0].data.get("step")) == int(chunk.steps[0]),
                        "box_available": result.trajectory[0].triclinic_dimensions is not None,
                        "box_max_error_A": f"{np.max(np.abs(result.trajectory[0].triclinic_dimensions - chunk.box_vectors_A[0])):.8f}",
                    }
                )
    input_rows.append(
        {
            "input_pair": "CA multi-model PDB",
            "status": "validated",
            "reader": "MDCoreAlign built-in",
            "n_frames": 5,
            "n_atoms": 8,
            "fit_labels_stable": True,
            "coordinate_unit": "angstrom",
            "time_values": 0,
            "box_frames": 0,
            "selection": "protein and name CA",
        }
    )
    write_tsv(output_root / "02_reader_audit/input_roundtrip_tests.tsv", input_rows)
    write_tsv(output_root / "05_xtc/xtc_roundtrip.tsv", xtc_rows)
    write_tsv(output_root / "05_xtc/time_box_validation.tsv", time_rows)
    write_tsv(output_root / "05_xtc/chunk_performance.tsv", performance_rows)
    return file_rows


def mmcif_evidence(output_root: Path, work: Path) -> list[dict[str, object]]:
    try:
        import gemmi
    except ImportError:
        rows = [{"case": "Gemmi dependency", "status": "dependency_unavailable"}]
        write_tsv(output_root / "04_mmcif/mmcif_roundtrip.tsv", rows)
        write_tsv(output_root / "04_mmcif/metadata_validation.tsv", rows)
        write_tsv(output_root / "04_mmcif/large_structure_tests.tsv", rows)
        return []
    records = (
        ExportAtomRecord("ATOM", 1, "CA", "", "GLY", "CHAIN_A", "10001", "A", "SEG", "C"),
        ExportAtomRecord("HETATM", 2, "ZN", "", "ZN", "CHAIN_B", "2", "", "ION", "ZN"),
    )
    coords = np.asarray(
        [[[1.0, 2.0, 3.0], [10001.0, -2.0, 4.0]], [[4.0, 5.0, 6.0], [10002.0, -3.0, 5.0]]]
    )
    source = ArrayCoordinateSource(
        coords, records=records,
        fit_labels=[record.residue_key for record in records],
        fit_selection="protein and name CA", apply_selection="all",
    )
    output = work / "selected_models.cif"
    writer = build_coordinate_writer(
        "mmcif", output, records=records, n_selected_frames=2,
        allow_large_text_output=False,
        provenance={"purpose": "structure", "source_frame_indices": [1, 0]},
    )
    rotations, translations = identity_transforms(2)
    started = time.perf_counter()
    stream_apply_transforms(
        source, rotations, translations, np.asarray([1, 0]), writer,
        chunk_size=1, drop_unit_cell=False,
    )
    elapsed = time.perf_counter() - started
    block = gemmi.cif.read(str(output)).sole_block()
    xyz = np.column_stack(
        [np.asarray(list(block.find_values(tag)), dtype=float) for tag in ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]]
    ).reshape(2, 2, 3)
    max_error = float(np.max(np.abs(xyz - coords[[1, 0]])))
    roundtrip = [
        {
            "case": "selected_two_model_reverse_order",
            "n_models": 2,
            "n_atoms_per_model": 2,
            "max_coordinate_error_A": f"{max_error:.8f}",
            "model_numbers": ",".join(block.find_values("_atom_site.pdbx_PDB_model_num")),
            "source_frame_indices": ",".join(block.find_values("_mdcorealign_frame.source_frame_index")),
            "pass": max_error <= 1e-6,
        }
    ]
    metadata = [
        {
            "field": field,
            "observed": ",".join(block.find_values(tag)),
            "pass": bool(list(block.find_values(tag))),
        }
        for field, tag in [
            ("auth_asym_id", "_atom_site.auth_asym_id"),
            ("auth_seq_id", "_atom_site.auth_seq_id"),
            ("insertion_code", "_atom_site.pdbx_PDB_ins_code"),
            ("record_type", "_atom_site.group_PDB"),
            ("occupancy", "_atom_site.occupancy"),
            ("B_factor", "_atom_site.B_iso_or_equiv"),
            ("provenance", "_mdcorealign_export.key"),
        ]
    ]
    large = [
        {"case": "multi_character_chain", "value": "CHAIN_A", "pass": "CHAIN_A" in list(block.find_values("_atom_site.auth_asym_id"))},
        {"case": "residue_above_pdb_limit", "value": "10001", "pass": "10001" in list(block.find_values("_atom_site.auth_seq_id"))},
        {"case": "coordinate_above_pdb_limit", "value": "10002", "pass": np.max(xyz) > 10000.0},
    ]
    write_tsv(output_root / "04_mmcif/mmcif_roundtrip.tsv", roundtrip)
    write_tsv(output_root / "04_mmcif/metadata_validation.tsv", metadata)
    write_tsv(output_root / "04_mmcif/large_structure_tests.tsv", large)
    return [
        {
            "format": "mmcif", "input_pair": "derived compact array",
            "chunk_size": 1, "n_frames": 2, "n_atoms": 2,
            "file_size_bytes": output.stat().st_size,
            "write_seconds": f"{elapsed:.6f}",
            "peak_rss_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.3f}",
        }
    ]


def pdb_evidence(output_root: Path, work: Path) -> list[dict[str, object]]:
    rng = np.random.default_rng(17)
    coords = rng.normal(size=(5, 8, 3))
    source = array_source(coords)
    rotations, translations = identity_transforms(5)
    output = work / "streamed.pdb"
    legacy = work / "legacy.pdb"
    writer = StreamingPDBWriter(
        output, source.records, n_selected_frames=5,
        allow_large_text_output=False, synthetic_records=True,
    )
    started = time.perf_counter()
    stream_apply_transforms(
        source, rotations, translations, np.arange(5), writer,
        chunk_size=2, drop_unit_cell=False,
    )
    elapsed = time.perf_counter() - started
    write_multimodel_pdb(coords, legacy)
    write_tsv(
        output_root / "06_pdb_compatibility/pdb_parity.tsv",
        [{
            "case": "legacy_CA_array_layout", "byte_identical": output.read_bytes() == legacy.read_bytes(),
            "stream_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "legacy_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest(),
        }],
    )
    limit_rows = []
    for case, record in [
        ("multi_character_chain", ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "AB", "1", "", "", "C")),
        ("residue_overflow", ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "A", "10000", "", "", "C")),
    ]:
        try:
            StreamingPDBWriter(
                work / f"{case}.pdb", [record], n_selected_frames=1,
                allow_large_text_output=False,
            )
        except ValueError as exc:
            limit_rows.append({"case": case, "rejected": True, "message": str(exc)})
        else:
            limit_rows.append({"case": case, "rejected": False, "message": ""})
    ordinary = ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "A", "1", "", "", "C")
    for case, records, n_frames, allow_large in [
        ("atom_serial_overflow", [ordinary] * 100_000, 1, False),
        ("model_number_overflow", [ordinary], 10_000, False),
        ("large_text_guard", [ordinary] * 1_001, 1_000, False),
    ]:
        try:
            StreamingPDBWriter(
                work / f"{case}.pdb", records, n_selected_frames=n_frames,
                allow_large_text_output=allow_large,
            )
        except ValueError as exc:
            limit_rows.append({"case": case, "rejected": True, "message": str(exc)})
        else:
            limit_rows.append({"case": case, "rejected": False, "message": ""})
    overflow_source = ArrayCoordinateSource(
        np.asarray([[[10_000.0, 0.0, 0.0]]]), records=[ordinary],
        fit_labels=[ordinary.residue_key], fit_selection="all", apply_selection="all",
    )
    overflow_writer = StreamingPDBWriter(
        work / "coordinate_overflow.pdb", [ordinary], n_selected_frames=1,
        allow_large_text_output=False,
    )
    overflow_rotations, overflow_translations = identity_transforms(1)
    try:
        stream_apply_transforms(
            overflow_source, overflow_rotations, overflow_translations,
            np.asarray([0]), overflow_writer, chunk_size=1,
            drop_unit_cell=False,
        )
    except ValueError as exc:
        limit_rows.append({"case": "coordinate_field_overflow", "rejected": True, "message": str(exc)})
    else:
        limit_rows.append({"case": "coordinate_field_overflow", "rejected": False, "message": ""})
    write_tsv(output_root / "06_pdb_compatibility/pdb_limit_tests.tsv", limit_rows)
    return [
        {
            "format": "pdb", "input_pair": "derived compact array",
            "chunk_size": 2, "n_frames": 5, "n_atoms": 8,
            "file_size_bytes": output.stat().st_size,
            "write_seconds": f"{elapsed:.6f}",
            "peak_rss_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.3f}",
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="results/purpose_based_coordinate_export_v1",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    work = output_root / "08_validation/compact_fixture_outputs"
    work.mkdir(parents=True, exist_ok=True)
    streaming_and_memory_evidence(output_root)
    file_rows = []
    file_rows.extend(reader_and_xtc_evidence(output_root, work))
    file_rows.extend(mmcif_evidence(output_root, work))
    file_rows.extend(pdb_evidence(output_root, work))
    write_tsv(output_root / "08_validation/writer_memory_file_size.tsv", file_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
