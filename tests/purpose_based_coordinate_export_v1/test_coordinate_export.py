"""Focused tests for purpose-based coordinate export."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from mdcorealign.cli import _load_pdb_models, run
from mdcorealign.coordinate_export import (
    ArrayCoordinateSource,
    ExportAtomRecord,
    MDAnalysisXTCWriter,
    StreamingPDBWriter,
    build_coordinate_writer,
    parse_frame_indices,
    resolve_export_contract,
    stream_apply_transforms,
    validate_transforms,
)
from mdsuperpose import write_multimodel_pdb

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"


def _identity_transforms(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(np.eye(3)[None, :, :], n_frames, axis=0), np.zeros((n_frames, 3))


def _source(coords: np.ndarray, *, boxes=None, times=None, steps=None) -> ArrayCoordinateSource:
    return ArrayCoordinateSource(
        coords,
        records=None,
        fit_labels=[f"A:ALA{index + 1}" for index in range(coords.shape[1])],
        fit_selection="protein and name CA",
        apply_selection="protein and name CA",
        box_vectors_A=boxes,
        times_ps=times,
        steps=steps,
    )


def test_purpose_defaults_and_conflicts() -> None:
    assert resolve_export_contract(None, None) == ("compatibility", "pdb", True)
    assert resolve_export_contract("structure", None) == ("structure", "mmcif", False)
    assert resolve_export_contract("trajectory", None) == ("trajectory", "xtc", False)
    with pytest.raises(ValueError, match="does not support"):
        resolve_export_contract("trajectory", "pdb")


def test_frame_selection_is_explicit_and_deterministic() -> None:
    np.testing.assert_array_equal(
        parse_frame_indices(10, purpose="structure"), np.asarray([0])
    )
    np.testing.assert_array_equal(
        parse_frame_indices(10, purpose="trajectory", stride=3), np.asarray([0, 3, 6, 9])
    )
    np.testing.assert_array_equal(
        parse_frame_indices(10, purpose="structure", frames="1,4,8"), np.asarray([1, 4, 8])
    )
    with pytest.raises(ValueError, match="only one"):
        parse_frame_indices(10, purpose="structure", frames="1", frame_range="0:2")


def test_transform_validation_rejects_reflection() -> None:
    rotations, translations = _identity_transforms(2)
    rotations[1, 0, 0] = -1.0
    with pytest.raises(ValueError, match="right-handed"):
        validate_transforms(rotations, translations, n_frames=2)


def test_transform_and_box_convention_matches_row_vector_reference(tmp_path: Path) -> None:
    class CaptureWriter:
        preserves_boxes = True

        def __init__(self) -> None:
            self.coords = None
            self.boxes = None

        def write_chunk(self, frame_indices, coordinates_A, times_ps, steps, box_vectors_A):
            self.coords = coordinates_A.copy()
            self.boxes = tuple(box_vectors_A)
            return sum(box is not None for box in box_vectors_A)

        def close(self) -> None:
            return None

    coords = np.asarray([[[1.0, 0.0, 0.0]]])
    box = np.asarray([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    source = _source(coords, boxes=[box])
    rotation = np.asarray([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    translation = np.asarray([[2.0, 3.0, 4.0]])
    writer = CaptureWriter()
    stream_apply_transforms(
        source, rotation, translation, np.asarray([0]), writer,
        chunk_size=1, drop_unit_cell=False,
    )
    np.testing.assert_allclose(writer.coords[0, 0], [2.0, 4.0, 4.0])
    np.testing.assert_allclose(writer.boxes[0], box @ rotation[0].T)


def test_streaming_pdb_preserves_legacy_coordinates_without_inventing_residues(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(4, 6, 3))
    rotations, translations = _identity_transforms(4)
    source = _source(coords)
    current = tmp_path / "streamed.pdb"
    legacy = tmp_path / "legacy.pdb"
    writer = StreamingPDBWriter(
        current,
        source.records,
        n_selected_frames=4,
        allow_large_text_output=False,
        synthetic_records=True,
    )
    stats = stream_apply_transforms(
        source,
        rotations,
        translations,
        np.arange(4),
        writer,
        chunk_size=2,
        drop_unit_cell=False,
    )
    write_multimodel_pdb(coords, legacy)
    # The legacy numerical writer invents ALA metadata. Public array exports
    # now declare synthetic CA/UNK rows while preserving the coordinate/model
    # layout and rounding contract, including the compatibility path.
    current_coords, current_records = _load_pdb_models(current)
    legacy_coords, _ = _load_pdb_models(legacy)
    np.testing.assert_array_equal(current_coords, legacy_coords)
    assert {record.resname for record in current_records} == {"UNK"}
    assert {record.atom_name for record in current_records} == {"CA"}
    assert "SYNTHETIC CA/UNK" in current.read_text()
    assert stats.peak_chunk_frames == 2
    assert stats.n_frames == 4


def test_pdb_preserves_atom_metadata_and_selected_model_numbers(tmp_path: Path) -> None:
    coords = np.arange(3 * 2 * 3, dtype=float).reshape(3, 2, 3) / 10.0
    records = (
        ExportAtomRecord("ATOM", 11, "CA", "", "GLY", "B", "42", "", "SEG", "C"),
        ExportAtomRecord("HETATM", 12, "ZN", "", "ZN", "B", "90", "", "SEG", "ZN"),
    )
    source = ArrayCoordinateSource(
        coords,
        records=records,
        fit_labels=("B:GLY42", "B:ZN90"),
        fit_selection="all",
        apply_selection="all",
    )
    output = tmp_path / "metadata.pdb"
    rotations, translations = _identity_transforms(3)
    writer = StreamingPDBWriter(
        output,
        source.records,
        n_selected_frames=2,
        allow_large_text_output=False,
    )
    stream_apply_transforms(
        source,
        rotations,
        translations,
        np.asarray([0, 2]),
        writer,
        chunk_size=1,
        drop_unit_cell=False,
    )
    text = output.read_text()
    assert "MODEL        1" in text
    assert "MODEL        3" in text
    assert "GLY B  42" in text
    assert "HETATM" in text


def test_xtc_roundtrip_preserves_coordinates_time_and_rotated_box(tmp_path: Path) -> None:
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.XTC import XTCReader
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    coords = np.asarray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ],
        dtype=float,
    )
    boxes = [np.diag([20.0, 30.0, 40.0]), np.diag([21.0, 31.0, 41.0])]
    source = _source(
        coords,
        boxes=boxes,
        times=np.asarray([2.5, 5.0]),
        steps=np.asarray([10, 20]),
    )
    rotations, translations = _identity_transforms(2)
    rotations[1] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translations[0] = [0.5, -0.5, 1.0]
    translations[1] = [2.0, 3.0, 4.0]
    output = tmp_path / "aligned.xtc"
    writer = MDAnalysisXTCWriter(output, n_atoms=2)
    stats = stream_apply_transforms(
        source,
        rotations,
        translations,
        np.arange(2),
        writer,
        chunk_size=1,
        drop_unit_cell=False,
    )
    with XTCReader(str(output), n_atoms=2, convert_units=True) as reader:
        observed = []
        observed_times = []
        observed_steps = []
        observed_boxes = []
        for timestep in reader:
            observed.append(timestep.positions.copy())
            observed_times.append(timestep.time)
            observed_steps.append(timestep.data.get("step"))
            observed_boxes.append(timestep.triclinic_dimensions.copy())
    with XTCFile(str(output), "r") as reader:
        raw_boxes_A = np.asarray([frame.box.copy() * 10.0 for frame in reader])
    expected_coords = coords @ np.swapaxes(rotations, 1, 2) + translations[:, None, :]
    expected_boxes = np.asarray(boxes) @ np.swapaxes(rotations, 1, 2)
    np.testing.assert_allclose(observed, expected_coords, atol=1.1e-2)
    np.testing.assert_allclose(observed_times, [2.5, 5.0], atol=1e-5)
    assert observed_steps == [10, 20]
    np.testing.assert_allclose(raw_boxes_A, expected_boxes, atol=2e-5)
    assert np.all(np.linalg.norm(observed_boxes, axis=2) > 0.0)
    assert stats.box_frames_written == 2
    assert mda.__version__


def test_mmcif_preserves_multichar_chain_and_provenance(tmp_path: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    coords = np.asarray(
        [
            [[1.0, 2.0, 3.0], [10001.0, -2.0, 4.0]],
            [[4.0, 5.0, 6.0], [10002.0, -3.0, 5.0]],
        ]
    )
    records = (
        ExportAtomRecord("ATOM", 1, "CA", "", "GLY", "CHAIN_A", "10001", "A", "SEGX", "C"),
        ExportAtomRecord("HETATM", 2, "ZN", "", "ZN", "CHAIN_B", "2", "", "ION", "ZN"),
    )
    source = ArrayCoordinateSource(
        coords,
        records=records,
        fit_labels=tuple(record.residue_key for record in records),
        fit_selection="protein and name CA",
        apply_selection="all",
    )
    output = tmp_path / "structure.cif"
    writer = build_coordinate_writer(
        "mmcif",
        output,
        records=source.records,
        n_selected_frames=2,
        allow_large_text_output=False,
        provenance={"purpose": "structure", "test": True},
    )
    rotations, translations = _identity_transforms(2)
    stream_apply_transforms(
        source,
        rotations,
        translations,
        np.asarray([1, 0]),
        writer,
        chunk_size=1,
        drop_unit_cell=False,
    )
    document = gemmi.cif.read(str(output))
    block = document.sole_block()
    assert list(block.find_values("_atom_site.auth_asym_id")) == ["CHAIN_A", "CHAIN_B", "CHAIN_A", "CHAIN_B"]
    assert list(block.find_values("_atom_site.auth_seq_id"))[:2] == ["10001", "2"]
    assert list(block.find_values("_atom_site.group_PDB"))[:2] == ["ATOM", "HETATM"]
    assert list(block.find_values("_atom_site.pdbx_PDB_model_num")) == ["1", "1", "2", "2"]
    assert list(block.find_values("_mdcorealign_frame.source_frame_index")) == ["1", "0"]
    assert "structure" in output.read_text()


def test_missing_optional_writer_dependencies_are_actionable(monkeypatch, tmp_path: Path) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"gemmi", "MDAnalysis"} or name.startswith("MDAnalysis."):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    record = ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "A", "1", "", "", "C")
    with pytest.raises(ValueError, match="Gemmi"):
        build_coordinate_writer(
            "mmcif", tmp_path / "x.cif", records=[record], n_selected_frames=1,
            allow_large_text_output=False, provenance={},
        )
    with pytest.raises(ValueError, match="MDAnalysis"):
        build_coordinate_writer(
            "xtc", tmp_path / "x.xtc", records=[record], n_selected_frames=1,
            allow_large_text_output=False, provenance={},
        )


def test_pdb_limit_errors_recommend_safer_format(tmp_path: Path) -> None:
    long_chain = ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "AB", "1", "", "", "C")
    with pytest.raises(ValueError, match="multi-character"):
        StreamingPDBWriter(
            tmp_path / "chain.pdb", [long_chain], n_selected_frames=1,
            allow_large_text_output=False,
        )
    large_resid = ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "A", "10000", "", "", "C")
    with pytest.raises(ValueError, match="residue number"):
        StreamingPDBWriter(
            tmp_path / "resid.pdb", [large_resid], n_selected_frames=1,
            allow_large_text_output=False,
        )
    ordinary = ExportAtomRecord("ATOM", 1, "CA", "", "ALA", "A", "1", "", "", "C")
    with pytest.raises(ValueError, match="99,999"):
        StreamingPDBWriter(
            tmp_path / "atoms.pdb", [ordinary] * 100_000,
            n_selected_frames=1, allow_large_text_output=False,
        )
    with pytest.raises(ValueError, match="9,999 model"):
        StreamingPDBWriter(
            tmp_path / "models.pdb", [ordinary],
            n_selected_frames=10_000, allow_large_text_output=False,
        )
    overflow_source = ArrayCoordinateSource(
        np.asarray([[[10_000.0, 0.0, 0.0]]]), records=[ordinary],
        fit_labels=[ordinary.residue_key], fit_selection="all", apply_selection="all",
    )
    writer = StreamingPDBWriter(
        tmp_path / "coords.pdb", [ordinary], n_selected_frames=1,
        allow_large_text_output=False,
    )
    rotations, translations = _identity_transforms(1)
    with pytest.raises(ValueError, match="8.3 coordinate"):
        stream_apply_transforms(
            overflow_source, rotations, translations, np.asarray([0]), writer,
            chunk_size=1, drop_unit_cell=False,
        )


def test_apply_cli_keeps_legacy_default_and_writes_sidecars(tmp_path: Path) -> None:
    transform_path = tmp_path / "transforms.npz"
    rotations, translations = _identity_transforms(5)
    np.savez(transform_path, rotations=rotations, translations=translations)
    output_dir = tmp_path / "output"
    assert run(
        [
            "apply", "--input", str(TOY), "--input-format", "ca-pdb",
            "--transforms", str(transform_path), "--output-dir", str(output_dir),
            "--chunk-size", "2",
        ]
    ) == 0
    summary = json.loads((output_dir / "apply_summary.json").read_text())
    assert summary["legacy_contract"] is True
    assert summary["output_format"] == "pdb"
    assert summary["legacy_metadata"]["format"] == "pdb"
    assert summary["peak_chunk_frames"] == 2
    for name in [
        "aligned_trajectory.pdb", "coordinate_export_provenance.json",
        "selected_frames.tsv", "export_atom_metadata.tsv",
    ]:
        assert (output_dir / name).stat().st_size > 0


def test_formats_command_has_separate_public_sections(capsys) -> None:
    assert run(["formats"]) == 0
    output = capsys.readouterr().out
    for heading in [
        "Validated inputs",
        "Validated outputs",
        "Standard runtime dependencies",
    ]:
        assert heading in output
    for value in ["PDB + XTC", "GRO + XTC", "PDBx/mmCIF structure", "XTC aligned trajectory"]:
        assert value in output


def test_structure_cli_defaults_to_first_frame(tmp_path: Path) -> None:
    pytest.importorskip("gemmi")
    transform_path = tmp_path / "transforms.npz"
    rotations, translations = _identity_transforms(5)
    np.savez(transform_path, rotations=rotations, translations=translations)
    output_dir = tmp_path / "structure"
    assert run(
        [
            "apply", "--input", str(TOY), "--input-format", "ca-pdb",
            "--transforms", str(transform_path), "--purpose", "structure",
            "--output-dir", str(output_dir),
        ]
    ) == 0
    summary = json.loads((output_dir / "apply_summary.json").read_text())
    assert summary["n_frames"] == 1
    assert summary["selected_frame_indices"] == [0]
    assert (output_dir / "aligned_structure.cif").exists()


@pytest.mark.parametrize("topology_kind", ["pdb", "gro"])
def test_streaming_topology_xtc_input_matrix(tmp_path: Path, topology_kind: str) -> None:
    mda = pytest.importorskip("MDAnalysis")
    topology_pdb = ROOT / "examples/toy_md/toy_ca_topology.pdb"
    trajectory_pdb = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    universe = mda.Universe(str(topology_pdb), str(trajectory_pdb))
    source_xtc = tmp_path / "source.xtc"
    with mda.Writer(str(source_xtc), n_atoms=universe.atoms.n_atoms) as writer:
        for timestep in universe.trajectory:
            timestep.dimensions = np.asarray([30.0, 31.0, 32.0, 90.0, 90.0, 90.0])
            timestep.time = float(timestep.frame) * 2.0
            timestep.data["step"] = int(timestep.frame) * 20
            writer.write(universe.atoms)
    if topology_kind == "gro":
        topology = tmp_path / "topology.gro"
        universe.trajectory[0]
        with mda.Writer(str(topology), n_atoms=universe.atoms.n_atoms) as writer:
            writer.write(universe.atoms)
    else:
        topology = topology_pdb

    rotations, translations = _identity_transforms(5)
    transform_path = tmp_path / f"{topology_kind}_transforms.npz"
    np.savez(transform_path, rotations=rotations, translations=translations)
    output_dir = tmp_path / f"{topology_kind}_output"
    output_xtc = output_dir / "aligned.xtc"
    arguments = [
        "apply", "--topology", str(topology), "--trajectory", str(source_xtc),
        "--transforms", str(transform_path), "--purpose", "trajectory",
        "--output", str(output_xtc), "--output-dir", str(output_dir),
        "--chunk-size", "2",
    ]
    if topology_kind == "gro":
        arguments.extend(["--topology-format", "GRO", "--trajectory-format", "XTC"])
    assert run(arguments) == 0
    summary = json.loads((output_dir / "apply_summary.json").read_text())
    assert summary["streaming_transform_application"] is True
    assert summary["n_frames"] == 5
    assert summary["peak_chunk_frames"] == 2
    assert summary["unit_cell_policy"] == "rotated_and_preserved"
    result = mda.Universe(str(topology), str(output_xtc))
    assert len(result.trajectory) == 5
    assert result.atoms.n_atoms == 8
    assert np.all(result.trajectory[0].dimensions[:3] > 0.0)
