"""VT02 self-sufficient, full-protein annotated export tests."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mdcorealign.cli import run
from mdcorealign.result_provenance import SOURCE_MANIFEST_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    xyz: np.ndarray,
    *,
    occupancy: float = 0.55,
    b_factor: float = 12.5,
) -> str:
    element = atom_name[0]
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}\n"
    )


def _write_full_atom_multimodel_pdb(
    path: Path,
    *,
    n_frames: int = 5,
    chains: tuple[str, ...] = ("A", "B"),
) -> tuple[int, int]:
    residue_names = ("ALA", "GLY", "SER", "VAL", "LEU", "THR")
    atoms = ("N", "CA", "C", "O")
    lines: list[str] = []
    for frame in range(n_frames):
        lines.append(f"MODEL     {frame + 1:4d}\n")
        serial = 0
        for residue_index, resname in enumerate(residue_names, start=1):
            chain = chains[(residue_index - 1) % len(chains)]
            theta = residue_index * 0.73
            base = np.asarray(
                [
                    residue_index * 2.2,
                    np.sin(theta) * 3.0,
                    np.cos(theta) * 2.0,
                ]
            )
            mobile = np.asarray(
                [0.0, 0.0, 0.35 * frame if residue_index >= 5 else 0.02 * frame]
            )
            for atom_index, atom_name in enumerate(atoms):
                serial += 1
                offset = np.asarray([0.35 * atom_index, 0.17 * atom_index, -0.11 * atom_index])
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        resname,
                        chain,
                        residue_index,
                        base + mobile + offset,
                    )
                )
        lines.extend(["TER\n", "ENDMDL\n"])
    lines.append("END\n")
    path.write_text("".join(lines))
    return len(residue_names) * len(atoms), len(residue_names)


def _align_full_atom(source: Path, result: Path) -> None:
    assert (
        run(
            [
                "align",
                "--input",
                str(source),
                "--input-format",
                "ca-pdb",
                "--mode",
                "throughput",
                "--backend",
                "cpu",
                "--fast-representative-count",
                "5",
                "--output-dir",
                str(result),
            ]
        )
        == 0
    )


def _profile_rows(result: Path) -> list[dict[str, str]]:
    with (result / "residue_profile.tsv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_align_writes_versioned_source_manifest_without_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "full.pdb"
    n_atoms, n_ca = _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    summary = json.loads((result / "summary.json").read_text())
    manifest = json.loads((result / "diagnostics/source_manifest.json").read_text())
    assert summary["schema_version"] == "1.1.0"
    assert summary["software_version"] == "0.1.0rc6"
    assert summary["fit_selection"] == "protein and name CA"
    assert summary["analysis_coordinate_selection"] == "protein and name CA"
    assert summary["n_fit_atoms"] == n_ca
    assert summary["n_analysis_coordinate_atoms"] == n_ca
    assert summary["default_export_selection"] == "protein"
    assert summary["n_exportable_atoms"] == n_atoms
    assert summary["source_is_ca_only"] is False
    assert "apply_selection" not in summary
    assert "n_apply_atoms" not in summary
    assert manifest["schema_version"] == SOURCE_MANIFEST_SCHEMA_VERSION
    assert manifest["software_version"] == "0.1.0rc6"
    assert manifest["source_result_schema_version"] == "1.1.0"
    assert manifest["fit_selection"] == "protein and name CA"
    assert manifest["analysis_coordinate_selection"] == "protein and name CA"
    assert manifest["default_export_selection"] == "protein"
    assert manifest["n_fit_atoms"] == n_ca
    assert manifest["n_analysis_coordinate_atoms"] == n_ca
    assert manifest["n_exportable_atoms"] == n_atoms
    assert manifest["source_is_ca_only"] is False
    assert summary["n_fit_atoms"] == manifest["n_fit_atoms"]
    assert summary["n_analysis_coordinate_atoms"] == manifest["n_analysis_coordinate_atoms"]
    assert summary["default_export_selection"] == manifest["default_export_selection"]
    assert summary["n_exportable_atoms"] == manifest["n_exportable_atoms"]
    assert manifest["input_sha256"] == manifest["single_input"]["sha256"]
    assert not list(result.glob("*.pdb"))
    assert not list(result.glob("*.cif"))
    assert not list(result.glob("*.xtc"))


def test_simple_pdb_export_is_full_protein_and_annotated(tmp_path: Path) -> None:
    mda = pytest.importorskip("MDAnalysis")
    source = tmp_path / "full.pdb"
    n_atoms, n_ca = _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    output = tmp_path / "aligned.pdb"
    assert run(["apply", str(result), "-o", str(output)]) == 0
    atom_lines = [line for line in output.read_text().splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == n_atoms
    assert len(atom_lines) > n_ca
    profile = {
        (row["chain"], row["resname"], row["residue_id"]): row
        for row in _profile_rows(result)
    }
    per_residue: dict[tuple[str, str, str], set[tuple[float, float]]] = {}
    for line in atom_lines:
        key = (line[21].strip() or "_", line[17:20].strip(), line[22:26].strip())
        observed = (float(line[54:60]), float(line[60:66]))
        per_residue.setdefault(key, set()).add(observed)
        expected = profile[key]
        assert observed[0] == float(expected["core_label"])
        assert observed[1] == pytest.approx(float(expected["robust_RMSF"]), abs=0.006)
    assert all(len(values) == 1 for values in per_residue.values())
    source_xyz = np.asarray(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in source.read_text().splitlines()
            if line.startswith("ATOM")
        ][:n_atoms]
    )
    observed_xyz = np.asarray(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in atom_lines
        ]
    )
    with np.load(result / "transforms.npz") as transforms:
        expected_xyz = source_xyz @ transforms["rotations"][0].T + transforms[
            "translations"
        ][0]
    np.testing.assert_allclose(observed_xyz, expected_xyz, atol=6e-4)
    text = output.read_text()
    assert "OCCUPANCY = REPORTING CORE LABEL" in text
    assert "B-FACTOR = STABLE-CORE-ALIGNED RMSF IN ANGSTROM" in text
    assert "--plain" not in text
    assert mda.Universe(str(output)).atoms.n_atoms == n_atoms

    all_models = tmp_path / "aligned_all_models.pdb"
    assert run(["apply", str(result), "-o", str(all_models), "--all-frames"]) == 0
    annotations_by_serial: dict[int, set[tuple[float, float]]] = {}
    for line in all_models.read_text().splitlines():
        if line.startswith("ATOM"):
            annotations_by_serial.setdefault(int(line[6:11]), set()).add(
                (float(line[54:60]), float(line[60:66]))
            )
    assert annotations_by_serial
    assert all(len(values) == 1 for values in annotations_by_serial.values())


def test_simple_mmcif_export_has_profile_and_original_fields(tmp_path: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    source = tmp_path / "full.pdb"
    n_atoms, _ = _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    output = tmp_path / "aligned.cif"
    assert run(["apply", str(result), "-o", str(output)]) == 0
    block = gemmi.cif.read(str(output)).sole_block()
    assert len(list(block.find_values("_atom_site.id"))) == n_atoms
    assert set(block.find_values("_atom_site.occupancy")) <= {"0.000000", "1.000000"}
    assert len(list(block.find_values("_mdcorealign_residue_profile.residue_id"))) == 6
    assert list(block.find_values("_mdcorealign_residue_profile.schema_version")) == [
        "2.0.0"
    ] * 6
    assert len(list(block.find_values("_mdcorealign_source_atom_metadata.atom_index"))) == n_atoms
    assert set(block.find_values("_mdcorealign_source_atom_metadata.original_occupancy")) == {
        "0.55"
    }
    cif_xyz = np.column_stack(
        [
            np.asarray(list(block.find_values(field)), dtype=float)
            for field in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z")
        ]
    )
    source_xyz = np.asarray(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in source.read_text().splitlines()
            if line.startswith("ATOM")
        ][:n_atoms]
    )
    with np.load(result / "transforms.npz") as transforms:
        expected_xyz = source_xyz @ transforms["rotations"][0].T + transforms[
            "translations"
        ][0]
    np.testing.assert_allclose(cif_xyz, expected_xyz, atol=1e-6)


def test_simple_xtc_export_writes_all_frames_and_annotated_topology(tmp_path: Path) -> None:
    mda = pytest.importorskip("MDAnalysis")
    source = tmp_path / "full.pdb"
    n_atoms, _ = _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    output = tmp_path / "aligned.xtc"
    assert run(["apply", str(result), "-o", str(output)]) == 0
    companion = tmp_path / "aligned.topology.pdb"
    sidecar = tmp_path / "aligned.xtc.mdcorealign.json"
    assert companion.is_file()
    assert sidecar.is_file()
    observed = mda.Universe(str(companion), str(output))
    assert observed.atoms.n_atoms == n_atoms
    assert len(observed.trajectory) == 5
    first_frame = observed.trajectory[0].positions.copy()
    source_universe = mda.Universe(str(source))
    source_xyz = source_universe.trajectory[0].positions.copy()
    with np.load(result / "transforms.npz") as transforms:
        expected_xyz = source_xyz @ transforms["rotations"][0].T + transforms[
            "translations"
        ][0]
    np.testing.assert_allclose(first_frame, expected_xyz, atol=1.1e-2)
    metadata = json.loads(sidecar.read_text())
    assert metadata["topology_companion"] == str(companion)
    assert metadata["atom_count"] == n_atoms
    assert metadata["frame_count"] == 5

    chunked = tmp_path / "aligned_chunked.xtc"
    assert (
        run(
            [
                "apply",
                str(result),
                "-o",
                str(chunked),
                "--chunk-size",
                "2",
                "--output-dir",
                str(tmp_path / "chunked_metadata"),
            ]
        )
        == 0
    )
    chunked_universe = mda.Universe(
        str(tmp_path / "aligned_chunked.topology.pdb"), str(chunked)
    )
    observed_frames = np.asarray(
        [timestep.positions.copy() for timestep in observed.trajectory]
    )
    chunked_frames = np.asarray(
        [timestep.positions.copy() for timestep in chunked_universe.trajectory]
    )
    np.testing.assert_allclose(chunked_frames, observed_frames, atol=1e-6)
    chunked_summary = json.loads(
        (tmp_path / "chunked_metadata/apply_summary.json").read_text()
    )
    assert chunked_summary["peak_chunk_frames"] <= 2


def test_ca_only_source_warns_and_never_invents_atoms(tmp_path: Path, capsys) -> None:
    source = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    result = tmp_path / "result"
    assert (
        run(
            [
                "align",
                "--input",
                str(source),
                "--input-format",
                "ca-pdb",
                "--fast-representative-count",
                "5",
                "--output-dir",
                str(result),
            ]
        )
        == 0
    )
    capsys.readouterr()
    summary = json.loads((result / "summary.json").read_text())
    manifest = json.loads((result / "diagnostics/source_manifest.json").read_text())
    assert summary["source_is_ca_only"] is True
    assert summary["default_export_selection"] == "all_available_atoms"
    assert summary["n_exportable_atoms"] == summary["n_fit_atoms"] == 8
    assert manifest["source_is_ca_only"] is True
    assert manifest["default_export_selection"] == "all_available_atoms"
    assert manifest["n_exportable_atoms"] == manifest["n_fit_atoms"] == 8
    output = tmp_path / "ca.pdb"
    assert run(["apply", str(result), "-o", str(output)]) == 0
    captured = capsys.readouterr()
    assert "CA-only source detected" in captured.err
    atom_lines = [line for line in output.read_text().splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == 8
    assert {line[12:16].strip() for line in atom_lines} == {"CA"}


def test_source_override_accepts_matching_move_and_rejects_hash_mismatch(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "full.pdb"
    _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    moved = tmp_path / "moved.pdb"
    shutil.copy2(source, moved)
    source.unlink()
    assert (
        run(
            [
                "apply",
                str(result),
                "-o",
                str(tmp_path / "moved.cif"),
                "--input",
                str(moved),
            ]
        )
        == 0
    )
    changed = tmp_path / "changed.pdb"
    changed.write_text(moved.read_text() + "REMARK source hash mismatch fixture\n")
    assert (
        run(
            [
                "apply",
                str(result),
                "-o",
                str(tmp_path / "bad.cif"),
                "--input",
                str(changed),
            ]
        )
        == 2
    )
    assert "hash does not match" in capsys.readouterr().err


def test_suffix_and_legacy_conflicts_fail_clearly(tmp_path: Path, capsys) -> None:
    source = tmp_path / "full.pdb"
    _write_full_atom_multimodel_pdb(source)
    result = tmp_path / "result"
    _align_full_atom(source, result)
    assert (
        run(
            [
                "apply",
                str(result),
                "-o",
                str(tmp_path / "bad.xtc"),
                "--purpose",
                "structure",
            ]
        )
        == 2
    )
    assert "conflicts with output suffix" in capsys.readouterr().err
    assert run(["apply", str(result), "-o", str(tmp_path / "bad.xyz")]) == 2
    assert "output suffix" in capsys.readouterr().err


def test_full_atom_pdb_xtc_and_gro_xtc_sources_export_all_protein(tmp_path: Path) -> None:
    mda = pytest.importorskip("MDAnalysis")
    source_pdb = tmp_path / "full.pdb"
    n_atoms, n_ca = _write_full_atom_multimodel_pdb(source_pdb, chains=("A",))
    universe = mda.Universe(str(source_pdb))
    xtc = tmp_path / "source.xtc"
    with mda.Writer(str(xtc), n_atoms=n_atoms) as writer:
        for timestep in universe.trajectory:
            timestep.dimensions = np.asarray([40.0, 41.0, 42.0, 90.0, 90.0, 90.0])
            writer.write(universe.atoms)
    universe.trajectory[0]
    gro = tmp_path / "topology.gro"
    with mda.Writer(str(gro), n_atoms=n_atoms) as writer:
        writer.write(universe.atoms)

    for label, topology in (("pdb", source_pdb), ("gro", gro)):
        result = tmp_path / f"result_{label}"
        assert (
            run(
                [
                    "align",
                    "--topology",
                    str(topology),
                    "--trajectory",
                    str(xtc),
                    "--fast-representative-count",
                    "5",
                    "--output-dir",
                    str(result),
                ]
            )
            == 0
        )
        manifest = json.loads((result / "diagnostics/source_manifest.json").read_text())
        alignment_summary = json.loads((result / "summary.json").read_text())
        assert manifest["n_fit_atoms"] == n_ca
        assert manifest["n_exportable_atoms"] == n_atoms
        assert alignment_summary["n_fit_atoms"] == n_ca
        assert alignment_summary["n_analysis_coordinate_atoms"] == n_ca
        assert alignment_summary["n_exportable_atoms"] == n_atoms
        assert alignment_summary["default_export_selection"] == "protein"
        output = tmp_path / f"{label}.xtc"
        apply_metadata = tmp_path / f"apply_{label}"
        assert (
            run(
                [
                    "apply",
                    str(result),
                    "-o",
                    str(output),
                    "--output-dir",
                    str(apply_metadata),
                ]
            )
            == 0
        )
        summary = json.loads((apply_metadata / "apply_summary.json").read_text())
        assert summary["n_exported_atoms"] == n_atoms
        assert summary["export_selection"] == "protein"
        assert summary["source_result_schema_version"] == "1.1.0"
        assert summary["n_fit_atoms"] == n_ca
        companions = list(tmp_path.glob(f"{label}.topology.*"))
        assert len(companions) == 1
        if companions[0].suffix == ".pdb":
            readback = mda.Universe(str(companions[0]), str(output))
            assert readback.atoms.n_atoms == n_atoms
        else:
            gemmi = pytest.importorskip("gemmi")
            block = gemmi.cif.read(str(companions[0])).sole_block()
            assert len(list(block.find_values("_atom_site.id"))) == n_atoms
            from MDAnalysis.coordinates.XTC import XTCReader

            with XTCReader(str(output), n_atoms=n_atoms) as reader:
                assert len(reader) == 5
