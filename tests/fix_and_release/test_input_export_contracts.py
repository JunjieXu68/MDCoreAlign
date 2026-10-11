"""Public input/read-replay/export contracts, using small independent fixtures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from mdcorealign.cli import _load_pdb_models, _prepare_input, build_parser, run


def _coordinates() -> np.ndarray:
    rng = np.random.default_rng(195)
    xyz = rng.normal(size=(5, 8, 3)) * 0.08
    xyz += rng.normal(size=(8, 3))[None] * 3
    xyz += np.arange(5)[:, None, None] * np.array([0.4, -0.3, 0.1])
    return xyz


def _write_pdb(path: Path, *, mixed: bool = False) -> np.ndarray:
    ca = _coordinates()
    xyz = np.stack((ca, ca + np.array([0.3, 0.2, -0.1])), axis=2).reshape(5, 16, 3)
    lines = []
    for frame, coordinates in enumerate(xyz):
        lines.append(f"MODEL     {frame + 1:4d}\n")
        for index, point in enumerate(coordinates):
            resid = index // 2 + 1
            chain = "A" if resid <= 4 else "B"
            name = "CA" if index % 2 == 0 else "N"
            resname = "GLY"
            record = "ATOM"
            if mixed:
                resname = {1: "MSE", 2: "MSE", 7: "DA", 8: "LIG"}.get(resid, "GLY")
                if resid in {2, 8}:
                    record = "HETATM"
            lines.append(
                f"{record:<6}{index + 1:5d} {name:>4} {resname:>3} {chain}{resid:4d}    "
                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  0.55 12.50           C\n"
            )
        lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))
    return np.round(xyz, 3)


def _args(arguments: list[str]):
    return build_parser().parse_args(["align", *arguments])


def _align(path: Path, result: Path, *extra: str) -> None:
    assert run(["align", "--input", str(path), "--mode", "throughput", "--backend", "cpu",
                "--fast-representative-count", "5", "--output-dir", str(result), *extra]) == 0


def _apply(result: Path, path: Path, *extra: str) -> int:
    return run(["apply", str(result), "-o", str(path), "--output-dir", str(path.parent / path.stem), *extra])


def _direct(result: Path, xyz: np.ndarray) -> np.ndarray:
    with np.load(result / "transforms.npz") as transforms:
        return np.einsum("tni,tji->tnj", xyz, transforms["rotations"]) + transforms["translations"][:, None]


def test_chain_selects_fit_only_and_absent_chain_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "two_chains.pdb"
    _write_pdb(source)
    prepared = _prepare_input(_args(["--input", str(source), "--chain", "A"]))
    assert prepared.fit_trajectory.n_atoms == 4
    assert all(label.startswith("A:") for label in prepared.fit_labels)
    assert prepared.n_exportable_atoms == 16
    with pytest.raises(ValueError, match="chain"):
        _prepare_input(_args(["--input", str(source), "--chain", "Z"]))


def test_chain_selection_replays_without_shrinking_default_export(tmp_path: Path) -> None:
    source, result = tmp_path / "two_chains.pdb", tmp_path / "fit_A"
    xyz = _write_pdb(source)
    _align(source, result, "--chain", "A")
    assert json.loads((result / "summary.json").read_text())["n_fit_atoms"] == 4
    exported = tmp_path / "all_chains.pdb"
    assert _apply(result, exported, "--all-frames") == 0
    observed, records = _load_pdb_models(exported)
    assert {r.chain for r in records} == {"A", "B"}
    assert len(records) == 16
    np.testing.assert_allclose(observed, _direct(result, xyz), atol=5.1e-4)
    summary = json.loads((tmp_path / "all_chains/apply_summary.json").read_text())
    assert summary["visualization_annotations"] is False
    assert any("no invented RMSF" in warning for warning in summary["warnings"])
    assert all(record.occupancy == 0.55 and record.b_factor == 12.50 for record in records)


@pytest.mark.parametrize("kind", ["npy", "npz"])
@pytest.mark.parametrize("compatibility", [False, True])
def test_array_nm_read_configuration_roundtrip_and_unknown_metadata(
    tmp_path: Path, kind: str, compatibility: bool,
) -> None:
    source, result = tmp_path / ("coordinates." + kind), tmp_path / "result"
    xyz_nm = _coordinates() / 10
    extra = []
    if kind == "npy":
        np.save(source, xyz_nm)
    else:
        np.savez(source, coords=xyz_nm + 8, selected=xyz_nm)
        extra = ["--array-key", "selected"]
    _align(source, result, "--source-unit", "nanometer", *extra)
    exported = tmp_path / "aligned.pdb"
    apply_extra = ["--purpose", "compatibility"] if compatibility else []
    assert _apply(result, exported, "--all-frames", *apply_extra) == 0
    observed, records = _load_pdb_models(exported)
    assert {r.resname for r in records} == {"UNK"}
    assert {r.atom_name for r in records} == {"CA"}
    np.testing.assert_allclose(observed, _direct(result, xyz_nm * 10), atol=5.1e-4)
    manifest = json.loads((result / "diagnostics/source_manifest.json").read_text())
    config = manifest["read_configuration"]
    assert config["source_unit"] == "nanometer"
    assert config["array_key"] == ("selected" if kind == "npz" else None)
    assert config["source_frame_indices"] == list(range(5))
    assert manifest["source_metadata_kind"] == "synthetic_array"


def test_replay_rejects_conflicting_array_key_units_and_wrong_file(tmp_path: Path, capsys) -> None:
    source, result = tmp_path / "coordinates.npz", tmp_path / "result"
    np.savez(source, first=_coordinates(), second=_coordinates() + 15)
    _align(source, result, "--array-key", "first", "--source-unit", "angstrom")
    assert _apply(result, tmp_path / "wrong_key.pdb", "--array-key", "second") == 2
    assert "array-key conflicts" in capsys.readouterr().err
    assert _apply(result, tmp_path / "wrong_unit.pdb", "--source-unit", "nanometer") == 2
    assert "source-unit conflicts" in capsys.readouterr().err
    other = tmp_path / "other.npz"
    np.savez(other, first=_coordinates() + 0.1, second=_coordinates() + 15)
    assert _apply(result, tmp_path / "wrong_source.pdb", "--input", str(other)) == 2
    assert "hash does not match" in capsys.readouterr().err


def test_array_selection_without_real_topology_is_not_silently_ignored(tmp_path: Path) -> None:
    source = tmp_path / "array.npy"
    np.save(source, _coordinates())
    with pytest.raises(ValueError, match="selection|metadata|topology"):
        _prepare_input(_args(["--input", str(source), "--fit-selection", "name O"]))
    with pytest.raises(ValueError, match="chain|metadata|topology"):
        _prepare_input(_args(["--input", str(source), "--chain", "A"]))


def test_export_ca_subset_preserves_source_identity_and_frame_order(tmp_path: Path) -> None:
    source, result = tmp_path / "full.pdb", tmp_path / "result"
    xyz = _write_pdb(source)
    _align(source, result)
    exported = tmp_path / "ca.pdb"
    assert _apply(result, exported, "--export-selection", "protein and name CA", "--frames", "3,1") == 0
    observed, records = _load_pdb_models(exported)
    assert len(records) == 8 and all(r.atom_name == "CA" for r in records)
    np.testing.assert_allclose(observed, _direct(result, xyz[:, ::2])[[3, 1]], atol=5.1e-4)
    summary = json.loads((tmp_path / "ca/apply_summary.json").read_text())
    assert summary["complete_protein_export"] is False
    assert _apply(result, tmp_path / "empty.pdb", "--export-selection", "name O") == 2


@pytest.mark.parametrize("overrides", ["none", "topology_only", "both_renamed"])
def test_topology_and_trajectory_format_overrides_are_independent(tmp_path: Path, overrides: str) -> None:
    pdb = tmp_path / "topology.pdb"
    xyz = _write_pdb(pdb)
    universe = mda.Universe(str(pdb))
    xtc = tmp_path / "trajectory.xtc"
    with mda.Writer(str(xtc), universe.atoms.n_atoms) as writer:
        for _ts in universe.trajectory:
            writer.write(universe.atoms)
    extra = []
    if overrides == "topology_only":
        extra = ["--topology-format", "PDB"]
    if overrides == "both_renamed":
        renamed_top, renamed_traj = tmp_path / "topology.weird", tmp_path / "trajectory.weird"
        shutil.copyfile(pdb, renamed_top)
        shutil.copyfile(xtc, renamed_traj)
        pdb, xtc = renamed_top, renamed_traj
        extra = ["--topology-format", "PDB", "--trajectory-format", "XTC"]
    args = _args(["--topology", str(pdb), "--trajectory", str(xtc), *extra])
    prepared = _prepare_input(args)
    np.testing.assert_allclose(prepared.fit_trajectory.coords, xyz[:, ::2], atol=0.006)
    result = tmp_path / "result"
    assert run(["align", "--topology", str(pdb), "--trajectory", str(xtc), "--mode", "throughput",
                "--output-dir", str(result), *extra]) == 0
    output = tmp_path / "aligned.pdb"
    assert _apply(result, output, "--all-frames", "--export-selection", "protein and name CA") == 0
    observed, records = _load_pdb_models(output)
    assert len(records) == 8
    np.testing.assert_allclose(observed, _direct(result, prepared.fit_trajectory.coords), atol=5.1e-4)


def test_mse_and_nonprotein_selection_is_reader_independent(tmp_path: Path) -> None:
    pdb = tmp_path / "mixed.pdb"
    _write_pdb(pdb, mixed=True)
    builtin = _prepare_input(_args(["--input", str(pdb)]))
    backend = _prepare_input(_args(["--topology", str(pdb), "--trajectory", str(pdb)]))
    expected = ("A:MSE1", "A:MSE2", "A:GLY3", "A:GLY4", "B:GLY5", "B:GLY6")
    assert builtin.fit_labels == backend.fit_labels == expected
    assert builtin.n_exportable_atoms == backend.n_exportable_atoms == 12
    np.testing.assert_allclose(builtin.fit_trajectory.coords, backend.fit_trajectory.coords, atol=1e-6)


def test_mdanalysis_chain_scope_and_full_export_replay(tmp_path: Path) -> None:
    pdb = tmp_path / "two_chain.pdb"
    xyz = _write_pdb(pdb)
    result = tmp_path / "result"
    assert run(["align", "--topology", str(pdb), "--trajectory", str(pdb),
                "--chain", "A", "--output-dir", str(result)]) == 0
    assert json.loads((result / "summary.json").read_text())["n_fit_atoms"] == 4
    output = tmp_path / "all.pdb"
    assert _apply(result, output, "--all-frames") == 0
    observed, records = _load_pdb_models(output)
    assert len(records) == 16
    np.testing.assert_allclose(observed, _direct(result, xyz), atol=5.1e-4)
    with pytest.raises(ValueError, match="chain"):
        _prepare_input(_args(["--topology", str(pdb), "--trajectory", str(pdb), "--chain", "Z"]))
    assert _apply(result, tmp_path / "changed_chain.pdb", "--chain", "B") == 2
    assert _apply(result, tmp_path / "invalid.pdb", "--export-selection", "not-a-selector") == 2
    assert _apply(result, tmp_path / "empty.pdb", "--export-selection", "name DOES_NOT_EXIST") == 2
    xtc = tmp_path / "all.xtc"
    assert _apply(result, xtc) == 0
    summary = json.loads((tmp_path / "all/apply_summary.json").read_text())
    assert summary["visualization_annotations"] is False
    companion = Path(summary["outputs"]["topology_companion"])
    restored = mda.Universe(str(companion), str(xtc))
    assert restored.atoms.n_atoms == 16
    observed_xtc = np.asarray([restored.atoms.positions.copy() for _ in restored.trajectory])
    np.testing.assert_allclose(observed_xtc, _direct(result, xyz), atol=0.006)


def test_topology_only_pdb_can_replay_without_switching_reader(tmp_path: Path) -> None:
    pdb, result = tmp_path / "topology.pdb", tmp_path / "result"
    xyz = _write_pdb(pdb)
    assert run(["align", "--topology", str(pdb), "--output-dir", str(result)]) == 0
    output = tmp_path / "aligned.pdb"
    assert _apply(result, output, "--all-frames") == 0
    observed, records = _load_pdb_models(output)
    assert len(records) == 16
    np.testing.assert_allclose(observed, _direct(result, xyz), atol=5.1e-4)


@pytest.mark.parametrize("field", ["source_atom_identities", "fit_atom_order_fingerprint", "source_frame_indices"])
def test_subset_export_keeps_identity_and_order_guards(tmp_path: Path, capsys, field: str) -> None:
    pdb, result = tmp_path / "topology.pdb", tmp_path / "result"
    _write_pdb(pdb)
    _align(pdb, result)
    path = result / "diagnostics/source_manifest.json"
    manifest = json.loads(path.read_text())
    if field == "source_atom_identities":
        manifest[field][0], manifest[field][1] = manifest[field][1], manifest[field][0]
    elif field == "source_frame_indices":
        manifest["read_configuration"][field] = [1, 0, 2, 3, 4]
    else:
        manifest[field] = "0" * 64
    path.write_text(json.dumps(manifest))
    assert _apply(result, tmp_path / "ca.pdb", "--export-selection", "name CA") == 2
    error = capsys.readouterr().err
    assert any(word in error for word in ("identity/order", "frame ordering"))


def test_standard_legacy_pdb_manifest_remains_readable(tmp_path: Path) -> None:
    pdb, result = tmp_path / "standard.pdb", tmp_path / "result"
    xyz = _write_pdb(pdb)
    _align(pdb, result)
    path = result / "diagnostics/source_manifest.json"
    manifest = json.loads(path.read_text())
    for field in ("read_configuration", "source_atom_identities", "source_atom_order_fingerprint", "fit_atom_order_fingerprint"):
        manifest.pop(field)
    manifest["schema_version"] = "2.1.0"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "aligned.pdb"
    assert _apply(result, output, "--all-frames") == 0
    np.testing.assert_allclose(_load_pdb_models(output)[0], _direct(result, xyz), atol=5.1e-4)


def test_legacy_array_manifest_without_replay_configuration_is_rejected(tmp_path: Path) -> None:
    source, result = tmp_path / "coordinates.npy", tmp_path / "result"
    np.save(source, _coordinates() / 10)
    _align(source, result, "--source-unit", "nanometer")
    path = result / "diagnostics/source_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("read_configuration", None)
    manifest["schema_version"] = "2.1.0"
    path.write_text(json.dumps(manifest))
    assert _apply(result, tmp_path / "unsafe.pdb", "--purpose", "compatibility") == 2


def test_legacy_apply_rejects_mixed_source_conventions_before_reading(tmp_path: Path, capsys) -> None:
    """A manual transform archive must not silently choose one of two inputs."""

    pdb = tmp_path / "topology.pdb"
    _write_pdb(pdb)
    array = tmp_path / "different_coordinates.npy"
    np.save(array, _coordinates() + 100)
    transforms = tmp_path / "identity.npz"
    np.savez(transforms, rotations=np.repeat(np.eye(3)[None], 5, axis=0),
             translations=np.zeros((5, 3)))
    output = tmp_path / "ambiguous.pdb"
    output_dir = tmp_path / "export_result"
    assert run([
        "apply", "--transforms", str(transforms), "--input", str(array),
        "--topology", str(pdb), "--trajectory", str(pdb), "--purpose", "compatibility",
        "-o", str(output), "--output-dir", str(output_dir),
    ]) == 2
    assert "use --input or --topology/--trajectory, not both" in capsys.readouterr().err
    assert not output.exists()
    assert not output_dir.exists()


def test_materialized_all_count_does_not_change_future_default_export(tmp_path: Path) -> None:
    """Legacy in-command all export and later default protein export differ."""

    pdb = tmp_path / "protein_and_ligand.pdb"
    lines = []
    for frame_index, frame in enumerate(_coordinates()[:, :5]):
        lines.append(f"MODEL     {frame_index + 1:4d}\n")
        for index, xyz in enumerate(frame):
            protein = index < 4
            record, name, residue = ("ATOM", "CA", "GLY") if protein else ("HETATM", "C1", "LIG")
            lines.append(
                f"{record:<6}{index + 1:5d} {name:>4} {residue:>3} A{index + 1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
        lines.append("ENDMDL\n")
    lines.append("END\n")
    pdb.write_text("".join(lines))
    result = tmp_path / "result"
    _align(pdb, result, "--materialize", "--apply-selection", "all")
    summary = json.loads((result / "summary.json").read_text())
    assert summary["n_exportable_atoms"] == 5
    assert summary["n_exportable_atoms_scope"] == "materialized_selection"
    assert summary["n_materialized_atoms"] == 5
    assert summary["materialized_selection"] == "all"
    assert summary["n_default_export_atoms"] == 4
    assert len(_load_pdb_models(result / "aligned_trajectory.pdb")[1]) == 5
    manifest = json.loads((result / "diagnostics/source_manifest.json").read_text())
    assert manifest["n_exportable_atoms"] == 4
    output = tmp_path / "default.pdb"
    assert _apply(result, output, "--all-frames") == 0
    observed, records = _load_pdb_models(output)
    assert len(records) == 4 and all(record.resname == "GLY" for record in records)
    source = _load_pdb_models(pdb)[0][:, :4]
    np.testing.assert_allclose(observed, _direct(result, source), atol=5.1e-4)
