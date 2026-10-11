"""Small regression checks for the bounded biological validation analysis."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.analysis import biological_core_validation_v1 as analysis


def residues(count: int) -> list[dict]:
    return [
        {"chain": "A", "source_resid": i + 1, "sequence_position": i + 1, "resname": "ALA"}
        for i in range(count)
    ]


def test_pair_variance_is_population_variance_and_excludes_three_neighbors() -> None:
    coords = np.zeros((3, 5, 3))
    coords[:, :, 0] = np.arange(5)
    coords[:, 4, 0] = [4, 6, 8]
    metadata = residues(5)
    # Gaps in source numbering must not turn sequence neighbors into distant pairs.
    for row in metadata:
        row["source_resid"] *= 10
    pairs = analysis.pair_variances(coords, metadata, frame_block=2)
    np.testing.assert_array_equal(pairs["left"], [0])
    np.testing.assert_array_equal(pairs["right"], [4])
    np.testing.assert_allclose(pairs["mean_A"], [6])
    np.testing.assert_allclose(pairs["variance_A2"], [8 / 3])
    assert int(pairs["n_frames"]) == 3
    score, count = analysis.geometry_score(np.ones(5, dtype=bool), pairs)
    assert score == pytest.approx(np.sqrt(8 / 3))
    assert count == 1
    whole = analysis.pair_variances(coords, metadata, frame_block=3)
    np.testing.assert_allclose(pairs["variance_A2"], whole["variance_A2"])


def test_cross_chain_pairs_are_not_excluded_as_sequence_neighbors() -> None:
    coords = np.asarray([[[0, 0, 0], [2, 0, 0]], [[0, 0, 0], [4, 0, 0]]])
    metadata = residues(2)
    metadata[1].update(chain="B", sequence_position=1)
    pairs = analysis.pair_variances(coords, metadata)
    np.testing.assert_array_equal(pairs["left"], [0])
    np.testing.assert_array_equal(pairs["right"], [1])
    np.testing.assert_allclose(pairs["variance_A2"], [1])


def test_no_eligible_pair_returns_none_instead_of_a_perfect_score() -> None:
    coords = np.zeros((2, 4, 3))
    pairs = analysis.pair_variances(coords, residues(4))
    assert analysis.geometry_score(np.ones(4, dtype=bool), pairs) == (None, 0)
    assert analysis.geometry_score(np.zeros(4, dtype=bool), pairs) == (None, 0)


def test_pair_variance_requires_multiple_heldout_frames() -> None:
    with pytest.raises(ValueError, match="at least two"):
        analysis.pair_variances(np.zeros((1, 5, 3)), residues(5))


def test_distance_variance_is_invariant_to_independent_frame_rigid_transforms() -> None:
    rng = np.random.default_rng(43)
    coords = rng.normal(size=(9, 8, 3)) * 5
    rotation = Rotation.random(9, random_state=rng).as_matrix()
    transformed = coords @ rotation + rng.normal(size=(9, 1, 3)) * 40
    original = analysis.pair_variances(coords, residues(8), frame_block=4)
    perturbed = analysis.pair_variances(transformed, residues(8), frame_block=3)
    np.testing.assert_allclose(original["variance_A2"], perturbed["variance_A2"], atol=1e-12)


def test_rmsf_uses_temporal_mean_and_is_invariant_to_one_common_rigid_transform() -> None:
    coords = np.asarray([[[-1, 0, 0], [0, -2, 0]], [[1, 0, 0], [0, 2, 0]]])
    np.testing.assert_allclose(analysis.rmsf(coords), [1, 2])
    transformed = coords @ Rotation.from_euler("xyz", [30, 70, 10], degrees=True).as_matrix()
    transformed += np.asarray([15, -20, 9])
    np.testing.assert_allclose(analysis.rmsf(transformed), [1, 2])


def test_unknown_annotations_are_removed_from_region_statistic_denominators() -> None:
    core = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    region = np.asarray([1, 0, 1, 1, 1, 0], dtype=bool)
    evaluable = np.asarray([1, 1, 0, 0, 1, 1], dtype=bool)
    result = analysis.region_statistics(core, region, evaluable)
    assert result["core_size"] == 4
    assert result["evaluated_core_size"] == 2
    assert result["unannotated_core_fraction"] == 0.5
    assert result["evaluable_residues"] == 4
    assert result["region_size"] == 2
    assert result["overlap"] == 1
    assert result["purity"] == result["coverage"] == 0.5
    assert result["enrichment"] == 1
    region[~evaluable] = False
    assert result == analysis.region_statistics(core, region, evaluable)


def test_empty_core_has_undefined_purity_and_unknown_fraction() -> None:
    result = analysis.region_statistics(
        np.zeros(5, dtype=bool), np.ones(5, dtype=bool), np.ones(5, dtype=bool)
    )
    assert result["evaluated_core_size"] == result["core_size"] == 0
    assert result["purity"] is None
    assert result["unannotated_core_fraction"] is None
    assert result["enrichment"] is None
    assert result["coverage"] == 0


def test_random_reference_is_reproducible_and_has_exact_cardinality(monkeypatch) -> None:
    observed = []

    def score(mask, pairs):
        observed.append(mask.copy())
        return float(np.flatnonzero(mask).sum()), 1

    monkeypatch.setattr(analysis, "geometry_score", score)
    first = analysis.random_scores(12, 7, {}, count=25)
    second = analysis.random_scores(12, 7, {}, count=25)
    assert first == second
    assert len(first) == 25
    assert all(mask.dtype == bool and mask.sum() == 7 for mask in observed)
    assert any(not np.array_equal(observed[0], mask) for mask in observed[1:25])


def test_random_reference_keeps_unevaluable_draws_without_redrawing() -> None:
    pairs = analysis.pair_variances(np.zeros((2, 5, 3)), residues(5))
    observed = analysis.random_scores(5, 4, pairs, count=40)
    rng = np.random.default_rng(analysis.SEED)
    expected = []
    for _ in range(40):
        selected = rng.choice(5, 4, replace=False)
        expected.append(0.0 if 0 in selected and 4 in selected else None)
    assert observed == expected
    assert None in observed
    assert 0.0 in observed
    assert analysis.random_scores(5, 1, pairs, count=11) == [None] * 11


def test_ca_writer_preserves_real_residue_ids_insertions_and_chains(tmp_path: Path) -> None:
    metadata = [
        {"chain": "A", "source_resid": -2, "sequence_position": 1, "resname": "GLY"},
        {"chain": "A", "source_resid": 42, "icode": "B", "sequence_position": 2, "resname": "SER"},
        {"chain": "B", "source_resid": 105, "sequence_position": 1, "resname": "LYS"},
    ]
    coords = np.arange(18, dtype=float).reshape(2, 3, 3) / 4
    path = tmp_path / "mapped.pdb"
    analysis.write_ca(coords, path, metadata)
    mask, frame_count, labels = analysis.parse_mdlovofit_mask(path, 3)
    assert frame_count == 2
    assert mask.tolist() == [True, True, True]
    assert labels == [("A", "-2", "GLY"), ("A", "42B", "SER"), ("B", "105", "LYS")]
    parsed, _ = analysis.load_ca_multimodel_pdb_minimal(path, expected_frames=2, expected_residues=3)
    np.testing.assert_allclose(parsed, coords)


def test_writer_rejects_identifier_overflow_instead_of_truncating(tmp_path: Path) -> None:
    metadata = residues(3)
    metadata[0]["source_resid"] = 10000
    with pytest.raises(ValueError, match="overflow"):
        analysis.write_ca(np.zeros((2, 3, 3)), tmp_path / "overflow.pdb", metadata)


def test_read_export_retains_variable_per_frame_membership(tmp_path: Path) -> None:
    path = tmp_path / "export.pdb"
    rows = []
    for frame in range(2):
        rows.append(f"REMARK FRAME: {frame + 1}\n")
        for atom in range(3):
            occupancy = float(atom != frame)
            rows.append(
                f"ATOM  {atom + 1:5d}  CA  ALA A{atom + 1:4d}    "
                f"{float(atom):8.3f}{float(frame):8.3f}{0.0:8.3f}"
                f"{occupancy:6.2f}{0.0:6.2f}          C\n"
            )
        rows.append("END\n")
    path.write_text("".join(rows))
    labels = [("A", str(i), "ALA") for i in range(1, 4)]
    aligned, masks = analysis.read_export(path, 2, 3, labels)
    assert aligned.shape == (2, 3, 3)
    assert masks.tolist() == [[False, True, True], [True, False, True]]
    changed = path.read_text().splitlines(keepends=True)
    # Wrong identity in the second frame must be rejected even when core masks vary.
    atom_lines = [i for i, line in enumerate(changed) if line.startswith("ATOM")]
    target = atom_lines[4]
    changed[target] = changed[target][:22] + "  99" + changed[target][26:]
    path.write_text("".join(changed))
    with pytest.raises(ValueError, match="identity mismatch"):
        analysis.read_export(path, 2, 3, labels)


def test_export_identity_compares_insertion_code_as_part_of_residue_id(tmp_path: Path) -> None:
    metadata = residues(5)
    metadata[2].update(source_resid=42, icode="B")
    path = tmp_path / "insertion.pdb"
    analysis.write_ca(np.zeros((2, 5, 3)), path, metadata)
    labels = [("A", str(i), "ALA") for i in range(1, 6)]
    labels[2] = ("A", "42B", "ALA")
    _, masks = analysis.read_export(path, 2, 5, labels)
    assert masks.all()
    labels[2] = ("A", "42", "ALA")
    with pytest.raises(ValueError, match="identity mismatch"):
        analysis.read_export(path, 2, 5, labels)


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_variable_stock_core_is_explicit_consensus_and_still_checks_identity(
    tmp_path: Path, monkeypatch, wrong_identity: bool
) -> None:
    metadata = residues(5)
    metadata[0].update(source_resid=42, icode="B")
    coords = np.random.default_rng(93).normal(size=(2, 5, 3))
    input_path = tmp_path / "input.pdb"
    analysis.write_ca(coords, input_path, metadata)
    executable = tmp_path / "stock"
    executable.write_bytes(b"fake stock executable for parser regression")
    monkeypatch.setattr(analysis, "STOCK_MDLOVOFIT", executable)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("-t") + 1])
        lines = input_path.read_text().splitlines(keepends=True)
        indices = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
        masks = [[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]]
        for serial, line_index in enumerate(indices):
            frame, atom = divmod(serial, 5)
            line = lines[line_index]
            lines[line_index] = line[:54] + f"{float(masks[frame][atom]):6.2f}" + line[60:]
        if wrong_identity:
            line_index = indices[5]
            lines[line_index] = lines[line_index][:21] + "B" + lines[line_index][22:]
        output.write_text("".join(lines))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(analysis.subprocess, "run", fake_run)
    destination = tmp_path / "export"
    if wrong_identity:
        with pytest.raises(ValueError, match="identity mismatch"):
            analysis.mdlovo(coords, input_path, destination)
        assert (destination / "aligned.pdb").is_file()
    else:
        result = analysis.mdlovo(coords, input_path, destination)
        assert result["metadata"]["core_kind"] == "derived_frequency_consensus_core"
        assert result["metadata"]["constant_across_frames"] is False
        np.testing.assert_allclose(result["frequency"], [0.5, 1, 1, 0.5, 0])
        assert result["mask"].tolist() == [True, True, True, False, False]
        assert not (destination / "aligned.pdb").exists()
        with np.load(destination / "membership.npz") as saved:
            assert saved["masks"].tolist() == [[True, True, True, False, False], [False, True, True, True, False]]


@pytest.mark.parametrize(
    ("key", "indices"),
    [
        ("train_indices", []),
        ("test_indices", []),
        ("train_indices", [0.0, 1.0]),
        ("train_indices", [False, True]),
        ("train_indices", ["0", "1"]),
        ("train_indices", [1, 0]),
        ("train_indices", [0, 0]),
        ("train_indices", [-1, 0]),
        ("test_indices", [4, 6]),
        ("test_indices", [2, 3]),
        ("test_indices", [5]),
    ],
)
def test_split_rejects_invalid_or_leaking_indices(key: str, indices: list) -> None:
    case = {"coords_path": "source.npy", "train_indices": [0, 1, 2], "test_indices": [3, 4, 5]}
    case[key] = indices
    with pytest.raises(ValueError):
        analysis.validate_split(case, 6)


def test_independent_replicate_split_allows_same_frame_numbers_but_not_same_file(tmp_path: Path) -> None:
    case = {
        "coords_path": str(tmp_path / "R1.npy"),
        "test_coords_path": str(tmp_path / "R2.npy"),
        "train_indices": [0, 1, 2],
        "test_indices": [0, 1, 2],
    }
    analysis.validate_split(case, 3, n_test_frames=3)
    case["test_coords_path"] = str(tmp_path / "." / "R1.npy")
    with pytest.raises(ValueError, match="paths must differ"):
        analysis.validate_split(case, 3, n_test_frames=3)


def test_independent_replicate_split_uses_test_source_frame_bound() -> None:
    case = {
        "coords_path": "R1.npy",
        "test_coords_path": "R2.npy",
        "train_indices": [0, 1, 2],
        "test_indices": [0, 2],
    }
    with pytest.raises(ValueError, match="source bounds"):
        analysis.validate_split(case, 10, n_test_frames=2)


def test_heldout_frames_do_not_enter_training_estimation(tmp_path: Path, monkeypatch) -> None:
    class HeldoutReached(Exception):
        pass

    solve_inputs = []
    heldout_inputs = []
    external_inputs = []

    def solve(coords, **kwargs):
        solve_inputs.append(coords.copy())
        weights = np.linspace(0.1, 1.0, coords.shape[1])
        return SimpleNamespace(
            frozen_weights=weights,
            core_labels=weights > 0.45,
            rotations=np.repeat(np.eye(3)[None], len(coords), axis=0),
            translations=np.zeros((len(coords), 3)),
        )

    def external(coords, *args):
        external_inputs.append(coords.copy())
        mask = np.arange(coords.shape[1]) >= 3
        return {"aligned": coords.copy(), "mask": mask, "frequency": mask.astype(float)}

    def stop_at_heldout(coords, metadata):
        heldout_inputs.append(coords.copy())
        raise HeldoutReached

    monkeypatch.setattr(analysis, "solve_minimal_cpu", solve)
    monkeypatch.setattr(analysis, "mdlovo", external)
    monkeypatch.setattr(
        analysis,
        "fit_all_frames_fixed_weights",
        lambda coords, *args, **kwargs: SimpleNamespace(aligned_coords=coords.copy()),
    )
    monkeypatch.setattr(analysis, "pair_variances", stop_at_heldout)
    raw = np.random.default_rng(9).normal(size=(6, 8, 3))
    for repeat in range(2):
        values = raw.copy()
        if repeat:
            values[3:, 0] += 50
        run = tmp_path / str(repeat)
        run.mkdir()
        np.save(run / "coords.npy", values)
        case = {
            "case_id": "TEST",
            "coords_path": str(run / "coords.npy"),
            "residues": residues(8),
            "train_indices": [0, 1, 2],
            "test_indices": [3, 4, 5],
            "regions": {"CORE": [True] * 5 + [False] * 3},
            "region_roles": {"CORE": "test annotation"},
            "literature_evaluable": [True] * 8,
        }
        config = run / "config.json"
        config.write_text(json.dumps({"run_directory": str(run), "cases": [case]}))
        with pytest.raises(HeldoutReached):
            analysis.analyze_case(config, "TEST")
    assert [len(value) for value in solve_inputs] == [6, 3, 6, 3]
    assert [len(value) for value in external_inputs] == [6, 3, 6, 3]
    np.testing.assert_array_equal(solve_inputs[1], solve_inputs[3])
    np.testing.assert_array_equal(external_inputs[1], external_inputs[3])
    assert not np.array_equal(heldout_inputs[0], heldout_inputs[1])
    for offset, heldout in zip((0, 2), heldout_inputs, strict=True):
        np.testing.assert_array_equal(solve_inputs[offset][:3], solve_inputs[offset + 1])
        np.testing.assert_array_equal(solve_inputs[offset][3:], heldout)
