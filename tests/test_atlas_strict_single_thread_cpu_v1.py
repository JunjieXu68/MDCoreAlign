"""Focused tests for the strict single-thread ATLAS benchmark support code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.atlas_strict_single_thread_cpu_v1 import orchestrate, run_timing
from scripts.atlas_strict_single_thread_cpu_v1.common import (
    METHOD_MDCA,
    METHOD_MDLOVOFIT,
    OUT,
    command_for,
    read_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.extract_cores import (
    disagreement_segments,
    parse_mdlovofit_mask,
    top_k,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import (
    snapshot_has_active_conflict,
)


def test_locked_panel_has_expected_shape_and_prefixes() -> None:
    rows = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    assert len(rows) == 48
    assert len({(row["system_id"], row["prefix_id"]) for row in rows}) == 48
    assert len({row["system_id"] for row in rows}) == 8
    for system in {row["system_id"] for row in rows}:
        assert len([row for row in rows if row["system_id"] == system]) == 6


def test_run_order_is_counterbalanced() -> None:
    rows = read_tsv(OUT / "02_preregistration/run_order.tsv")
    assert len(rows) == 576
    measured_first = [
        row for row in rows if row["phase"] == "measured" and row["order_position"] == "1"
    ]
    assert sum(row["method"] == METHOD_MDCA for row in measured_first) == 120
    assert sum(row["method"] == METHOD_MDLOVOFIT for row in measured_first) == 120


def test_commands_share_input_and_do_not_materialise() -> None:
    row = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")[0]
    mdca = command_for(METHOD_MDCA, row, 17)
    mdlovo = command_for(METHOD_MDLOVOFIT, row, 17)
    assert row["input_path"] in " ".join(mdca)
    assert row["input_path"] in " ".join(mdlovo)
    assert "-t" not in mdlovo
    assert "rmsf" not in " ".join(mdca).lower()
    assert "w5" not in " ".join(mdca).lower()


def test_top_k_uses_exact_cardinality_and_input_ties() -> None:
    weights = np.asarray([0.8, 0.9, 0.9, 0.2, 0.1])
    selected = top_k(weights, 2)
    assert selected.sum() == 2
    assert np.flatnonzero(selected).tolist() == [1, 2]


def _atom_line(serial: int, residue: int, occupancy: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{residue:4d}    "
        f"{float(residue):8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{1.0:6.2f}"
        "          C\n"
    )


def test_mdlovofit_mask_parser_checks_all_models(tmp_path: Path) -> None:
    path = tmp_path / "core.pdb"
    lines = []
    for _ in range(2):
        lines.extend([_atom_line(1, 1, 1.0), _atom_line(2, 2, 0.0), "ENDMDL\n"])
    path.write_text("".join(lines))
    mask, models, labels = parse_mdlovofit_mask(path, 2)
    assert mask.tolist() == [True, False]
    assert models == 2
    assert labels == [("A", "1", "ALA"), ("A", "2", "ALA")]


def test_mdlovofit_stock_remark_frame_end_format(tmp_path: Path) -> None:
    """Match the persistent 1jnr_C/f10001 stock 20.1.1 export structure."""

    path = tmp_path / "stock_core.pdb"
    path.write_text(
        "".join(
            f"REMARK FRAME: {frame:8d}\n"
            + _atom_line(1, 1, 1.0)
            + _atom_line(2, 2, 0.0)
            + "END\n"
            for frame in (1, 2, 3)
        )
    )
    mask, models, labels = parse_mdlovofit_mask(path, 2)
    assert models == 3
    assert mask.tolist() == [True, False]
    assert labels == [("A", "1", "ALA"), ("A", "2", "ALA")]


def test_mdlovofit_standard_terminal_end_is_not_an_extra_model(tmp_path: Path) -> None:
    path = tmp_path / "standard_core.pdb"
    path.write_text("MODEL        1\n" + _atom_line(1, 1, 1.0) + "ENDMDL\nEND\n")
    mask, models, _ = parse_mdlovofit_mask(path, 1)
    assert mask.tolist() == [True]
    assert models == 1


@pytest.mark.parametrize(
    ("second_model", "message"),
    [
        (_atom_line(1, 1, 1.0), "truncated final"),
        (_atom_line(1, 1, 1.0) + "END\n", "1 atoms; expected 2"),
        (_atom_line(1, 1, 0.0) + _atom_line(2, 2, 1.0) + "END\n", "core differs"),
        (_atom_line(1, 1, 1.0) + _atom_line(2, 3, 0.0) + "END\n", "mapping differs"),
        (_atom_line(1, 1, 0.5) + _atom_line(2, 2, 0.0) + "END\n", "not binary"),
        (_atom_line(1, 1, float("nan")) + _atom_line(2, 2, 0.0) + "END\n", "not binary"),
        (_atom_line(1, 1, 1.0).replace(" CA ", " CB "), "non-CA"),
    ],
)
def test_mdlovofit_stock_parser_rejects_corrupt_models(
    tmp_path: Path, second_model: str, message: str
) -> None:
    path = tmp_path / "invalid_core.pdb"
    path.write_text(
        "REMARK FRAME:        1\n"
        + _atom_line(1, 1, 1.0)
        + _atom_line(2, 2, 0.0)
        + "END\nREMARK FRAME:        2\n"
        + second_model
    )
    with pytest.raises(ValueError, match=message):
        parse_mdlovofit_mask(path, 2)


def test_disagreement_segments_do_not_cross_chains() -> None:
    first = np.asarray([True, False, True, False])
    second = np.asarray([False, False, False, False])
    metadata = [
        {"chain": "A", "residue_id": "1"},
        {"chain": "A", "residue_id": "2"},
        {"chain": "B", "residue_id": "1"},
        {"chain": "B", "residue_id": "2"},
    ]
    segments = disagreement_segments(first, second, metadata)
    assert [(row["chain"], row["length"]) for row in segments] == [("A", 1), ("B", 1)]


def test_preregistration_hash_file_lists_all_contracts() -> None:
    rows = (OUT / "02_preregistration/preregistration.sha256").read_text().splitlines()
    assert len(rows) == 5
    assert all(len(row.split()[0]) == 64 for row in rows)


def test_interrupted_pair_is_invalidated_and_restarted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "raw.tsv"
    monkeypatch.setattr(run_timing, "RAW", raw)
    partial = {field: "" for field in run_timing.FIELDS}
    partial.update(
        {
            "setting_id": "system__f00250",
            "phase": "measured",
            "repeat": 1,
            "pair_attempt": 1,
            "method": METHOD_MDCA,
            "exit_code": 0,
            "timeout": False,
            "invalidated": False,
        }
    )
    run_timing.write_tsv(raw, [partial], fields=run_timing.FIELDS)
    assert run_timing.reconcile_and_completed_pairs() == set()
    updated = read_tsv(raw)[0]
    assert updated["invalidated"] == "True"
    assert updated["invalidation_reason"] == "incomplete_or_failed_paired_attempt"
    assert run_timing.next_pair_attempt("system__f00250", "measured", 1) == 2


def test_process_guard_distinguishes_stopped_from_active_work() -> None:
    header = "PID PPID STAT ELAPSED %CPU %MEM RSS COMMAND COMMAND"
    stopped = header + "\n10 1 Tl 100 0.0 1.0 100 python build_exact_temporal_panel --workers 12"
    running = header + "\n10 1 Rl 100 100.0 1.0 100 python build_exact_temporal_panel --workers 12"
    assert not snapshot_has_active_conflict(stopped)
    assert snapshot_has_active_conflict(running)


def test_resume_phase_order_does_not_repeat_completed_freeze(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        orchestrate,
        "parse_args",
        lambda: type("Args", (), {"start_at": "AT3"})(),
    )
    monkeypatch.setattr(orchestrate, "wait_for_idle", lambda: None)
    monkeypatch.setattr(orchestrate, "run_phase", lambda name, *args: calls.append(name))
    monkeypatch.setattr(orchestrate, "run_validation", lambda: calls.append("VALIDATION"))
    monkeypatch.setattr(orchestrate, "log", lambda message: None)
    orchestrate.main()
    assert "AT0_AT2_FINAL_FREEZE" not in calls
    assert calls[0] == "AT3"


def test_resume_at6_preserves_completed_timing(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        orchestrate,
        "parse_args",
        lambda: type("Args", (), {"start_at": "AT6"})(),
    )
    monkeypatch.setattr(orchestrate, "wait_for_idle", lambda: None)
    monkeypatch.setattr(orchestrate, "run_phase", lambda name, *args: calls.append(name))
    monkeypatch.setattr(orchestrate, "run_validation", lambda: calls.append("VALIDATION"))
    monkeypatch.setattr(orchestrate, "log", lambda message: None)
    orchestrate.main()
    assert calls == ["AT6", "AT7_AT9", "VALIDATION", "AT10_ARCHIVE"]
