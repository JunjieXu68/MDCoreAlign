"""Small, isolated regression checks for untimed AT6 export recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.atlas_strict_single_thread_cpu_v1 import extract_cores, prefetch_cores
from scripts.atlas_strict_single_thread_cpu_v1.common import (
    OUT,
    THREAD_ENV,
    read_tsv,
    setting_id,
    sha256,
    write_tsv,
)


def atom_line(serial: int, occupancy: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{serial:4d}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{1.0:6.2f}"
        "          C\n"
    )


@pytest.fixture
def tiny_export(tmp_path: Path, monkeypatch):
    """Freeze a two-frame input and replace only the external program call."""

    source = tmp_path / "input.pdb"
    frame = atom_line(1, 1.0) + atom_line(2, 0.0)
    source.write_text("".join(f"MODEL {i}\n{frame}ENDMDL\n" for i in (1, 2)))
    executable = tmp_path / "stock_mdlovofit"
    executable.write_bytes(b"fake frozen executable")
    out = tmp_path / "results"
    row = {
        "system_id": "tiny_A",
        "prefix_id": "f00002",
        "n_frames": "2",
        "n_ca": "2",
        "input_path": str(source),
        "input_size_bytes": str(source.stat().st_size),
        "input_sha256": sha256(source),
    }
    panel = out / "01_panel/atlas_48_setting_manifest.tsv"
    write_tsv(panel, [row])
    panel.with_suffix(".sha256").write_text(f"{sha256(panel)}  {panel.name}\n")
    write_tsv(
        out / "00_freeze/executable_hashes.tsv",
        [{"path": str(executable), "sha256": sha256(executable)}],
    )
    monkeypatch.setattr(extract_cores, "ROOT", tmp_path)
    monkeypatch.setattr(extract_cores, "OUT", out)
    monkeypatch.setattr(extract_cores, "STOCK_MDLOVOFIT", executable)
    monkeypatch.setattr(extract_cores.os, "sched_getaffinity", lambda pid: {17, 18})
    state = SimpleNamespace(
        row=row,
        source=source,
        executable=executable,
        out=out,
        calls=[],
        output="".join(f"REMARK FRAME: {i:8d}\n{frame}END\n" for i in (1, 2)),
        returncode=0,
    )

    def fake_run(command, **kwargs):
        state.calls.append((command, kwargs))
        Path(command[command.index("-t") + 1]).write_text(state.output)
        kwargs["stdout"].write(b"mock stock output\n")
        kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=state.returncode)

    monkeypatch.setattr(extract_cores.subprocess, "run", fake_run)
    return state


def test_prefetch_has_exact_original_64_export_identities() -> None:
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    schedule = prefetch_cores.export_schedule(panel)
    identities = [(setting_id(row), extraction) for row, extraction in schedule]
    expected = {(setting_id(row), "registered_1") for row in panel}
    expected.update(
        (setting_id(row), extraction)
        for row in panel
        if row["prefix_id"] == "f10001"
        for extraction in ("self_audit_1", "self_audit_2")
    )
    assert len(schedule) == len(set(identities)) == 64
    assert set(identities) == expected
    workloads = [int(row["n_frames"]) * int(row["n_ca"]) for row, _ in schedule]
    assert workloads == sorted(workloads, reverse=True)


def test_prefetch_rejects_duplicate_setting_identity() -> None:
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    panel[0] = panel[1]
    with pytest.raises(ValueError, match="locked 48-setting"):
        prefetch_cores.export_schedule(panel)


def test_validated_export_is_reused_without_another_external_call(tiny_export) -> None:
    first = extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    second = extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=18)
    assert len(tiny_export.calls) == 1
    assert first["checkpoint_reused"] is False
    assert second["checkpoint_reused"] is True
    assert second["logical_cpu"] == 17
    assert second["mask"].tolist() == [True, False]
    assert second["model_count"] == 2
    command, kwargs = tiny_export.calls[0]
    assert command[:3] == ["taskset", "-c", "17"]
    assert command[command.index("-f") + 1] == "0.600000"
    assert command[-1] == str(tiny_export.source)
    assert all(kwargs["env"][key] == value for key, value in THREAD_ENV.items())
    assert not Path(command[command.index("-t") + 1]).exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("input_sha256", "incorrect"),
        ("model_count", 1),
        ("mask", [1, 0]),
        ("mask_sha256", "incorrect"),
        ("status", "failed"),
    ],
)
def test_corrupt_checkpoint_is_rejected_without_rerunning(tiny_export, field, bad_value) -> None:
    extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    checkpoint = tiny_export.out / "06_core_extraction/export_records/tiny_A__f00002__registered_1.json"
    record = json.loads(checkpoint.read_text())
    record[field] = bad_value
    checkpoint.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checkpoint"):
        extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    assert len(tiny_export.calls) == 1


def test_corrupt_checkpoint_log_is_rejected_without_rerunning(tiny_export) -> None:
    record = extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    (tiny_export.out.parent / record["stdout_path"]).write_text("changed stdout")
    with pytest.raises(ValueError, match="log hash mismatch"):
        extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    assert len(tiny_export.calls) == 1


@pytest.mark.parametrize("changed", ["input", "executable"])
def test_cached_export_rechecks_frozen_input_and_executable(tiny_export, changed) -> None:
    extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    path = tiny_export.source if changed == "input" else tiny_export.executable
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(ValueError, match="SHA-256|frozen executable"):
        extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    assert len(tiny_export.calls) == 1


def test_wrong_frame_count_preserves_failed_export_and_does_not_publish_cache(tiny_export) -> None:
    tiny_export.output = "REMARK FRAME:        1\n" + atom_line(1, 1.0) + atom_line(2, 0.0) + "END\n"
    with pytest.raises(ValueError, match="mapping/mask mismatch"):
        extract_cores.run_mdlovofit_export(tiny_export.row, "registered_1", logical_cpu=17)
    assert not list((tiny_export.out / "06_core_extraction/export_records").glob("*.json"))
    attempts = list((tiny_export.out / "06_core_extraction/export_attempts").iterdir())
    assert len(attempts) == 1
    failure = json.loads((attempts[0] / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert (attempts[0] / "failed_aligned_core.pdb").read_text() == tiny_export.output
    assert (attempts[0] / "stdout.txt").is_file()
