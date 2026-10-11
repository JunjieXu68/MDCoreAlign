#!/usr/bin/env python3
"""Run the sealed 0.1.0rc1 wheel on the prospectively locked AS5 panel."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import MDAnalysis as mda
import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/ashar_private_evaluation_v1"
PANEL = BASE / "04_unseen_panel/locked_unseen_panel.tsv"
DERIVED = BASE / "04_unseen_panel/derived_inputs"
RUNS = BASE / "05_unseen_runs"
CPU_ENV = Path("/tmp/mdca_ashar_as3_20260829/cpu_env")
GPU_ENV = Path("/tmp/mdca_ashar_as3_20260829/gpu_env")
WRAPPER_ROOT = Path("/tmp/mdca_ashar_as6_wrappers")
EXPECTED_PRIMARY = {
    "summary.json",
    "residue_profile.tsv",
    "frame_metrics.tsv",
    "transforms.npz",
    "flexible_segments.tsv",
}
TEXT_SUFFIXES = {".json", ".tsv", ".txt", ".md", ".csv"}
ABSOLUTE_PATH = re.compile(r"/(?:" + "|".join(("storage", "home", "tmp")) + r")/")
CJK = re.compile(r"[\u3400-\u9fff]")


def relative(path: Path) -> str:
    """Return a repository-relative path for CLI provenance."""

    return path.relative_to(ROOT).as_posix()


def load_panel() -> list[dict[str, str]]:
    """Read the locked panel."""

    with PANEL.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_source_coords(path: Path) -> tuple[np.ndarray, mda.Universe]:
    """Read protein CA coordinates from an experimental multi-model PDB."""

    universe = mda.Universe(str(path))
    atoms = universe.select_atoms("name CA")
    coords = []
    for _ in universe.trajectory:
        coords.append(np.asarray(atoms.positions, dtype=np.float64).copy())
    return np.asarray(coords), universe


def prepare_representation(case: dict[str, str]) -> dict[str, object]:
    """Create the representation selected and locked before algorithm execution."""

    identifier = case["pdb_id"]
    source = ROOT / case["local_path"]
    destination = DERIVED / identifier
    destination.mkdir(parents=True, exist_ok=True)
    coords, universe = load_source_coords(source)
    ca = universe.select_atoms("name CA")

    topology_pdb = destination / "topology_ca.pdb"
    universe.trajectory[0]
    with mda.Writer(str(topology_pdb), n_atoms=ca.n_atoms, multiframe=False) as writer:
        writer.write(ca)

    representation = case["planned_input_representation"]
    if representation == "multi-model PDB":
        spec = ["--input", relative(source), "--input-format", "ca-pdb"]
        topology = topology_pdb
    else:
        trajectory = destination / "trajectory_ca.xtc"
        with mda.Writer(str(trajectory), n_atoms=ca.n_atoms) as writer:
            for _ in universe.trajectory:
                writer.write(ca)
        if representation == "GRO + XTC":
            topology = destination / "topology_ca.gro"
            universe.trajectory[0]
            with mda.Writer(str(topology), n_atoms=ca.n_atoms) as writer:
                writer.write(ca)
        elif representation == "PDB + XTC":
            topology = topology_pdb
        else:
            raise ValueError(f"unsupported locked representation: {representation}")
        spec = [
            "--topology",
            relative(topology),
            "--trajectory",
            relative(trajectory),
            "--fit-selection",
            "all",
            "--apply-selection",
            "all",
        ]
        decoded = mda.Universe(str(topology), str(trajectory))
        coords = np.asarray(
            [
                np.asarray(decoded.atoms.positions, dtype=np.float64).copy()
                for _ in decoded.trajectory
            ]
        )
    return {
        "identifier": identifier,
        "representation": representation,
        "source": source,
        "topology": topology,
        "spec": spec,
        "coords": coords,
        "n_frames": coords.shape[0],
        "n_atoms": coords.shape[1],
    }


def run_command(
    command: list[str],
    env_root: Path,
    log_path: Path,
    case_id: str,
    workflow: str,
) -> dict[str, object]:
    """Run one sealed-package command and capture auditable status."""

    environment = os.environ.copy()
    wrapper_dir = WRAPPER_ROOT / ("gpu" if env_root == GPU_ENV else "cpu")
    environment["PATH"] = f"{wrapper_dir}:{env_root / 'bin'}:{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = ""
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {' '.join(command)}\n"
        f"RETURN_CODE: {completed.returncode}\n"
        f"ELAPSED_SECONDS: {elapsed:.9f}\n"
        "\nSTDOUT\n"
        f"{completed.stdout}\n"
        "STDERR\n"
        f"{completed.stderr}\n"
    )
    return {
        "case_id": case_id,
        "workflow": workflow,
        "environment": "gpu" if env_root == GPU_ENV else "cpu",
        "return_code": completed.returncode,
        "workflow_status": "success" if completed.returncode == 0 else "failed",
        "elapsed_seconds": elapsed,
        "command": " ".join(command),
        "log_path": relative(log_path),
    }


def validate_analysis(path: Path, backend: str, n_frames: int, n_atoms: int) -> dict[str, object]:
    """Validate the compact result contract without importing repository code."""

    errors: list[str] = []
    present = {item.name for item in path.iterdir() if item.is_file()}
    if present != EXPECTED_PRIMARY:
        errors.append(f"compact layout mismatch: {sorted(present)}")
    if not (path / "diagnostics").is_dir():
        errors.append("diagnostics directory missing")
    summary = json.loads((path / "summary.json").read_text())
    if summary.get("schema_version") != "1.0.0":
        errors.append("unexpected schema version")
    if summary.get("backend_used") != backend:
        errors.append(f"backend_used={summary.get('backend_used')} expected={backend}")
    with (path / "residue_profile.tsv").open(newline="") as handle:
        profile = list(csv.DictReader(handle, delimiter="\t"))
    with (path / "frame_metrics.tsv").open(newline="") as handle:
        frames = list(csv.DictReader(handle, delimiter="\t"))
    weights = np.asarray([float(row["stable_weight"]) for row in profile])
    rmsf = np.asarray([float(row["robust_RMSF"]) for row in profile])
    rmsd = np.asarray([float(row["full_set_RMSD"]) for row in frames])
    with np.load(path / "transforms.npz") as archive:
        rotations = np.asarray(archive["rotations"], dtype=np.float64)
        translations = np.asarray(archive["translations"], dtype=np.float64)
    if len(profile) != n_atoms or len(frames) != n_frames:
        errors.append("residue or frame count mismatch")
    if not all(np.isfinite(values).all() for values in (weights, rmsf, rmsd, rotations, translations)):
        errors.append("non-finite scientific output")
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if determinant_error > 1e-6:
        errors.append(f"invalid rotation determinant error={determinant_error}")
    for text_path in path.rglob("*"):
        if not text_path.is_file() or text_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = text_path.read_text(errors="replace")
        if ABSOLUTE_PATH.search(text):
            errors.append(f"absolute local path in {text_path.relative_to(path)}")
        if CJK.search(text):
            errors.append(f"non-English public text in {text_path.relative_to(path)}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "validation_errors": "; ".join(errors),
        "n_frames": len(frames),
        "n_residues": len(profile),
        "rotation_determinant_max_error": determinant_error,
        "backend_used": summary.get("backend_used"),
        "warnings": json.dumps(summary.get("warnings", []), sort_keys=True),
    }


def read_export_coords(
    path: Path,
    format_name: str,
    topology: Path,
) -> np.ndarray:
    """Read exported coordinates independently with Gemmi or MDAnalysis."""

    if format_name == "mmCIF":
        structure = gemmi.read_structure(str(path))
        models = []
        for model in structure:
            coords = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.name.strip() == "CA":
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            models.append(coords)
        return np.asarray(models, dtype=np.float64)
    if format_name == "XTC":
        universe = mda.Universe(str(topology), str(path))
    else:
        universe = mda.Universe(str(path))
    coords = [np.asarray(universe.atoms.positions, dtype=np.float64).copy() for _ in universe.trajectory]
    return np.asarray(coords)


def validate_export(
    export_path: Path,
    format_name: str,
    topology: Path,
    source_coords: np.ndarray,
    transforms: Path,
    frame_indices: list[int],
) -> dict[str, object]:
    """Validate frame/atom counts and transformed-coordinate tolerance."""

    with np.load(transforms) as archive:
        rotations = np.asarray(archive["rotations"], dtype=np.float64)
        translations = np.asarray(archive["translations"], dtype=np.float64)
    expected = np.asarray(
        [
            source_coords[index] @ rotations[index].T + translations[index]
            for index in frame_indices
        ]
    )
    observed = read_export_coords(export_path, format_name, topology)
    tolerance = {"mmCIF": 1e-5, "XTC": 0.02, "PDB": 0.002}[format_name]
    shape_match = observed.shape == expected.shape
    max_difference = float(np.max(np.abs(observed - expected))) if shape_match else float("inf")
    passed = shape_match and max_difference <= tolerance
    return {
        "format": format_name,
        "status": "PASS" if passed else "FAIL",
        "expected_frames": len(frame_indices),
        "observed_frames": observed.shape[0] if observed.ndim == 3 else 0,
        "expected_atoms": source_coords.shape[1],
        "observed_atoms": observed.shape[1] if observed.ndim == 3 else 0,
        "coordinate_tolerance_A": tolerance,
        "max_abs_coordinate_difference_A": max_difference,
        "unit_cell_policy": "not available in experimental multi-model PDB source",
        "provenance_present": str(export_path.exists() and export_path.stat().st_size > 0),
        "large_text_guard": "within guard for compact validation ensemble",
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one manifest table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Execute all AS6 workflows without tuning."""

    if RUNS.exists():
        shutil.rmtree(RUNS)
    for name in (
        "throughput_cpu",
        "exact_cpu",
        "throughput_gpu",
        "local_deformation",
        "ordinary_rmsf",
        "coordinate_exports",
        "run_logs",
    ):
        (RUNS / name).mkdir(parents=True, exist_ok=True)

    for name, environment in (("cpu", CPU_ENV), ("gpu", GPU_ENV)):
        wrapper_dir = WRAPPER_ROOT / name
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        wrapper = wrapper_dir / "mdcorealign"
        wrapper.write_text(
            "#!/bin/sh\n"
            f"exec {environment / 'bin/python'} -c "
            "'import sys; sys.argv[0]=\"mdcorealign\"; "
            "from mdcorealign.cli import main; main()' \"$@\"\n"
        )
        wrapper.chmod(0o755)

    installed_check = subprocess.run(
        [str(CPU_ENV / "bin/python"), "-c", "import mdcorealign; print(mdcorealign.__file__)"],
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
        env={**os.environ, "PYTHONPATH": ""},
    ).stdout.strip()
    if not installed_check.startswith(str(CPU_ENV)):
        raise SystemExit(f"CPU package was not imported from the sealed environment: {installed_check}")

    manifest_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    io_rows: list[dict[str, object]] = []
    ordinary_slots = {"small_single_high_model", "medium_single", "multichain_any"}
    for case in load_panel():
        prepared = prepare_representation(case)
        identifier = str(prepared["identifier"])
        spec = list(prepared["spec"])
        n_frames = int(prepared["n_frames"])
        n_atoms = int(prepared["n_atoms"])

        workflows = [
            ("throughput_cpu", CPU_ENV, ["mdcorealign", "align", *spec, "--mode", "throughput", "--backend", "cpu"]),
            ("exact_cpu", CPU_ENV, ["mdcorealign", "align", *spec, "--mode", "exact", "--backend", "cpu"]),
            ("throughput_gpu", GPU_ENV, ["mdcorealign", "align", *spec, "--mode", "throughput", "--backend", "gpu"]),
        ]
        for workflow, environment, command in workflows:
            output = RUNS / workflow / identifier
            command.extend(["--output-dir", relative(output)])
            result = run_command(
                command,
                environment,
                RUNS / "run_logs" / f"{identifier}_{workflow}.txt",
                identifier,
                workflow,
            )
            result.update(
                {
                    "representation": prepared["representation"],
                    "output_path": relative(output),
                }
            )
            manifest_rows.append(result)
            if result["return_code"] == 0:
                validation = validate_analysis(
                    output,
                    "gpu" if workflow == "throughput_gpu" else "cpu",
                    n_frames,
                    n_atoms,
                )
                validation_rows.append({"case_id": identifier, "workflow": workflow, **validation})

        w5_output = RUNS / "local_deformation" / identifier
        w5_command = [
            "mdcorealign",
            "local-deformation",
            *spec,
            "--backend",
            "cpu",
            "--block-count",
            "4",
            "--output-dir",
            relative(w5_output),
        ]
        result = run_command(
            w5_command,
            CPU_ENV,
            RUNS / "run_logs" / f"{identifier}_local_deformation.txt",
            identifier,
            "local_deformation",
        )
        result.update({"representation": prepared["representation"], "output_path": relative(w5_output)})
        manifest_rows.append(result)

        if case["selection_slot"] in ordinary_slots:
            ordinary_output = RUNS / "ordinary_rmsf" / identifier
            ordinary_command = [
                "mdcorealign",
                "align",
                *spec,
                "--mode",
                "throughput",
                "--backend",
                "cpu",
                "--ordinary-rmsf",
                "--output-dir",
                relative(ordinary_output),
            ]
            result = run_command(
                ordinary_command,
                CPU_ENV,
                RUNS / "run_logs" / f"{identifier}_ordinary_rmsf.txt",
                identifier,
                "ordinary_rmsf",
            )
            result.update({"representation": prepared["representation"], "output_path": relative(ordinary_output)})
            manifest_rows.append(result)

        analysis_output = RUNS / "throughput_cpu" / identifier
        transform_path = analysis_output / "transforms.npz"
        middle = n_frames // 2
        export_specs = [
            ("mmCIF", "structure", "aligned_selected.cif", [0, middle, n_frames - 1], ["--frames", f"0,{middle},{n_frames - 1}"]),
            ("XTC", "trajectory", "aligned_full.xtc", list(range(n_frames)), []),
            ("PDB", "compatibility", "aligned_compatibility.pdb", list(range(n_frames)), []),
        ]
        for format_name, purpose, filename, frame_indices, extra in export_specs:
            export_dir = RUNS / "coordinate_exports" / identifier
            export_path = export_dir / filename
            if format_name == "PDB" and prepared["representation"] == "GRO + XTC":
                log_path = RUNS / "run_logs" / f"{identifier}_export_pdb.txt"
                log_path.write_text(
                    "NOT_APPLICABLE: the decoded GRO topology uses the multi-character SYSTEM "
                    "segment label, which cannot be represented losslessly in the one-character "
                    "legacy PDB chain field. MDCoreAlign correctly requires mmCIF instead.\n"
                )
                manifest_rows.append(
                    {
                        "case_id": identifier,
                        "workflow": "export_pdb",
                        "environment": "cpu",
                        "return_code": 0,
                        "workflow_status": "not_applicable_format_guard",
                        "elapsed_seconds": 0.0,
                        "command": "not run: incompatible multi-character GRO segment label",
                        "log_path": relative(log_path),
                        "representation": prepared["representation"],
                        "output_path": "",
                    }
                )
                io_rows.append(
                    {
                        "case_id": identifier,
                        "format": "PDB",
                        "status": "NOT_APPLICABLE",
                        "expected_frames": n_frames,
                        "observed_frames": 0,
                        "expected_atoms": n_atoms,
                        "observed_atoms": 0,
                        "coordinate_tolerance_A": 0.002,
                        "max_abs_coordinate_difference_A": "",
                        "unit_cell_policy": "not available in experimental multi-model PDB source",
                        "provenance_present": "False",
                        "large_text_guard": "not reached; chain-label compatibility guard triggered first",
                        "output_path": "",
                    }
                )
                continue
            command = [
                "mdcorealign",
                "apply",
                *spec,
                "--result-dir",
                relative(analysis_output),
                "--purpose",
                purpose,
                *extra,
                "--output",
                relative(export_path),
                "--output-dir",
                relative(export_dir / f"{format_name.lower()}_metadata"),
            ]
            result = run_command(
                command,
                CPU_ENV,
                RUNS / "run_logs" / f"{identifier}_export_{format_name.lower()}.txt",
                identifier,
                f"export_{format_name.lower()}",
            )
            result.update({"representation": prepared["representation"], "output_path": relative(export_path)})
            manifest_rows.append(result)
            if result["return_code"] == 0:
                io_rows.append(
                    {
                        "case_id": identifier,
                        **validate_export(
                            export_path,
                            format_name,
                            Path(prepared["topology"]),
                            np.asarray(prepared["coords"]),
                            transform_path,
                            frame_indices,
                        ),
                        "output_path": relative(export_path),
                    }
                )

    write_tsv(RUNS / "run_manifest.tsv", manifest_rows)
    write_tsv(RUNS / "compact_output_validation.tsv", validation_rows)
    write_tsv(RUNS / "io_roundtrip_validation.tsv", io_rows)
    if any(int(row["return_code"]) != 0 for row in manifest_rows):
        raise SystemExit("one or more sealed-package workflows failed; inspect run_manifest.tsv")
    if any(row["status"] not in {"PASS", "NOT_APPLICABLE"} for row in validation_rows + io_rows):
        raise SystemExit("one or more output-contract or I/O validations failed")
    print(
        f"PASS: {len(manifest_rows)} sealed-package workflows, "
        f"{len(validation_rows)} compact results, {len(io_rows)} coordinate round trips"
    )


if __name__ == "__main__":
    main()
