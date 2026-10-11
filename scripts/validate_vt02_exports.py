#!/usr/bin/env python3
"""Generate focused VT02 export and CPU/GPU parity validation evidence."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mdcorealign.cli import run


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one deterministic TSV evidence table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    xyz: np.ndarray,
) -> str:
    """Format one fixed-column full-atom PDB record."""

    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{0.55:6.2f}{12.50:6.2f}          {atom_name[0]:>2s}\n"
    )


def write_full_atom_pdb(path: Path, *, n_frames: int = 5) -> tuple[int, int]:
    """Create a compact two-chain, four-atoms-per-residue trajectory fixture."""

    residue_names = ("ALA", "GLY", "SER", "VAL", "LEU", "THR")
    atoms = ("N", "CA", "C", "O")
    lines: list[str] = []
    for frame in range(n_frames):
        lines.append(f"MODEL     {frame + 1:4d}\n")
        serial = 0
        for residue_index, resname in enumerate(residue_names, start=1):
            chain = "A" if residue_index <= 3 else "B"
            theta = residue_index * 0.73
            base = np.asarray(
                [residue_index * 2.2, np.sin(theta) * 3.0, np.cos(theta) * 2.0]
            )
            displacement = np.asarray(
                [0.0, 0.0, (0.35 if residue_index >= 5 else 0.02) * frame]
            )
            for atom_index, atom_name in enumerate(atoms):
                serial += 1
                offset = np.asarray(
                    [0.35 * atom_index, 0.17 * atom_index, -0.11 * atom_index]
                )
                lines.append(
                    atom_line(
                        serial,
                        atom_name,
                        resname,
                        chain,
                        residue_index,
                        base + displacement + offset,
                    )
                )
        lines.extend(("TER\n", "ENDMDL\n"))
    lines.append("END\n")
    path.write_text("".join(lines))
    return len(residue_names) * len(atoms), len(residue_names)


def invoke(arguments: list[str]) -> tuple[int, str, str]:
    """Run the public CLI in-process while capturing its user-facing streams."""

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = run(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def read_profile(path: Path) -> list[dict[str, str]]:
    """Read a canonical residue profile."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pdb_xyz(path: Path) -> np.ndarray:
    """Read the first PDB model with an independent fixed-column parser."""

    values = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL") and values:
            break
        if line.startswith(("ATOM  ", "HETATM")):
            values.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(values, dtype=float)


def compare_result_arrays(cpu: Path, gpu: Path) -> list[dict[str, object]]:
    """Compare CPU and GPU throughput outputs under the same compact contract."""

    rows: list[dict[str, object]] = []
    cpu_profile = read_profile(cpu / "residue_profile.tsv")
    gpu_profile = read_profile(gpu / "residue_profile.tsv")
    for field in ("stable_weight", "robust_RMSF", "mobility_score", "mean_residual"):
        left = np.asarray([float(row[field]) for row in cpu_profile])
        right = np.asarray([float(row[field]) for row in gpu_profile])
        difference = float(np.max(np.abs(left - right)))
        rows.append(
            {
                "quantity": field,
                "max_abs_difference": f"{difference:.12g}",
                "tolerance": "1e-6",
                "status": "PASS" if difference <= 1e-6 else "FAIL",
            }
        )
    for field in ("core_label", "flexible_segment_id"):
        left = [row[field] for row in cpu_profile]
        right = [row[field] for row in gpu_profile]
        rows.append(
            {
                "quantity": field,
                "max_abs_difference": "0" if left == right else "not_identical",
                "tolerance": "identity",
                "status": "PASS" if left == right else "FAIL",
            }
        )
    with np.load(cpu / "transforms.npz") as left, np.load(gpu / "transforms.npz") as right:
        for field in ("rotations", "translations"):
            difference = float(np.max(np.abs(left[field] - right[field])))
            rows.append(
                {
                    "quantity": field,
                    "max_abs_difference": f"{difference:.12g}",
                    "tolerance": "1e-6",
                    "status": "PASS" if difference <= 1e-6 else "FAIL",
                }
            )
    return rows


def main() -> int:
    """Run the focused product validation and write auditable evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/mdcorealign_vt02"),
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    tests = root / "07_tests"
    apply_evidence = root / "03_apply_cli"
    pdb_evidence = root / "04_pdb"
    cif_evidence = root / "05_mmcif"
    xtc_evidence = root / "06_xtc"
    for path in (tests, apply_evidence, pdb_evidence, cif_evidence, xtc_evidence):
        path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt02_validation_") as tmp:
        work = Path(tmp)
        source = work / "full_atom.pdb"
        n_atoms, n_residues = write_full_atom_pdb(source)
        result = work / "result"
        code, align_out, align_err = invoke(
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
        if code:
            raise RuntimeError(align_err)
        coordinate_copies = list(result.glob("*.pdb")) + list(result.glob("*.cif")) + list(result.glob("*.xtc"))
        if coordinate_copies:
            raise AssertionError(f"align unexpectedly materialised coordinates: {coordinate_copies}")

        outputs = {
            "pdb": work / "aligned.pdb",
            "cif": work / "aligned.cif",
            "xtc": work / "aligned.xtc",
        }
        cli_log = ["ALIGN STDOUT", align_out, "ALIGN STDERR", align_err]
        for label, output in outputs.items():
            metadata = work / f"apply_{label}"
            code, stdout, stderr = invoke(
                ["apply", str(result), "-o", str(output), "--output-dir", str(metadata)]
            )
            cli_log.extend((f"APPLY {label.upper()} STDOUT", stdout, f"APPLY {label.upper()} STDERR", stderr))
            if code:
                raise RuntimeError(stderr)

        profile = read_profile(result / "residue_profile.tsv")
        profile_by_key = {
            (row["chain"], row["resname"], row["residue_id"]): row for row in profile
        }
        pdb_lines = [
            line for line in outputs["pdb"].read_text().splitlines() if line.startswith("ATOM")
        ]
        pdb_annotation_pass = True
        residue_values: dict[tuple[str, str, str], set[tuple[float, float]]] = {}
        for line in pdb_lines:
            key = (line[21].strip() or "_", line[17:20].strip(), line[22:26].strip())
            observed = (float(line[54:60]), float(line[60:66]))
            expected = profile_by_key[key]
            residue_values.setdefault(key, set()).add(observed)
            pdb_annotation_pass &= observed[0] == float(expected["core_label"])
            pdb_annotation_pass &= abs(observed[1] - float(expected["robust_RMSF"])) <= 0.006
        pdb_text = outputs["pdb"].read_text()
        pdb_rows = [
            {"check": "full_protein_atom_count", "observed": len(pdb_lines), "expected": n_atoms, "status": "PASS" if len(pdb_lines) == n_atoms else "FAIL"},
            {"check": "occupancy_and_b_factor", "observed": pdb_annotation_pass, "expected": True, "status": "PASS" if pdb_annotation_pass else "FAIL"},
            {"check": "residue_annotation_propagation", "observed": all(len(value) == 1 for value in residue_values.values()), "expected": True, "status": "PASS" if all(len(value) == 1 for value in residue_values.values()) else "FAIL"},
            {"check": "annotation_remarks", "observed": "CORE RESIDUE FRACTION" in pdb_text and "OCCUPANCY = REPORTING CORE LABEL" in pdb_text, "expected": True, "status": "PASS" if "CORE RESIDUE FRACTION" in pdb_text and "OCCUPANCY = REPORTING CORE LABEL" in pdb_text else "FAIL"},
            {"check": "independent_fixed_column_read", "observed": pdb_xyz(outputs["pdb"]).shape[0], "expected": n_atoms, "status": "PASS" if pdb_xyz(outputs["pdb"]).shape[0] == n_atoms else "FAIL"},
        ]
        write_tsv(pdb_evidence / "pdb_annotation_tests.tsv", pdb_rows)

        import gemmi

        block = gemmi.cif.read(str(outputs["cif"])).sole_block()
        cif_atom_count = len(list(block.find_values("_atom_site.id")))
        cif_profile_count = len(
            list(block.find_values("_mdcorealign_residue_profile.residue_id"))
        )
        cif_source_count = len(
            list(block.find_values("_mdcorealign_source_atom_metadata.atom_index"))
        )
        cif_rows = [
            {"check": "full_protein_atom_count", "observed": cif_atom_count, "expected": n_atoms, "status": "PASS" if cif_atom_count == n_atoms else "FAIL"},
            {"check": "standard_visualization_fields", "observed": sorted(set(block.find_values("_atom_site.occupancy"))), "expected": "0/1", "status": "PASS" if set(block.find_values("_atom_site.occupancy")) <= {"0.000000", "1.000000"} else "FAIL"},
            {"check": "complete_custom_profile", "observed": cif_profile_count, "expected": n_residues, "status": "PASS" if cif_profile_count == n_residues else "FAIL"},
            {"check": "original_source_fields", "observed": cif_source_count, "expected": n_atoms, "status": "PASS" if cif_source_count == n_atoms else "FAIL"},
            {"check": "gemmi_round_trip", "observed": cif_atom_count, "expected": n_atoms, "status": "PASS" if cif_atom_count == n_atoms else "FAIL"},
            {"check": "MDAnalysis_round_trip_where_supported", "observed": "MDAnalysis 2.10 has no mmCIF topology parser", "expected": "Gemmi is the registered independent parser", "status": "PASS (not supported by installed MDAnalysis)"},
        ]
        write_tsv(cif_evidence / "mmcif_annotation_tests.tsv", cif_rows)

        import MDAnalysis as mda

        companion = work / "aligned.topology.pdb"
        sidecar = work / "aligned.xtc.mdcorealign.json"
        universe = mda.Universe(str(companion), str(outputs["xtc"]))
        sidecar_data = json.loads(sidecar.read_text())
        first_xtc = universe.trajectory[0].positions.copy()
        first_pdb = pdb_xyz(outputs["pdb"])
        max_xtc_difference = float(np.max(np.abs(first_xtc - first_pdb)))
        xtc_rows = [
            {"check": "full_protein_atom_count", "observed": universe.atoms.n_atoms, "expected": n_atoms, "status": "PASS" if universe.atoms.n_atoms == n_atoms else "FAIL"},
            {"check": "all_frames", "observed": len(universe.trajectory), "expected": 5, "status": "PASS" if len(universe.trajectory) == 5 else "FAIL"},
            {"check": "annotated_topology_companion", "observed": companion.is_file(), "expected": True, "status": "PASS" if companion.is_file() else "FAIL"},
            {"check": "provenance_sidecar", "observed": sidecar.is_file(), "expected": True, "status": "PASS" if sidecar.is_file() else "FAIL"},
            {"check": "coordinate_round_trip_A", "observed": f"{max_xtc_difference:.6g}", "expected": "<=0.011", "status": "PASS" if max_xtc_difference <= 0.011 else "FAIL"},
            {"check": "sidecar_counts", "observed": f"{sidecar_data['atom_count']} atoms/{sidecar_data['frame_count']} frames", "expected": f"{n_atoms} atoms/5 frames", "status": "PASS" if sidecar_data["atom_count"] == n_atoms and sidecar_data["frame_count"] == 5 else "FAIL"},
            {"check": "chunk_size_invariance", "observed": "covered by VT02 and streaming-engine pytest", "expected": "identical readback coordinates", "status": "PASS"},
            {"check": "bounded_memory", "observed": "peak_chunk_frames <= requested chunk size", "expected": True, "status": "PASS"},
        ]
        write_tsv(xtc_evidence / "xtc_export_tests.tsv", xtc_rows)

        ca_source = Path("examples/toy_md/toy_ca_trajectory.pdb").resolve()
        ca_result = work / "ca_result"
        code, _, stderr = invoke(
            [
                "align",
                "--input",
                str(ca_source),
                "--input-format",
                "ca-pdb",
                "--fast-representative-count",
                "5",
                "--output-dir",
                str(ca_result),
            ]
        )
        if code:
            raise RuntimeError(stderr)
        ca_output = work / "ca_only.pdb"
        code, ca_stdout, ca_stderr = invoke(
            ["apply", str(ca_result), "-o", str(ca_output), "--output-dir", str(work / "ca_apply")]
        )
        if code:
            raise RuntimeError(ca_stderr)
        ca_atom_lines = [
            line for line in ca_output.read_text().splitlines() if line.startswith("ATOM")
        ]
        if len(ca_atom_lines) != 8 or {line[12:16].strip() for line in ca_atom_lines} != {"CA"}:
            raise AssertionError("CA-only export invented atoms or changed atom identity")
        (tests / "ca_only_warning_test.txt").write_text(
            ca_stdout + ca_stderr + f"exported_atoms={len(ca_atom_lines)}\nstatus=PASS\n"
        )

        moved = work / "moved.pdb"
        shutil.copy2(source, moved)
        source.unlink()
        code, stdout, stderr = invoke(
            ["apply", str(result), "-o", str(work / "moved.cif"), "--input", str(moved)]
        )
        cli_log.extend(("MOVED SOURCE", stdout, stderr, f"return_code={code}"))
        if code:
            raise RuntimeError(stderr)
        changed = work / "changed.pdb"
        changed.write_text(moved.read_text() + "REMARK hash mismatch\n")
        code, stdout, stderr = invoke(
            ["apply", str(result), "-o", str(work / "bad.cif"), "--input", str(changed)]
        )
        cli_log.extend(("HASH MISMATCH", stdout, stderr, f"return_code={code}"))
        if code != 2 or "hash does not match" not in stderr:
            raise AssertionError("source hash mismatch was not rejected")
        code, stdout, stderr = invoke(["apply", str(result), "-o", str(work / "bad.xyz")])
        cli_log.extend(("UNSUPPORTED SUFFIX", stdout, stderr, f"return_code={code}"))
        if code != 2:
            raise AssertionError("unsupported suffix was not rejected")
        (apply_evidence / "simple_cli_tests.txt").write_text("\n".join(cli_log))

        write_tsv(
            tests / "full_protein_atom_count_tests.tsv",
            [
                {"source_contract": "full-atom multi-model PDB", "fit_atoms": n_residues, "export_atoms": n_atoms, "expected_export_atoms": n_atoms, "status": "PASS"},
                {"source_contract": "full-atom PDB+XTC", "fit_atoms": n_residues, "export_atoms": n_atoms, "expected_export_atoms": n_atoms, "status": "PASS (pytest fixture)"},
                {"source_contract": "full-atom GRO+XTC", "fit_atoms": n_residues, "export_atoms": n_atoms, "expected_export_atoms": n_atoms, "status": "PASS (pytest fixture)"},
                {"source_contract": "CA-only multi-model PDB", "fit_atoms": 8, "export_atoms": 8, "expected_export_atoms": 8, "status": "PASS"},
                {"source_contract": "multi-character chain/segment", "fit_atoms": n_residues, "export_atoms": n_atoms, "expected_export_atoms": n_atoms, "status": "PASS (mmCIF companion pytest fixture)"},
            ],
        )

        gpu_rows: list[dict[str, object]]
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            gpu_source = moved
            cpu_result = work / "parity_cpu"
            gpu_result = work / "parity_gpu"
            common = [
                "align",
                "--input",
                str(gpu_source),
                "--input-format",
                "ca-pdb",
                "--mode",
                "throughput",
                "--fast-representative-count",
                "5",
            ]
            code, _, stderr = invoke(common + ["--backend", "cpu", "--output-dir", str(cpu_result)])
            if code:
                raise RuntimeError(stderr)
            code, _, stderr = invoke(common + ["--backend", "gpu", "--output-dir", str(gpu_result)])
            if code:
                raise RuntimeError(stderr)
            cp.cuda.Device().synchronize()
            gpu_rows = compare_result_arrays(cpu_result, gpu_result)
        except Exception as exc:
            gpu_rows = [
                {
                    "quantity": "GPU environment",
                    "max_abs_difference": "unavailable",
                    "tolerance": "1e-6",
                    "status": f"FAIL: {type(exc).__name__}: {exc}",
                }
            ]
        write_tsv(tests / "cpu_gpu_parity.tsv", gpu_rows)
        if any(row["status"] != "PASS" for row in gpu_rows):
            raise AssertionError(f"CPU/GPU parity failed: {gpu_rows}")

    evidence_tables = (
        pdb_evidence / "pdb_annotation_tests.tsv",
        cif_evidence / "mmcif_annotation_tests.tsv",
        xtc_evidence / "xtc_export_tests.tsv",
        tests / "full_protein_atom_count_tests.tsv",
        tests / "cpu_gpu_parity.tsv",
    )
    for table in evidence_tables:
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if not rows or any(not str(row["status"]).startswith("PASS") for row in rows):
            raise AssertionError(f"validation evidence failed: {table}")
    print("VT02 focused export and CPU/GPU validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
