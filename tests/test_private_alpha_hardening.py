"""Private-alpha hardening tests for user-facing MDCoreAlign behavior."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mdsuperpose.flexibility import detect_flexible_segments

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    """Return an environment that imports the local src package."""

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    xyz: tuple[float, float, float],
    *,
    element: str = "C",
) -> str:
    """Format a minimal PDB atom line compatible with the private-alpha parser."""

    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {atom_name[:4]:>4} {resname[:3]:>3} {chain[:1]:1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element[:2]:>2}\n"
    )


def _write_multimodel(path: Path, frames: list[list[str]]) -> None:
    """Write a minimal multi-model PDB."""

    lines: list[str] = []
    for model_index, atom_lines in enumerate(frames, start=1):
        lines.append(f"MODEL     {model_index:4d}\n")
        lines.extend(atom_lines)
        lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def _protein_with_ligand_pdb(path: Path) -> None:
    """Create a small protein CA plus HETATM trajectory."""

    base = [
        ("ATOM", 1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
        ("ATOM", 2, "CA", "GLY", "A", 2, (1.0, 0.0, 0.0), "C"),
        ("ATOM", 3, "CA", "SER", "A", 3, (0.0, 1.0, 0.0), "C"),
        ("ATOM", 4, "CA", "THR", "A", 4, (0.0, 0.0, 1.0), "C"),
        ("HETATM", 5, "C1", "LIG", "L", 1, (2.0, 2.0, 2.0), "C"),
    ]
    frames: list[list[str]] = []
    for shift in (0.0, 0.2):
        frames.append(
            [
                _pdb_line(record, serial, atom, resname, chain, resid, (xyz[0] + shift, xyz[1], xyz[2]), element=element)
                for record, serial, atom, resname, chain, resid, xyz, element in base
            ]
        )
    _write_multimodel(path, frames)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class PrivateAlphaHardeningTest(unittest.TestCase):
    """Validate private-alpha usability safeguards."""

    def _run(self, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=_env(),
            check=check,
            capture_output=True,
            text=True,
        )

    def test_fit_apply_selection_transforms_extra_atoms_without_changing_weights(self) -> None:
        """Apply-selection can include HETATM records without changing fit weights."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "protein_ligand.pdb"
            _protein_with_ligand_pdb(pdb)
            fit_only = tmp / "fit_only"
            apply_all = tmp / "apply_all"
            common = [
                sys.executable,
                "-m",
                "mdcorealign",
                "align",
                "--input",
                str(pdb),
                "--input-format",
                "pdb",
                "--fit-selection",
                "protein and name CA",
                "--mode",
                "default",
                "--legacy-artifacts",
            ]
            self._run([*common, "--apply-selection", "protein and name CA", "--output-dir", str(fit_only)])
            self._run([*common, "--apply-selection", "all", "--output-dir", str(apply_all)])

            fit_weights = [float(row["soft_stability_weight"]) for row in _read_csv(fit_only / "weights.csv")]
            all_weights = [float(row["soft_stability_weight"]) for row in _read_csv(apply_all / "weights.csv")]
            np.testing.assert_allclose(fit_weights, all_weights)
            self.assertEqual(len(fit_weights), 4)
            self.assertIn("HETATM", (apply_all / "aligned_trajectory.pdb").read_text())
            summary = json.loads((apply_all / "summary.json").read_text())
            self.assertEqual(summary["n_fit_atoms"], 4)
            self.assertEqual(summary["n_analysis_coordinate_atoms"], 4)
            self.assertEqual(summary["n_exportable_atoms"], 5)
            self.assertEqual(summary["n_materialized_atoms"], 5)
            self.assertEqual(summary["materialized_selection"], "all")
            self.assertEqual(summary["n_exportable_atoms_scope"], "materialized_selection")
            self.assertEqual(summary["n_default_export_atoms"], 4)
            manifest = json.loads((apply_all / "diagnostics/source_manifest.json").read_text())
            self.assertEqual(manifest["n_exportable_atoms"], 4)
            self.assertEqual(summary["legacy_metadata"]["n_analysis_loaded_atoms"], 5)
            self.assertIn("apply-selection includes atoms outside fit-selection", "; ".join(summary["warnings"]))

    def test_dry_run_writes_audit_without_alignment(self) -> None:
        """Dry-run validates selections and does not create aligned trajectory output."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "protein_ligand.pdb"
            out = tmp / "dry"
            _protein_with_ligand_pdb(pdb)
            self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "align",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "pdb",
                    "--legacy-artifacts",
                    "--output-dir",
                    str(out),
                    "--dry-run",
                ]
            )
            self.assertTrue((out / "input_audit.tsv").exists())
            self.assertTrue((out / "input_audit.json").exists())
            self.assertTrue((out / "chain_mapping_report.tsv").exists())
            self.assertTrue((out / "diagnostics" / "run_metadata.json").exists())
            self.assertFalse((out / "aligned_trajectory.pdb").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["alignment_run"])

    def test_empty_fit_selection_error_is_actionable(self) -> None:
        """A zero-atom fit selection should fail with a useful message."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "hetatm_only.pdb"
            frames = [
                [_pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0.0, 0.0, 0.0))],
                [_pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0.1, 0.0, 0.0))],
            ]
            _write_multimodel(pdb, frames)
            completed = self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "align",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "pdb",
                    "--output-dir",
                    str(tmp / "out"),
                    "--dry-run",
                ],
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("zero atoms", completed.stderr)
            self.assertIn("audit-preprocessing", completed.stderr)

    def test_empty_apply_selection_error_is_actionable(self) -> None:
        """A zero-atom apply selection should fail before alignment."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "hetatm_only.pdb"
            frames = [
                [_pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0.0, 0.0, 0.0))],
                [_pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0.1, 0.0, 0.0))],
            ]
            _write_multimodel(pdb, frames)
            completed = self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "align",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "pdb",
                    "--fit-selection",
                    "all",
                    "--apply-selection",
                    "protein",
                    "--output-dir",
                    str(tmp / "out"),
                    "--dry-run",
                ],
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("apply-selection", completed.stderr)
            self.assertIn("zero atoms", completed.stderr)

    def test_array_input_with_nonfinite_coordinates_is_actionable(self) -> None:
        """Array input containing NaN or Inf values should be rejected clearly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for bad_value in (np.nan, np.inf):
                coords = np.zeros((2, 3, 3), dtype=float)
                coords[1, 1, 2] = bad_value
                array_path = tmp / f"bad_{'nan' if np.isnan(bad_value) else 'inf'}.npy"
                np.save(array_path, coords)
                completed = self._run(
                    [
                        sys.executable,
                        "-m",
                        "mdcorealign",
                        "audit-preprocessing",
                        "--input",
                        str(array_path),
                        "--input-format",
                        "array",
                        "--output-dir",
                        str(tmp / f"audit_{array_path.stem}"),
                    ],
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("NaN or infinite values", completed.stderr)
                self.assertIn("clean the trajectory", completed.stderr)

    def test_mismatched_pdb_model_atom_count_is_actionable(self) -> None:
        """Changing atom counts across models should be rejected clearly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "mismatch.pdb"
            frames = [
                [
                    _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
                    _pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1.0, 0.0, 0.0)),
                ],
                [_pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0.1, 0.0, 0.0))],
            ]
            _write_multimodel(pdb, frames)
            completed = self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "audit-preprocessing",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "ca-pdb",
                    "--output-dir",
                    str(tmp / "audit"),
                ],
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("atom counts mismatch", completed.stderr)

    def test_large_jump_detection_in_input_audit(self) -> None:
        """Input audit should flag very large frame-to-frame CA jumps."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "jump.pdb"
            frames = [
                [
                    _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
                    _pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1.0, 0.0, 0.0)),
                ],
                [
                    _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (100.0, 0.0, 0.0)),
                    _pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1.0, 0.0, 0.0)),
                ],
            ]
            _write_multimodel(pdb, frames)
            out = tmp / "audit"
            self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "audit-preprocessing",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "ca-pdb",
                    "--output-dir",
                    str(out),
                    "--jump-threshold",
                    "15",
                ]
            )
            row = _read_tsv(out / "input_audit.tsv")[0]
            self.assertEqual(row["suspected_PBC_imaging_issue"], "True")
            self.assertGreater(float(row["large_frame_to_frame_displacement_A"]), 15.0)

    def test_missing_chain_ids_are_reported(self) -> None:
        """Input audit should report missing chain identifiers."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb = tmp / "missing_chain.pdb"
            frames = [
                [
                    _pdb_line("ATOM", 1, "CA", "ALA", "", 1, (0.0, 0.0, 0.0)),
                    _pdb_line("ATOM", 2, "CA", "GLY", "", 2, (1.0, 0.0, 0.0)),
                ],
                [
                    _pdb_line("ATOM", 1, "CA", "ALA", "", 1, (0.1, 0.0, 0.0)),
                    _pdb_line("ATOM", 2, "CA", "GLY", "", 2, (1.1, 0.0, 0.0)),
                ],
            ]
            _write_multimodel(pdb, frames)
            out = tmp / "audit"
            self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "audit-preprocessing",
                    "--input",
                    str(pdb),
                    "--input-format",
                    "ca-pdb",
                    "--output-dir",
                    str(out),
                ]
            )
            row = _read_tsv(out / "input_audit.tsv")[0]
            chain_row = _read_tsv(out / "chain_mapping_report.tsv")[0]
            self.assertEqual(row["missing_chain_ids"], "2")
            self.assertEqual(chain_row["missing_chain_id"], "1")

    def test_summary_and_run_metadata_schema(self) -> None:
        """Full flexibility run should expose stable summary and metadata fields."""

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "flex"
            self._run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "flexibility",
                    "--input",
                    str(ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"),
                    "--input-format",
                    "ca-pdb",
                    "--output-dir",
                    str(out),
                    "--system-name",
                    "toy_schema",
                ]
            )
            summary = json.loads((out / "summary.json").read_text())
            metadata = json.loads((out / "run_metadata.json").read_text())
            for field in [
                "method",
                "fit_selection",
                "analysis_coordinate_selection",
                "n_frames",
                "n_fit_atoms",
                "n_fit_residues",
                "n_analysis_coordinate_atoms",
                "default_export_selection",
                "n_exportable_atoms",
                "source_is_ca_only",
                "core_threshold",
                "flexible_percentile",
                "converged",
                "n_iterations",
                "effective_core_size",
                "weighted_RMSD",
                "unified_posthoc_RMSD",
                "warnings",
            ]:
                self.assertIn(field, summary)
            for field in [
                "package_version",
                "python_version",
                "platform",
                "command",
                "timestamp",
                "input_files",
                "output_files",
                "external_tools_detected",
                "random_seed_if_any",
            ]:
                self.assertIn(field, metadata)

    def test_flexible_segments_do_not_merge_across_chains(self) -> None:
        """Flexible segment detection must respect chain boundaries."""

        mobility = np.asarray([1.0, 1.0, 0.1, 1.0, 1.0])
        chains = ["A", "A", "A", "B", "B"]
        segment_ids = detect_flexible_segments(
            mobility,
            percentile_cutoff=0.5,
            min_segment_length=2,
            merge_gap=0,
            chain=chains,
        )
        self.assertEqual(segment_ids.tolist(), [1, 1, 0, 2, 2])

    def test_private_alpha_bundle_template_schema(self) -> None:
        """Advisor feedback template should have stable, user-facing columns."""

        script = ROOT / "scripts" / "package_private_alpha_bundle.py"
        spec = importlib.util.spec_from_file_location("package_private_alpha_bundle", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["package_private_alpha_bundle"] = module
        spec.loader.exec_module(module)
        module.main()
        template = ROOT / "results" / "private_alpha_hardening" / "advisor_testing_template.tsv"
        with template.open(newline="") as handle:
            columns = list(csv.DictReader(handle, delimiter="\t").fieldnames or [])
        self.assertEqual(columns, module.FEEDBACK_FIELDS)


if __name__ == "__main__":
    unittest.main()
