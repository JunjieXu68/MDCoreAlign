"""Tests for the external MDLovoFit wrapper."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    build_mdlovofit_command,
    parse_mdlovofit_stdout,
    read_multimodel_pdb_coords,
    read_multimodel_pdb_with_occupancy,
    resolve_mdlovofit_executable,
    write_multimodel_pdb,
)


class MdlovofitWrapperTest(unittest.TestCase):
    """Check wrapper configuration and parsing behavior."""

    def test_parse_stdout_leaves_missing_fields_as_none(self) -> None:
        """Missing MDLovoFit fields should remain explicitly unavailable."""

        stdout = (
            "Fraction of best aligned atoms to consider: 0.600000\n"
            "Total number of atoms: 131\n"
            "Average RMSD of least mobile atoms: 0.812\n"
            "Average RMSD of all atoms: 1.527\n"
        )
        parsed = parse_mdlovofit_stdout(stdout)

        self.assertAlmostEqual(parsed.fraction_of_best_aligned_atoms, 0.6)
        self.assertEqual(parsed.total_number_of_atoms, 131)
        self.assertAlmostEqual(parsed.average_rmsd_least_mobile, 0.812)
        self.assertIsNone(parsed.average_rmsd_most_mobile)
        self.assertAlmostEqual(parsed.average_rmsd_all_atoms, 1.527)
        self.assertIsNone(parsed.number_of_atoms_of_selection)
        self.assertIsNone(parsed.frame_rmsd_row_count)

    def test_parse_stdout_extracts_frame_row_count_and_control_fields(self) -> None:
        """The parser should extract the visible MDLovoFit control and frame-table fields."""

        stdout = (
            "# Fraction of best aligned atoms to consider:  0.60\n"
            "# Different assignment for each frame: F\n"
            "# Number of trials for each alignment:   100\n"
            "# Total number of atoms:      131\n"
            "# Number of atoms of selection:      131\n"
            "# Total number of atoms found:      131\n"
            "          1          0.000000000          0.000000000          0.000000000\n"
            "          2          0.675922039          1.883027069          1.164292776\n"
            "# Average RMSD of least mobile atoms:      0.67592\n"
            "# Average RMSD of most mobile atoms:       1.88303\n"
            "# Average RMSD of all atoms:               1.16429\n"
        )

        parsed = parse_mdlovofit_stdout(stdout)

        self.assertAlmostEqual(parsed.fraction_of_best_aligned_atoms, 0.6)
        self.assertEqual(parsed.different_assignment_for_each_frame, "F")
        self.assertEqual(parsed.number_of_trials_for_each_alignment, 100)
        self.assertEqual(parsed.total_number_of_atoms, 131)
        self.assertEqual(parsed.number_of_atoms_of_selection, 131)
        self.assertEqual(parsed.total_number_of_atoms_found, 131)
        self.assertEqual(parsed.frame_rmsd_row_count, 2)
        self.assertAlmostEqual(parsed.average_rmsd_least_mobile, 0.67592)
        self.assertAlmostEqual(parsed.average_rmsd_most_mobile, 1.88303)
        self.assertAlmostEqual(parsed.average_rmsd_all_atoms, 1.16429)

    def test_command_builder_uses_fraction_output_and_inputs(self) -> None:
        """The external command line should contain the requested paths."""

        command = build_mdlovofit_command(
            "/opt/mdlovofit",
            fraction=0.6,
            output_path="aligned.pdb",
            input_paths=["traj1.pdb", "traj2.pdb"],
        )

        self.assertEqual(
            command,
            ["/opt/mdlovofit", "-f", "0.600000", "-t", "aligned.pdb", "traj1.pdb", "traj2.pdb"],
        )

    def test_resolve_executable_prefers_explicit_then_env_then_config_then_path(self) -> None:
        """Executable resolution should follow the documented precedence."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            explicit = tmp / "explicit_mdlovofit"
            env_exec = tmp / "env_mdlovofit"
            config_exec = tmp / "config_mdlovofit"
            path_exec = tmp / "path_mdlovofit"
            for executable in (explicit, env_exec, config_exec, path_exec):
                executable.write_text("#!/bin/sh\n")
                executable.chmod(0o755)

            config_path = tmp / "config.json"
            config_path.write_text(json.dumps({"mdlovofit_executable": str(config_exec)}))

            with patch.dict(os.environ, {"MDLOVOFIT_EXECUTABLE": str(env_exec)}, clear=False):
                with patch("shutil.which", return_value=str(path_exec)):
                    resolved = resolve_mdlovofit_executable(
                        explicit,
                        config_path=config_path,
                    )
                    self.assertEqual(resolved, explicit)

                    resolved = resolve_mdlovofit_executable(
                        None,
                        config_path=config_path,
                    )
                    self.assertEqual(resolved, env_exec)

            with patch.dict(os.environ, {}, clear=True):
                with patch("shutil.which", return_value=str(path_exec)):
                    resolved = resolve_mdlovofit_executable(None, config_path=config_path)
                    self.assertEqual(resolved, config_exec)

                    resolved = resolve_mdlovofit_executable(None, config_path=None)
                    self.assertEqual(resolved, path_exec)

    def test_missing_executable_fails_clearly(self) -> None:
        """Resolution should raise a clear error when nothing can be found."""

        with patch.dict(os.environ, {}, clear=True):
            with patch("shutil.which", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    resolve_mdlovofit_executable(None, config_path=None)

    def test_multimodel_pdb_roundtrip_preserves_coordinates(self) -> None:
        """The helper PDB writer and reader should round-trip trajectory data."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],
                [[0.1, -0.1, 0.2], [0.9, 0.6, -0.4]],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.pdb"
            write_multimodel_pdb(coords, path)
            parsed = read_multimodel_pdb_coords(path, n_atoms=2)

        np.testing.assert_allclose(parsed, coords, atol=1e-3)

    def test_reader_can_extract_binary_occupancy_mask(self) -> None:
        """The MDLovoFit output reader should expose occupancy values for reconstruction."""

        pdb_text = (
            "REMARK FRAME:        1\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  0.00  0.00           C\n"
            "REMARK FRAME:        2\n"
            "ATOM      1  CA  ALA A   1       0.100   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2       1.100   0.000   0.000  0.00  0.00           C\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aligned.pdb"
            path.write_text(pdb_text)
            coords, occupancies = read_multimodel_pdb_with_occupancy(path, n_atoms=2)

        np.testing.assert_allclose(coords[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(coords[1, 1], [1.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(occupancies, np.array([[1.0, 0.0], [1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
