"""Tests for minimal visualization/export support."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    Trajectory,
    export_external_mdlovofit_visualization,
    export_soft_trajectory_visualization,
)


def _read_single_model_fields(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read occupancy and B-factor fields from one exported PDB."""

    occupancies: list[float] = []
    bfactors: list[float] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            occupancies.append(float(line[54:60]))
            bfactors.append(float(line[60:66]))
    return np.asarray(occupancies, dtype=float), np.asarray(bfactors, dtype=float)


class ExportSupportTest(unittest.TestCase):
    """Check soft and external visualization export helpers."""

    def test_soft_export_writes_expected_files_and_fields(self) -> None:
        """Soft export should produce PDB and CSV files with separated timings."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [1.0, -0.1, 0.0], [2.3, 0.0, 0.2], [3.4, 0.0, -0.1]],
                [[0.0, -0.1, 0.0], [1.0, 0.1, 0.0], [2.2, 0.2, 0.1], [3.3, -0.2, 0.0]],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "toy"
            artifacts = export_soft_trajectory_visualization(
                Trajectory(coords=coords),
                prefix,
                core_fraction=0.5,
                residue_labels=("ALA1", "GLY2", "SER3", "TYR4"),
                atom_names=("CA", "CA", "CA", "CA"),
                rule="huber",
                delta=0.25,
            )

            self.assertTrue(artifacts.occupancy_pdb_path.exists())
            self.assertTrue(artifacts.bfactor_pdb_path.exists())
            self.assertTrue(artifacts.csv_path.exists())
            self.assertTrue(artifacts.summary_json_path.exists())
            self.assertIsNotNone(artifacts.algorithm_runtime_seconds)
            self.assertGreaterEqual(artifacts.export_runtime_seconds, 0.0)
            self.assertIsNone(artifacts.render_runtime_seconds)

            occupancies, occupancy_bfactors = _read_single_model_fields(artifacts.occupancy_pdb_path)
            self.assertTrue(set(np.unique(occupancies)).issubset({0.0, 1.0}))
            self.assertTrue(np.allclose(occupancy_bfactors, 0.0))

            _, mobility_bfactors = _read_single_model_fields(artifacts.bfactor_pdb_path)
            self.assertGreaterEqual(float(np.min(mobility_bfactors)), 0.0)
            self.assertLessEqual(float(np.max(mobility_bfactors)), 100.0)

            with artifacts.csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                list(rows[0]),
                [
                    "atom_index",
                    "atom_name",
                    "residue_index",
                    "residue_id",
                    "weight",
                    "stability_score",
                    "mobility_score",
                    "residual",
                    "hard_core_label",
                    "aggregated_residue_score",
                ],
            )

            summary = json.loads(artifacts.summary_json_path.read_text())
            self.assertEqual(summary["occupancy_semantics"], "binary stable-core mask; 1 = selected core, 0 = non-core")
            self.assertEqual(summary["bfactor_field"], "mobility_score")
            self.assertIsNone(summary["render_runtime_seconds"])
            self.assertGreaterEqual(summary["algorithm_runtime_seconds"], 0.0)
            self.assertGreaterEqual(summary["export_runtime_seconds"], 0.0)

    def test_external_export_preserves_binary_occupancy_and_reports_summary(self) -> None:
        """External export should preserve occupancy and report mask availability cleanly."""

        pdb_text = (
            "REMARK FRAME:        1\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  CA  ALA A   3       2.000   0.000   0.000  0.00  0.00           C\n"
            "ATOM      4  CA  ALA A   4       3.000   0.000   0.000  0.00  0.00           C\n"
            "REMARK FRAME:        2\n"
            "ATOM      1  CA  ALA A   1       0.100   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2       1.100   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  CA  ALA A   3       2.100   0.000   0.000  0.00  0.00           C\n"
            "ATOM      4  CA  ALA A   4       3.100   0.000   0.000  0.00  0.00           C\n"
            "END\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            aligned_path = tmp / "aligned.pdb"
            aligned_path.write_text(pdb_text)
            artifacts = export_external_mdlovofit_visualization(
                aligned_path,
                tmp / "toy_external",
                n_atoms=4,
                residue_labels=("ALA1", "ALA2", "ALA3", "ALA4"),
                atom_names=("CA", "CA", "CA", "CA"),
                algorithm_runtime_seconds=1.23,
                benchmark_overhead_seconds=0.11,
            )

            self.assertTrue(artifacts.occupancy_pdb_path.exists())
            self.assertTrue(artifacts.csv_path.exists())
            self.assertTrue(artifacts.summary_json_path.exists())
            self.assertEqual(artifacts.algorithm_runtime_seconds, 1.23)
            self.assertGreaterEqual(artifacts.export_runtime_seconds, 0.0)
            self.assertIsNone(artifacts.render_runtime_seconds)

            occupancies, bfactors = _read_single_model_fields(artifacts.occupancy_pdb_path)
            np.testing.assert_allclose(occupancies, np.array([1.0, 1.0, 0.0, 0.0]))
            np.testing.assert_allclose(bfactors, np.zeros(4))

            with artifacts.csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                list(rows[0]),
                [
                    "atom_index",
                    "atom_name",
                    "residue_index",
                    "residue_id",
                    "raw_occupancy",
                    "raw_bfactor",
                    "hard_core_label",
                ],
            )

            summary = json.loads(artifacts.summary_json_path.read_text())
            self.assertTrue(summary["occupancy_mask_binary"])
            self.assertTrue(summary["occupancy_mask_constant_across_frames"])
            self.assertEqual(summary["output_subset_atom_count"], 2)
            self.assertAlmostEqual(summary["output_subset_atom_fraction"], 0.5)
            self.assertEqual(summary["bfactor_unique_values"], [0.0])
            self.assertTrue(summary["bfactor_constant_across_frames"])


if __name__ == "__main__":
    unittest.main()
