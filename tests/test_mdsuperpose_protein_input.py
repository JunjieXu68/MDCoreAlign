"""Tests for minimal real-protein coordinate input helpers."""

from __future__ import annotations

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
    load_ca_coords_from_pdb,
    load_paired_ca_coords_from_pdb,
    load_paired_protein_coords_from_arrays,
    solve_rigid_alignment,
)


def _write_text(path: Path, text: str) -> None:
    """Write a small helper text file for input tests."""

    path.write_text(text.strip() + "\n")


class ProteinInputTest(unittest.TestCase):
    """Check minimal paired protein coordinate loading behavior."""

    def test_array_pair_loader_assumes_order_or_validates_labels(self) -> None:
        """Array input should support explicit labels and preserve coordinates."""

        reference = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [2.9, 0.4, 0.0],
                [4.1, 0.8, 0.1],
            ],
            dtype=float,
        )
        mobile = reference + np.array([0.3, -0.1, 0.2], dtype=float)
        paired = load_paired_protein_coords_from_arrays(
            mobile,
            reference,
            mobile_residue_labels=["A:MET1", "A:GLU2", "A:LEU3", "A:ARG4"],
            reference_residue_labels=["A:MET1", "A:GLU2", "A:LEU3", "A:ARG4"],
        )

        self.assertEqual(paired.correspondence_mode, "residue_labels")
        self.assertEqual(paired.n_atoms, 4)
        np.testing.assert_allclose(paired.mobile_coords, mobile)
        np.testing.assert_allclose(paired.reference_coords, reference)

    def test_ca_pdb_pair_loader_matches_residue_labels(self) -> None:
        """CA-only PDB input should round-trip simple protein-like coordinates."""

        mobile_pdb = """
ATOM      1  CA  MET A   1      0.300  -0.100   0.200  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.800   0.100   0.200  1.00 20.00           C
ATOM      3  CA  LEU A   3      3.200   0.300   0.200  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.400   0.700   0.300  1.00 20.00           C
END
"""
        reference_pdb = """
ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.500   0.200   0.000  1.00 20.00           C
ATOM      3  CA  LEU A   3      2.900   0.400   0.000  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.100   0.800   0.100  1.00 20.00           C
END
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mobile_path = tmp / "mobile.pdb"
            reference_path = tmp / "reference.pdb"
            _write_text(mobile_path, mobile_pdb)
            _write_text(reference_path, reference_pdb)

            mobile_only = load_ca_coords_from_pdb(mobile_path, chain_id="A")
            paired = load_paired_ca_coords_from_pdb(mobile_path, reference_path)

        self.assertEqual(mobile_only.residue_labels, ("A:MET1", "A:GLU2", "A:LEU3", "A:ARG4"))
        self.assertEqual(paired.residue_labels, ("A:MET1", "A:GLU2", "A:LEU3", "A:ARG4"))
        result = solve_rigid_alignment(paired.mobile_coords, paired.reference_coords)
        self.assertLess(result.rmsd, 1e-10)

    def test_missing_residue_is_rejected_clearly(self) -> None:
        """Mismatched residue sets should raise a clear correspondence error."""

        mobile_pdb = """
ATOM      1  CA  MET A   1      0.300  -0.100   0.200  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.800   0.100   0.200  1.00 20.00           C
ATOM      3  CA  LEU A   3      3.200   0.300   0.200  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.400   0.700   0.300  1.00 20.00           C
END
"""
        reference_pdb = """
ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  LEU A   3      2.900   0.400   0.000  1.00 20.00           C
ATOM      3  CA  ARG A   4      4.100   0.800   0.100  1.00 20.00           C
ATOM      4  CA  TYR A   5      5.300   1.100   0.100  1.00 20.00           C
END
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mobile_path = tmp / "mobile_missing.pdb"
            reference_path = tmp / "reference_missing.pdb"
            _write_text(mobile_path, mobile_pdb)
            _write_text(reference_path, reference_pdb)

            with self.assertRaisesRegex(ValueError, "consistent atom correspondence|missing residues"):
                load_paired_ca_coords_from_pdb(mobile_path, reference_path)


if __name__ == "__main__":
    unittest.main()
