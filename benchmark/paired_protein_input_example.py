"""Minimal example for loading paired protein-like coordinates."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import load_paired_ca_coords_from_pdb, solve_rigid_alignment


MOBILE_PDB = """
ATOM      1  CA  MET A   1      0.300  -0.100   0.200  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.800   0.100   0.200  1.00 20.00           C
ATOM      3  CA  LEU A   3      3.200   0.300   0.200  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.400   0.700   0.300  1.00 20.00           C
END
"""


REFERENCE_PDB = """
ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.500   0.200   0.000  1.00 20.00           C
ATOM      3  CA  LEU A   3      2.900   0.400   0.000  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.100   0.800   0.100  1.00 20.00           C
END
"""


def main() -> None:
    """Load two small CA-only PDBs and run baseline rigid alignment."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mobile_path = tmp / "mobile_example.pdb"
        reference_path = tmp / "reference_example.pdb"
        mobile_path.write_text(MOBILE_PDB.strip() + "\n")
        reference_path.write_text(REFERENCE_PDB.strip() + "\n")

        paired = load_paired_ca_coords_from_pdb(mobile_path, reference_path)
        result = solve_rigid_alignment(paired.mobile_coords, paired.reference_coords)

    print("Paired protein input example")
    print(f"n_atoms={paired.n_atoms}")
    print(f"correspondence_mode={paired.correspondence_mode}")
    print(f"residue_labels={list(paired.residue_labels)}")
    print(f"rmsd={result.rmsd:.6f}")


if __name__ == "__main__":
    main()
