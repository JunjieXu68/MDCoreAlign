"""Tests for conservative non-protein support policy."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _atom_line(serial: int, atom: str, resname: str, chain: str, resid: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom[:4]:>4} {resname[:3]:>3} {chain[:1]:1s}{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.00:6.2f}           P\n"
    )


class NonProteinPolicyTest(unittest.TestCase):
    """Validate RNA/DNA and non-protein policy docs/errors."""

    def test_non_protein_docs_are_explicit(self) -> None:
        text = (ROOT / "docs" / "non_protein_systems.md").read_text()
        self.assertIn("Not formally supported or validated", text)
        self.assertIn("protein C-alpha", text)
        self.assertIn("Ligands are not used for stable-core inference", text)

    def test_rna_like_no_ca_error_mentions_not_formally_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb = Path(tmpdir) / "rna_like.pdb"
            pdb.write_text(
                "MODEL        1\n"
                + _atom_line(1, "P", "A", "A", 1, 0.0)
                + _atom_line(2, "P", "U", "A", 2, 1.0)
                + "ENDMDL\nMODEL        2\n"
                + _atom_line(1, "P", "A", "A", 1, 0.1)
                + _atom_line(2, "P", "U", "A", 2, 1.1)
                + "ENDMDL\nEND\n"
            )
            completed = subprocess.run(
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
                    str(Path(tmpdir) / "out"),
                    "--dry-run",
                ],
                cwd=ROOT,
                env=_env(),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("RNA/DNA-only systems are not formally validated", completed.stderr)
            self.assertIn("protein C-alpha default does not apply", completed.stderr)


if __name__ == "__main__":
    unittest.main()
