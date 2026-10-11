"""Tests for the public `mdcorealign formats` command."""

from __future__ import annotations

import csv
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


class FormatsCliTest(unittest.TestCase):
    """Validate the format support reporting command."""

    def test_formats_help_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "mdcorealign", "formats", "--help"],
            cwd=ROOT,
            env=_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("usage:", completed.stdout)

    def test_formats_table_contains_expected_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "supported_formats.tsv"
            subprocess.run(
                [sys.executable, "-m", "mdcorealign", "formats", "--out", str(out)],
                cwd=ROOT,
                env=_env(),
                check=True,
                capture_output=True,
                text=True,
            )
            with out.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            text = "\n".join(row["format_or_pair"] for row in rows)
            for expected in ["PDB", "XTC", "DCD", "PRMTOP/PARM7", "NC/NetCDF", "CSV/TSV", "JSON"]:
                self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
