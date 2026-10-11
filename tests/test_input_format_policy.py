"""Documentation and CLI tests for input-format policy."""

from __future__ import annotations

import json
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


class InputFormatPolicyTest(unittest.TestCase):
    """Validate format docs and user-facing format errors."""

    def test_docs_exist_and_define_support_boundaries(self) -> None:
        support = ROOT / "docs" / "support_matrix.md"
        output = ROOT / "docs" / "output_formats.md"
        comparison = ROOT / "docs" / "mdlovofit_comparison.md"
        for path in [support, output, comparison]:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 500, path)
        support_text = support.read_text()
        for phrase in ["Validated", "Expected", "Experimental", "Not recommended / not formally supported"]:
            self.assertIn(phrase, support_text)
        self.assertIn("PDB trajectory", comparison.read_text())
        self.assertIn("C-alpha", comparison.read_text())

    def test_unknown_extension_produces_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown = Path(tmpdir) / "trajectory.unknown"
            unknown.write_text("not coordinates\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "align",
                    "--input",
                    str(unknown),
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
            self.assertIn("Format detection failed", completed.stderr)
            self.assertIn("--topology-format / --trajectory-format", completed.stderr)

    def test_format_override_recorded_in_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out"
            toy = ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "align",
                    "--topology",
                    str(toy),
                    "--trajectory",
                    str(toy),
                    "--topology-format",
                    "pdb",
                    "--trajectory-format",
                    "pdb",
                    "--output-dir",
                    str(out),
                    "--dry-run",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads(
                (out / "diagnostics" / "run_metadata.json").read_text()
            )
            self.assertEqual(metadata["format_overrides"]["topology_format"], "pdb")
            self.assertEqual(metadata["format_overrides"]["trajectory_format"], "pdb")


if __name__ == "__main__":
    unittest.main()
