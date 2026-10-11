"""Tests for input readiness classification and audit output."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mdcorealign.input_validation import (
    NEEDS_CONVERSION,
    NEEDS_PREPROCESSING,
    READY_TO_RUN,
    RUNNABLE_WITH_WARNINGS,
    UNSUPPORTED,
    classify_input,
)

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class InputReadinessStatusTest(unittest.TestCase):
    """Validate readiness status categories."""

    def test_classify_ready_warning_preprocessing_conversion_and_unsupported(self) -> None:
        toy = ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"
        self.assertEqual(classify_input(input_path=toy, selection_size=8).status, READY_TO_RUN)
        self.assertEqual(classify_input(input_path=toy, missing_chain_ids=1, selection_size=8).status, RUNNABLE_WITH_WARNINGS)
        self.assertEqual(classify_input(input_path=toy, large_jump_detected=True, selection_size=8).status, NEEDS_PREPROCESSING)
        self.assertEqual(classify_input(input_path=toy, selection_size=0).status, UNSUPPORTED)
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown = Path(tmpdir) / "coords.foo"
            unknown.write_text("x\n")
            self.assertEqual(classify_input(input_path=unknown).status, NEEDS_CONVERSION)

    def test_missing_topology_cli_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory = Path(tmpdir) / "traj.xtc"
            trajectory.write_bytes(b"not really xtc")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "audit-preprocessing",
                    "--trajectory",
                    str(trajectory),
                    "--output-dir",
                    str(Path(tmpdir) / "out"),
                ],
                cwd=ROOT,
                env=_env(),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("reliable topology is required", completed.stderr)

    def test_input_readiness_status_written_to_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "audit"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdcorealign",
                    "audit-preprocessing",
                    "--input",
                    str(ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"),
                    "--input-format",
                    "ca-pdb",
                    "--output-dir",
                    str(out),
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                capture_output=True,
                text=True,
            )
            with (out / "input_audit.tsv").open(newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(row["input_readiness_status"], READY_TO_RUN)
            self.assertNotEqual(row["input_readiness_recommended_action"], "")


if __name__ == "__main__":
    unittest.main()
