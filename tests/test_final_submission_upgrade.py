"""Tests for final submission-upgrade outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if (path != SRC or os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product") and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_script(name: str):
    """Load one script module."""

    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _columns(path: Path) -> list[str]:
    """Return TSV columns."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or [])


class FinalSubmissionUpgradeTest(unittest.TestCase):
    """Validate final submission-upgrade package."""

    def test_release_metadata_audit_schema(self) -> None:
        """Release metadata audit should be explicit about pending values."""

        module = _load_script("update_release_metadata")
        table = ROOT / "results" / "final_submission_upgrade" / "release_metadata" / "release_metadata_audit.tsv"
        rows = _read_tsv(table)
        self.assertEqual(_columns(table), module.FIELDS)
        self.assertGreaterEqual(len(rows), 4)
        self.assertTrue(any(row["field"] == "doi" and row["value"] == "pending" for row in rows))
        self.assertTrue((ROOT / "results" / "final_submission_upgrade" / "release_metadata" / "manifest.json").exists())

    def test_release_archive_has_no_blockers(self) -> None:
        """Build and twine check should pass in release archive readiness."""

        base = ROOT / "results" / "final_submission_upgrade" / "release_archive"
        blockers = json.loads((base / "blockers.json").read_text())
        rows = _read_tsv(base / "release_archive_checklist.tsv")
        self.assertEqual(blockers["blockers"], [])
        self.assertIn("GitHub remote", blockers["warnings"])
        self.assertTrue(any(row["check"] == "python -m build" and row["status"] == "pass" for row in rows))
        self.assertTrue(any(row["check"] == "twine check" and row["status"] == "pass" for row in rows))

    def test_testpypi_pending_is_not_blocker(self) -> None:
        """Missing TestPyPI upload should be marked pending, not failed."""

        module = _load_script("validate_testpypi_install")
        base = ROOT / "results" / "final_submission_upgrade" / "testpypi_validation"
        report = json.loads((base / "testpypi_validation_report.json").read_text())
        self.assertEqual(report["status"], "not_applicable_pending_upload")
        self.assertFalse(report["external_tools_required_for_core_commands"])
        self.assertEqual(_columns(base / "toy_outputs_check.tsv"), module.TOY_FIELDS)

    def test_user_testing_template_schema(self) -> None:
        """User feedback template should expose all required columns."""

        module = _load_script("summarize_user_testing_feedback")
        base = ROOT / "results" / "final_submission_upgrade" / "user_testing"
        self.assertEqual(_columns(base / "user_feedback_template.tsv"), module.FEEDBACK_FIELDS)
        rows = _read_tsv(base / "user_testing_summary.tsv")
        self.assertEqual(rows[0]["feedback_status"], "pending")

    def test_reproducibility_capsule(self) -> None:
        """Reproducibility capsule should rerun toy outputs and validate cached tables."""

        base = ROOT / "results" / "final_submission_upgrade" / "reproducibility_capsule"
        report = json.loads((base / "reproduce_core_results_report.json").read_text())
        rows = _read_tsv(base / "reproduced_files.tsv")
        self.assertEqual(report["status"], "pass")
        self.assertTrue(any(row["mode"] == "rerun_toy_quickstart" and row["status"] == "present" for row in rows))
        self.assertTrue(any(row["mode"] == "cached_official_table" and row["status"] == "validated_by_checksum" for row in rows))

    def test_residue_consensus_unavailable_is_explicit(self) -> None:
        """Residue-level reference consensus should not be inferred without data."""

        module = _load_script("analyze_reference_consensus_residue_level")
        base = ROOT / "results" / "final_submission_upgrade" / "reference_consensus_residue_level"
        self.assertEqual(_columns(base / "reference_consensus_residues.tsv"), module.RESIDUE_FIELDS)
        self.assertEqual(_columns(base / "reference_consensus_system_summary.tsv"), module.SYSTEM_FIELDS)
        self.assertEqual(len(_read_tsv(base / "reference_consensus_residues.tsv")), 0)
        self.assertIn("No reference-independence claim", (base / "reference_consensus_residue_summary.md").read_text())

    def test_final_figure_data_and_paper_text(self) -> None:
        """Final figure data and paper text snippets should contain required caveats."""

        fig = ROOT / "results" / "final_submission_upgrade" / "final_figures"
        self.assertGreater(len(_read_tsv(fig / "table_user_recommendations.tsv")), 0)
        self.assertIn("RMSF is alignment-frame dependent", (fig / "main_figure_captions_final.md").read_text())
        text_dir = ROOT / "results" / "final_submission_upgrade" / "paper_text_final"
        text = "\n".join(path.read_text() for path in text_dir.glob("*.md"))
        for phrase in [
            "Do not claim THESEUS was rerun locally",
            "Do not claim PBC/nojump was proven",
            "Do not claim reference independence",
            "Do not overclaim YiiP biological validation",
            "RMSF is alignment-frame dependent",
            "CA-level selection is the formal default",
        ]:
            self.assertIn(phrase, text)

    def test_final_submission_audit_when_generated(self) -> None:
        """Final submission audit should report no blockers once generated."""

        base = ROOT / "results" / "final_submission_upgrade" / "final_audit"
        blockers_path = base / "blockers.json"
        if not blockers_path.exists():
            self.skipTest("final submission audit has not been generated yet")
        module = _load_script("audit_final_submission_upgrade")
        blockers = json.loads(blockers_path.read_text())
        self.assertEqual(_columns(base / "final_submission_upgrade_checklist.tsv"), module.FIELDS)
        self.assertEqual(blockers["blockers"], [])
        self.assertIn("ready", blockers["recommendation"].lower())


if __name__ == "__main__":
    unittest.main()
