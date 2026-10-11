"""Tests for final quality-upgrade outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_script(name: str):
    """Load one script module by filename stem."""

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


class FinalQualityUpgradeTest(unittest.TestCase):
    """Validate final quality-upgrade result package."""

    def test_preprocessing_final_schema_and_limitation(self) -> None:
        """Final preprocessing verification should be explicit and conservative."""

        module = _load_script("verify_md_preprocessing_pbc_centering")
        base = ROOT / "results" / "final_quality_upgrade" / "preprocessing_verification"
        table = base / "pbc_centering_verification.tsv"
        recentered = base / "recentered_comparison.tsv"
        rows = _read_tsv(table)

        self.assertEqual(_columns(table), module.FIELDS)
        self.assertEqual(_columns(recentered), module.RECENTER_FIELDS)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["pbc_nojump_status"] == "not_independently_verified_from_metadata" for row in rows))
        self.assertIn(module.LIMITATION_SENTENCE, (base / "pbc_centering_verification_summary.md").read_text())
        self.assertGreaterEqual(len(_read_tsv(recentered)), 1)

    def test_temporal_final_classification(self) -> None:
        """Final temporal convergence classification should avoid hidden failures."""

        module = _load_script("classify_temporal_block_convergence")
        base = ROOT / "results" / "final_quality_upgrade" / "temporal_convergence"
        table = base / "temporal_convergence_classification.tsv"
        rows = _read_tsv(table)
        labels = {row["classification"] for row in rows}

        self.assertEqual(_columns(table), module.FIELDS)
        self.assertGreater(len(rows), 0)
        self.assertTrue(labels <= module.CLASSES)
        self.assertFalse(labels & {"true_failure", "weight_collapse"})
        self.assertIn("conservative max-iteration stopping", (base / "temporal_convergence_summary.md").read_text())

    def test_reference_consensus_status_is_not_overclaimed(self) -> None:
        """Reference-consensus output should not infer unavailable residue-level labels."""

        module = _load_script("analyze_reference_consensus")
        base = ROOT / "results" / "final_quality_upgrade" / "reference_consensus"
        residue = base / "reference_consensus_residues.tsv"
        system = base / "reference_consensus_system_summary.tsv"
        system_rows = _read_tsv(system)
        summary = (base / "reference_consensus_summary.md").read_text()

        self.assertEqual(_columns(residue), module.RESIDUE_FIELDS)
        self.assertEqual(_columns(system), module.SYSTEM_FIELDS)
        self.assertEqual(len(_read_tsv(residue)), 0)
        self.assertEqual(len(system_rows), 4)
        self.assertTrue(all(row["n_residues"] for row in system_rows))
        self.assertIn("no reference-independence claim is made", summary)

    def test_optional_status_tables_are_explicit(self) -> None:
        """Optional analyses should be marked unavailable rather than guessed."""

        external = ROOT / "results" / "final_quality_upgrade" / "external_md_mini_validation"
        atom = ROOT / "results" / "final_quality_upgrade" / "atom_selection_sensitivity"
        self.assertIn("not_applicable", (external / "external_md_mini_validation_summary.md").read_text())
        self.assertEqual(len(_read_tsv(external / "external_md_mini_validation.tsv")), 0)
        atom_rows = _read_tsv(atom / "atom_selection_sensitivity.tsv")
        self.assertGreaterEqual(len(atom_rows), 2)
        self.assertTrue(any(row["selection_label"] == "backbone" and row["converged"] == "unavailable" for row in atom_rows))

    def test_user_readiness_reports_exist(self) -> None:
        """Clean install, toy quickstart, and build reports should exist."""

        base = ROOT / "results" / "final_quality_upgrade" / "user_readiness"
        for name in [
            "clean_install_report.md",
            "clean_install_report.json",
            "toy_quickstart_report.md",
            "package_build_report.md",
            "package_build_report.json",
        ]:
            path = base / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)
        clean = json.loads((base / "clean_install_report.json").read_text())
        self.assertEqual(clean["status"], "pass")
        self.assertFalse(clean["external_tools_required_for_core_commands"])

    def test_paper_text_required_caveats(self) -> None:
        """Final text snippets should include the requested reviewer-risk language."""

        text_dir = ROOT / "results" / "final_quality_upgrade" / "paper_text_updates"
        texts = "\n".join(path.read_text() for path in text_dir.glob("*.md"))
        for phrase in [
            "PBC/nojump and centering status could not be independently verified",
            "Do not claim reference independence",
            "Warm start is an efficiency option",
            "not be described as rerun locally",
            "YiiP should not be overclaimed",
            "RMSF is alignment-frame dependent",
            "NMR500 is supplementary structural-ensemble evidence",
        ]:
            self.assertIn(phrase, texts)

    def test_final_audit_has_no_blockers_when_generated(self) -> None:
        """Final audit should report no blockers after the full run is frozen."""

        base = ROOT / "results" / "final_quality_upgrade" / "final_audit"
        blockers_path = base / "blockers.json"
        if not blockers_path.exists():
            self.skipTest("final quality audit has not been generated yet")
        module = _load_script("audit_final_quality_upgrade")
        checklist = base / "final_quality_checklist.tsv"
        blockers = json.loads(blockers_path.read_text())

        self.assertEqual(_columns(checklist), module.FIELDS)
        self.assertEqual(blockers["blockers"], [])
        self.assertIn("Ready", blockers["recommendation"])


if __name__ == "__main__":
    unittest.main()
