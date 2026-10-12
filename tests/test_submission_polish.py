"""Tests for submission-polish outputs and audit checks."""

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


class SubmissionPolishTest(unittest.TestCase):
    """Validate final submission-polish package."""

    def test_preprocessing_verification_schema_and_caveat(self) -> None:
        """PBC/centering verification should be explicit and non-overclaiming."""

        base = ROOT / "results" / "submission_polish" / "preprocessing_verification"
        table = base / "pbc_centering_verification.tsv"
        rows = _read_tsv(table)
        columns = _columns(table)

        self.assertIn("system", columns)
        self.assertIn("selection", columns)
        self.assertIn("notes", columns)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["selection"] == "protein and name CA" for row in rows))
        summary = (base / "pbc_centering_verification_summary.md").read_text()
        self.assertIn("PBC/nojump and centering status could not be independently verified", summary)
        self.assertTrue((base / "manifest.json").exists())

    def test_temporal_convergence_classification_schema(self) -> None:
        """Temporal non-convergence should be classified without hidden failures."""

        base = ROOT / "results" / "submission_polish" / "temporal_convergence"
        table = base / "temporal_convergence_classification.tsv"
        rows = _read_tsv(table)
        labels = {row["classification"] for row in rows}
        columns = _columns(table)

        self.assertIn("system", columns)
        self.assertIn("block_id", columns)
        self.assertIn("classification", columns)
        self.assertGreater(len(rows), 0)
        self.assertFalse(labels & {"true_failure", "weight_collapse"})
        summary = (base / "temporal_convergence_summary.md").read_text()
        self.assertIn("conservative max-iteration stopping", summary)

    def test_main_figure_data_schemas(self) -> None:
        """Main-figure data tables should expose stable schemas and required diagnostics."""

        module = _load_script("make_md_trajectory_main_figure_data")
        base = ROOT / "results" / "submission_polish" / "main_figures"
        fig2 = base / "fig2_benchmark_summary.tsv"
        fig3 = base / "fig3_md_diagnostics_summary.tsv"
        fig4 = base / "fig4_flexibility_profile_summary.tsv"

        self.assertEqual(_columns(fig2), module.FIG2_FIELDS)
        self.assertEqual(_columns(fig3), module.FIG3_FIELDS)
        self.assertEqual(_columns(fig4), module.FIG4_FIELDS)
        self.assertEqual(len(_read_tsv(fig2)), 12)
        self.assertEqual(len(_read_tsv(fig4)), 4)
        diagnostics = {row["diagnostic"] for row in _read_tsv(fig3)}
        self.assertIn("reference_sensitivity", diagnostics)
        self.assertIn("warm_start_ablation", diagnostics)
        self.assertIn("trajectory_length_dependence", diagnostics)
        captions = (base / "main_figure_captions_draft.md").read_text()
        self.assertIn("not claimed to be reference independent", captions)

    def test_final_text_caveats_are_present(self) -> None:
        """Final text snippets should include required reviewer-risk caveats."""

        text_dir = ROOT / "results" / "submission_polish" / "paper_text"
        limitations = (text_dir / "limitations_final.md").read_text().lower()
        reference = (text_dir / "results_reference_sensitivity_final.md").read_text().lower()
        warm = (text_dir / "results_warm_start_final.md").read_text().lower()
        supplement = (text_dir / "supplementary_allocation_final.md").read_text().lower()

        self.assertIn("rmsf is alignment-frame dependent", limitations)
        self.assertIn("do not imply theseus was rerun locally", limitations)
        self.assertIn("limited external biological-proxy support", limitations)
        self.assertIn("nmr500 should be treated as supplementary structural-ensemble evidence", limitations)
        self.assertIn("not be described as reference independent", reference)
        self.assertIn("efficiency option", warm)
        self.assertIn("not be described as scientifically necessary", warm)
        self.assertIn("nmr500 structural-ensemble benchmark", supplement)

    def test_final_submission_audit_has_no_blockers(self) -> None:
        """Submission-polish audit should report no blockers."""

        module = _load_script("audit_submission_polish")
        base = ROOT / "results" / "submission_polish"
        checklist = base / "final_submission_polish_checklist.tsv"
        rows = _read_tsv(checklist)
        blockers = json.loads((base / "blockers.json").read_text())

        self.assertEqual(_columns(checklist), module.FIELDS)
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        self.assertEqual(blockers["blockers"], [])
        self.assertIn("Ready", blockers["recommendation"])


if __name__ == "__main__":
    unittest.main()
