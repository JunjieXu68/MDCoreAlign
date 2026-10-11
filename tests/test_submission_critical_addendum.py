"""Tests for the submission-critical analysis addendum."""

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


class SubmissionCriticalAddendumTest(unittest.TestCase):
    """Validate reviewer-critical rerun outputs."""

    def test_regularization_rerun_schema_and_method_completeness(self) -> None:
        """Regularization rerun should include all six methods for all systems."""

        module = _load_script("rerun_regularization_formal_systems")
        path = ROOT / "results" / "submission_critical" / "regularization_rerun" / "regularization_rerun_four_systems.tsv"
        rows = _read_tsv(path)
        self.assertEqual(_columns(path), module.FIELDS)
        expected_methods = {method for method, _, _ in module.METHODS}
        by_system: dict[str, set[str]] = {}
        for row in rows:
            by_system.setdefault(row["system"], set()).add(row["method"])
            self.assertNotEqual(row["input_coordinate_file"], "")
            self.assertNotEqual(row["topology_file"], "")
            self.assertNotEqual(row["selection"], "")
        self.assertEqual(len(by_system), 4)
        for methods in by_system.values():
            self.assertEqual(methods, expected_methods)

    def test_diagnostics_trace_runtime_and_nochange_validation(self) -> None:
        """Diagnostics rerun should have traces, runtime profiles, and no-change pass rows."""

        module = _load_script("rerun_diagnostics_formal_systems")
        base = ROOT / "results" / "submission_critical" / "diagnostics_rerun"
        trace_path = base / "alignment_traces_four_systems.tsv"
        rows = _read_tsv(trace_path)
        self.assertEqual(_columns(trace_path), module.TRACE_FIELDS)
        self.assertGreater(len(rows), 1000)
        runtime = json.loads((base / "runtime_profiles_four_systems.json").read_text())
        self.assertIn("profiles", runtime)
        self.assertEqual(len(runtime["profiles"]), 4)
        for profile in runtime["profiles"].values():
            for field in (
                "residual_computation_seconds",
                "weight_update_seconds",
                "sequence_regularization_seconds",
                "contact_regularization_seconds",
                "weighted_kabsch_or_refit_seconds",
                "rmsf_computation_seconds",
                "flexible_segment_detection_seconds",
            ):
                self.assertIn(field, profile)
        validation_rows = _read_tsv(base / "diagnostics_nochange_validation.tsv")
        self.assertEqual(len(validation_rows), 4)
        self.assertTrue(all(row["passed"] == "True" for row in validation_rows))

    def test_theseus_missing_validation_schema(self) -> None:
        """THESEUS validation should clearly mark executable-missing status when absent."""

        module = _load_script("validate_theseus_reproducibility")
        path = ROOT / "results" / "submission_critical" / "theseus_reproducibility" / "theseus_rerun_or_validation.tsv"
        rows = _read_tsv(path)
        self.assertEqual(_columns(path), module.VALIDATION_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["required_columns_present"] == "True" for row in rows))
        if all(row["executable_detected"] == "False" for row in rows):
            self.assertTrue(all(row["status"] == "not_rerun_executable_missing" for row in rows))

    def test_lambda_sensitivity_grid_schema_and_default_marking(self) -> None:
        """Lambda sensitivity rerun should contain a complete 4x4 grid per system."""

        module = _load_script("rerun_lambda_sensitivity_formal_systems")
        path = ROOT / "results" / "submission_critical" / "lambda_sensitivity_rerun" / "lambda_sensitivity_rerun.tsv"
        rows = _read_tsv(path)
        self.assertEqual(_columns(path), module.FIELDS)
        self.assertEqual(len(rows), 4 * len(module.parameter_grid()))
        self.assertEqual(sum(row["is_default_setting"] == "yes" for row in rows), 4)
        self.assertTrue(all(row["notes"] == "formal_lambda_rerun_completed" for row in rows))

    def test_yiip_proxy_is_conservative(self) -> None:
        """YiiP proxy should avoid strong unsupported biological claims."""

        path = ROOT / "results" / "submission_critical" / "yiip_biology_proxy" / "yiip_proxy_summary.tsv"
        rows = _read_tsv(path)
        self.assertGreater(len(rows), 0)
        allowed = {"weak_support", "inconclusive", "negative"}
        self.assertTrue(all(row["status"] in allowed for row in rows))
        summary = (path.parent / "yiip_proxy_summary.md").read_text()
        self.assertIn("not strong external biological validation", summary)

    def test_readiness_audit_has_no_blockers(self) -> None:
        """Readiness audit should exist and report no detected blockers."""

        base = ROOT / "results" / "submission_critical" / "readiness_audit"
        rows = _read_tsv(base / "submission_readiness_checklist.tsv")
        self.assertGreater(len(rows), 0)
        payload = json.loads((base / "blockers.json").read_text())
        self.assertEqual(payload["blockers"], [])
        self.assertIn("Almost ready", payload["recommendation"])


if __name__ == "__main__":
    unittest.main()
