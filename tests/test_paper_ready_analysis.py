"""Tests for reviewer-facing pre-manuscript analysis scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose import (  # noqa: E402
    RUNTIME_PROFILE_FIELDS,
    TRACE_FIELDS,
    Trajectory,
    final_solution_signature,
    solve_trajectory_with_optional_diagnostics,
)


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


def _table_columns(path: Path) -> list[str]:
    """Return TSV/CSV column names."""

    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or [])


class PaperReadyAnalysisTest(unittest.TestCase):
    """Validate pre-manuscript analysis outputs and helpers."""

    def test_final_statistics_output_schemas(self) -> None:
        """Final statistics tables should expose stable manuscript-facing schemas."""

        module = _load_script("analyze_final_statistics")
        checks = [
            (ROOT / "results" / "final_statistics" / "trajectory_benchmark_statistics.tsv", module.TRAJECTORY_FIELDS),
            (ROOT / "results" / "final_statistics" / "pair_benchmark_statistics.tsv", module.PAIR_FIELDS),
            (ROOT / "results" / "final_statistics" / "nmr500_benchmark_statistics.tsv", module.NMR_FIELDS),
            (ROOT / "results" / "final_statistics" / "flexibility_statistics_final.tsv", module.FLEX_FIELDS),
            (ROOT / "results" / "final_statistics" / "biology_support_statistics.tsv", module.BIOLOGY_FIELDS),
        ]
        for path, expected in checks:
            self.assertTrue(path.exists(), path)
            self.assertEqual(_table_columns(path), expected)
            self.assertGreater(path.stat().st_size, 0)

    def test_regularization_ablation_output_schema(self) -> None:
        """Regularization ablation output should keep the formal schema."""

        module = _load_script("run_regularization_ablation_formal_systems")
        path = ROOT / "results" / "ablation" / "regularization" / "regularization_ablation_four_systems.tsv"
        self.assertTrue(path.exists())
        self.assertEqual(_table_columns(path), module.FIELDS)

    def test_diagnostics_trace_and_runtime_json_schema(self) -> None:
        """Diagnostics outputs should expose trace and runtime schemas."""

        trace_path = ROOT / "results" / "diagnostics" / "alignment_traces_four_systems.tsv"
        runtime_path = ROOT / "results" / "diagnostics" / "runtime_profiles_four_systems.json"
        self.assertTrue(trace_path.exists())
        self.assertEqual(_table_columns(trace_path), [*TRACE_FIELDS, "diagnostic_source", "notes"])
        payload = json.loads(runtime_path.read_text())
        self.assertIn("profiles", payload)
        self.assertGreater(len(payload["profiles"]), 0)
        for field in RUNTIME_PROFILE_FIELDS:
            self.assertIn(field, payload["profiles"][0])

    def test_parameter_sensitivity_grid_and_schema(self) -> None:
        """Sensitivity grid should be small, deterministic, and schema-stable."""

        module = _load_script("run_parameter_sensitivity_formal_systems")
        grid = module.parameter_grid()
        self.assertEqual(len(grid), 3 * 4 * 4 * 4)
        path = ROOT / "results" / "sensitivity" / "sensitivity_summary.tsv"
        self.assertTrue(path.exists())
        self.assertEqual(_table_columns(path), module.FIELDS)

    def test_paper_output_audit_passes(self) -> None:
        """The paper-output audit should pass after required tables are generated."""

        module = _load_script("audit_paper_figures_tables")
        rows, failed = module.audit_rows()
        self.assertFalse(failed)
        self.assertGreater(len(rows), 0)

    def test_stale_test_status_detection(self) -> None:
        """Project-state audit should detect stale old test-count mentions."""

        module = _load_script("audit_project_state")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stale.md"
            path.write_text("Current regression status: 80 tests passed.\n")
            warnings = module.stale_status_warnings([str(path)])
        self.assertEqual(len(warnings), 1)
        self.assertIn("stale test-count", warnings[0])

    def test_figure_data_tables_are_non_empty(self) -> None:
        """Formal figure-data tables should exist and contain at least one row."""

        for path in (ROOT / "results" / "flexibility" / "figure_data").glob("*.tsv"):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertGreater(len(rows), 0, path)

    def test_diagnostics_do_not_change_numerical_results(self) -> None:
        """Running with diagnostics should not alter final RMSD or weights."""

        base = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        frames = np.stack([base, base + np.array([0.1, -0.2, 0.0])], axis=0)
        trajectory = Trajectory(coords=frames)
        kwargs = {"max_iterations": 5, "rule": "huber", "delta": 0.25, "warm_start": True}
        plain = solve_trajectory_with_optional_diagnostics(trajectory, diagnostics=False, **kwargs)
        traced = solve_trajectory_with_optional_diagnostics(trajectory, diagnostics=True, **kwargs)
        plain_sig = final_solution_signature(plain.result)
        traced_sig = final_solution_signature(traced.result)
        self.assertGreater(len(traced.trace_rows), 0)
        self.assertAlmostEqual(plain_sig["mean_rmsd"], traced_sig["mean_rmsd"], places=12)
        np.testing.assert_allclose(plain_sig["weights"], traced_sig["weights"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
