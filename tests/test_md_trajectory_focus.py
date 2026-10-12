"""Tests for MD-trajectory-focused validation analyses."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


class MDTrajectoryFocusTest(unittest.TestCase):
    """Validate MD-focus scripts and outputs."""

    def test_block_stride_prefix_and_overlap_helpers(self) -> None:
        """Core MD-focus helper definitions should be deterministic."""

        common = _load_script("md_trajectory_focus_common")
        blocks = common.md_block_definitions(100)
        self.assertEqual([row["block_id"] for row in blocks], [
            "block_0_25",
            "block_25_50",
            "block_50_75",
            "block_75_100",
            "first_half",
            "second_half",
            "full",
        ])
        self.assertEqual(common.stride_values(), [1, 2, 5, 10, 20])
        prefixes = common.prefix_definitions(100)
        self.assertEqual([row["frame_end"] for row in prefixes], [10, 25, 50, 75, 100])
        self.assertAlmostEqual(common.jaccard(np.array([1, 1, 0]), np.array([1, 0, 1])), 1 / 3)

    def test_temporal_stride_length_reference_and_warm_schemas(self) -> None:
        """Main MD-focus output tables should expose required schemas."""

        checks = [
            ("analyze_md_temporal_block_stability", "temporal_block_stability", "temporal_block_stability.tsv", 28),
            ("analyze_md_stride_sensitivity", "stride_sensitivity", "stride_sensitivity.tsv", 20),
            ("analyze_md_trajectory_length_dependence", "trajectory_length_dependence", "trajectory_length_dependence.tsv", 20),
            ("analyze_md_reference_sensitivity", "reference_sensitivity", "reference_sensitivity.tsv", 20),
            ("analyze_md_warm_start_ablation", "warm_start_ablation", "warm_start_ablation.tsv", 8),
        ]
        for module_name, directory, filename, expected_rows in checks:
            module = _load_script(module_name)
            path = ROOT / "results" / "md_trajectory_focus" / directory / filename
            self.assertEqual(_columns(path), module.FIELDS)
            self.assertEqual(len(_read_tsv(path)), expected_rows)

    def test_preprocessing_manifest_and_per_frame_residuals(self) -> None:
        """Preprocessing and per-frame residual tables should be populated."""

        prep = _load_script("audit_md_preprocessing_manifest")
        prep_path = ROOT / "results" / "md_trajectory_focus" / "preprocessing_manifest" / "md_preprocessing_manifest.tsv"
        prep_rows = _read_tsv(prep_path)
        self.assertEqual(_columns(prep_path), prep.FIELDS)
        self.assertEqual(len(prep_rows), 4)
        self.assertTrue(all(row["selection"] == "protein and name CA" for row in prep_rows))
        residuals = _load_script("analyze_md_per_frame_residuals")
        residual_path = ROOT / "results" / "md_trajectory_focus" / "per_frame_residuals" / "per_frame_residuals.tsv"
        residual_rows = _read_tsv(residual_path)
        self.assertEqual(_columns(residual_path), residuals.FIELDS)
        self.assertEqual(len(residual_rows), 500 + 500 + 201 + 181)

    def test_optional_adk_missing_behavior_and_pca_toy_schema(self) -> None:
        """Optional scripts should handle missing annotation and PCA toy data."""

        adk = _load_script("analyze_adk_domain_motion")
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = adk.build_rows(Path(tmpdir))
        self.assertEqual(rows[0]["domain"], "unavailable")
        pca = _load_script("analyze_md_pca_alignment_effect")
        coords = np.random.default_rng(0).normal(size=(6, 4, 3))
        projections, fractions, loadings = pca.pca_components(coords)
        self.assertEqual(projections.shape, (6, 3))
        self.assertEqual(loadings.shape, (3, 4))
        self.assertLessEqual(float(np.sum(fractions)), 1.0 + 1e-12)

    def test_text_updates_and_readiness_audit(self) -> None:
        """Required text updates and readiness audit should exist."""

        text_dir = ROOT / "results" / "md_trajectory_focus" / "paper_text_updates"
        expected = [
            "result_md_trajectory_benchmark_update.md",
            "result_temporal_stability_update.md",
            "result_stride_length_reference_update.md",
            "result_flexibility_profile_update.md",
            "result_warm_start_runtime_update.md",
            "methods_md_preprocessing_update.md",
            "limitations_md_trajectory_update.md",
            "supplementary_allocation_update.md",
        ]
        for name in expected:
            self.assertTrue((text_dir / name).exists(), name)
            self.assertGreater((text_dir / name).stat().st_size, 20)
        blockers = json.loads(
            (ROOT / "results" / "md_trajectory_focus" / "readiness_audit" / "blockers.json").read_text()
        )
        self.assertEqual(blockers["blockers"], [])
        self.assertIn("ready", blockers["recommendation"].lower())


if __name__ == "__main__":
    unittest.main()
