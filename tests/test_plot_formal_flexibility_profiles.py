"""Tests for formal flexibility-profile figure generation."""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_formal_flexibility_profiles.py"

TOY_FIELDS = [
    "residue_id",
    "resname",
    "chain",
    "robust_aligned_RMSF",
    "ordinary_aligned_RMSF",
    "soft_stability_weight",
    "mobility_score",
    "flexible_segment_id",
    "rank",
    "percentile",
    "core_label",
    "is_flexible",
    "is_terminal",
]


def _load_plot_module():
    """Load the plotting script as an importable test module."""

    spec = importlib.util.spec_from_file_location("plot_formal_flexibility_profiles", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load plot_formal_flexibility_profiles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_profile(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a tiny flexibility profile with the formal profile schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _toy_rows(chains: tuple[str, ...] = ("A", "A", "A", "A", "A", "A")) -> list[dict[str, object]]:
    """Return deterministic toy flexibility rows."""

    segment_ids = (0, 1, 1, 0, 2, 2)
    rows: list[dict[str, object]] = []
    for index, (chain, segment_id) in enumerate(zip(chains, segment_ids, strict=True), start=1):
        mobility = 0.12 * index
        stability = 1.0 - mobility
        rows.append(
            {
                "residue_id": str(index if chain == "A" else index - 3),
                "resname": ("ALA", "GLY", "SER", "ASP", "LEU", "LYS")[index - 1],
                "chain": chain,
                "robust_aligned_RMSF": 0.2 + 0.15 * index,
                "ordinary_aligned_RMSF": 0.3 + 0.11 * index,
                "soft_stability_weight": stability,
                "mobility_score": mobility,
                "flexible_segment_id": segment_id,
                "rank": index,
                "percentile": (index - 1) / 5.0,
                "core_label": 1 if index <= 3 else 0,
                "is_flexible": 1 if segment_id > 0 else 0,
                "is_terminal": 1 if index in (1, 6) else 0,
            }
        )
    return rows


class FormalFlexibilityPlotTest(unittest.TestCase):
    """Validate figure-data generation and figure-script execution."""

    def test_figure_data_schema_from_toy_profile(self) -> None:
        """Plot-ready tables should expose stable schemas."""

        module = _load_plot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "toy_flexibility_profile.csv"
            _write_profile(profile_path, _toy_rows())
            profiles = module.load_profiles(profile_files=[profile_path], system_labels=["Toy"])

        rmsf_rows = module.build_rmsf_figure_rows(profiles)
        mobility_rows = module.build_mobility_figure_rows(profiles)
        core_rows = module.build_core_noncore_rows(profiles)
        statistics_rows = module.build_statistics_rows(profiles)

        self.assertEqual(len(rmsf_rows), 12)
        self.assertEqual(set(rmsf_rows[0]), set(module.RMSF_FIGURE_FIELDS))
        self.assertEqual(set(mobility_rows[0]), set(module.MOBILITY_FIGURE_FIELDS))
        self.assertEqual(set(core_rows[0]), set(module.CORE_NONCORE_FIELDS))
        self.assertEqual(set(statistics_rows[0]), set(module.STATISTICS_FIELDS))
        self.assertEqual({row["alignment"] for row in rmsf_rows}, {"ordinary", "robust"})

    def test_flexible_segment_shading_intervals_are_contiguous(self) -> None:
        """Flexible shading intervals should follow contiguous positive segment IDs."""

        module = _load_plot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "toy_flexibility_profile.csv"
            _write_profile(profile_path, _toy_rows())
            profiles = module.load_profiles(profile_files=[profile_path], system_labels=["Toy"])

        panel = module.split_profile_panels(profiles)[0]
        intervals = module.flexible_segment_intervals(panel)

        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]["segment_id"], 1)
        self.assertEqual(intervals[0]["start_position"], 2)
        self.assertEqual(intervals[0]["end_position"], 3)
        self.assertEqual(intervals[1]["segment_id"], 2)
        self.assertEqual(intervals[1]["start_position"], 5)
        self.assertEqual(intervals[1]["end_position"], 6)

    def test_chain_splitting_for_nhaa_and_yiip(self) -> None:
        """NhaA and YiiP short should be split into chain-specific panels."""

        module = _load_plot_module()
        nhaa_path = ROOT / "results" / "flexibility" / "nhaa_flexibility_profile.csv"
        yiip_path = ROOT / "results" / "flexibility" / "yiip_short_flexibility_profile.csv"
        self.assertTrue(nhaa_path.exists())
        self.assertTrue(yiip_path.exists())

        profiles = module.load_profiles(profile_files=[nhaa_path, yiip_path], system_labels=["NhaA", "YiiP short"])
        panels = module.split_profile_panels(profiles)
        observed = {(panel.system_label, panel.chain) for panel in panels}

        self.assertIn(("NhaA", "A"), observed)
        self.assertIn(("NhaA", "B"), observed)
        self.assertIn(("YiiP short", "A"), observed)
        self.assertIn(("YiiP short", "B"), observed)
        for panel in panels:
            self.assertEqual({row["chain"] for row in panel.rows}, {panel.chain})

    def test_no_cross_chain_flexible_segment_merging(self) -> None:
        """The same segment ID in two chains should not become one cross-chain interval."""

        module = _load_plot_module()
        rows = _toy_rows(chains=("A", "A", "A", "B", "B", "B"))
        for index, row in enumerate(rows, start=1):
            row["flexible_segment_id"] = 1
            row["is_flexible"] = 1
            row["residue_id"] = str(index if row["chain"] == "A" else index - 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "toy_flexibility_profile.csv"
            _write_profile(profile_path, rows)
            profiles = module.load_profiles(profile_files=[profile_path], system_labels=["Toy"])

        panels = module.split_profile_panels(profiles)
        intervals = [interval for panel in panels for interval in module.flexible_segment_intervals(panel)]

        self.assertEqual(len(panels), 2)
        self.assertEqual(len(intervals), 2)
        self.assertEqual({interval["chain"] for interval in intervals}, {"A", "B"})
        self.assertEqual(intervals[0]["end_residue"], "3")
        self.assertEqual(intervals[1]["start_residue"], "1")

    def test_script_execution_on_small_toy_dataset(self) -> None:
        """The CLI should generate all expected outputs for a small toy profile."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            profile_path = tmp_path / "toy_flexibility_profile.csv"
            figure_dir = tmp_path / "figures"
            figure_data_dir = tmp_path / "figure_data"
            _write_profile(profile_path, _toy_rows())

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["MPLBACKEND"] = "Agg"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(tmp_path),
                    "--profile-files",
                    str(profile_path),
                    "--system-labels",
                    "Toy",
                    "--output-dir",
                    str(figure_dir),
                    "--figure-data-dir",
                    str(figure_data_dir),
                    "--dpi",
                    "80",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            expected_files = [
                figure_dir / "fig_rmsf_profiles_four_systems.pdf",
                figure_dir / "fig_rmsf_profiles_four_systems.png",
                figure_dir / "fig_mobility_vs_robust_rmsf.pdf",
                figure_dir / "fig_core_vs_noncore_rmsf.pdf",
                figure_dir / "figure_captions_draft.md",
                figure_data_dir / "fig_rmsf_profiles.tsv",
                figure_data_dir / "fig_mobility_vs_rmsf.tsv",
                figure_data_dir / "fig_core_noncore_rmsf.tsv",
                tmp_path / "flexibility_statistics_four_systems.tsv",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
