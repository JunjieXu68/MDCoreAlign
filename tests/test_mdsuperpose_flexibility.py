"""Tests for RMSF and flexible-segment profiling."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    FlexibilityProfile,
    Trajectory,
    align_trajectory_ordinary,
    build_extended_flexibility_rows,
    build_flexibility_profile,
    build_rmsf_long_rows,
    compute_rmsf,
    detect_flexible_segments,
    load_ca_multimodel_pdb_trajectory,
    percentile_scores,
    summarize_flexibility_profile,
    summarize_flexible_segments,
    write_flexibility_profile_csv,
)


def _toy_flexible_trajectory() -> Trajectory:
    """Build a small trajectory with a flexible C-terminal segment."""

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    frames = []
    for frame_index in range(12):
        frame = base.copy()
        frame += np.array([0.05 * frame_index, -0.02 * frame_index, 0.0])
        displacement = 0.35 * np.sin(0.7 * frame_index)
        frame[4:, 1] += displacement
        frames.append(frame)
    return Trajectory(coords=np.asarray(frames, dtype=float))


class FlexibilityProfileTest(unittest.TestCase):
    """Validate flexibility-profile helpers."""

    def test_compute_rmsf_matches_definition(self) -> None:
        """RMSF should be the RMS distance to the per-atom mean position."""

        aligned = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            ],
            dtype=float,
        )
        rmsf = compute_rmsf(aligned)
        np.testing.assert_allclose(rmsf, np.array([1.0, 1.0]))

    def test_ordinary_alignment_removes_global_translation(self) -> None:
        """Ordinary alignment should remove rigid translations before RMSF."""

        base = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        coords = np.stack([base, base + np.array([5.0, -2.0, 1.0])])
        aligned = align_trajectory_ordinary(Trajectory(coords=coords))
        np.testing.assert_allclose(compute_rmsf(aligned), np.zeros(3), atol=1e-12)

    def test_detect_flexible_segments_is_contiguous_and_thresholded(self) -> None:
        """Flexible segment IDs should require contiguous high-mobility residues."""

        mobility = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.95], dtype=float)
        segments = detect_flexible_segments(mobility, percentile_cutoff=0.6, min_segment_length=2)
        np.testing.assert_array_equal(segments, np.array([0, 0, 1, 1, 0, 0]))

    def test_detect_flexible_segments_can_merge_short_gaps(self) -> None:
        """Short gaps should be mergeable within a chain but not across chains."""

        mobility = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.85], dtype=float)
        segments = detect_flexible_segments(
            mobility,
            percentile_cutoff=0.3,
            min_segment_length=3,
            merge_gap=1,
            chain=("A", "A", "A", "B", "B", "B"),
        )
        np.testing.assert_array_equal(segments, np.array([1, 1, 1, 0, 0, 0]))

    def test_percentile_scores_give_larger_values_larger_percentiles(self) -> None:
        """Percentiles should be monotone with the input values."""

        percentiles = percentile_scores(np.array([10.0, 30.0, 20.0]))
        np.testing.assert_allclose(percentiles, np.array([0.0, 1.0, 0.5]))

    def test_build_profile_and_csv_columns(self) -> None:
        """The profile should expose the manuscript-facing RMSF table columns."""

        trajectory = _toy_flexible_trajectory()
        profile = build_flexibility_profile(
            trajectory,
            residue_labels=("A:ALA1", "A:GLY2", "A:SER3", "A:THR4", "A:LYS5", "A:ASP6"),
            rule="huber",
            delta=0.12,
            max_iterations=12,
            tolerance=1e-6,
            flexible_percentile_cutoff=0.65,
            min_segment_length=1,
        )

        self.assertEqual(profile.n_residues, 6)
        self.assertEqual(profile.resname[0], "ALA")
        self.assertEqual(profile.chain[0], "A")
        self.assertGreater(float(np.mean(profile.robust_aligned_rmsf[4:])), float(np.mean(profile.robust_aligned_rmsf[:4])))
        self.assertTrue(np.any(profile.flexible_segment_id > 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flexibility.csv"
            write_flexibility_profile_csv(profile, path)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            list(rows[0]),
            [
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
            ],
        )

        extended_rows = build_extended_flexibility_rows(profile, core_fraction=0.5, terminal_window=1)
        self.assertIn("delta_RMSF", extended_rows[0])
        self.assertIn("core_label", extended_rows[0])
        self.assertIn("is_terminal", extended_rows[0])

        segment_rows = summarize_flexible_segments(profile, system="toy", core_fraction=0.5, terminal_window=1)
        self.assertGreaterEqual(len(segment_rows), 1)
        self.assertEqual(segment_rows[0]["system"], "toy")
        self.assertIn("mean_delta_RMSF", segment_rows[0])

        summary = summarize_flexibility_profile(profile, system="toy", core_fraction=0.5)
        self.assertEqual(summary["system"], "toy")
        self.assertEqual(summary["n_residues"], 6)
        self.assertIn("spearman_RMSF_vs_mobility", summary)

        long_rows = build_rmsf_long_rows(profile, system="toy", core_fraction=0.5)
        self.assertEqual(len(long_rows), 12)
        self.assertEqual({row["alignment"] for row in long_rows}, {"ordinary", "robust"})

    def test_load_ca_multimodel_pdb_trajectory_checks_correspondence(self) -> None:
        """CA-only multi-model PDB loading should preserve residue labels."""

        pdb_text = (
            "MODEL        1\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  CA  SER A   3       0.000   1.000   0.000  1.00  0.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "ATOM      1  CA  ALA A   1       0.100   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY A   2       1.100   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  CA  SER A   3       0.100   1.000   0.000  1.00  0.00           C\n"
            "ENDMDL\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "toy.pdb"
            path.write_text(pdb_text)
            trajectory, labels = load_ca_multimodel_pdb_trajectory(path)

        self.assertEqual(trajectory.n_frames, 2)
        self.assertEqual(trajectory.n_atoms, 3)
        self.assertEqual(labels, ("A:ALA1", "A:GLY2", "A:SER3"))

    def test_known_flexible_loop_is_recovered_from_profile(self) -> None:
        """A toy profile with one known flexible loop should recover that loop."""

        stability = np.array([0.9, 0.88, 0.86, 0.15, 0.12, 0.18, 0.84, 0.87], dtype=float)
        mobility = 1.0 - stability
        segment_ids = detect_flexible_segments(
            mobility,
            percentile_cutoff=0.65,
            min_segment_length=3,
            merge_gap=0,
            chain=("A",) * 8,
        )
        profile = FlexibilityProfile(
            residue_id=tuple(str(index) for index in range(1, 9)),
            resname=("ALA", "GLY", "SER", "ASN", "GLY", "ASP", "LEU", "LYS"),
            chain=("A",) * 8,
            robust_aligned_rmsf=np.array([0.2, 0.25, 0.22, 1.4, 1.6, 1.3, 0.3, 0.24], dtype=float),
            ordinary_aligned_rmsf=np.array([0.3, 0.35, 0.31, 1.1, 1.2, 1.0, 0.4, 0.34], dtype=float),
            soft_stability_weight=stability,
            mobility_score=mobility,
            flexible_segment_id=segment_ids,
            rank=np.array([8, 7, 6, 2, 1, 3, 5, 4], dtype=int),
            percentile=percentile_scores(mobility),
        )
        segment_rows = summarize_flexible_segments(profile, system="toy_loop", core_fraction=0.5)

        self.assertEqual(len(segment_rows), 1)
        self.assertEqual(segment_rows[0]["start_residue"], "4")
        self.assertEqual(segment_rows[0]["end_residue"], "6")
        self.assertEqual(segment_rows[0]["length"], 3)


if __name__ == "__main__":
    unittest.main()
