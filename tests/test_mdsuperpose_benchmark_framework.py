"""Tests for benchmark serialization and lightweight trajectory I/O."""

from __future__ import annotations

import json
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
    BenchmarkResult,
    Trajectory,
    benchmark_hard_trim_control_alignment,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    benchmark_soft_weight_robust_alignment,
    load_trajectory_from_array_file,
    load_trajectory_from_multimodel_pdb,
    summarize_benchmark_rows,
    write_benchmark_json,
    write_multimodel_pdb,
)


class BenchmarkFrameworkTest(unittest.TestCase):
    """Check benchmark rows, summaries, and simple input loaders."""

    def test_benchmark_rows_and_summary_include_new_fields(self) -> None:
        """Benchmark rows should carry weighted RMSD and iteration information."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.1], [2.2, 0.0, -0.1]],
                [[-0.1, 0.0, 0.0], [0.9, 0.0, -0.1], [1.8, 0.0, 0.2]],
            ],
            dtype=float,
        )
        trajectory = Trajectory(coords=coords)
        result = benchmark_soft_weight_robust_alignment(
            trajectory,
            core_fraction=2.0 / 3.0,
            rule="huber",
            delta=0.2,
            sequence_continuity_strength=0.4,
            sequence_continuity_passes=2,
        )

        rows = benchmark_results_to_rows(
            [result],
            benchmark_mode="synthetic",
            dataset_name="toy_case",
            repeat_index=0,
            preprocessing_steps=["source=unit_test"],
        )
        summary = summarize_benchmark_rows(rows)

        self.assertEqual(rows[0]["method_name"], "soft_weight_robust_alignment")
        self.assertIsNotNone(rows[0]["weighted_rmsd"])
        self.assertIsNotNone(rows[0]["iteration_count"])
        self.assertEqual(rows[0]["unified_posthoc_rmsd"], rows[0]["rmsd"])
        self.assertIsNone(rows[0]["mdlovofit_native_rmsd"])
        self.assertIsNone(rows[0]["mdlovofit_native_least_mobile_rmsd"])
        self.assertIsNone(rows[0]["mdlovofit_output_subset_atom_count"])
        self.assertEqual(summary[0]["n_repeats"], 1)
        self.assertIsNotNone(summary[0]["weighted_rmsd_mean"])
        self.assertEqual(summary[0]["unified_posthoc_rmsd_mean"], summary[0]["rmsd_mean"])
        self.assertIsNone(summary[0]["mdlovofit_native_least_mobile_rmsd_mean"])
        self.assertEqual(result.extra_metadata["sequence_continuity_strength"], 0.4)
        self.assertEqual(result.extra_metadata["sequence_continuity_passes"], 2)
        self.assertIn("core_transition_count", result.extra_metadata)

    def test_hard_trim_control_benchmark_reports_binary_core_metadata(self) -> None:
        """The hard-trim control benchmark should emit comparable summary fields."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [2.1, 0.0, 0.0], [3.6, 0.0, 0.0]],
                [[-0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0], [3.4, 0.0, 0.0]],
            ],
            dtype=float,
        )
        trajectory = Trajectory(coords=coords)
        result = benchmark_hard_trim_control_alignment(
            trajectory,
            core_fraction=0.5,
            max_iterations=12,
            tolerance=1e-6,
        )

        self.assertEqual(result.method_name, "hard_trim_control_alignment")
        self.assertIsNotNone(result.weighted_rmsd)
        self.assertIsNotNone(result.iteration_count)
        self.assertEqual(result.core_fraction, 0.5)
        self.assertEqual(result.extra_metadata["warm_start"], False)
        self.assertIn("selected_core_indices", result.extra_metadata)

    def test_benchmark_rows_and_summary_include_theseus_fields(self) -> None:
        """THESEUS-native fields should survive row flattening and summarization."""

        result = BenchmarkResult(
            method_name="external_theseus",
            runtime_seconds=1.2,
            algorithm_runtime_seconds=None,
            overhead_seconds=0.1,
            rmsd=2.5,
            weighted_rmsd=None,
            iteration_count=None,
            core_size=10,
            core_fraction=0.5,
            parsing_success=True,
            transform_if_available=None,
            extra_metadata={
                "reported_classical_pairwise_rmsd": 2.1,
                "reported_maximum_likelihood_rmsd": 1.9,
                "reported_least_squares_sigma": 0.8,
                "reported_maximum_likelihood_sigma": 0.7,
                "output_core_atom_count": 10,
                "output_core_atom_fraction": 0.5,
            },
        )

        rows = benchmark_results_to_rows(
            [result],
            benchmark_mode="real",
            dataset_name="toy_case",
            repeat_index=0,
            preprocessing_steps=["source=unit_test"],
        )
        summary = summarize_benchmark_rows(rows)

        self.assertEqual(rows[0]["method_name"], "external_theseus")
        self.assertEqual(rows[0]["theseus_native_classical_pairwise_rmsd"], 2.1)
        self.assertEqual(rows[0]["theseus_native_maximum_likelihood_rmsd"], 1.9)
        self.assertEqual(rows[0]["theseus_output_core_atom_count"], 10)
        self.assertEqual(summary[0]["theseus_native_classical_pairwise_rmsd_mean"], 2.1)
        self.assertEqual(summary[0]["theseus_output_core_atom_fraction_mean"], 0.5)
        self.assertEqual(summary[0]["parsing_success_rate"], 1.0)

    def test_benchmark_result_reports_contact_metadata_when_enabled(self) -> None:
        """Soft benchmark metadata should expose sparse contact-graph diagnostics."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.2, 0.0], [2.0, 2.2, 0.0]],
                [[0.1, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.9, 0.0], [2.0, 2.2, 0.0]],
            ],
            dtype=float,
        )
        result = benchmark_soft_weight_robust_alignment(
            Trajectory(coords=coords),
            core_fraction=0.5,
            rule="huber",
            delta=0.25,
            contact_consistency_strength=0.5,
            contact_consistency_passes=2,
            contact_distance_cutoff=2.5,
            contact_min_sequence_separation=2,
            contact_max_neighbors=1,
        )

        self.assertEqual(result.extra_metadata["contact_consistency_strength"], 0.5)
        self.assertEqual(result.extra_metadata["contact_consistency_passes"], 2)
        self.assertIsNotNone(result.extra_metadata["contact_graph_edge_count"])
        self.assertIsNotNone(result.extra_metadata["contact_graph_mean_degree"])
        self.assertIsNotNone(result.extra_metadata["contact_graph_disagreement"])

    def test_array_and_pdb_loaders_roundtrip_coordinates(self) -> None:
        """Lightweight real-input loaders should parse array and PDB sources."""

        coords = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.5, 0.1], [2.1, 0.0, 0.0]],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            npy_path = tmp / "coords.npy"
            pdb_path = tmp / "coords.pdb"
            np.save(npy_path, coords)
            write_multimodel_pdb(coords, pdb_path)

            from_array = load_trajectory_from_array_file(npy_path)
            from_pdb = load_trajectory_from_multimodel_pdb(pdb_path)

        np.testing.assert_allclose(from_array.coords, coords)
        np.testing.assert_allclose(from_pdb.coords, coords, atol=1e-3)

    def test_json_writer_handles_serialized_benchmark_records(self) -> None:
        """JSON output should remain machine-readable for later plotting."""

        record = benchmark_result_to_record(
            benchmark_soft_weight_robust_alignment(
                Trajectory(
                    coords=np.array(
                        [
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                            [[0.1, 0.0, 0.0], [1.0, 0.1, 0.0], [1.9, 0.0, 0.0]],
                        ],
                        dtype=float,
                    )
                ),
                core_fraction=2.0 / 3.0,
                rule="huber",
                delta=0.2,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            write_benchmark_json(path, {"records": [record]})
            payload = json.loads(path.read_text())

        self.assertEqual(len(payload["records"]), 1)
        self.assertIn("weighted_rmsd", payload["records"][0])

    def test_soft_benchmark_supports_warm_start_and_frame_reordering(self) -> None:
        """Warm-start benchmark modes should remain numerically stable."""

        base = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        frames = []
        for time_index in range(8):
            theta = 0.03 * time_index
            rotation = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            translation = np.array([0.05 * time_index, -0.02 * time_index, 0.0], dtype=float)
            frame = base @ rotation.T + translation
            frame[[4, 5]] += np.array([0.0, 0.1 * np.sin(0.4 * time_index), 0.0], dtype=float)
            frames.append(frame)
        trajectory = Trajectory(coords=np.asarray(frames, dtype=float))

        baseline = benchmark_soft_weight_robust_alignment(
            trajectory,
            core_fraction=0.5,
            rule="huber",
            delta=0.16,
            max_iterations=20,
            tolerance=1e-6,
        )
        for frame_order in ("reference_rmsd", "reference_centered_rmsd"):
            with self.subTest(frame_order=frame_order):
                optimized = benchmark_soft_weight_robust_alignment(
                    trajectory,
                    core_fraction=0.5,
                    rule="huber",
                    delta=0.16,
                    max_iterations=20,
                    tolerance=1e-6,
                    warm_start=True,
                    frame_order=frame_order,
                )

                self.assertAlmostEqual(baseline.rmsd, optimized.rmsd, places=6)
                self.assertAlmostEqual(baseline.weighted_rmsd, optimized.weighted_rmsd, places=6)
                self.assertEqual(optimized.extra_metadata["warm_start"], True)
                self.assertEqual(optimized.extra_metadata["frame_order"], frame_order)
                self.assertEqual(len(optimized.extra_metadata["frame_permutation"]), trajectory.n_frames)
                self.assertIn("ordering_runtime_seconds", optimized.extra_metadata)
                self.assertIn("solver_runtime_seconds", optimized.extra_metadata)
                self.assertGreaterEqual(optimized.extra_metadata["ordering_runtime_seconds"], 0.0)
                self.assertGreaterEqual(
                    optimized.runtime_seconds,
                    optimized.extra_metadata["solver_runtime_seconds"],
                )


if __name__ == "__main__":
    unittest.main()
