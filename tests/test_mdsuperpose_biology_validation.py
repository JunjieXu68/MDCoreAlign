"""Tests for lightweight biological-validation helpers."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    DomainRange,
    assign_domains_from_ranges,
    build_residue_labels,
    enrichment_ratio,
    infer_sequence_blocks,
    overlap_fraction,
    pearson_correlation,
    residue_contact_density,
    residue_interblock_contact_density,
    spearman_correlation,
    top_fraction_mask,
)


class BiologyValidationHelperTests(unittest.TestCase):
    """Check small statistical and domain-assignment helpers."""

    def test_spearman_and_pearson_perfect_order(self) -> None:
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        y = np.asarray([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0)
        self.assertAlmostEqual(spearman_correlation(x, y), 1.0)

    def test_top_fraction_mask_is_deterministic(self) -> None:
        values = np.asarray([3.0, 1.0, 2.0, 4.0])
        mask = top_fraction_mask(values, 0.5, highest=True)
        np.testing.assert_array_equal(mask, np.asarray([True, False, False, True]))

    def test_overlap_fraction_uses_smaller_selected_set(self) -> None:
        first = np.asarray([True, True, False, False])
        second = np.asarray([True, False, True, False])
        self.assertAlmostEqual(overlap_fraction(first, second), 0.5)

    def test_enrichment_ratio(self) -> None:
        selection = np.asarray([True, True, False, False, False])
        group = np.asarray([True, False, False, False, False])
        self.assertAlmostEqual(enrichment_ratio(selection, group), 2.5)

    def test_assign_domains_from_ranges(self) -> None:
        residue_numbers = np.asarray([1, 30, 70, 130, 170, 220])
        ranges = [
            DomainRange("CORE", 1, 29),
            DomainRange("NMP", 30, 67),
            DomainRange("CORE", 68, 117),
            DomainRange("LID", 118, 160),
            DomainRange("CORE", 161, 214),
        ]
        labels = assign_domains_from_ranges(residue_numbers, ranges)
        np.testing.assert_array_equal(
            labels,
            np.asarray(["CORE", "NMP", "CORE", "LID", "CORE", "unassigned"], dtype=object),
        )

    def test_infer_sequence_blocks_splits_on_residue_number_wrap(self) -> None:
        residue_numbers = np.asarray([10, 11, 12, 380, 381, 10, 11, 12])
        block_ids = infer_sequence_blocks(residue_numbers)
        np.testing.assert_array_equal(block_ids, np.asarray([0, 0, 0, 0, 0, 1, 1, 1]))

    def test_build_residue_labels_disambiguates_duplicate_numbering_blocks(self) -> None:
        residue_names = np.asarray(["ALA", "GLY", "ALA", "GLY"], dtype=object)
        residue_numbers = np.asarray([10, 11, 10, 11])
        block_ids = np.asarray([0, 0, 1, 1])
        labels = build_residue_labels(residue_names, residue_numbers, block_ids=block_ids)
        self.assertEqual(labels, ["ALA10_block1", "GLY11_block1", "ALA10_block2", "GLY11_block2"])

    def test_residue_contact_density_respects_block_boundaries(self) -> None:
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        residue_numbers = np.asarray([10, 11, 10, 11])
        block_ids = np.asarray([0, 0, 1, 1])
        density = residue_contact_density(
            coords,
            residue_numbers=residue_numbers,
            block_ids=block_ids,
            cutoff=2.0,
            min_sequence_separation=2,
        )
        np.testing.assert_array_equal(density, np.asarray([0.0, 1.0, 1.0, 0.0]))

    def test_residue_interblock_contact_density_counts_only_other_blocks(self) -> None:
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]
        )
        block_ids = np.asarray([0, 0, 1, 1])
        density = residue_interblock_contact_density(coords, block_ids=block_ids, cutoff=2.0)
        np.testing.assert_array_equal(density, np.asarray([1.0, 0.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
