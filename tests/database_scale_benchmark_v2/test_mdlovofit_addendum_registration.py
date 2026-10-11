"""Tests for prospective MDLovoFit addendum registration helpers."""

from scripts.database_scale_benchmark_v2.register_mdlovofit_addendum import (
    frame_bin,
    residue_bin,
)


def test_residue_bins_have_locked_boundaries():
    assert residue_bin(99) == "lt100"
    assert residue_bin(100) == "100_199"
    assert residue_bin(199) == "100_199"
    assert residue_bin(200) == "200_399"
    assert residue_bin(400) == "ge400"


def test_frame_bins_have_locked_boundaries():
    assert frame_bin(449) == "lt450"
    assert frame_bin(450) == "450_499"
    assert frame_bin(499) == "450_499"
    assert frame_bin(500) == "ge500"
