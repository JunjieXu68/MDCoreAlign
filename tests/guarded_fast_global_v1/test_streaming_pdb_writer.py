"""Tests for the allocation-lean matched-output PDB writer."""

from __future__ import annotations

import numpy as np

from mdcorealign.pdb_writer import write_multimodel_ca_pdb_streaming
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_coords,
    write_multimodel_pdb,
)


def test_streaming_writer_preserves_legacy_pdb_coordinates(tmp_path) -> None:
    rng = np.random.default_rng(902)
    coords = rng.normal(size=(4, 7, 3))
    legacy = tmp_path / "legacy.pdb"
    streaming = tmp_path / "streaming.pdb"

    write_multimodel_pdb(coords, legacy)
    write_multimodel_ca_pdb_streaming(coords, streaming)

    legacy_coords = read_multimodel_pdb_coords(legacy, 7)
    streaming_coords = read_multimodel_pdb_coords(streaming, 7)
    np.testing.assert_array_equal(streaming_coords, legacy_coords)
    assert streaming.read_text().count("MODEL") == 4
    assert streaming.read_text().endswith("END\n")


def test_streaming_writer_rejects_unrepresentable_coordinates(
    tmp_path,
) -> None:
    coords = np.zeros((2, 5, 3), dtype=float)
    coords[1, 2, 0] = 10000.0

    with np.testing.assert_raises_regex(ValueError, "8.3"):
        write_multimodel_ca_pdb_streaming(
            coords,
            tmp_path / "overflow.pdb",
        )

