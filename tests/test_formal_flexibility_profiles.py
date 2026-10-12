"""Tests for formal flexibility-profile benchmark helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark" / "formal_flexibility_profiles.py"
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_formal_script_module():
    """Load the formal flexibility script as a testable module."""

    spec = importlib.util.spec_from_file_location("formal_flexibility_profiles", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load formal_flexibility_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAtom:
    """Small fake atom object with the attributes used by metadata inference."""

    def __init__(self, *, resname: str, resid: int, segid: str = "", chainID: str = "") -> None:
        self.name = "CA"
        self.resname = resname
        self.resid = resid
        self.segid = segid
        self.chainID = chainID


class FormalFlexibilityProfilesTest(unittest.TestCase):
    """Check chain metadata inference used by formal RMSF benchmark outputs."""

    def test_distinct_segids_map_to_distinct_chain_ids(self) -> None:
        """PROA/PROB-style segment IDs should become A/B chain IDs."""

        module = _load_formal_script_module()
        atoms = [
            FakeAtom(resname="TYR", resid=7, segid="PROA", chainID="P"),
            FakeAtom(resname="ASP", resid=8, segid="PROA", chainID="P"),
            FakeAtom(resname="TYR", resid=7, segid="PROB", chainID="P"),
            FakeAtom(resname="ASP", resid=8, segid="PROB", chainID="P"),
        ]
        self.assertEqual(module.infer_chain_values(atoms), ["A", "A", "B", "B"])

    def test_duplicate_residue_blocks_without_chain_metadata_get_split(self) -> None:
        """Repeated residue IDs in chainless topology should be split into pseudochains."""

        module = _load_formal_script_module()
        atoms = [
            FakeAtom(resname="SER", resid=10, segid="SYSTEM"),
            FakeAtom(resname="ASP", resid=11, segid="SYSTEM"),
            FakeAtom(resname="SER", resid=10, segid="SYSTEM"),
            FakeAtom(resname="ASP", resid=11, segid="SYSTEM"),
        ]
        self.assertEqual(module.infer_chain_values(atoms), ["A", "A", "B", "B"])


if __name__ == "__main__":
    unittest.main()
