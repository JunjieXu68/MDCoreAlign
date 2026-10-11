#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EX="${ROOT}/examples"
WORK="$(mktemp -d)"
python -c 'import MDAnalysis, gemmi'
mdcorealign align --topology "${EX}/02_pdb_xtc_cpu/topology.pdb" --trajectory "${EX}/02_pdb_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/pdbxtc"
mdcorealign align --topology "${EX}/01_gromacs_xtc_cpu/topology.gro" --trajectory "${EX}/01_gromacs_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/groxtc"
mdcorealign align --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/pdb"
mdcorealign apply --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --result-dir "${WORK}/pdb" --purpose structure --frames 0,2,4 --output "${WORK}/aligned.cif" --output-dir "${WORK}/mmcif"
mdcorealign apply --topology "${EX}/02_pdb_xtc_cpu/topology.pdb" --trajectory "${EX}/02_pdb_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --result-dir "${WORK}/pdbxtc" --purpose trajectory --output "${WORK}/aligned.xtc" --output-dir "${WORK}/xtc"
mdcorealign apply --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --result-dir "${WORK}/pdb" --purpose compatibility --output "${WORK}/aligned.pdb" --output-dir "${WORK}/compatibility"
python - "${WORK}" <<'PY'
import sys
from pathlib import Path
import gemmi
from MDAnalysis.coordinates.XTC import XTCReader
root=Path(sys.argv[1])
assert len(gemmi.cif.read(str(root/'aligned.cif'))) > 0
with XTCReader(str(root/'aligned.xtc'), n_atoms=8) as reader:
    assert len(reader)==5
assert (root/'aligned.pdb').read_text().count('MODEL')==5
print('PASS: GRO+XTC, PDB+XTC, multi-model PDB, mmCIF, XTC and compatibility PDB')
PY
