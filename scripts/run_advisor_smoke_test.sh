#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then ROOT="${SCRIPT_DIR}"; else ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"; fi
if [[ -d "${ROOT}/04_examples" ]]; then EX="${ROOT}/04_examples"; else EX="${ROOT}/examples"; fi
WORK="$(mktemp -d)"
echo "Advisor smoke work directory: ${WORK}"
python - <<'PY'
import MDAnalysis, gemmi, mdcorealign, numpy
print(
    "Standard dependencies:",
    f"NumPy={numpy.__version__}",
    f"MDAnalysis={MDAnalysis.__version__}",
    f"Gemmi={gemmi.__version__}",
)
PY
mdcorealign align --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/throughput"
mdcorealign align --input "${EX}/04_exact_audit/trajectory.pdb" --input-format ca-pdb --mode exact --backend cpu --exact-representative-count 5 --output-dir "${WORK}/exact"
python "${EX}/compare_throughput_exact.py" "${WORK}/throughput" "${WORK}/exact" > "${WORK}/comparison.json"
mdcorealign local-deformation --input "${EX}/06_local_deformation/trajectory.pdb" --input-format ca-pdb --backend cpu --block-count 2 --output-dir "${WORK}/w5"
mdcorealign align --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --ordinary-rmsf --output-dir "${WORK}/ordinary"
mdcorealign apply --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --result-dir "${WORK}/throughput" --purpose structure --frames 0,2,4 --output "${WORK}/aligned.cif" --output-dir "${WORK}/mmcif"
mdcorealign align --topology "${EX}/02_pdb_xtc_cpu/topology.pdb" --trajectory "${EX}/02_pdb_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/pdbxtc"
mdcorealign align --topology "${EX}/01_gromacs_xtc_cpu/topology.gro" --trajectory "${EX}/01_gromacs_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/groxtc"
mdcorealign apply --topology "${EX}/02_pdb_xtc_cpu/topology.pdb" --trajectory "${EX}/02_pdb_xtc_cpu/trajectory.xtc" --fit-selection all --apply-selection all --result-dir "${WORK}/pdbxtc" --purpose trajectory --output "${WORK}/aligned.xtc" --output-dir "${WORK}/xtc"
mdcorealign apply --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --result-dir "${WORK}/throughput" --purpose compatibility --output "${WORK}/aligned.pdb" --output-dir "${WORK}/pdb"
python - "${WORK}" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
primary={'summary.json','residue_profile.tsv','frame_metrics.tsv','transforms.npz','flexible_segments.tsv'}
for result_name in ('throughput','exact','pdbxtc','groxtc'):
    result=root/result_name
    assert {path.name for path in result.iterdir() if path.is_file()} == primary, result
    assert (result/'diagnostics').is_dir()
required=[root/'throughput/transforms.npz',root/'throughput/residue_profile.tsv',root/'exact/transforms.npz',root/'w5/local_deformation.tsv',root/'aligned.cif',root/'aligned.xtc',root/'aligned.pdb']
assert all(p.exists() and p.stat().st_size for p in required), required
with (root/'ordinary/residue_profile.tsv').open() as h: assert 'ordinary_RMSF' in next(csv.reader(h, delimiter='\t'))
import gemmi
assert len(gemmi.cif.read(str(root/'aligned.cif'))) > 0
from MDAnalysis.coordinates.XTC import XTCReader
with XTCReader(str(root/'aligned.xtc'), n_atoms=8) as reader: assert len(reader) == 5
summary=json.loads((root/'throughput/summary.json').read_text())
assert summary['mode']=='throughput' and summary['backend_used']=='cpu' and summary['experimental_mode'] is False
assert summary['converged'] is None and summary['converged_fraction'] is None
assert summary['convergence_interpretation']=='not_applicable_fixed_budget'
print('PASS: full standard CPU product; throughput, exact, W5, ordinary RMSF, GRO+XTC, PDB+XTC, multi-model PDB, mmCIF, XTC, compatibility PDB, compact outputs')
PY
