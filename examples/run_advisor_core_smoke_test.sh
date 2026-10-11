#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EX="${ROOT}/examples"
WORK="$(mktemp -d)"
mdcorealign align --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/throughput"
mdcorealign align --input "${EX}/04_exact_audit/trajectory.pdb" --input-format ca-pdb --mode exact --backend cpu --exact-representative-count 5 --output-dir "${WORK}/exact"
mdcorealign local-deformation --input "${EX}/06_local_deformation/trajectory.pdb" --input-format ca-pdb --backend cpu --block-count 2 --output-dir "${WORK}/w5"
mdcorealign align --input "${EX}/03_multimodel_pdb/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --ordinary-rmsf --output-dir "${WORK}/ordinary"
python - "${WORK}" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
assert json.loads((root/'throughput/summary.json').read_text())['mode']=='throughput'
assert json.loads((root/'exact/summary.json').read_text())['mode']=='exact'
assert (root/'w5/local_deformation.tsv').stat().st_size > 0
with (root/'ordinary/residue_profile.tsv').open() as handle:
    assert 'ordinary_RMSF' in next(csv.reader(handle, delimiter='\t'))
print('PASS: throughput CPU, exact CPU, W5 and ordinary RMSF opt-in')
PY
