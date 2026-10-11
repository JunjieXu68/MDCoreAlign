#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then ROOT="${SCRIPT_DIR}"; else ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"; fi
if [[ -d "${ROOT}/04_examples" ]]; then EX="${ROOT}/04_examples"; else EX="${ROOT}/examples"; fi
WORK="$(mktemp -d)"
mdcorealign align --input "${EX}/05_gpu/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir "${WORK}/cpu"
mdcorealign align --input "${EX}/05_gpu/trajectory.pdb" --input-format ca-pdb --mode throughput --backend gpu --fast-representative-count 5 --output-dir "${WORK}/gpu"
python - "${WORK}" <<'PY'
import csv, json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); data=json.loads((root/'gpu/summary.json').read_text())
assert data['backend_requested']=='gpu' and data['backend_used']=='gpu', data
assert (root/'gpu/transforms.npz').exists()
def profile(path):
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle, delimiter='\t'))
    return (
        np.asarray([float(row['stable_weight']) for row in rows]),
        np.asarray([row['core_label'].lower() in {'1','true'} for row in rows]),
        np.asarray([float(row['robust_RMSF']) for row in rows]),
    )
cpu=profile(root/'cpu/residue_profile.tsv'); gpu=profile(root/'gpu/residue_profile.tsv')
np.testing.assert_allclose(gpu[0], cpu[0], atol=1e-6, rtol=0)
np.testing.assert_array_equal(gpu[1], cpu[1])
np.testing.assert_allclose(gpu[2], cpu[2], atol=1e-6, rtol=0)
def rmsd(path):
    with path.open(newline='') as handle:
        return np.asarray([float(row['full_set_RMSD']) for row in csv.DictReader(handle, delimiter='\t')])
np.testing.assert_allclose(rmsd(root/'gpu/frame_metrics.tsv'), rmsd(root/'cpu/frame_metrics.tsv'), atol=1e-6, rtol=0)
print('PASS: explicit GPU used without fallback and compact CPU/GPU outputs are scientifically equivalent')
PY
