#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${1:-${REPO_ROOT}/results/throughput_v2_sparse_keyframe/09_decision/validation/gpu_clean_install}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"
VENV_DIR="${TMPDIR:-/tmp}/mdcorealign-tv10-gpu-${RUN_ID}"
OUTPUT_DIR="${TMPDIR:-/tmp}/mdcorealign-tv10-gpu-output-${RUN_ID}"
WHEEL="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name 'mdcorealign-*.whl' | sort | tail -n 1)"

mkdir -p "${REPORT_DIR}" "${OUTPUT_DIR}"
python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >"${REPORT_DIR}/pip_upgrade.log" 2>&1
python -m pip install "${WHEEL}[gpu]" >"${REPORT_DIR}/pip_install_gpu.log" 2>&1
python - <<'PY' >"${REPORT_DIR}/gpu_environment.json"
import json
import cupy as cp
import mdcorealign

properties = cp.cuda.runtime.getDeviceProperties(0)
name = properties["name"]
if isinstance(name, bytes):
    name = name.decode()
print(json.dumps({
    "mdcorealign": mdcorealign.__version__,
    "cupy": cp.__version__,
    "device_count": cp.cuda.runtime.getDeviceCount(),
    "device": name,
}, sort_keys=True))
PY
mdcorealign align \
  --input "${REPO_ROOT}/examples/toy_md/toy_ca_trajectory.pdb" \
  --input-format ca-pdb \
  --mode throughput \
  --backend gpu \
  --output-dir "${OUTPUT_DIR}" \
  >"${REPORT_DIR}/gpu_execution.log" 2>&1
python - "${OUTPUT_DIR}/summary.json" <<'PY' >"${REPORT_DIR}/backend_check.txt"
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
assert metadata["backend_requested"] == "gpu"
assert metadata["backend_used"] == "gpu"
print("backend_requested=gpu")
print("backend_used=gpu")
print("gpu_clean_install=pass")
PY
cat >"${REPORT_DIR}/gpu_clean_install_report.md" <<EOF
# GPU Clean-Install Report

- Status: pass
- Wheel: $(basename "${WHEEL}")
- Installation: fresh temporary virtual environment with the optional GPU extra
- Input: bundled CA-only toy trajectory
- Requested backend: GPU
- Used backend: GPU
- Explicit GPU requests did not fall back to CPU.
EOF
rm -rf "${VENV_DIR}" "${OUTPUT_DIR}"
printf 'GPU clean install passed. Report: %s\n' "${REPORT_DIR}/gpu_clean_install_report.md"
