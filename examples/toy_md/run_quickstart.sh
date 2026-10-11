#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/output}"

mkdir -p "${OUTPUT_DIR}"

if command -v mdcorealign >/dev/null 2>&1; then
  MDCOREALIGN_CMD=(mdcorealign)
else
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  MDCOREALIGN_CMD=(python -m mdcorealign)
fi

"${MDCOREALIGN_CMD[@]}" align \
  --input "${SCRIPT_DIR}/toy_ca_trajectory.pdb" \
  --input-format ca-pdb \
  --output-dir "${OUTPUT_DIR}"

printf '\nToy compact analysis completed. Open %s/summary.json first.\n' "${OUTPUT_DIR}"
