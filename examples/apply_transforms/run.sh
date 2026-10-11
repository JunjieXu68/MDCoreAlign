#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${1:-$(dirname "${BASH_SOURCE[0]}")/output}"
ANALYSIS="${BASE}/analysis"
MATERIALISED="${BASE}/materialised"
mdcorealign align --input "${ROOT}/examples/toy_md/toy_ca_trajectory.pdb" \
  --input-format ca-pdb --mode throughput --backend cpu \
  --fast-representative-count 5 --output-dir "${ANALYSIS}"
mdcorealign apply --input "${ROOT}/examples/toy_md/toy_ca_trajectory.pdb" \
  --input-format ca-pdb --transforms "${ANALYSIS}/transforms.npz" \
  --output-dir "${MATERIALISED}"
