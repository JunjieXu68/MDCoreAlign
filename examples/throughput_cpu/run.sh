#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$(dirname "${BASH_SOURCE[0]}")/output}"
mdcorealign align --input "${ROOT}/examples/toy_md/toy_ca_trajectory.pdb" \
  --input-format ca-pdb --mode throughput --backend cpu \
  --fast-representative-count 5 --output-dir "${OUT}"
