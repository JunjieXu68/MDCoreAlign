#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$(dirname "${BASH_SOURCE[0]}")/output}"
mdcorealign local-deformation \
  --input "${ROOT}/examples/toy_md/toy_ca_trajectory.pdb" \
  --input-format ca-pdb --backend cpu --block-count 2 --output-dir "${OUT}"
