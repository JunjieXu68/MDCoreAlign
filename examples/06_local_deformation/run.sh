#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mdcorealign local-deformation --input trajectory.pdb --input-format ca-pdb --backend cpu --block-count 2 --output-dir output
