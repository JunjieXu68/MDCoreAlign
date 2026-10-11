#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mdcorealign apply --input trajectory.pdb --input-format ca-pdb --result-dir ../03_multimodel_pdb/output --purpose structure --frames 0,2,4 --output aligned.cif --output-dir output
