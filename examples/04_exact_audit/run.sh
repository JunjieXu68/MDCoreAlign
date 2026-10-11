#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mdcorealign align --input trajectory.pdb --input-format ca-pdb --mode exact --backend cpu --exact-representative-count 5 --output-dir output
