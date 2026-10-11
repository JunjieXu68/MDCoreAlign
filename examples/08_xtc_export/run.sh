#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mdcorealign apply --topology ../02_pdb_xtc_cpu/topology.pdb --trajectory ../02_pdb_xtc_cpu/trajectory.xtc --fit-selection all --apply-selection all --result-dir ../02_pdb_xtc_cpu/output --purpose trajectory --output aligned.xtc --output-dir output
