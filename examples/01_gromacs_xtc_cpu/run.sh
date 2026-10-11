#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mdcorealign align --topology topology.gro --trajectory trajectory.xtc --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir output
