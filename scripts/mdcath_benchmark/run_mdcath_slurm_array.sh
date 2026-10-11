#!/usr/bin/env bash
set -euo pipefail

# Example:
#   sbatch --array=0-99 scripts/mdcath_benchmark/run_mdcath_slurm_array.sh

MODE="${MDCATH_MODE:-full_trajectory}"
MANIFEST="${MDCATH_MANIFEST:-results/mdcath_benchmark/manifest/mdcath_trajectory_manifest.tsv}"
CACHE_DIR="${MDCATH_CACHE:-results/mdcath_benchmark/cache}"
N_WORKERS="${MDCATH_N_WORKERS:-1}"
MAX_TRAJECTORIES_PER_TASK="${MDCATH_MAX_TRAJECTORIES_PER_TASK:-100}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OFFSET=$(( TASK_ID * MAX_TRAJECTORIES_PER_TASK ))
STOP=$(( OFFSET + MAX_TRAJECTORIES_PER_TASK ))

python scripts/mdcath_benchmark/run_mdcath_benchmark.py \
  --mode "$MODE" \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --n-workers "$N_WORKERS" \
  --row-start "$OFFSET" \
  --row-stop "$STOP" \
  --resume
