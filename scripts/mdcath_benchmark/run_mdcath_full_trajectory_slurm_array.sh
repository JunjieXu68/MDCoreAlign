#!/usr/bin/env bash
#SBATCH --job-name=mdcath_full_trajectory
#SBATCH --output=results/mdcath_benchmark/logs/slurm/full_trajectory/%A_%a.out
#SBATCH --error=results/mdcath_benchmark/logs/slurm/full_trajectory/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-999

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/storage/junjie/projects/mdlovofit}"
mkdir -p results/mdcath_benchmark/logs/slurm/full_trajectory

export MDCATH_CACHE="${MDCATH_CACHE:-/storage/junjie/projects/mdlovofit/results/mdcath_benchmark/cache}"

python scripts/mdcath_benchmark/run_mdcath_benchmark.py \
  --mode full_trajectory \
  --cache-dir "$MDCATH_CACHE" \
  --allow-download \
  --max-cache-gb 1000 \
  --purge-after-success \
  --resume \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --n-jobs "$SLURM_ARRAY_TASK_COUNT" \
  --timeout-sec 600

