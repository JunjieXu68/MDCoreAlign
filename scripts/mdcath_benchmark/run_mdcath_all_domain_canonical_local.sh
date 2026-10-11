#!/usr/bin/env bash
set -euo pipefail

cd /storage/junjie/projects/mdlovofit

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MDCATH_CACHE=results/mdcath_benchmark/cache

mkdir -p results/mdcath_benchmark/all_domain_canonical/run_fullframes

PYTHONPATH=. python scripts/mdcath_benchmark/run_mdcath_subset_benchmark.py \
  --subset-manifest results/mdcath_benchmark/all_domain_canonical/mdcath_all_domain_canonical_manifest.tsv \
  --output-dir results/mdcath_benchmark/all_domain_canonical/run_fullframes \
  --cache-dir results/mdcath_benchmark/cache \
  --allow-download \
  --include-fullset \
  --include-mdcorealign \
  --include-mdlovofit \
  --include-theseus \
  --mdlovofit-exe /usr/local/bin/mdlovofit \
  --theseus-exe /home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus \
  --n-workers 2 \
  --resume \
  2>&1 | tee -a results/mdcath_benchmark/all_domain_canonical/run_fullframes/all_domain_canonical.log
