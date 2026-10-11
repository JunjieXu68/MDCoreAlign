#!/usr/bin/env bash
set -uo pipefail

cd /storage/junjie/projects/mdlovofit

while true; do
  date
  timeout 300 python scripts/mdcath_benchmark/monitor_mdcath_jobs.py --mode domain_canonical || true
  if pgrep -f "aggregate_mdcath_results.py --mode domain_canonical" >/dev/null; then
    echo "aggregate already running; skipping this cycle"
  else
    timeout 300 python scripts/mdcath_benchmark/aggregate_mdcath_results.py --mode domain_canonical || true
  fi
  if pgrep -f "validate_mdcath_outputs.py --mode domain_canonical" >/dev/null; then
    echo "validation already running; skipping this cycle"
  else
    timeout 300 python scripts/mdcath_benchmark/validate_mdcath_outputs.py \
      --mode domain_canonical \
      --min-success 5000 \
      --require-completion || true
  fi
  echo "watch cycle complete"
  sleep 600
done
