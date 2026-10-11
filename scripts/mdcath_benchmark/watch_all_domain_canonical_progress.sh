#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROGRESS_DIR="results/mdcath_benchmark/all_domain_canonical/progress"
mkdir -p "$PROGRESS_DIR"
HISTORY="$PROGRESS_DIR/progress_history.tsv"

while true; do
  PYTHONPATH=. python scripts/mdcath_benchmark/monitor_mdcath_all_domain_canonical.py
  if [[ -f "$PROGRESS_DIR/current_status.tsv" ]]; then
    if [[ ! -f "$HISTORY" ]]; then
      cat "$PROGRESS_DIR/current_status.tsv" > "$HISTORY"
    else
      tail -n +2 "$PROGRESS_DIR/current_status.tsv" >> "$HISTORY"
    fi
  fi
  sleep 600
done
