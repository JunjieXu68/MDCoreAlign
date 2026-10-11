#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

mkdir -p results/project_completion/logs \
  results/project_completion/status \
  results/project_completion/tables \
  results/project_completion/figures \
  results/project_completion/manuscript \
  docs/project_completion

export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"

echo "[run_completion_pipeline] started at $(date --iso-8601=seconds)"
echo "[run_completion_pipeline] working directory: ${PWD}"
python scripts/project_completion/completion_pipeline.py
status=$?
echo "[run_completion_pipeline] finished at $(date --iso-8601=seconds) with status ${status}"
exit "${status}"
