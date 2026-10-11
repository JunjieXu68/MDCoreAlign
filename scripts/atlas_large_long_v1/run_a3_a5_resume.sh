#!/usr/bin/env bash
set -euo pipefail

ROOT="/storage/junjie/projects/mdlovofit"
OUT="$ROOT/results/atlas_large_long_v1"
GPU_PYTHON="$ROOT/.venv-gpu-cu12/bin/python"
MDA_PYTHON="/home/jjxu/miniconda3/envs/mdlovofit/bin/python"
CURRENT_EXTERNAL_PID="${ATLAS_EXTERNAL_PID:-2688925}"

cd "$ROOT"
export PYTHONPATH="src:."
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

mkdir -p "$OUT/orchestration"
STATUS="$OUT/orchestration/a3_a5_resume_status.txt"

stage() {
    printf '%s\t%s\n' "$(date -Ins)" "$1" | tee -a "$STATUS"
}

trap 'stage "failed_at_line_${LINENO}"' ERR

stage "waiting_for_existing_external_pid_${CURRENT_EXTERNAL_PID}"
while kill -0 "$CURRENT_EXTERNAL_PID" 2>/dev/null; do
    sleep 60
done

stage "resume_or_verify_external"
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a3_external.py run
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a3_external.py finalize

stage "measure_raw_source_contract"
"$MDA_PYTHON" scripts/atlas_large_long_v1/prepare_a3_external_inputs.py measure-raw-source

stage "run_materialised_output_contract"
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a3_internal_contracts.py run
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a3_internal_contracts.py finalize

stage "run_same_shape_batch_controls"
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a5_batching.py run
"$GPU_PYTHON" scripts/atlas_large_long_v1/run_a5_batching.py finalize

stage "finalize_A4_A5_publication_package"
"$GPU_PYTHON" scripts/atlas_large_long_v1/finalize_a4_a5.py

stage "run_full_test_suite"
if "$GPU_PYTHON" -m pytest -q >"$OUT/orchestration/full_test_suite.log" 2>&1; then
    stage "full_test_suite_passed"
else
    stage "full_test_suite_failed"
    exit 1
fi

stage "complete"
