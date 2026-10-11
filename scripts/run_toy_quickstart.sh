#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/examples/toy_md/output}"

bash "${REPO_ROOT}/examples/toy_md/run_quickstart.sh" "${OUTPUT_DIR}"
