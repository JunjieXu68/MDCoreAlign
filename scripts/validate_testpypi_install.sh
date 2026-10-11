#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${REPO_ROOT}/scripts/validate_testpypi_install.py" "$@"
