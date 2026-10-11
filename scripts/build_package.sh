#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${1:-${REPO_ROOT}/results/final_quality_upgrade/user_readiness}"
LOG_DIR="${REPORT_DIR}/logs"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
STARTED_AT="$(date --iso-8601=seconds)"
python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("build", "twine") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing packaging modules:", ", ".join(missing), file=sys.stderr)
    print('Install them with: pip install ".[dev]"', file=sys.stderr)
    raise SystemExit(2)
PY

BUILD_CWD="$(mktemp -d "${TMPDIR:-/tmp}/mdcorealign-build.XXXXXX")"
(
  cd "${BUILD_CWD}"
  python -m build --outdir "${REPO_ROOT}/dist" "${REPO_ROOT}"
) >"${LOG_DIR}/python_build.log" 2>&1
cd "${REPO_ROOT}"
python -m twine check dist/* >"${LOG_DIR}/twine_check.log" 2>&1
FINISHED_AT="$(date --iso-8601=seconds)"

WHEEL="$(find dist -maxdepth 1 -name '*.whl' -printf '%f' | sort | head -n 1)"
SDIST="$(find dist -maxdepth 1 -name '*.tar.gz' -printf '%f' | sort | head -n 1)"

cat >"${REPORT_DIR}/package_build_report.md" <<MD
# Package Build Report

- Started at: \`${STARTED_AT}\`
- Finished at: \`${FINISHED_AT}\`
- Status: \`pass\`
- Build command: \`python -m build\`
- Twine check command: \`python -m twine check dist/*\`
- Wheel: \`${WHEEL}\`
- Source distribution: \`${SDIST}\`
MD

cat >"${REPORT_DIR}/package_build_report.json" <<JSON
{
  "status": "pass",
  "started_at": "${STARTED_AT}",
  "finished_at": "${FINISHED_AT}",
  "build_command": "python -m build",
  "twine_check_command": "python -m twine check dist/*",
  "wheel": "${WHEEL}",
  "sdist": "${SDIST}",
  "logs_dir": "${LOG_DIR}"
}
JSON

printf '\nPackage build and twine check passed. Reports written to %s\n' "${REPORT_DIR}"
