#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${1:-${REPO_ROOT}/results/final_quality_upgrade/user_readiness}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"
VENV_DIR="${TMPDIR:-/tmp}/mdcorealign-clean-install-${RUN_ID}"
TOY_OUTPUT="${TMPDIR:-/tmp}/mdcorealign-clean-install-toy-${RUN_ID}"
LOG_DIR="${REPORT_DIR}/logs"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

STARTED_AT="$(date --iso-8601=seconds)"
python -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >"${LOG_DIR}/pip_upgrade.log" 2>&1
python -m pip install "${REPO_ROOT}" >"${LOG_DIR}/pip_install_core.log" 2>&1

IMPORT_OUTPUT="$(python -c "import mdcorealign; print(mdcorealign.__version__)")"
mdcorealign --help >"${LOG_DIR}/mdcorealign_help.txt"
mdcorealign align --help >"${LOG_DIR}/mdcorealign_align_help.txt"
mdcorealign flexibility --help >"${LOG_DIR}/mdcorealign_flexibility_help.txt"
mdcorealign diagnostics --help >"${LOG_DIR}/mdcorealign_diagnostics_help.txt"
mdcorealign audit-preprocessing --help >"${LOG_DIR}/mdcorealign_audit_preprocessing_help.txt"
mdcorealign local-deformation --help >"${LOG_DIR}/mdcorealign_local_deformation_help.txt"
mdcorealign apply --help >"${LOG_DIR}/mdcorealign_apply_help.txt"

bash "${REPO_ROOT}/examples/toy_md/run_quickstart.sh" "${TOY_OUTPUT}" >"${LOG_DIR}/toy_quickstart.log" 2>&1

python -m pip install pytest build twine >"${LOG_DIR}/pip_install_dev_tools.log" 2>&1
python -m pytest -q \
  "${REPO_ROOT}/tests/test_mdcorealign_release_readiness.py" \
  "${REPO_ROOT}/tests/test_product_contract_cli.py" \
  >"${LOG_DIR}/pytest_release_readiness.log" 2>&1

FINISHED_AT="$(date --iso-8601=seconds)"
cat >"${REPORT_DIR}/clean_install_report.json" <<JSON
{
  "status": "pass",
  "started_at": "${STARTED_AT}",
  "finished_at": "${FINISHED_AT}",
  "install_command": "python -m pip install ${REPO_ROOT}",
  "import_name": "mdcorealign",
  "import_version": "${IMPORT_OUTPUT}",
  "cli_name": "mdcorealign",
  "external_tools_required_for_core_commands": false,
  "toy_output_dir": "${TOY_OUTPUT}",
  "logs_dir": "${LOG_DIR}",
  "expected_toy_outputs": {
    "aligned_trajectory.pdb": "$(test -s "${TOY_OUTPUT}/aligned_trajectory.pdb" && echo present || echo missing)",
    "weights.csv": "$(test -s "${TOY_OUTPUT}/weights.csv" && echo present || echo missing)",
    "flexibility_profile.csv": "$(test -s "${TOY_OUTPUT}/flexibility_profile.csv" && echo present || echo missing)",
    "flexible_segments.tsv": "$(test -s "${TOY_OUTPUT}/flexible_segments.tsv" && echo present || echo missing)",
    "summary.json": "$(test -s "${TOY_OUTPUT}/summary.json" && echo present || echo missing)"
  }
}
JSON

cat >"${REPORT_DIR}/clean_install_report.md" <<MD
# Clean Install Report

- Started at: \`${STARTED_AT}\`
- Finished at: \`${FINISHED_AT}\`
- Status: \`pass\`
- Install command: \`python -m pip install .\`
- Import check: \`import mdcorealign\` -> \`${IMPORT_OUTPUT}\`
- CLI checks: \`mdcorealign --help\`, \`align --help\`, \`flexibility --help\`, \`diagnostics --help\`, and \`audit-preprocessing --help\` passed.
- Toy quickstart: passed.
- Reduced release-readiness tests: passed.
- External MDLovoFit/THESEUS required for core commands: \`false\`.

## Manuscript Installation Command

\`\`\`bash
pip install mdcorealign
\`\`\`

For a source checkout:

\`\`\`bash
pip install .
\`\`\`
MD

printf '\nClean install check passed. Reports written to %s\n' "${REPORT_DIR}"
