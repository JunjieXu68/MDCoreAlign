#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/ashar_private_evaluation_v1/02_archive_validation"
ARCHIVE="${ROOT}/results/ashar_private_evaluation_v1/01_package/MDCoreAlign_Ashar_evaluation_package_provisional_20260829.tar.gz"
WORK="/tmp/mdca_ashar_as3_20260829"
CPU_ENV="${WORK}/cpu_env"
GPU_ENV="${WORK}/gpu_env"
EXTRACTED="${WORK}/extracted/MDCoreAlign_Ashar_evaluation_package_provisional"

rm -rf "${WORK}"
mkdir -p "${OUT}" "${WORK}/extracted"
tar -xzf "${ARCHIVE}" -C "${WORK}/extracted"

python - "${EXTRACTED}" > "${OUT}/manifest_verification.txt" <<'PY'
import csv
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "sha256_manifest.tsv").open(), delimiter="\t"))
errors = []
for row in rows:
    path = root / row["path"]
    if not path.is_file():
        errors.append(f"missing: {row['path']}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        errors.append(f"hash mismatch: {row['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: verified {len(rows)} archived files against sha256_manifest.tsv")
PY

python -m venv "${CPU_ENV}"
"${CPU_ENV}/bin/python" -m pip install --upgrade pip > "${OUT}/cpu_pip_install.txt" 2>&1
"${CPU_ENV}/bin/python" -m pip install "${EXTRACTED}/dist/mdcorealign-0.1.0rc1-py3-none-any.whl[dev]" >> "${OUT}/cpu_pip_install.txt" 2>&1
"${CPU_ENV}/bin/python" -m pip check > "${OUT}/pip_check_cpu.txt" 2>&1
"${CPU_ENV}/bin/mdcorealign" doctor > "${OUT}/doctor_cpu.txt" 2>&1

(
  cd "${EXTRACTED}"
  PATH="${CPU_ENV}/bin:${PATH}" bash examples/run_advisor_smoke_test.sh
) > "${OUT}/advisor_smoke_output.txt" 2>&1

(
  cd "${EXTRACTED}/source"
  "${CPU_ENV}/bin/python" -m pytest -q tests
) > "${OUT}/shipped_pytest_output.txt" 2>&1

{
  printf 'command\tstatus\n'
  for command in \
    "mdcorealign --help" \
    "mdcorealign align --help" \
    "mdcorealign local-deformation --help" \
    "mdcorealign apply --help" \
    "mdcorealign doctor --help"; do
    if PATH="${CPU_ENV}/bin:${PATH}" bash -c "${command}" >/dev/null 2>&1; then
      printf '%s\tPASS\n' "${command}"
    else
      printf '%s\tFAIL\n' "${command}"
      exit 1
    fi
  done
} > "${OUT}/documented_command_audit.tsv"

IO_WORK="${WORK}/io_roundtrip"
mkdir -p "${IO_WORK}"
(
  cd "${EXTRACTED}"
  PATH="${CPU_ENV}/bin:${PATH}" bash examples/run_advisor_io_smoke_test.sh
) > "${OUT}/io_smoke_output.txt" 2>&1

"${CPU_ENV}/bin/python" - "${EXTRACTED}" "${IO_WORK}" > "${OUT}/io_roundtrip.tsv" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

import MDAnalysis as mda
import gemmi

root = Path(sys.argv[1])
work = Path(sys.argv[2])
exe = Path(sys.executable).with_name("mdcorealign")
trajectory = root / "examples/03_multimodel_pdb/trajectory.pdb"
result = work / "analysis"
subprocess.run(
    [str(exe), "align", "--input", str(trajectory), "--input-format", "ca-pdb",
     "--mode", "throughput", "--backend", "cpu", "--fast-representative-count", "5",
     "--output-dir", str(result)], check=True, capture_output=True, text=True
)
outputs = {
    "mmCIF": (work / "aligned.cif", ["--purpose", "structure", "--frames", "0,2,4"]),
    "XTC": (work / "aligned.xtc", ["--purpose", "trajectory"]),
    "PDB": (work / "aligned.pdb", ["--purpose", "compatibility"]),
}
rows = []
for name, (output, options) in outputs.items():
    command = [str(exe), "apply", "--input", str(trajectory), "--input-format", "ca-pdb",
               "--result-dir", str(result), *options, "--output", str(output),
               "--output-dir", str(work / f"apply_{name.lower()}")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    if name == "mmCIF":
        count = len(gemmi.cif.read(str(output)))
        passed = count > 0
        detail = f"blocks={count}"
    elif name == "XTC":
        universe = mda.Universe(str(root / "examples/02_pdb_xtc_cpu/topology.pdb"), str(output))
        count = len(universe.trajectory)
        passed = count == 5 and universe.atoms.n_atoms == 8
        detail = f"frames={count};atoms={universe.atoms.n_atoms}"
    else:
        universe = mda.Universe(str(output))
        count = len(universe.trajectory)
        passed = count == 5 and universe.atoms.n_atoms == 8
        detail = f"models={count};atoms={universe.atoms.n_atoms}"
    rows.append({"format": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    if not passed:
        raise SystemExit(f"{name} round trip failed: {detail}")
writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], delimiter="\t")
writer.writeheader()
writer.writerows(rows)
PY

python -m venv "${GPU_ENV}"
"${GPU_ENV}/bin/python" -m pip install --upgrade pip > "${OUT}/gpu_pip_install.txt" 2>&1
"${GPU_ENV}/bin/python" -m pip install "${EXTRACTED}/dist/mdcorealign-0.1.0rc1-py3-none-any.whl" "cupy-cuda12x==13.6.0" >> "${OUT}/gpu_pip_install.txt" 2>&1
"${GPU_ENV}/bin/python" -m pip check > "${OUT}/pip_check_gpu.txt" 2>&1
"${GPU_ENV}/bin/mdcorealign" doctor > "${OUT}/doctor_gpu.txt" 2>&1
(
  cd "${EXTRACTED}"
  PATH="${GPU_ENV}/bin:${PATH}" bash examples/run_gpu_smoke_test.sh
) > "${OUT}/gpu_smoke_output.txt" 2>&1

cat > "${OUT}/archive_only_install_report.md" <<EOF
# AS3 Archive-Only Installation Report

- Provisional archive: \`results/ashar_private_evaluation_v1/01_package/$(basename "${ARCHIVE}")\`
- Extracted package: fresh temporary directory outside the repository
- CPU environment: fresh \`venv\`; MDCoreAlign installed only from the shipped wheel
- GPU environment: separate fresh \`venv\`; the same shipped wheel plus \`cupy-cuda12x==13.6.0\`
- Manifest verification: PASS
- CPU \`pip check\`: PASS
- CPU doctor and advisor smoke: PASS
- Shipped tests: PASS
- Documented commands: PASS
- mmCIF/XTC/PDB round trips: PASS
- GPU \`pip check\`: PASS
- Explicit GPU smoke with no CPU fallback: PASS

Detailed stdout and command-level evidence are stored in this directory.
EOF

echo "AS3 archive-only validation passed"
