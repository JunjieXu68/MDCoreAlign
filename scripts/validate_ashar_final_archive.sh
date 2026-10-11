#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/ashar_private_evaluation_v1/08_final_package/final_archive_validation"
ARCHIVE="${ROOT}/results/ashar_private_evaluation_v1/08_final_package/MDCoreAlign_Ashar_private_evaluation_package_20260829.tar.gz"
WORK="/tmp/mdca_ashar_as9_20260829"
PACKAGE="${WORK}/extracted/MDCoreAlign_Ashar_private_evaluation_package"
CPU_ENV="${WORK}/cpu_env"
GPU_ENV="${WORK}/gpu_env"

rm -rf "${WORK}" "${OUT}"
mkdir -p "${WORK}/extracted" "${OUT}"
tar -xzf "${ARCHIVE}" -C "${WORK}/extracted"

python - "${PACKAGE}" > "${OUT}/manifest_verification.txt" <<'PY'
import csv, hashlib, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=list(csv.DictReader((root/'sha256_manifest.tsv').open(), delimiter='\t'))
errors=[]
for row in rows:
    path=root/row['path']
    if not path.is_file(): errors.append(f"missing:{row['path']}"); continue
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: errors.append(f"hash:{row['path']}")
if errors: raise SystemExit('\n'.join(errors))
print(f"PASS: {len(rows)} files match the internal SHA-256 manifest")
PY

python - "${PACKAGE}" > "${OUT}/language_secret_path_scan.tsv" <<'PY'
import re, sys
from pathlib import Path
root=Path(sys.argv[1]); text_suffixes={'','.cff','.csv','.json','.md','.patch','.py','.rst','.sh','.toml','.tsv','.txt','.yml','.yaml'}
patterns={
 'CJK':re.compile(r'[\u3400-\u9fff]'),
 'local_absolute_path':re.compile(r'/(?:storage|home)/|/tmp/(?:tmp|mdca_)'),
 'Hugging_Face_token':re.compile(r'hf_[A-Za-z0-9]{10,}'),
 'credential_assignment':re.compile(r'(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s<]+'),
}
rows=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.suffix.lower() not in text_suffixes: continue
 try: text=path.read_text()
 except UnicodeDecodeError: continue
 for name,pattern in patterns.items():
  if pattern.search(text) or (name=='CJK' and pattern.search(path.relative_to(root).as_posix())):
   rows.append((path.relative_to(root).as_posix(),name,'FAIL'))
print('path\tcheck\tstatus')
for row in rows: print('\t'.join(row))
print(f'SUMMARY\tall_scans\t{"PASS" if not rows else "FAIL"}')
if rows: raise SystemExit(1)
PY

python -m venv "${CPU_ENV}"
"${CPU_ENV}/bin/python" -m pip install --upgrade pip > "${OUT}/cpu_install.txt" 2>&1
"${CPU_ENV}/bin/python" -m pip install "${PACKAGE}/dist/mdcorealign-0.1.0rc1-py3-none-any.whl[dev]" >> "${OUT}/cpu_install.txt" 2>&1
"${CPU_ENV}/bin/python" -m pip check > "${OUT}/pip_check_cpu.txt" 2>&1
"${CPU_ENV}/bin/mdcorealign" doctor > "${OUT}/doctor_cpu.txt" 2>&1
(
  cd "${PACKAGE}"
  PATH="${CPU_ENV}/bin:${PATH}" bash examples/run_advisor_smoke_test.sh
) > "${OUT}/advisor_smoke_output.txt" 2>&1
(
  cd "${PACKAGE}/source"
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
    PATH="${CPU_ENV}/bin:${PATH}" bash -c "${command}" >/dev/null
    printf '%s\tPASS\n' "${command}"
  done
} > "${OUT}/documented_command_audit.tsv"

REPLAY="${WORK}/unseen_replay"
(
  cd "${PACKAGE}"
  PATH="${CPU_ENV}/bin:${PATH}" mdcorealign align \
    --input examples/unseen_release_check/9ZDL.pdb \
    --input-format ca-pdb --mode throughput --backend cpu \
    --output-dir "${REPLAY}"
) > "${OUT}/unseen_replay_output.txt" 2>&1
"${CPU_ENV}/bin/python" - "${REPLAY}" > "${OUT}/unseen_replay_validation.txt" <<'PY'
import csv,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); required={'summary.json','residue_profile.tsv','frame_metrics.tsv','transforms.npz','flexible_segments.tsv'}
assert {p.name for p in root.iterdir() if p.is_file()}==required
summary=json.loads((root/'summary.json').read_text()); assert summary['backend_used']=='cpu' and summary['n_frames']==20 and summary['n_fit_residues']==53
with (root/'residue_profile.tsv').open() as h: rows=list(csv.DictReader(h,delimiter='\t'))
assert len(rows)==53 and np.isfinite([float(r['stable_weight']) for r in rows]).all()
print('PASS: bundled strictly input-unseen 9ZDL replay completed from the shipped wheel')
PY

python -m venv "${GPU_ENV}"
"${GPU_ENV}/bin/python" -m pip install --upgrade pip > "${OUT}/gpu_install.txt" 2>&1
"${GPU_ENV}/bin/python" -m pip install "${PACKAGE}/dist/mdcorealign-0.1.0rc1-py3-none-any.whl" "cupy-cuda12x==13.6.0" >> "${OUT}/gpu_install.txt" 2>&1
"${GPU_ENV}/bin/python" -m pip check > "${OUT}/pip_check_gpu.txt" 2>&1
"${GPU_ENV}/bin/mdcorealign" doctor > "${OUT}/doctor_gpu.txt" 2>&1
(
  cd "${PACKAGE}"
  PATH="${GPU_ENV}/bin:${PATH}" bash examples/run_gpu_smoke_test.sh
) > "${OUT}/gpu_smoke_output.txt" 2>&1

cat > "${ROOT}/results/ashar_private_evaluation_v1/08_final_package/final_archive_validation_report.md" <<EOF
# Final Ashar Archive-Only Validation

Status: **PASS**

- Archive extraction and internal manifest: PASS
- English-only, secret, credential, and local absolute-path scans: PASS
- Fresh CPU environment installed only the shipped wheel: PASS
- CPU \`pip check\`, doctor, advisor smoke, and purpose-based I/O: PASS
- Shipped source tests: PASS
- Documented public commands: PASS
- Bundled strictly input-unseen \`9ZDL\` replay: PASS
- Fresh GPU environment installed the same wheel plus \`cupy-cuda12x==13.6.0\`: PASS
- Explicit GPU smoke with \`backend_used=gpu\` and no fallback: PASS

All commands used files contained in the independently extracted final archive.
EOF

echo "AS9 final archive-only validation passed"
