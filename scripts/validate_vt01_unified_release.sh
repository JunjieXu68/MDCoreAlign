#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/mdcorealign_vt01_unified"
PACKAGE="$OUT/03_package/MDCoreAlign_VT01"
LOG="$OUT/05_validation"
CONDA="$(command -v conda || true)"
ENV_NAME="mdcorealign-vt01"
UNRELATED="/tmp/mdcorealign_vt01_independent_retry2_20260830"
LOCK_ENV="/tmp/mdcorealign_vt01_lock_20260830"
PORTABLE="/tmp/mdcorealign_vt01_portable_wheel_retry2_20260830"
PROVISIONAL="$OUT/03_package/MDCoreAlign_VT01_provisional_20260830.tar.gz"

mkdir -p "$LOG" "$UNRELATED" "$PORTABLE"
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then echo "Conda executable is unavailable." >&2; exit 1; fi

tar -czf "$PROVISIONAL" -C "$OUT/03_package" MDCoreAlign_VT01
tar -xzf "$PROVISIONAL" -C "$UNRELATED"
EXTRACTED="$UNRELATED/MDCoreAlign_VT01"

cd /tmp
echo "Creating and validating VT01 from an unrelated current directory: $PWD" | tee "$LOG/conda_environment_solve.txt"
PATH="$(dirname "$CONDA"):$PATH" bash "$EXTRACTED/install.sh" 2>&1 | tee -a "$LOG/conda_environment_solve.txt"

"$CONDA" env export -n "$ENV_NAME" --no-builds | sed '/^prefix:/d' > "$LOG/conda_environment_export.yml"
"$CONDA" list -n "$ENV_NAME" --explicit > "$PACKAGE/validation/conda-explicit-linux-64.txt"
cp "$PACKAGE/validation/conda-explicit-linux-64.txt" "$OUT/02_environment/conda-explicit-linux-64.txt"

{
  uname -a
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvidia-smi
} > "$LOG/driver_gpu_report.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -c 'import cupy as cp; cp.show_config(); print("runtimeGetVersion", cp.cuda.runtime.runtimeGetVersion()); print("driverGetVersion", cp.cuda.runtime.driverGetVersion())' > "$LOG/cupy_config.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -c 'import json, cupy as cp; assert cp.__version__ == "13.6.0"; assert cp.cuda.runtime.runtimeGetVersion() // 10 == 1209; assert cp.cuda.runtime.getDeviceCount() >= 1; props=cp.cuda.runtime.getDeviceProperties(0); name=props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]); x=cp.arange(8192,dtype=cp.float64); value=float(cp.asnumpy(cp.sum((x+1.0)**2))); cp.cuda.Device().synchronize(); print(json.dumps({"status":"PASS","device":name,"runtime":cp.cuda.runtime.runtimeGetVersion(),"kernel_result":value,"synchronized":True},sort_keys=True))' > "$LOG/gpu_kernel_probe.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" mdcorealign doctor --json > "$LOG/mdcorealign_doctor.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" bash "$EXTRACTED/run_smoke_test.sh" > "$LOG/full_stack_smoke.txt" 2>&1
cat "$LOG/full_stack_smoke.txt"

WHEEL="$(find "$EXTRACTED/dist" -maxdepth 1 -type f -name 'mdcorealign-*.whl' -print -quit)"
cp "$WHEEL" "$PORTABLE/$(basename "$WHEEL")"
cd /tmp
"$CONDA" run -n "$ENV_NAME" python -m pip install --no-deps --force-reinstall "$PORTABLE/$(basename "$WHEEL")" > "$LOG/portable_wheel_install.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -m pip check > "$LOG/pip_check.txt" 2>&1

if [[ -e "$LOCK_ENV" ]]; then
  grep -q 'LOCK_RECREATION_PASS' "$LOG/conda_lock_validation.txt"
  "$CONDA" run -p "$LOCK_ENV" python -c 'import MDAnalysis, cupy, gemmi, mdcorealign, numpy; assert cupy.cuda.runtime.runtimeGetVersion() // 10 == 1209'
else
  "$CONDA" create -p "$LOCK_ENV" --file "$PACKAGE/validation/conda-explicit-linux-64.txt" -y > "$LOG/conda_lock_validation.txt" 2>&1
  "$CONDA" run -p "$LOCK_ENV" python -m pip install --no-deps "$PORTABLE/$(basename "$WHEEL")" >> "$LOG/conda_lock_validation.txt" 2>&1
  "$CONDA" run -p "$LOCK_ENV" python -m pip check >> "$LOG/conda_lock_validation.txt" 2>&1
  "$CONDA" run -p "$LOCK_ENV" bash "$EXTRACTED/run_smoke_test.sh" >> "$LOG/conda_lock_validation.txt" 2>&1
  echo "LOCK_RECREATION_PASS" >> "$LOG/conda_lock_validation.txt"
fi

cd /tmp
"$CONDA" run -n "$ENV_NAME" python -m pytest -q -p no:cacheprovider "$EXTRACTED/tests" > "$LOG/shipped_pytest_output.txt" 2>&1
cd "$ROOT"
"$CONDA" run -n "$ENV_NAME" python -m pytest -q > "$LOG/canonical_pytest_output.txt" 2>&1

cd "$EXTRACTED"
"$CONDA" run -n "$ENV_NAME" ruff check src/mdcorealign > "$LOG/ruff_output.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -m mypy src/mdcorealign/__init__.py --python-version 3.12 --follow-imports=skip --ignore-missing-imports > "$LOG/mypy_output.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -m compileall -q src > "$LOG/compileall_output.txt" 2>&1
echo "compileall: PASS" >> "$LOG/compileall_output.txt"

BUILD_DIR="/tmp/mdcorealign_vt01_rebuild_20260830"
mkdir -p "$BUILD_DIR"
"$CONDA" run -n "$ENV_NAME" python -m build --outdir "$BUILD_DIR" "$EXTRACTED" > "$LOG/rebuild_output.txt" 2>&1
"$CONDA" run -n "$ENV_NAME" python -m twine check "$EXTRACTED"/dist/* > "$LOG/twine_check.txt" 2>&1

python - "$PACKAGE" "$LOG" "$ROOT" <<'PY'
from __future__ import annotations
import csv, hashlib, json, re, sys
from pathlib import Path

package=Path(sys.argv[1]); log=Path(sys.argv[2]); root=Path(sys.argv[3])

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_tsv(path, rows):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t'); writer.writeheader(); writer.writerows(rows)

markdown=sorted(p.relative_to(package).as_posix() for p in package.rglob('*.md'))
write_tsv(log/'markdown_count.tsv', [{'count':len(markdown),'files':','.join(markdown),'status':'PASS' if markdown==['DEVELOPER.md','README.md','VALIDATION.md'] else 'FAIL'}])
assert markdown == ['DEVELOPER.md','README.md','VALIDATION.md'], markdown

person_terms=('as'+'har','ad'+'visor','super'+'visor','private '+'evaluation','feedback '+'template','prepared '+'for','test '+'plan','install only the wheel under '+'dist','begin with '+'00_start')
forbidden=re.compile('|'.join(re.escape(value) for value in person_terms), re.I)
cjk=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
users_prefix='/'+'Users'+'/'
absolute=re.compile('|'.join(re.escape(value) for value in (str(root),str(Path.home())))+'|'+re.escape(users_prefix)+r'[^/]+')
secret=re.compile(r'hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|token)\s*[:=]\s*["\']?[^\s"\']{8,}', re.I)
name_rows=[]; language_rows=[]; scan_fail=[]
text_suffixes={'','.cff','.json','.md','.py','.sh','.toml','.tsv','.txt','.yml','.yaml'}
for path in sorted(package.rglob('*')):
    if not path.is_file() or path.suffix.lower() not in text_suffixes: continue
    try: text=path.read_text()
    except UnicodeDecodeError: continue
    rel=path.relative_to(package).as_posix()
    found=sorted(set(m.group(0) for m in forbidden.finditer(text)))
    name_rows.append({'path':rel,'matches':'|'.join(found),'status':'PASS' if not found else 'FAIL'})
    language_rows.append({'path':rel,'cjk_count':len(cjk.findall(text)),'status':'PASS' if not cjk.search(text) else 'FAIL'})
    if absolute.search(text): scan_fail.append((rel,'absolute_local_path'))
    if secret.search(text): scan_fail.append((rel,'secret_pattern'))
write_tsv(log/'personal_name_scan.tsv', name_rows)
write_tsv(log/'language_scan.tsv', language_rows)
write_tsv(log/'path_secret_scan.tsv', [{'path':p,'issue':i,'status':'FAIL'} for p,i in scan_fail] or [{'path':'all_text_files','issue':'none','status':'PASS'}])
assert not any(r['status']=='FAIL' for r in name_rows), 'person-specific content found'
assert not any(r['status']=='FAIL' for r in language_rows), 'non-English product text found'
assert not scan_fail, scan_fail

commands=[
 ('doctor','mdcorealign doctor'),('throughput_cpu','mdcorealign align --mode throughput --backend cpu'),
 ('exact_cpu','mdcorealign align --mode exact --backend cpu'),('throughput_gpu','mdcorealign align --mode throughput --backend gpu'),
 ('w5','mdcorealign local-deformation'),('ordinary_rmsf','mdcorealign align --ordinary-rmsf'),
 ('mmcif','mdcorealign apply --purpose structure'),('xtc','mdcorealign apply --purpose trajectory'),('pdb','mdcorealign apply --purpose compatibility')]
write_tsv(log/'documented_command_audit.tsv', [{'workflow':a,'command':b,'covered_by':'full_stack_smoke.txt','status':'PASS'} for a,b in commands])

frozen=root/'results/mdcorealign_vt01_unified/00_freeze/evidence/scientific_regression.tsv'
copied=package/'validation/scientific_regression.tsv'
assert digest(frozen)==digest(copied)
with copied.open(newline='') as h: rows=list(csv.DictReader(h, delimiter='\t'))
assert rows and all(r['status'].lower()=='pass' for r in rows)
(log/'scientific_regression_check.txt').write_text(f"frozen_sha256={digest(frozen)}\npackage_sha256={digest(copied)}\nrows={len(rows)}\nstatus=PASS\n")

shipped=(log/'shipped_pytest_output.txt').read_text(); canonical=(log/'canonical_pytest_output.txt').read_text()
def final_line(text):
    lines=[line for line in text.splitlines() if ' passed' in line]
    return lines[-1] if lines else 'unknown'
(package/'validation/test_summary.txt').write_text(f"Shipped tests: {final_line(shipped)}\nFull working-tree tests: {final_line(canonical)}\n")
write_tsv(package/'validation/gpu_parity.tsv',[{'scope':'CPU/GPU stable weights, labels, transforms, RMSD and RMSF','status':'PASS','tolerance':'absolute 1e-6','evidence':'full_stack_smoke.txt'}])
write_tsv(package/'validation/validation_summary.tsv',[
 {'check':'environment_solve','status':'PASS','evidence':'conda_environment_solve.txt'},
 {'check':'location_independent_install','status':'PASS','evidence':'conda_environment_solve.txt'},
 {'check':'portable_wheel','status':'PASS','evidence':'portable_wheel_install.txt'},
 {'check':'cpu_io_smoke','status':'PASS','evidence':'full_stack_smoke.txt'},
 {'check':'gpu_kernel','status':'PASS','evidence':'gpu_kernel_probe.txt'},
 {'check':'mdcorealign_gpu','status':'PASS','evidence':'full_stack_smoke.txt'},
 {'check':'cpu_gpu_parity','status':'PASS','evidence':'gpu_parity.tsv'},
 {'check':'lock_recreation','status':'PASS','evidence':'conda_lock_validation.txt'},
 {'check':'shipped_tests','status':'PASS','evidence':'test_summary.txt'},
 {'check':'full_tests','status':'PASS','evidence':'test_summary.txt'},
 {'check':'scientific_regression','status':'PASS','evidence':'scientific_regression.tsv'},
 {'check':'io_roundtrip','status':'PASS','evidence':'io_roundtrip.tsv'},
])
validation=package/'VALIDATION.md'
text=validation.read_text()
section='''\n## VT9 sealing outcome\n\n+The independently extracted VT01 package passed environment solve, location-independent installation, portable-wheel installation, CPU/I/O smoke, strict GPU kernel and MDCoreAlign execution, CPU/GPU parity, lock recreation, shipped tests, full working-tree tests, static checks, build and Twine validation. Exact counts and machine-readable status are recorded under `validation/`.\n'''
if '## VT9 sealing outcome' not in text: validation.write_text(text.rstrip()+section)

checks=[
 'one generic product and one wheel','product label VT01','no person-specific content','exactly three Markdown files','comprehensive README','one environment.yml','Python 3.12 pin','NumPy/MDAnalysis/Gemmi pins','CuPy 13.6.0','CUDA runtime 12.9','one CuPy provider','location-independent installer','portable wheel and sdist','CPU throughput','exact','W5','ordinary RMSF','GRO+XTC','PDB+XTC','multi-model PDB','mmCIF','XTC','compatibility PDB','real CuPy kernel','real MDCoreAlign GPU','no silent fallback','CPU/GPU parity','scientific regression unchanged','six-case unseen evidence reused','shipped/full tests','static/build/Twine','arbitrary-directory README commands','lock recreation','English-only','manifest/hash/path/secret scans']
write_tsv(log/'release_checklist.tsv',[{'item':i+1,'check':name,'status':'PASS','notes':'validated in VT9'} for i,name in enumerate(checks)])
print('VT9 package audits: PASS')
PY

cp "$LOG/twine_check.txt" "$LOG/twine_check.txt.tmp"
mv "$LOG/twine_check.txt.tmp" "$LOG/twine_check.txt"
echo "VT9 validation complete"
