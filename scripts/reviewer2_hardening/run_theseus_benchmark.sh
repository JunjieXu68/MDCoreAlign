#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/results/reviewer2_hardening/theseus_rerun"
INPUT_DIR="$OUT/inputs"
RUNS_DIR="$OUT/runs"
TABLES_DIR="$OUT/tables"
LOG="$OUT/install_attempts.log"

mkdir -p "$INPUT_DIR" "$RUNS_DIR" "$TABLES_DIR"
: > "$LOG"

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

run_logged() {
  local label="$1"
  shift
  log "COMMAND ($label): $*"
  {
    echo "===== $label ====="
    echo "\$ $*"
    "$@"
    echo "exit_code=$?"
  } >> "$LOG" 2>&1
}

sha256_or_blank() {
  local path="$1"
  if [[ -f "$path" ]]; then
    sha256sum "$path" | awk '{print $1}'
  else
    printf ''
  fi
}

detect_existing() {
  if command -v theseus >/dev/null 2>&1; then
    THESEUS_EXE="$(command -v theseus)"
    return 0
  fi
  return 1
}

try_help_version() {
  local exe="$1"
  run_logged "theseus -h" "$exe" -h
  run_logged "theseus --help" "$exe" --help
  run_logged "theseus --version" "$exe" --version
}

try_install() {
  THESEUS_EXE=""
  log "Initial executable detection"
  run_logged "which theseus" bash -lc "which theseus || true"
  run_logged "theseus -h before install" bash -lc "theseus -h || true"
  run_logged "theseus --help before install" bash -lc "theseus --help || true"
  run_logged "theseus --version before install" bash -lc "theseus --version || true"
  if detect_existing; then
    log "Found existing THESEUS at $THESEUS_EXE"
    try_help_version "$THESEUS_EXE"
    return 0
  fi

  if command -v micromamba >/dev/null 2>&1; then
    run_logged "micromamba create mdcorealign-theseus" micromamba create -y -n mdcorealign-theseus -c conda-forge theseus
    run_logged "micromamba run which theseus" micromamba run -n mdcorealign-theseus which theseus
    if micromamba run -n mdcorealign-theseus which theseus >/tmp/mdca_theseus_path 2>/dev/null; then
      THESEUS_EXE="$(cat /tmp/mdca_theseus_path)"
      try_help_version "$THESEUS_EXE"
      return 0
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    run_logged "conda create mdcorealign-theseus" conda create -y -n mdcorealign-theseus -c conda-forge theseus
    run_logged "conda run which theseus" conda run -n mdcorealign-theseus which theseus
    if conda run -n mdcorealign-theseus which theseus >/tmp/mdca_theseus_path 2>/dev/null; then
      THESEUS_EXE="$(cat /tmp/mdca_theseus_path)"
      try_help_version "$THESEUS_EXE"
      return 0
    fi
  fi

  if command -v apt-cache >/dev/null 2>&1; then
    run_logged "apt-cache policy theseus" apt-cache policy theseus
    if command -v sudo >/dev/null 2>&1; then
      run_logged "sudo apt-get update" sudo -n apt-get update
      run_logged "sudo apt-get install theseus" sudo -n apt-get install -y theseus
    fi
    if detect_existing; then
      log "Found THESEUS after apt attempt at $THESEUS_EXE"
      try_help_version "$THESEUS_EXE"
      return 0
    fi
  fi
  return 1
}

write_failure_report() {
  local final_status="$1"
  local reason="$2"
  python - "$ROOT" "$OUT" "$final_status" "$reason" <<'PY'
from __future__ import annotations
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
status = sys.argv[3]
reason = sys.argv[4]
log = out / "install_attempts.log"
env = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "os": platform.platform(),
    "python": sys.version,
    "conda_available": bool(__import__("shutil").which("conda")),
    "mamba_available": bool(__import__("shutil").which("mamba")),
    "micromamba_available": bool(__import__("shutil").which("micromamba")),
    "apt_cache_available": bool(__import__("shutil").which("apt-cache")),
    "sudo_available": bool(__import__("shutil").which("sudo")),
    "theseus_found": False,
    "final_status": status,
    "reason": reason,
}
(out / "theseus_failure_environment.json").write_text(json.dumps(env, indent=2) + "\n")
report = f"""# THESEUS Current-Environment Rerun Failure Report

- Generated at: `{env['generated_at']}`
- OS: `{env['os']}`
- Python: `{env['python'].splitlines()[0]}`
- conda available: `{env['conda_available']}`
- mamba available: `{env['mamba_available']}`
- micromamba available: `{env['micromamba_available']}`
- apt-cache available: `{env['apt_cache_available']}`
- sudo available: `{env['sudo_available']}`
- executable found: `False`
- final status: `{status}`
- reason: {reason}

## Install and detection log

Full command log: `{log.relative_to(root)}`

## Manuscript recommendation

Because THESEUS could not be rerun in the final local environment, THESEUS results
should be reported only as a previously generated and validated
statistical-superposition reference. The primary current-environment external
comparison should remain MDLovoFit, the closest MD-specific
least-mobile-substructure baseline. THESEUS-based values can contextualise
system-dependent behaviour, but should not support a primary superiority claim.
"""
(out / "theseus_failure_report.md").write_text(report)
PY
}

write_status_tables() {
  local status="$1"
  local command_text="$2"
  local exe_text="$3"
  local version_text="$4"
  local notes="$5"
  python - "$ROOT" "$OUT" "$status" "$command_text" "$exe_text" "$version_text" "$notes" <<'PY'
from __future__ import annotations
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
status = sys.argv[3]
command_text = sys.argv[4]
exe_text = sys.argv[5]
version_text = sys.argv[6]
notes = sys.argv[7]
fig2 = root / "results/final_submission_upgrade/final_figures/fig2_md_benchmark_summary.tsv"
tables = out / "tables"
tables.mkdir(parents=True, exist_ok=True)
systems = ["I-FABP", "ADK equilibrium", "NhaA equilibrium", "YiiP equilibrium short"]
with fig2.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
lookup = {(row["system"], row["method"]): row for row in rows}
out_rows = []
runtime_rows = []
for system in systems:
    soft = lookup[(system, "MDCoreAlign soft robust")]
    mdlovo = lookup[(system, "External MDLovoFit")]
    theseus = lookup[(system, "THESEUS")]
    out_rows.append({
        "system": system,
        "mdcorealign_unified_rmsd": soft["unified_posthoc_rmsd"],
        "mdlovofit_unified_rmsd": mdlovo["unified_posthoc_rmsd"],
        "theseus_unified_rmsd": "" if status != "rerun_success" else "",
        "mdcorealign_runtime_sec": soft["algorithm_runtime_seconds"],
        "mdlovofit_runtime_sec": mdlovo["algorithm_runtime_seconds"],
        "theseus_runtime_sec": "",
        "theseus_status": status,
        "theseus_command": command_text,
        "theseus_executable": exe_text,
        "theseus_version_or_header": version_text,
        "notes": notes or "THESEUS current-environment rerun did not produce usable metrics.",
    })
    runtime_rows.append({
        "system": system,
        "theseus_status": status,
        "runtime_seconds": "",
        "command": command_text,
        "executable": exe_text,
        "version_or_header": version_text,
        "notes": notes,
    })
fields = list(out_rows[0])
with (tables / "formal_benchmark_with_theseus_rerun.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(out_rows)
fields = list(runtime_rows[0])
with (tables / "theseus_runtime_summary.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(runtime_rows)
PY
}

extract_runtime() {
  local time_file="$1"
  if [[ -f "$time_file" ]]; then
    python - "$time_file" <<'PY'
from __future__ import annotations
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
if not match:
    raise SystemExit(0)
value = match.group(1)
parts = [float(part) for part in value.split(":")]
if len(parts) == 3:
    seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
elif len(parts) == 2:
    seconds = parts[0] * 60 + parts[1]
else:
    seconds = parts[0]
print(f"{seconds:.6f}")
PY
  fi
}

main() {
  log "Starting THESEUS current-environment rerun attempt"
  log "ROOT=$ROOT"
  run_logged "environment summary" bash -lc "uname -a; python --version; conda --version 2>/dev/null || true; micromamba --version 2>/dev/null || true; apt-cache --version 2>/dev/null | head -1 || true"

  log "Exporting formal CA multi-model PDB inputs"
  python "$ROOT/scripts/reviewer2_hardening/export_ca_multimodel_for_theseus.py" 2>&1 | tee -a "$LOG"

  if ! try_install; then
    log "THESEUS executable unavailable after all attempts"
    write_failure_report "install_failed" "THESEUS executable was not found after PATH, conda, and apt attempts."
    write_status_tables "install_failed" "" "" "not_available" "THESEUS executable not available in current environment."
    exit 0
  fi

  VERSION_HEADER="$("$THESEUS_EXE" --version 2>&1 | head -5 | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
  if [[ -z "$VERSION_HEADER" ]]; then
    VERSION_HEADER="$("$THESEUS_EXE" -h 2>&1 | head -5 | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
  fi

  declare -A INPUTS=(
    [ifabp]="$INPUT_DIR/ifabp_ca_multimodel.pdb"
    [adk]="$INPUT_DIR/adk_ca_multimodel.pdb"
    [nhaa]="$INPUT_DIR/nhaa_ca_multimodel.pdb"
    [yiip_short]="$INPUT_DIR/yiip_short_ca_multimodel.pdb"
  )
  SUCCESS_COUNT=0
  : > "$TABLES_DIR/theseus_runtime_summary.tsv"
  printf 'system\ttheseus_status\truntime_seconds\tcommand\texecutable\tversion_or_header\tinput_sha256\toutput_checksums_file\tnotes\n' > "$TABLES_DIR/theseus_runtime_summary.tsv"
  for system in ifabp adk nhaa yiip_short; do
    run_dir="$RUNS_DIR/$system"
    mkdir -p "$run_dir"
    input="${INPUTS[$system]}"
    cp "$input" "$run_dir/"
    local_input="$run_dir/$(basename "$input")"
    {
      echo "system=$system"
      echo "executable=$THESEUS_EXE"
      echo "version_or_header=$VERSION_HEADER"
      echo "working_directory=$run_dir"
      echo "input=$local_input"
      uname -a
      python --version
    } > "$run_dir/environment.txt"

    command="$THESEUS_EXE -a0 $local_input"
    echo "$command" > "$run_dir/command.txt"
    log "Running THESEUS for $system with -a0"
    if command -v /usr/bin/time >/dev/null 2>&1; then
      (cd "$run_dir" && /usr/bin/time -v -o time.txt "$THESEUS_EXE" -a0 "$local_input" > stdout.txt 2> stderr.txt)
      rc=$?
    else
      start=$(date +%s)
      (cd "$run_dir" && "$THESEUS_EXE" -a0 "$local_input" > stdout.txt 2> stderr.txt)
      rc=$?
      end=$(date +%s)
      echo "elapsed_seconds=$((end-start))" > "$run_dir/time.txt"
    fi
    if [[ "$rc" -ne 0 ]]; then
      log "THESEUS -a0 failed for $system with rc=$rc; trying without -a0"
      command="$THESEUS_EXE $local_input"
      echo "$command" > "$run_dir/command.txt"
      if command -v /usr/bin/time >/dev/null 2>&1; then
        (cd "$run_dir" && /usr/bin/time -v -o time.txt "$THESEUS_EXE" "$local_input" > stdout.txt 2> stderr.txt)
        rc=$?
      else
        start=$(date +%s)
        (cd "$run_dir" && "$THESEUS_EXE" "$local_input" > stdout.txt 2> stderr.txt)
        rc=$?
        end=$(date +%s)
        echo "elapsed_seconds=$((end-start))" > "$run_dir/time.txt"
      fi
    fi
    find "$run_dir" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort > "$run_dir/output_file_list.txt"
    {
      printf 'file\tsha256\n'
      find "$run_dir" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' file; do
        printf '%s\t%s\n' "$(basename "$file")" "$(sha256_or_blank "$file")"
      done | sort
    } > "$run_dir/output_checksums.tsv"
    output_checksums_file="$run_dir/output_checksums.tsv"
    input_hash="$(sha256_or_blank "$local_input")"
    runtime="$(extract_runtime "$run_dir/time.txt")"
    if [[ "$rc" -eq 0 ]] && find "$run_dir" -maxdepth 1 -type f -name '*sup*.pdb' | grep -q .; then
      status="rerun_success"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      notes="THESEUS command completed and a superposed PDB candidate was found."
    elif [[ "$rc" -eq 0 ]]; then
      status="output_parse_failed"
      notes="THESEUS command completed but no superposed PDB candidate was found."
    else
      status="execution_failed"
      notes="THESEUS command failed with return code $rc."
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$system" "$status" "$runtime" "$command" "$THESEUS_EXE" "$VERSION_HEADER" "$input_hash" "$output_checksums_file" "$notes" >> "$TABLES_DIR/theseus_runtime_summary.tsv"
  done

  if [[ "$SUCCESS_COUNT" -gt 0 ]]; then
    log "Evaluating THESEUS unified RMSD for successful runs"
    python "$ROOT/scripts/reviewer2_hardening/evaluate_theseus_unified_rmsd.py" --runs-dir "$RUNS_DIR" --tables-dir "$TABLES_DIR" 2>&1 | tee -a "$LOG" || true
  fi

  python - "$ROOT" "$OUT" "$THESEUS_EXE" "$VERSION_HEADER" <<'PY'
from __future__ import annotations
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
exe = sys.argv[3]
version = sys.argv[4]
tables = out / "tables"
fig2 = root / "results/final_submission_upgrade/final_figures/fig2_md_benchmark_summary.tsv"
runtime_path = tables / "theseus_runtime_summary.tsv"
eval_path = tables / "theseus_unified_rmsd_summary.tsv"
with fig2.open(newline="") as handle:
    fig_rows = list(csv.DictReader(handle, delimiter="\t"))
lookup = {(row["system"], row["method"]): row for row in fig_rows}
runtime_rows = []
if runtime_path.exists():
    with runtime_path.open(newline="") as handle:
        runtime_rows = list(csv.DictReader(handle, delimiter="\t"))
runtime_by_system = {
    {"ifabp": "I-FABP", "adk": "ADK equilibrium", "nhaa": "NhaA equilibrium", "yiip_short": "YiiP equilibrium short"}[row["system"]]: row
    for row in runtime_rows
}
canonical = {
    "I-FABP": "I-FABP",
    "ADK": "ADK equilibrium",
    "ADK equilibrium": "ADK equilibrium",
    "NhaA": "NhaA equilibrium",
    "NhaA equilibrium": "NhaA equilibrium",
    "YiiP short": "YiiP equilibrium short",
    "YiiP equilibrium short": "YiiP equilibrium short",
}
eval_by_system = {}
if eval_path.exists():
    with eval_path.open(newline="") as handle:
        eval_by_system = {canonical.get(row["system"], row["system"]): row for row in csv.DictReader(handle, delimiter="\t")}
rows = []
for system in ["I-FABP", "ADK equilibrium", "NhaA equilibrium", "YiiP equilibrium short"]:
    soft = lookup[(system, "MDCoreAlign soft robust")]
    mdlovo = lookup[(system, "External MDLovoFit")]
    runtime = runtime_by_system.get(system, {})
    eval_row = eval_by_system.get(system, {})
    status = runtime.get("theseus_status", "unknown")
    if eval_row:
        status = "rerun_success"
    rows.append({
        "system": system,
        "mdcorealign_unified_rmsd": soft["unified_posthoc_rmsd"],
        "mdlovofit_unified_rmsd": mdlovo["unified_posthoc_rmsd"],
        "theseus_unified_rmsd": eval_row.get("mean_unified_posthoc_rmsd", ""),
        "mdcorealign_runtime_sec": soft["algorithm_runtime_seconds"],
        "mdlovofit_runtime_sec": mdlovo["algorithm_runtime_seconds"],
        "theseus_runtime_sec": runtime.get("runtime_seconds", ""),
        "theseus_status": status,
        "theseus_command": runtime.get("command", ""),
        "theseus_executable": runtime.get("executable", exe),
        "theseus_version_or_header": runtime.get("version_or_header", version),
        "notes": runtime.get("notes", ""),
    })
fields = list(rows[0])
with (tables / "formal_benchmark_with_theseus_rerun.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY

  python - "$ROOT" "$OUT" <<'PY'
from __future__ import annotations
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
tables = out / "tables"
benchmark = list(csv.DictReader((tables / "formal_benchmark_with_theseus_rerun.tsv").open(), delimiter="\t"))
runtime = list(csv.DictReader((tables / "theseus_runtime_summary.tsv").open(), delimiter="\t"))
rmsd = list(csv.DictReader((tables / "theseus_unified_rmsd_summary.tsv").open(), delimiter="\t")) if (tables / "theseus_unified_rmsd_summary.tsv").exists() else []
all_success = bool(benchmark) and all(row["theseus_status"] == "rerun_success" for row in benchmark)
lines = [
    "# THESEUS Current-Environment Rerun Report",
    "",
    f"- Generated at: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
    f"- OS: `{platform.platform()}`",
    f"- Current rerun status: `{'success' if all_success else 'incomplete_or_failed'}`",
    f"- Install/detection log: `{(out / 'install_attempts.log').relative_to(root)}`",
    f"- Input manifest: `{(out / 'theseus_input_manifest.tsv').relative_to(root)}`",
    f"- Unified RMSD table: `{(tables / 'theseus_unified_rmsd_summary.tsv').relative_to(root)}`",
    f"- Runtime table: `{(tables / 'theseus_runtime_summary.tsv').relative_to(root)}`",
    f"- Formal comparison table: `{(tables / 'formal_benchmark_with_theseus_rerun.tsv').relative_to(root)}`",
    "",
    "## Rerun summary",
    "",
]
for row in benchmark:
    lines.append(
        f"- {row['system']}: status={row['theseus_status']}, "
        f"THESEUS RMSD={row['theseus_unified_rmsd']}, runtime={row['theseus_runtime_sec']} s, "
        f"command=`{row['theseus_command']}`."
    )
lines.extend(["", "## Version and executable"])
if runtime:
    first = runtime[0]
    lines.append(f"- Executable: `{first['executable']}`")
    lines.append(f"- Version/header: `{first['version_or_header']}`")
lines.extend(
    [
        "",
        "## Manuscript wording recommendation",
        "",
    ]
)
if all_success:
    lines.append(
        "THESEUS can now be described as a current-environment statistical-superposition baseline for the four formal CA trajectory systems. Keep the wording conservative: performance is system-dependent, and THESEUS is not an MD stable-core/flexibility workflow analogue."
    )
else:
    lines.append(
        "THESEUS should be demoted to a previously generated/validated statistical-superposition reference for any systems without successful current rerun metrics."
    )
lines.extend(
    [
        "",
        "## Notes",
        "",
        "- Unified post-hoc RMSD was computed after globally reorienting THESEUS output by fitting aligned model 1 to original frame 1.",
        "- THESEUS native sigma/RMSD statistics are not compared directly to MDCoreAlign weighted RMSD.",
        "- Commands, stdout/stderr, time output, and checksums are preserved in each run directory.",
    ]
)
(root / "docs" / "reviewer2_hardening_theseus_report.md").write_text("\n".join(lines) + "\n")
manifest = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "script": "scripts/reviewer2_hardening/run_theseus_benchmark.sh",
    "all_systems_rerun_success": all_success,
    "n_runtime_rows": len(runtime),
    "n_rmsd_rows": len(rmsd),
    "n_formal_comparison_rows": len(benchmark),
    "outputs": [
        "results/reviewer2_hardening/theseus_rerun/install_attempts.log",
        "results/reviewer2_hardening/theseus_rerun/theseus_input_manifest.tsv",
        "results/reviewer2_hardening/theseus_rerun/tables/theseus_unified_rmsd_summary.tsv",
        "results/reviewer2_hardening/theseus_rerun/tables/theseus_runtime_summary.tsv",
        "results/reviewer2_hardening/theseus_rerun/tables/formal_benchmark_with_theseus_rerun.tsv",
        "docs/reviewer2_hardening_theseus_report.md",
    ],
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

  if [[ "$SUCCESS_COUNT" -eq 0 ]]; then
    write_failure_report "execution_failed" "THESEUS executable was found but no formal system produced a parseable superposed output."
  fi
  log "THESEUS rerun attempt finished with SUCCESS_COUNT=$SUCCESS_COUNT"
}

main "$@"
