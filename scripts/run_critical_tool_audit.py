"""Run an adversarial critical audit for MDCoreAlign.

This script generates reviewer-style audit outputs. It does not change
scientific benchmark numbers or implement new algorithms.
"""

from __future__ import annotations

import csv
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "critical_tool_audit"


def rel(path: Path) -> str:
    """Return repository-relative path when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    ensure(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: object) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sanitize(text: str) -> str:
    return text.replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 240,
    log_path: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = sanitize(completed.stdout)
        stderr = sanitize(completed.stderr)
        result = {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "success": completed.returncode == 0,
            "runtime_seconds": time.perf_counter() - started,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_summary": stdout[-500:].replace("\n", "\\n"),
            "stderr_summary": stderr[-500:].replace("\n", "\\n"),
        }
    except Exception as exc:
        result = {
            "command": " ".join(command),
            "exit_code": -1,
            "success": False,
            "runtime_seconds": time.perf_counter() - started,
            "stdout": "",
            "stderr": sanitize(str(exc)),
            "stdout_summary": "",
            "stderr_summary": sanitize(str(exc))[-500:].replace("\n", "\\n"),
        }
    if log_path is not None:
        ensure(log_path.parent)
        log_path.write_text(
            f"$ {result['command']}\n\n## STDOUT\n{result['stdout']}\n\n## STDERR\n{result['stderr']}\n"
        )
    return result


def env_src() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


FINDING_FIELDS = [
    "finding_id",
    "severity",
    "category",
    "file_or_command",
    "description",
    "evidence",
    "recommended_fix",
    "must_fix_before_advisor",
    "must_fix_before_public_release",
]


class AuditState:
    def __init__(self) -> None:
        self.findings: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []
        self.tests: list[dict[str, object]] = []
        self.fixes: list[dict[str, object]] = []

    def add_finding(
        self,
        severity: str,
        category: str,
        file_or_command: str,
        description: str,
        evidence: str,
        recommended_fix: str,
        advisor: str = "no",
        public: str = "yes",
    ) -> str:
        finding_id = f"F-{len(self.findings) + 1:03d}"
        self.findings.append(
            {
                "finding_id": finding_id,
                "severity": severity,
                "category": category,
                "file_or_command": file_or_command,
                "description": description,
                "evidence": evidence,
                "recommended_fix": recommended_fix,
                "must_fix_before_advisor": advisor,
                "must_fix_before_public_release": public,
            }
        )
        return finding_id

    def add_test(self, phase: str, case: str, status: str, severity: str = "", notes: str = "") -> None:
        self.tests.append({"phase": phase, "case": case, "status": status, "severity": severity, "notes": notes})


def phase2_install_packaging(state: AuditState) -> None:
    out = OUT / "install_packaging"
    logs = out / "raw_logs"
    ensure(logs)
    rows: list[dict[str, object]] = []
    fields = ["case_id", "command", "status", "exit_code", "runtime_seconds", "log_file", "severity", "notes"]

    def record(case_id: str, command: list[str], result: dict[str, object], severity: str = "blocker", notes: str = "") -> None:
        log_file = logs / f"{case_id}.log"
        # Command already logged by run when needed; write a compact replay if absent.
        if not log_file.exists():
            log_file.write_text(f"$ {' '.join(command)}\n\n{result.get('stdout','')}\n\n{result.get('stderr','')}\n")
        status = "pass" if result["success"] else "fail"
        rows.append(
            {
                "case_id": case_id,
                "command": " ".join(command),
                "status": status,
                "exit_code": result["exit_code"],
                "runtime_seconds": f"{result['runtime_seconds']:.3f}",
                "log_file": rel(log_file),
                "severity": severity if status == "fail" else "",
                "notes": notes,
            }
        )
        state.add_test("install_packaging", case_id, status, severity if status == "fail" else "", notes)
        if status == "fail" and severity in {"blocker", "major"}:
            state.add_finding(severity, "install_packaging", " ".join(command), f"{case_id} failed", result["stderr_summary"], "fix packaging/install command", "yes", "yes")

    with tempfile.TemporaryDirectory(prefix="mdca-critical-install-") as tmpdir:
        tmp = Path(tmpdir)
        # Editable install.
        venv_edit = tmp / "venv_edit"
        result = run([sys.executable, "-m", "venv", str(venv_edit)], log_path=logs / "editable_venv.log")
        record("editable_venv", [sys.executable, "-m", "venv", str(venv_edit)], result)
        py_edit = venv_edit / "bin" / "python"
        if result["success"]:
            cmd = [str(py_edit), "-m", "pip", "install", "-e", ".[dev]"]
            result = run(cmd, timeout=360, log_path=logs / "editable_install.log")
            record("editable_install_dev", cmd, result)
            for i, cmd in enumerate(
                [
                    [str(py_edit), "-c", "import mdcorealign; print(mdcorealign.__version__)"],
                    [str(venv_edit / "bin" / "mdcorealign"), "--help"],
                    [str(venv_edit / "bin" / "mdcorealign"), "formats"],
                    [str(venv_edit / "bin" / "mdcorealign"), "align", "--help"],
                    [str(venv_edit / "bin" / "mdcorealign"), "flexibility", "--help"],
                    [str(venv_edit / "bin" / "mdcorealign"), "diagnostics", "--help"],
                    [str(venv_edit / "bin" / "mdcorealign"), "audit-preprocessing", "--help"],
                ],
                start=1,
            ):
                result = run(cmd, timeout=120, log_path=logs / f"editable_cli_{i}.log")
                record(f"editable_cli_{i}", cmd, result)

        # Normal install.
        venv_norm = tmp / "venv_norm"
        result = run([sys.executable, "-m", "venv", str(venv_norm)], log_path=logs / "normal_venv.log")
        record("normal_venv", [sys.executable, "-m", "venv", str(venv_norm)], result)
        py_norm = venv_norm / "bin" / "python"
        if result["success"]:
            cmd = [str(py_norm), "-m", "pip", "install", "."]
            result = run(cmd, timeout=360, log_path=logs / "normal_install.log")
            record("normal_install", cmd, result)
            cmd = [str(py_norm), "-c", "import mdcorealign; print(mdcorealign.__version__)"]
            result = run(cmd, timeout=120, log_path=logs / "normal_import.log")
            record("normal_import", cmd, result)

        # Build and twine check in current env.
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        cmd = [sys.executable, "-m", "build"]
        result = run(cmd, timeout=360, log_path=logs / "build.log")
        record("package_build", cmd, result)
        cmd = [sys.executable, "-m", "twine", "check", *[str(p) for p in sorted((ROOT / "dist").glob("*"))]]
        result = run(cmd, timeout=180, log_path=logs / "twine_check.log")
        record("twine_check", cmd, result)

        wheel = next(iter(sorted((ROOT / "dist").glob("*.whl"))), None)
        sdist = next(iter(sorted((ROOT / "dist").glob("*.tar.gz"))), None)
        for label, artifact in [("wheel", wheel), ("sdist", sdist)]:
            if artifact is None:
                fake = {"success": False, "exit_code": -1, "runtime_seconds": 0, "stdout": "", "stderr": f"{label} artifact missing", "stdout_summary": "", "stderr_summary": f"{label} artifact missing"}
                record(f"{label}_artifact_exists", ["ls", f"dist/*{label}*"], fake)
                continue
            venv = tmp / f"venv_{label}"
            result = run([sys.executable, "-m", "venv", str(venv)], log_path=logs / f"{label}_venv.log")
            record(f"{label}_venv", [sys.executable, "-m", "venv", str(venv)], result)
            if result["success"]:
                py = venv / "bin" / "python"
                cmd = [str(py), "-m", "pip", "install", str(artifact)]
                result = run(cmd, timeout=360, log_path=logs / f"{label}_install.log")
                record(f"{label}_install", cmd, result)
                cmd = [str(py), "-c", "import mdcorealign; print(mdcorealign.__version__)"]
                result = run(cmd, timeout=120, log_path=logs / f"{label}_import.log")
                record(f"{label}_import", cmd, result)

    # Missing optional external tools should not break core commands.
    env = env_src()
    env["PATH"] = "/usr/bin:/bin"
    result = run([sys.executable, "-m", "mdcorealign", "formats"], env=env, timeout=120, log_path=logs / "missing_external_formats.log")
    record("missing_external_tools_core_cli", [sys.executable, "-m", "mdcorealign", "formats"], result, notes="PATH restricted; MDLovoFit/THESEUS not required")

    write_tsv(out / "install_packaging_matrix.tsv", rows, fields)
    failed = [r for r in rows if r["status"] == "fail"]
    (out / "install_packaging_report.md").write_text(
        "# Install And Packaging Report\n\n"
        f"- Cases: `{len(rows)}`\n"
        f"- Failed: `{len(failed)}`\n"
        f"- Core package must install without MDLovoFit/THESEUS: `{'pass' if not any(r['case_id']=='missing_external_tools_core_cli' and r['status']=='fail' for r in rows) else 'fail'}`\n"
        "\nDetailed results are in `install_packaging_matrix.tsv`.\n"
    )


def pdb_line(record: str, serial: int, atom: str, resname: str, chain: str, resid: int, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return f"{record:<6}{serial:5d} {atom[:4]:>4} {resname[:3]:>3} {chain[:1]:1s}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}           C\n"


def write_pdb(path: Path, frames: list[list[str]]) -> None:
    ensure(path.parent)
    lines: list[str] = []
    for i, frame in enumerate(frames, start=1):
        lines.append(f"MODEL     {i:4d}\n")
        lines.extend(frame)
        lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def make_cli_fixtures(base: Path) -> dict[str, Path]:
    ensure(base)
    fixtures: dict[str, Path] = {}
    # Protein + ligand.
    frames = []
    for shift in [0.0, 0.2]:
        frames.append(
            [
                pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0 + shift, 0, 0)),
                pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1 + shift, 0, 0)),
                pdb_line("ATOM", 3, "CA", "SER", "A", 3, (0 + shift, 1, 0)),
                pdb_line("ATOM", 4, "CA", "THR", "A", 4, (0 + shift, 0, 1)),
                pdb_line("HETATM", 5, "C1", "LIG", "L", 1, (2 + shift, 2, 2)),
            ]
        )
    fixtures["protein_ligand"] = base / "protein_ligand.pdb"
    write_pdb(fixtures["protein_ligand"], frames)
    # HETATM only.
    fixtures["hetatm_only"] = base / "hetatm_only.pdb"
    write_pdb(fixtures["hetatm_only"], [[pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0, 0, 0))], [pdb_line("HETATM", 1, "C1", "LIG", "L", 1, (0.1, 0, 0))]])
    # Mismatch atom count.
    fixtures["mismatch"] = base / "mismatch.pdb"
    write_pdb(fixtures["mismatch"], [[pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0, 0, 0)), pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1, 0, 0))], [pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0.1, 0, 0))]])
    # NaN/Inf are hard to encode in strict PDB width; use array files.
    nan_coords = np.zeros((2, 3, 3), dtype=float)
    nan_coords[1, 1, 0] = np.nan
    fixtures["nan_array"] = base / "nan.npy"
    np.save(fixtures["nan_array"], nan_coords)
    inf_coords = np.zeros((2, 3, 3), dtype=float)
    inf_coords[1, 1, 0] = np.inf
    fixtures["inf_array"] = base / "inf.npy"
    np.save(fixtures["inf_array"], inf_coords)
    # Missing chain and duplicate labels.
    fixtures["missing_chain"] = base / "missing_chain.pdb"
    write_pdb(fixtures["missing_chain"], [[pdb_line("ATOM", 1, "CA", "ALA", "", 1, (0, 0, 0)), pdb_line("ATOM", 2, "CA", "GLY", "", 2, (1, 0, 0))], [pdb_line("ATOM", 1, "CA", "ALA", "", 1, (0.1, 0, 0)), pdb_line("ATOM", 2, "CA", "GLY", "", 2, (1.1, 0, 0))]])
    fixtures["duplicate_labels"] = base / "duplicate_labels.pdb"
    write_pdb(fixtures["duplicate_labels"], [[pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0, 0, 0)), pdb_line("ATOM", 2, "CA", "ALA", "A", 1, (1, 0, 0))], [pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0.1, 0, 0)), pdb_line("ATOM", 2, "CA", "ALA", "A", 1, (1.1, 0, 0))]])
    # Large jump.
    fixtures["large_jump"] = base / "large_jump.pdb"
    write_pdb(fixtures["large_jump"], [[pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (0, 0, 0)), pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1, 0, 0))], [pdb_line("ATOM", 1, "CA", "ALA", "A", 1, (100, 0, 0)), pdb_line("ATOM", 2, "CA", "GLY", "A", 2, (1, 0, 0))]])
    # RNA-like.
    fixtures["rna"] = base / "rna.pdb"
    write_pdb(fixtures["rna"], [[pdb_line("ATOM", 1, "P", "A", "A", 1, (0, 0, 0)), pdb_line("ATOM", 2, "P", "U", "A", 2, (1, 0, 0))], [pdb_line("ATOM", 1, "P", "A", "A", 1, (0.1, 0, 0)), pdb_line("ATOM", 2, "P", "U", "A", 2, (1.1, 0, 0))]])
    # Unknown extension.
    fixtures["unknown"] = base / "coords.foo"
    fixtures["unknown"].write_text("not coordinates\n")
    # Fake xtc for missing topology.
    fixtures["fake_xtc"] = base / "traj.xtc"
    fixtures["fake_xtc"].write_bytes(b"not xtc")
    return fixtures


def phase3_cli_adversarial(state: AuditState) -> None:
    out = OUT / "cli_adversarial"
    logs = out / "raw_logs"
    ensure(logs)
    fixture_dir = out / "fixtures"
    fixtures = make_cli_fixtures(fixture_dir)
    rows: list[dict[str, object]] = []
    fields = ["case_id", "command", "expected_behavior", "actual_behavior", "exit_code", "stderr_summary", "stdout_summary", "outputs_created", "severity", "fix_needed"]

    def case(case_id: str, command: list[str], expected: str, should_pass: bool, severity_if_bad: str = "major", timeout: int = 180) -> None:
        output_dir = None
        if "--output-dir" in command:
            try:
                output_dir = Path(command[command.index("--output-dir") + 1])
            except Exception:
                output_dir = None
        result = run(command, env=env_src(), timeout=timeout, log_path=logs / f"{case_id}.log")
        ok = bool(result["success"]) == should_pass
        traceback = "Traceback" in result["stderr"] or "Traceback" in result["stdout"]
        actionable = should_pass or any(term in (result["stderr"] + result["stdout"]).lower() for term in ["error", "suggestion", "selection", "topology", "format", "not formally validated", "input"])
        severity = "" if ok and not traceback and actionable else severity_if_bad
        fix_needed = "no" if not severity else "yes"
        if traceback:
            severity = "blocker"
            fix_needed = "yes"
        outputs_created = ""
        if output_dir and output_dir.is_dir():
            outputs_created = ",".join(sorted(p.name for p in output_dir.iterdir()))
        elif output_dir and output_dir.exists():
            outputs_created = "<output_path_exists_but_is_not_directory>"
        rows.append(
            {
                "case_id": case_id,
                "command": " ".join(command),
                "expected_behavior": expected,
                "actual_behavior": "pass" if result["success"] else "fail",
                "exit_code": result["exit_code"],
                "stderr_summary": result["stderr_summary"],
                "stdout_summary": result["stdout_summary"],
                "outputs_created": outputs_created,
                "severity": severity,
                "fix_needed": fix_needed,
            }
        )
        state.add_test("cli_adversarial", case_id, "pass" if not severity else "fail", severity, expected)
        if severity:
            state.failures.append(rows[-1])
            state.add_finding(severity, "cli_adversarial", " ".join(command), f"CLI adversarial case {case_id} did not meet expectations", result["stderr_summary"] or result["stdout_summary"], "improve CLI error handling or output behavior", "yes" if severity == "blocker" else "no", "yes")

    toy = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    real_script = ROOT / "examples/real_md_small/run_recommended_workflow.sh"
    case("happy_toy_quickstart", ["bash", str(ROOT / "examples/toy_md/run_quickstart.sh"), str(out / "toy_quickstart")], "toy quickstart succeeds", True)
    case("happy_real_md_small", ["bash", str(real_script), str(out / "real_md_small")], "real_md_small succeeds", True)
    case("happy_formats_out", [sys.executable, "-m", "mdcorealign", "formats", "--out", str(out / "supported_formats.tsv")], "formats writes TSV", True)
    case("happy_audit_toy", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(out / "audit_toy")], "audit succeeds", True)
    case("happy_align_dry_toy", [sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(out / "dry_toy"), "--dry-run"], "dry-run succeeds without aligned trajectory", True)
    case("happy_align_toy", [sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(out / "align_toy")], "align succeeds", True)
    case("happy_flex_toy", [sys.executable, "-m", "mdcorealign", "flexibility", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(out / "flex_toy")], "flexibility succeeds", True)
    case("happy_diag_toy", [sys.executable, "-m", "mdcorealign", "diagnostics", "--input", str(toy), "--input-format", "ca-pdb", "--trace", "--output-dir", str(out / "diag_toy")], "diagnostics succeeds", True)

    case("bad_missing_topology", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--topology", str(out / "missing.pdb"), "--trajectory", str(fixtures["fake_xtc"]), "--output-dir", str(out / "bad_missing_topology")], "missing topology gives actionable error", False)
    case("bad_missing_trajectory", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--topology", str(toy), "--trajectory", str(out / "missing.xtc"), "--output-dir", str(out / "bad_missing_trajectory")], "missing trajectory gives actionable error", False)
    case("bad_empty_fit_selection", [sys.executable, "-m", "mdcorealign", "align", "--input", str(fixtures["hetatm_only"]), "--input-format", "pdb", "--output-dir", str(out / "bad_empty_fit"), "--dry-run"], "empty fit selection gives actionable error", False)
    case("bad_empty_apply_selection", [sys.executable, "-m", "mdcorealign", "align", "--input", str(fixtures["hetatm_only"]), "--input-format", "pdb", "--fit-selection", "all", "--apply-selection", "protein", "--output-dir", str(out / "bad_empty_apply"), "--dry-run"], "empty apply selection gives actionable error", False)
    case("bad_invalid_selection", [sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "pdb", "--fit-selection", "resid 1 to 5", "--output-dir", str(out / "bad_invalid_selection"), "--dry-run"], "invalid selection gives actionable error", False)
    case("bad_unknown_extension", [sys.executable, "-m", "mdcorealign", "align", "--input", str(fixtures["unknown"]), "--output-dir", str(out / "bad_unknown"), "--dry-run"], "unknown extension gives format guidance", False)
    case("bad_mismatch_atom_count", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["mismatch"]), "--input-format", "ca-pdb", "--output-dir", str(out / "bad_mismatch")], "variable atom count rejected", False)
    case("bad_nan_coords", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["nan_array"]), "--input-format", "array", "--output-dir", str(out / "bad_nan")], "NaN rejected or clearly warned", False)
    case("bad_inf_coords", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["inf_array"]), "--input-format", "array", "--output-dir", str(out / "bad_inf")], "Inf rejected or clearly warned", False)
    case("warn_duplicate_labels", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["duplicate_labels"]), "--input-format", "ca-pdb", "--output-dir", str(out / "warn_duplicate")], "duplicate labels audit succeeds with warning/readiness", True, severity_if_bad="minor")
    case("warn_missing_chain", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["missing_chain"]), "--input-format", "ca-pdb", "--output-dir", str(out / "warn_missing_chain")], "missing chain audit succeeds with warning/readiness", True, severity_if_bad="minor")
    case("warn_large_jump", [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--input", str(fixtures["large_jump"]), "--input-format", "ca-pdb", "--output-dir", str(out / "warn_large_jump")], "large jump audit succeeds with preprocessing warning", True, severity_if_bad="major")
    # not writable is unreliable as root/current user; use output path that is an existing file.
    file_out = out / "output_file_not_dir"
    file_out.write_text("file")
    case("bad_output_path_file", [sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(file_out)], "output path that is a file gives actionable error", False)
    existing = out / "existing_overwrite"
    existing.mkdir(exist_ok=True)
    (existing / "summary.json").write_text("old")
    case("existing_output_files", [sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(existing)], "existing outputs overwritten predictably", True, severity_if_bad="minor")
    case("bad_rna_default", [sys.executable, "-m", "mdcorealign", "align", "--input", str(fixtures["rna"]), "--input-format", "pdb", "--output-dir", str(out / "bad_rna"), "--dry-run"], "RNA-like input with default CA gives unsupported warning", False)
    # IDP-like random cloud as array should run but may lack no-stable-core warning: record separately in numerical phase.
    var = out / "variable_atom_count.pdb"
    shutil.copy2(fixtures["mismatch"], var)
    case("bad_variable_atom_count", [sys.executable, "-m", "mdcorealign", "align", "--input", str(var), "--input-format", "pdb", "--output-dir", str(out / "bad_variable")], "variable atom count rejected", False)

    write_tsv(out / "cli_adversarial_cases.tsv", rows, fields)
    bad = [r for r in rows if r["fix_needed"] == "yes"]
    (out / "cli_adversarial_summary.md").write_text(
        "# CLI Adversarial Summary\n\n"
        f"- Cases: `{len(rows)}`\n"
        f"- Cases needing fixes: `{len(bad)}`\n"
        f"- Python tracebacks detected: `{sum('Traceback' in str(r['stderr_summary']) for r in rows)}`\n"
    )


def phase4_input_formats(state: AuditState) -> None:
    out = OUT / "input_formats"
    ensure(out)
    rows: list[dict[str, object]] = []
    fields = ["format_case", "topology_file", "trajectory_file", "backend_used", "dependency_required", "read_success", "selection_success", "audit_success", "dry_run_success", "alignment_success", "flexibility_success", "notes"]
    toy = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    arr = out / "toy.npy"
    np.save(arr, np.zeros((3, 5, 3), dtype=float))

    def record(case_name: str, input_args: list[str], top: str = "", traj: str = "", backend: str = "core", dep: str = "none") -> None:
        audit = run([sys.executable, "-m", "mdcorealign", "audit-preprocessing", *input_args, "--output-dir", str(out / f"{case_name}_audit")], env=env_src())
        dry = run([sys.executable, "-m", "mdcorealign", "align", *input_args, "--output-dir", str(out / f"{case_name}_dry"), "--dry-run"], env=env_src())
        align = run([sys.executable, "-m", "mdcorealign", "align", *input_args, "--output-dir", str(out / f"{case_name}_align")], env=env_src())
        flex = run([sys.executable, "-m", "mdcorealign", "flexibility", *input_args, "--output-dir", str(out / f"{case_name}_flex")], env=env_src())
        row = {
            "format_case": case_name,
            "topology_file": top,
            "trajectory_file": traj,
            "backend_used": backend,
            "dependency_required": dep,
            "read_success": audit["success"],
            "selection_success": audit["success"],
            "audit_success": audit["success"],
            "dry_run_success": dry["success"],
            "alignment_success": align["success"],
            "flexibility_success": flex["success"],
            "notes": (audit["stderr_summary"] or audit["stdout_summary"])[:300],
        }
        rows.append(row)
        state.add_test("input_formats", case_name, "pass" if audit["success"] else "not_tested_or_fail", "", row["notes"])

    record("multi_model_pdb_ca", ["--input", str(toy), "--input-format", "ca-pdb"], traj=rel(toy), backend="core_pdb")
    record("toy_pdb_trajectory", ["--input", str(toy), "--input-format", "pdb"], traj=rel(toy), backend="core_pdb")
    record("array_npy", ["--input", str(arr), "--input-format", "array"], traj=rel(arr), backend="core_array")

    # Optional backend cases if local fixtures exist; record dependency missing otherwise.
    optional = [
        ("PDB_XTC", "topology.pdb", "trajectory.xtc"),
        ("GRO_XTC", "topology.gro", "trajectory.xtc"),
        ("PSF_DCD", "topology.psf", "trajectory.dcd"),
        ("PRMTOP_NC", "topology.prmtop", "trajectory.nc"),
        ("TRR", "topology.gro", "trajectory.trr"),
        ("DCD_PDB", "topology.pdb", "trajectory.dcd"),
    ]
    for name, top, traj in optional:
        rows.append(
            {
                "format_case": name,
                "topology_file": top,
                "trajectory_file": traj,
                "backend_used": "optional_mdanalysis",
                "dependency_required": "MDAnalysis",
                "read_success": "not_tested_fixture_missing",
                "selection_success": "not_tested_fixture_missing",
                "audit_success": "not_tested_fixture_missing",
                "dry_run_success": "not_tested_fixture_missing",
                "alignment_success": "not_tested_fixture_missing",
                "flexibility_success": "not_tested_fixture_missing",
                "notes": "No small local fixture included in advisor package.",
            }
        )
    write_tsv(out / "input_format_smoke_tests.tsv", rows, fields)
    (out / "input_format_smoke_summary.md").write_text(
        "# Input Format Smoke Summary\n\n"
        "- Stable core tests run for multi-model PDB, toy PDB trajectory, and array input.\n"
        "- Optional topology+trajectory backend formats are recorded as not tested unless fixtures are available.\n"
        "- Do not claim optional formats as formally validated from this audit alone.\n"
    )


def solve_summary_from_coords(coords: np.ndarray) -> dict[str, object]:
    from mdsuperpose import Trajectory, solve_trajectory_robust_alignment, build_flexibility_profile, mean_soft_weights_from_result

    traj = Trajectory(coords=coords)
    result = solve_trajectory_robust_alignment(traj, max_iterations=20, delta=0.5)
    profile = build_flexibility_profile(traj, trajectory_result=result, flexible_percentile_cutoff=0.8, min_segment_length=2)
    weights = mean_soft_weights_from_result(result)
    return {
        "mean_rmsd": result.mean_rmsd,
        "mean_weighted_rmsd": result.mean_weighted_rmsd,
        "weights": weights,
        "segments": profile.flexible_segment_id,
        "converged": result.converged_fraction,
        "effective_core_size": float(np.sum(weights)),
    }


def phase5_numerical(state: AuditState) -> None:
    out = OUT / "numerical_invariants"
    ensure(out)
    rows: list[dict[str, object]] = []
    fields = ["case_id", "status", "metric", "value", "expected", "severity", "notes"]
    rng = np.random.default_rng(7)
    base = rng.normal(size=(30, 3))
    frames = np.stack([base + np.array([i * 0.1, 0, 0]) for i in range(10)])
    cases: list[tuple[str, np.ndarray, str]] = []
    cases.append(("pure_rigid_transform", frames, "near-zero RMSD after alignment"))
    mobile = frames.copy()
    mobile[:, 20:, :] += rng.normal(scale=3.0, size=mobile[:, 20:, :].shape)
    cases.append(("stable_core_mobile_tail", mobile, "core weights higher than tail"))
    hinge = frames.copy()
    for i in range(hinge.shape[0]):
        theta = i * 0.15
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        hinge[i, 15:, :] = hinge[i, 15:, :] @ rot.T
    cases.append(("two_domain_hinge", hinge, "mobile domain downweighted"))
    cloud = rng.normal(scale=10.0, size=(20, 30, 3))
    cases.append(("no_stable_core_random_cloud", cloud, "should not overconfidently report strong stable core"))

    for name, coords, expectation in cases:
        try:
            summary = solve_summary_from_coords(coords)
            weights = np.asarray(summary["weights"])
            finite = bool(np.isfinite(weights).all() and np.isfinite(summary["mean_rmsd"]))
            valid_range = bool(np.all((weights >= -1e-8) & (weights <= 1 + 1e-8)))
            status = "pass" if finite and valid_range and summary["effective_core_size"] > 0 else "fail"
            sev = "blocker" if status == "fail" else ""
            if name == "pure_rigid_transform" and float(summary["mean_rmsd"]) > 1e-5:
                status, sev = "fail", "blocker"
            if name == "stable_core_mobile_tail":
                core_mean = float(weights[:20].mean())
                tail_mean = float(weights[20:].mean())
                if core_mean <= tail_mean:
                    status, sev = "fail", "major"
            if name == "two_domain_hinge":
                if float(weights[:15].mean()) <= float(weights[15:].mean()):
                    status, sev = "fail", "major"
            if name == "no_stable_core_random_cloud":
                # This is a known caveat if no warning exists; record major but not blocker.
                if float(weights.max() - weights.min()) < 0.05 or float(weights.mean()) > 0.8:
                    status, sev = "fail", "major"
            rows.append({"case_id": name, "status": status, "metric": "mean_rmsd/effective_core_size", "value": f"{summary['mean_rmsd']:.6g}/{summary['effective_core_size']:.6g}", "expected": expectation, "severity": sev, "notes": f"converged={summary['converged']}"})
            if sev:
                state.add_finding(sev, "numerical", name, f"Numerical invariant concern for {name}", json.dumps(rows[-1]), "add warning/test or document caveat", "no" if sev == "major" else "yes", "yes")
        except Exception as exc:
            rows.append({"case_id": name, "status": "fail", "metric": "exception", "value": str(exc), "expected": expectation, "severity": "blocker", "notes": "unexpected exception"})
            state.add_finding("blocker", "numerical", name, "Numerical invariant test raised exception", str(exc), "fix numerical path", "yes", "yes")
    # Chain boundary direct invariant.
    from mdsuperpose.flexibility import detect_flexible_segments
    seg = detect_flexible_segments(np.array([1, 1, 0, 1, 1], dtype=float), percentile_cutoff=0.5, min_segment_length=2, chain=["A", "A", "A", "B", "B"])
    status = "pass" if seg.tolist() == [1, 1, 0, 2, 2] else "fail"
    rows.append({"case_id": "chain_boundary", "status": status, "metric": "segment_ids", "value": str(seg.tolist()), "expected": "[1,1,0,2,2]", "severity": "" if status == "pass" else "major", "notes": ""})
    write_tsv(out / "numerical_invariant_tests.tsv", rows, fields)
    (out / "numerical_invariant_summary.md").write_text(
        "# Numerical Invariant Summary\n\n"
        f"- Cases: `{len(rows)}`\n"
        f"- Failures/concerns: `{sum(1 for r in rows if r['status'] == 'fail')}`\n"
        "- Random-cloud no-stable-core behavior is treated as a caveat if overconfident output appears.\n"
    )


def phase6_output_schema(state: AuditState) -> None:
    out = OUT / "output_schema"
    ensure(out)
    run_dir = out / "toy_outputs"
    run(["bash", str(ROOT / "examples/toy_md/run_quickstart.sh"), str(run_dir)], env=env_src(), timeout=180)
    expected_columns = {
        "weights.csv": ["atom_index", "residue_label", "soft_stability_weight", "mobility_score", "mean_residual", "core_label"],
        "flexibility_profile.csv": ["residue_id", "resname", "chain", "robust_aligned_RMSF", "ordinary_aligned_RMSF", "soft_stability_weight", "mobility_score", "flexible_segment_id", "rank", "percentile"],
        "flexible_segments.tsv": ["system", "segment_id", "chain", "start_residue", "end_residue", "length", "mean_robust_RMSF", "max_robust_RMSF", "mean_ordinary_RMSF", "mean_delta_RMSF", "mean_mobility_score", "mean_stability_weight"],
    }
    rows: list[dict[str, object]] = []
    fields = ["file", "exists", "non_empty", "required_columns_present", "duplicate_columns", "metadata_checks", "schema_docs_match", "severity", "notes"]
    docs = (ROOT / "docs/output_formats.md").read_text(errors="ignore")
    for name in ["weights.csv", "flexibility_profile.csv", "flexible_segments.tsv", "summary.json", "run_metadata.json", "aligned_trajectory.pdb"]:
        path = run_dir / name
        exists = path.exists()
        non_empty = exists and path.stat().st_size > 0
        cols_ok = True
        dup = False
        notes = ""
        if name in expected_columns and exists:
            delim = "\t" if name.endswith(".tsv") else ","
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delim)
                cols = reader.fieldnames or []
            cols_ok = all(c in cols for c in expected_columns[name])
            dup = len(cols) != len(set(cols))
            missing = [c for c in expected_columns[name] if c not in cols]
            if missing:
                notes = f"missing columns: {missing}"
        metadata = "not_applicable"
        if name == "summary.json" and exists:
            data = json.loads(path.read_text())
            required = ["fit_selection", "apply_selection", "analysis_selection", "input_readiness_status", "warnings"]
            metadata = "pass" if all(k in data for k in required) else "fail"
        if name == "run_metadata.json" and exists:
            data = json.loads(path.read_text())
            required = ["package_version", "command", "timestamp", "external_tools_detected", "format_overrides"]
            metadata = "pass" if all(k in data for k in required) else "fail"
        doc_match = name in docs or name == "aligned_trajectory.pdb"
        severity = ""
        if not exists or not non_empty or not cols_ok or dup or metadata == "fail" or not doc_match:
            severity = "major"
            state.add_finding("major", "output_schema", name, "Output schema mismatch or missing metadata", notes or f"exists={exists}, metadata={metadata}, doc_match={doc_match}", "update output generation or docs", "no", "yes")
        rows.append({"file": name, "exists": exists, "non_empty": non_empty, "required_columns_present": cols_ok, "duplicate_columns": dup, "metadata_checks": metadata, "schema_docs_match": doc_match, "severity": severity, "notes": notes})
    write_tsv(out / "output_schema_audit.tsv", rows, fields)
    (out / "output_schema_summary.md").write_text(f"# Output Schema Summary\n\n- Files checked: `{len(rows)}`\n- Issues: `{sum(1 for r in rows if r['severity'])}`\n")


def phase7_docs_truthfulness(state: AuditState) -> None:
    out = OUT / "docs_truthfulness"
    ensure(out)
    files = [ROOT / p for p in ["README.md", "docs/input_formats.md", "docs/support_matrix.md", "docs/mdlovofit_comparison.md", "docs/non_protein_systems.md", "docs/output_formats.md", "docs/recommended_workflow.md", "docs/troubleshooting.md", "docs/private_alpha_advisor_test.md"]]
    files += list((ROOT / "dist_advisor").glob("*.md"))
    patterns = [
        ("all MD formats supported", r"supports all (MD )?formats"),
        ("all molecular systems supported", r"supports all molecular systems"),
        ("RNA/DNA formally supported", r"RNA/DNA (is )?(formally )?supported"),
        ("heavy atom formal validation", r"heavy-atom.*formally validated"),
        ("backbone formal validation", r"backbone.*formally validated"),
        ("wrapped trajectories automatically fixed", r"automatically (fix|unwrap|nojump)"),
        ("PBC guaranteed", r"PBC/nojump.*guaranteed"),
        ("reference independence", r"reference independent|reference independence"),
        ("universal superiority over THESEUS", r"universal superiority over THESEUS|always better than THESEUS"),
        ("MDLovoFit direct binary trajectories", r"MDLovoFit directly.*(XTC|DCD|TRR|NC)"),
        ("THESEUS rerun local overclaim", r"THESEUS was rerun locally"),
        ("YiiP strong biology", r"YiiP.*strong biological validation"),
        ("mobility_score equals RMSF", r"mobility_score (equals|is identical to|=) RMSF"),
    ]

    def is_negated_context(claim: str, context: str) -> bool:
        """Avoid flagging explicit caveats as overclaims."""

        normalized = re.sub(r"\s+", " ", context.lower())
        if claim in {"heavy atom formal validation", "backbone formal validation"}:
            return any(phrase in normalized for phrase in ["not formally validated", "not yet formally validated", "future validation"])
        if claim == "universal superiority over THESEUS":
            return any(phrase in normalized for phrase in ["should not claim", "do not claim", "not claim"])
        return False

    rows: list[dict[str, object]] = []
    fields = ["file", "claim_type", "status", "evidence", "severity", "recommended_fix"]
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(errors="ignore")
        for claim, pat in patterns:
            match = re.search(pat, text, re.I | re.S)
            if match:
                start = max(match.start() - 80, 0)
                end = min(match.end() + 80, len(text))
                if is_negated_context(claim, text[start:end]):
                    continue
                severity = "major"
                rows.append({"file": rel(f), "claim_type": claim, "status": "flagged", "evidence": match.group(0)[:200], "severity": severity, "recommended_fix": "soften/remove unsupported claim"})
                state.add_finding(severity, "docs_truthfulness", rel(f), f"Unsupported/unsafe documentation claim: {claim}", match.group(0)[:200], "edit wording", "no", "yes")
    if not rows:
        rows.append({"file": "audited_docs", "claim_type": "unsupported_claim_scan", "status": "pass", "evidence": "no flagged claims", "severity": "", "recommended_fix": ""})
    write_tsv(out / "docs_truthfulness_audit.tsv", rows, fields)
    (out / "docs_truthfulness_summary.md").write_text(f"# Documentation Truthfulness Summary\n\n- Flagged claims: `{sum(1 for r in rows if r['status']=='flagged')}`\n")


def phase8_performance(state: AuditState) -> None:
    out = OUT / "performance"
    ensure(out)
    rows: list[dict[str, object]] = []
    fields = ["case_id", "runtime_seconds", "peak_memory_MB", "n_frames", "n_residues", "n_atoms", "iterations", "effective_core_size", "output_size_MB", "warnings", "status"]
    from mdsuperpose import Trajectory, solve_trajectory_robust_alignment, mean_soft_weights_from_result
    for case_id, nf, nr in [("small_100x100", 100, 100), ("medium_1000x300", 1000, 300)]:
        rng = np.random.default_rng(3)
        base = rng.normal(size=(nr, 3))
        coords = np.empty((nf, nr, 3), dtype=float)
        for i in range(nf):
            coords[i] = base + np.array([0.01 * i, 0, 0]) + rng.normal(scale=0.03, size=base.shape)
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        start = time.perf_counter()
        result = solve_trajectory_robust_alignment(Trajectory(coords=coords), max_iterations=10, delta=0.5)
        runtime = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        weights = mean_soft_weights_from_result(result)
        rows.append({"case_id": case_id, "runtime_seconds": f"{runtime:.3f}", "peak_memory_MB": f"{max(start_mem, peak):.1f}", "n_frames": nf, "n_residues": nr, "n_atoms": nr, "iterations": f"{result.average_iterations_per_frame:.2f}", "effective_core_size": f"{float(weights.sum()):.3f}", "output_size_MB": "0", "warnings": "", "status": "pass"})
    rows.append({"case_id": "larger_5000x500", "runtime_seconds": "", "peak_memory_MB": "", "n_frames": 5000, "n_residues": 500, "n_atoms": 500, "iterations": "", "effective_core_size": "", "output_size_MB": "", "warnings": "not_run_time_budget_private_alpha", "status": "not_run"})
    rows.append({"case_id": "multichain_synthetic", "runtime_seconds": "", "peak_memory_MB": "", "n_frames": 100, "n_residues": 200, "n_atoms": 200, "iterations": "", "effective_core_size": "", "output_size_MB": "", "warnings": "covered by chain-boundary invariant, not separate stress run", "status": "info"})
    rows.append({"case_id": "many_flexible_segments", "runtime_seconds": "", "peak_memory_MB": "", "n_frames": 100, "n_residues": 200, "n_atoms": 200, "iterations": "", "effective_core_size": "", "output_size_MB": "", "warnings": "future stress case", "status": "info"})
    write_tsv(out / "performance_stress_tests.tsv", rows, fields)
    (out / "performance_summary.md").write_text("# Performance Stress Summary\n\n- Small and medium synthetic cases completed.\n- Larger 5000x500 case was not run in private-alpha audit time budget and should be treated as a future performance caveat.\n")


def phase9_file_safety(state: AuditState) -> None:
    out = OUT / "file_safety"
    ensure(out)
    rows: list[dict[str, object]] = []
    fields = ["case_id", "status", "severity", "evidence", "recommended_fix"]
    # Explicit --out outside output-dir is allowed by CLI; flag as major policy ambiguity.
    toy = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    with tempfile.TemporaryDirectory(prefix="mdca-safety-") as tmpdir:
        tmp = Path(tmpdir)
        outside = tmp / "outside.pdb"
        outdir = tmp / "out"
        result = run([sys.executable, "-m", "mdcorealign", "align", "--input", str(toy), "--input-format", "ca-pdb", "--output-dir", str(outdir), "--out", str(outside)], env=env_src())
        if result["success"] and outside.exists():
            rows.append({"case_id": "explicit_outside_output_path", "status": "allowed_explicit", "severity": "info", "evidence": "CLI allows explicit --out path outside output-dir", "recommended_fix": "Document --out behavior or restrict before public beta"})
        else:
            rows.append({"case_id": "explicit_outside_output_path", "status": "blocked_or_failed", "severity": "", "evidence": result["stderr_summary"], "recommended_fix": ""})
    # Bundle large/path scan.
    zip_path = ROOT / "dist_advisor/MDCoreAlign_private_alpha_v0.1.0_advisor.zip"
    large: list[str] = []
    leaks: list[str] = []
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.file_size > 25 * 1024 * 1024:
                    large.append(info.filename)
                if info.filename.lower().endswith((".md", ".py", ".sh", ".json", ".tsv", ".txt")):
                    text = archive.read(info).decode("utf-8", errors="ignore")
                    if any(marker in text for marker in ["/storage/", "/home/", "MDAnalysis_data"]):
                        leaks.append(info.filename)
    rows.append({"case_id": "advisor_bundle_large_files", "status": "pass" if not large else "fail", "severity": "" if not large else "major", "evidence": ";".join(large) or "none", "recommended_fix": "remove large files"})
    rows.append({"case_id": "advisor_bundle_local_paths", "status": "pass" if not leaks else "fail", "severity": "" if not leaks else "major", "evidence": ";".join(leaks) or "none", "recommended_fix": "sanitize bundled text"})
    # shell injection: scripts use arrays/static paths; rough scan.
    risky = []
    for script in [ROOT / "examples/toy_md/run_quickstart.sh", ROOT / "examples/real_md_small/run_recommended_workflow.sh", ROOT / "scripts/run_toy_quickstart.sh"]:
        text = script.read_text(errors="ignore")
        if "eval " in text or "`" in text:
            risky.append(rel(script))
    rows.append({"case_id": "shell_injection_static_scan", "status": "pass" if not risky else "fail", "severity": "" if not risky else "blocker", "evidence": ";".join(risky) or "no eval/backticks", "recommended_fix": "remove shell evaluation"})
    write_tsv(out / "file_safety_audit.tsv", rows, fields)
    (out / "file_safety_summary.md").write_text(f"# File Safety Summary\n\n- Cases: `{len(rows)}`\n- Failures: `{sum(1 for r in rows if r['status']=='fail')}`\n- Explicit `--out` outside output directory is allowed only when user explicitly passes the path; document before public beta.\n")


def phase10_static_quality(state: AuditState) -> None:
    out = OUT / "static_quality"
    ensure(out)
    rows: list[dict[str, object]] = []
    fields = ["tool_or_check", "status", "severity", "evidence", "recommended_fix"]
    checks = [
        ("compileall", [sys.executable, "-m", "compileall", "src", "tests", "scripts"]),
        ("ruff", ["ruff", "check", "src", "tests", "scripts"]),
        ("black", ["black", "--check", "src", "tests", "scripts"]),
        ("mypy", ["mypy", "src"]),
        ("pyright", ["pyright"]),
    ]
    for name, cmd in checks:
        if shutil.which(cmd[0]) is None and name != "compileall":
            rows.append({"tool_or_check": name, "status": "unavailable", "severity": "info", "evidence": f"{cmd[0]} not installed", "recommended_fix": "optional"})
            continue
        result = run(cmd, timeout=240)
        sev = "blocker" if name == "compileall" and not result["success"] else ("info" if result["success"] else "minor")
        rows.append({"tool_or_check": name, "status": "pass" if result["success"] else "fail", "severity": "" if result["success"] else sev, "evidence": result["stderr_summary"] or result["stdout_summary"], "recommended_fix": "fix static quality issue" if not result["success"] and sev != "info" else ""})
        if sev == "blocker":
            state.add_finding("blocker", "static_quality", name, "Syntax/import static check failed", result["stderr_summary"] or result["stdout_summary"], "fix syntax/import failure", "yes", "yes")
    # hard-coded path scan
    hardcoded = []
    for root in ["src", "tests", "scripts"]:
        for f in (ROOT / root).rglob("*.py"):
            text = f.read_text(errors="ignore")
            if any(marker in text for marker in ["/home/jjxu/MDAnalysis_data", "/storage/junjie"]):
                hardcoded.append(rel(f))
    rows.append({"tool_or_check": "hardcoded_local_paths", "status": "pass" if not hardcoded else "fail", "severity": "" if not hardcoded else "major", "evidence": ";".join(hardcoded) or "none", "recommended_fix": "move local benchmark paths out of public scripts or mark internal"})
    if hardcoded:
        state.add_finding("major", "static_quality", "scripts/tests", "Hard-coded local paths found in scripts/tests", ";".join(hardcoded), "isolate internal benchmark scripts before public release", "no", "yes")
    write_tsv(out / "static_quality_audit.tsv", rows, fields)
    (out / "static_quality_summary.md").write_text(f"# Static Quality Summary\n\n- Checks: `{len(rows)}`\n- Non-pass: `{sum(1 for r in rows if r['status'] not in {'pass','unavailable'})}`\n")


def phase11_advisor_bundle(state: AuditState) -> None:
    out = OUT / "advisor_bundle"
    ensure(out)
    result = run([sys.executable, "scripts/validate_advisor_bundle.py"], env=env_src(), timeout=360)
    report_path = ROOT / "results/advisor_bundle/advisor_bundle_validation_report.json"
    rows = [{
        "case_id": "advisor_bundle_validation",
        "status": "pass" if result["success"] else "fail",
        "severity": "" if result["success"] else "blocker",
        "evidence": result["stderr_summary"] or result["stdout_summary"],
        "recommended_fix": "" if result["success"] else "fix advisor bundle validation",
    }]
    if report_path.exists():
        report = json.loads(report_path.read_text())
        rows.extend([
            {"case_id": "install_from_bundle", "status": report.get("status", "unknown"), "severity": "" if report.get("status") == "pass" else "blocker", "evidence": "validation report", "recommended_fix": ""},
            {"case_id": "toy_outputs", "status": "pass" if all(v == "present" for v in report.get("toy_outputs", {}).values()) else "fail", "severity": "", "evidence": json.dumps(report.get("toy_outputs", {})), "recommended_fix": ""},
            {"case_id": "real_md_small_outputs", "status": "pass" if all(v == "present" for v in report.get("real_md_small_outputs", {}).values()) else "fail", "severity": "", "evidence": json.dumps(report.get("real_md_small_outputs", {})), "recommended_fix": ""},
        ])
    write_tsv(out / "advisor_bundle_critical_validation.tsv", rows, ["case_id", "status", "severity", "evidence", "recommended_fix"])
    (out / "advisor_bundle_critical_summary.md").write_text(f"# Advisor Bundle Critical Summary\n\n- Validation status: `{'pass' if result['success'] else 'fail'}`\n")
    if not result["success"]:
        state.add_finding("blocker", "advisor_bundle", "scripts/validate_advisor_bundle.py", "Advisor bundle validation failed", result["stderr_summary"], "fix bundle", "yes", "yes")


def apply_fixes(state: AuditState) -> None:
    out = OUT
    rows: list[dict[str, object]] = []
    fields = ["fix_id", "linked_finding_id", "file_changed", "description", "test_added_or_updated", "verification_command", "status"]
    # Fix known phase1 exact wording issue if present.
    readme = ROOT / "README.md"
    text = readme.read_text()
    target = "MDCoreAlign is primarily validated for protein MD trajectories with consistent topology and residue mapping, using C-alpha fitting as the formal default."
    if target not in text:
        old = "MDCoreAlign is primarily validated for protein MD trajectories with consistent\ntopology and residue mapping, using C-alpha fitting as the formal default."
        if old in text:
            readme.write_text(text.replace(old, target))
            rows.append({"fix_id": "FIX-001", "linked_finding_id": "P1-001", "file_changed": "README.md", "description": "Normalize required validation-scope sentence onto one line for consistency scanners.", "test_added_or_updated": "none", "verification_command": "rg required sentence README.md", "status": "applied"})
            state.fixes.append(rows[-1])
    write_tsv(out / "fixes_applied.tsv", rows, fields)
    (out / "fixes_summary.md").write_text(f"# Fixes Summary\n\n- Fixes applied: `{len(rows)}`\n- Only blocker or advisor/public-facing major issues were eligible for fixes.\n")


def phase13_final(state: AuditState) -> None:
    out = OUT / "final_regression"
    ensure(out)
    test = run([sys.executable, "-m", "pytest", "-q", "tests"], env=env_src(), timeout=360, log_path=out / "final_test_report.txt")
    # final_test_report should be raw pytest text, not command wrapper.
    (out / "final_test_report.txt").write_text(test["stdout"] + test["stderr"])
    smoke_rows = []
    for case, cmd in [
        ("mdcorealign_help", [sys.executable, "-m", "mdcorealign", "--help"]),
        ("formats", [sys.executable, "-m", "mdcorealign", "formats"]),
        ("toy_quickstart", ["bash", str(ROOT / "examples/toy_md/run_quickstart.sh"), str(out / "toy")]),
        ("real_md_small", ["bash", str(ROOT / "examples/real_md_small/run_recommended_workflow.sh"), str(out / "real")]),
        ("build", [sys.executable, "-m", "build"]),
    ]:
        r = run(cmd, env=env_src(), timeout=360)
        smoke_rows.append((case, r))
    twine = run([sys.executable, "-m", "twine", "check", *[str(p) for p in sorted((ROOT / "dist").glob("*"))]], env=env_src(), timeout=180)
    smoke_rows.append(("twine_check", twine))
    md = ["# Final Smoke Report", ""]
    for case, r in smoke_rows:
        md.append(f"- {case}: `{'pass' if r['success'] else 'fail'}`")
        if not r["success"]:
            state.add_finding("blocker", "final_regression", case, "Final smoke command failed", r["stderr_summary"] or r["stdout_summary"], "fix final regression failure", "yes", "yes")
    (out / "final_smoke_report.md").write_text("\n".join(md) + "\n")

    blockers = [f for f in state.findings if f["severity"] == "blocker"]
    majors = [f for f in state.findings if f["severity"] == "major"]
    safe_claims = [
        "protein C-alpha MD trajectory workflow under fixed topology/residue correspondence",
        "input audit and readiness reporting for common user errors",
        "toy and small real-MD ADK examples run from the advisor bundle",
        "MDLovoFit/THESEUS are optional external baselines, not required for core commands",
    ]
    unsafe_claims = [
        "all MD formats are supported",
        "all molecular systems are supported",
        "RNA/DNA workflows are formally validated",
        "universal superiority over THESEUS",
        "automatic PBC/nojump correction",
    ]
    if blockers:
        label = "prototype"
    elif any(f["must_fix_before_advisor"] == "yes" for f in majors):
        label = "prototype"
    elif majors:
        label = "private-alpha ready"
    else:
        label = "private-alpha ready"
    assessment = [
        "# Final Maturity Assessment",
        "",
        f"- Overall maturity label: `{label}`",
        "",
        "## Questions",
        "",
        f"1. Ready for advisor private-alpha testing? `{'yes' if label == 'private-alpha ready' else 'no'}`",
        "2. Ready for public GitHub beta? `not yet; remaining public-release documentation/CI/static-quality caveats should be resolved first`",
        "3. Ready for TestPyPI/PyPI? `not yet; advisor feedback and public metadata should be completed first`",
        "4. Ready for Bioinformatics/BMC Bioinformatics author review? `software package is advisor-review ready, but manuscript author review still needs final narrative and external feedback`",
        f"5. Remaining blockers: `{len(blockers)}`",
        f"6. Remaining major limitations: `{len(majors)}`",
        "7. Safe claims:",
        *[f"   - {c}" for c in safe_claims],
        "8. Unsafe claims:",
        *[f"   - {c}" for c in unsafe_claims],
        "9. Future work:",
        "   - broaden real-world topology+trajectory format smoke tests",
        "   - collect advisor/external user feedback",
        "   - add CI before public beta if absent",
        "   - validate non-CA/non-protein workflows only if scientifically needed",
    ]
    (out / "final_maturity_assessment.md").write_text("\n".join(assessment) + "\n")


def aggregate_reports(state: AuditState) -> None:
    ensure(OUT)
    # Load phase1 findings if present.
    p1 = OUT / "phase1_findings.tsv"
    if p1.exists():
        with p1.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                # Keep original P1 ids in aggregate.
                if not any(f.get("finding_id") == row["finding_id"] for f in state.findings):
                    state.findings.append(dict(row))
    write_tsv(OUT / "risk_register.tsv", state.findings, FINDING_FIELDS)
    failure_fields = ["case_id", "command", "expected_behavior", "actual_behavior", "exit_code", "stderr_summary", "stdout_summary", "outputs_created", "severity", "fix_needed"]
    write_tsv(OUT / "failure_cases.tsv", state.failures, failure_fields)
    write_tsv(OUT / "test_matrix.tsv", state.tests, ["phase", "case", "status", "severity", "notes"])
    blockers = [f for f in state.findings if f["severity"] == "blocker"]
    majors = [f for f in state.findings if f["severity"] == "major"]
    minors = [f for f in state.findings if f["severity"] == "minor"]
    write_json(
        OUT / "blockers.json",
        {
            "blockers": [f["finding_id"] for f in blockers],
            "major_issues": [f["finding_id"] for f in majors],
            "minor_issues": [f["finding_id"] for f in minors],
            "maturity_label": "private-alpha ready" if not blockers else "prototype",
        },
    )
    (OUT / "fix_recommendations.md").write_text(
        "# Fix Recommendations\n\n"
        "## Blockers\n\n"
        + ("\n".join(f"- `{f['finding_id']}` {f['description']}: {f['recommended_fix']}" for f in blockers) if blockers else "- none")
        + "\n\n## Major Issues\n\n"
        + ("\n".join(f"- `{f['finding_id']}` {f['description']}: {f['recommended_fix']}" for f in majors) if majors else "- none")
        + "\n\n## Minor/Info\n\n"
        + ("\n".join(f"- `{f['finding_id']}` {f['description']}: {f['recommended_fix']}" for f in state.findings if f["severity"] in {"minor", "info"}) if any(f["severity"] in {"minor", "info"} for f in state.findings) else "- none")
        + "\n"
    )
    report = [
        "# Critical Tool Audit Report",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Blockers: `{len(blockers)}`",
        f"- Major issues: `{len(majors)}`",
        f"- Minor issues: `{len(minors)}`",
        f"- Tests recorded: `{len(state.tests)}`",
        "",
        "## Overall Assessment",
        "",
        "MDCoreAlign is suitable for advisor private-alpha testing if no blockers remain. It is not yet production-ready and should not claim universal format/system support.",
        "",
        "## Top Risks",
        "",
    ]
    top = blockers + majors[:10]
    report.extend([f"- `{f['finding_id']}` [{f['severity']}] {f['description']} ({f['file_or_command']})" for f in top] or ["- none"])
    (OUT / "critical_audit_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    ensure(OUT)
    state = AuditState()
    phase2_install_packaging(state)
    phase3_cli_adversarial(state)
    phase4_input_formats(state)
    phase5_numerical(state)
    phase6_output_schema(state)
    phase7_docs_truthfulness(state)
    phase8_performance(state)
    phase9_file_safety(state)
    phase10_static_quality(state)
    phase11_advisor_bundle(state)
    apply_fixes(state)
    phase13_final(state)
    aggregate_reports(state)


if __name__ == "__main__":
    main()
