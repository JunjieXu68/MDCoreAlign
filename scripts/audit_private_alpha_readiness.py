"""Audit private-alpha readiness for MDCoreAlign.

This script checks user-facing packaging and documentation signals only. It
does not rerun official scientific benchmarks and does not change algorithms.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "private_alpha_hardening"
FIELDS = ["check", "status", "severity", "details", "recommended_action"]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(command: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=_env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive audit path
        return False, str(exc)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output[-1200:]


def _row(check: str, status: str, severity: str, details: str, action: str = "") -> dict[str, str]:
    return {
        "check": check,
        "status": status,
        "severity": severity,
        "details": details,
        "recommended_action": action,
    }


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _has_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _public_text_files() -> list[Path]:
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "examples"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".sh"}:
                if "/output/" not in str(path):
                    files.append(path)
    return files


def _local_path_leaks() -> list[str]:
    patterns = ["/storage/", "/home/", "MDAnalysis_data"]
    leaks: list[str] = []
    for path in _public_text_files():
        text = path.read_text(errors="ignore")
        for pattern in patterns:
            if pattern in text:
                leaks.append(f"{path.relative_to(ROOT)} contains {pattern}")
    return leaks


def _large_example_files() -> list[str]:
    large: list[str] = []
    for path in (ROOT / "examples").rglob("*"):
        if path.is_file() and "/output/" not in str(path):
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 5.0:
                large.append(f"{path.relative_to(ROOT)} ({size_mb:.1f} MB)")
    return large


def _test_status_from_report(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "warning", "test report not generated yet"
    text = path.read_text(errors="ignore")
    if re.search(r"\bfailed\b|ERROR|Traceback", text, flags=re.IGNORECASE):
        return "fail", "test report contains failure/error markers"
    match = re.search(r"(\d+) passed", text)
    if match:
        return "pass", f"{match.group(1)} tests passed"
    return "warning", "could not parse pytest pass count"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    ok, output = _run([sys.executable, "-c", "import mdcorealign; print(mdcorealign.__version__)"])
    rows.append(_row("import mdcorealign", "pass" if ok else "fail", "blocker", output, "fix package import"))

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "site"
        ok_install, install_output = _run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target), str(ROOT)],
            timeout=180,
        )
        if ok_install:
            env = _env()
            env["PYTHONPATH"] = str(target)
            completed = subprocess.run(
                [sys.executable, "-c", "import mdcorealign; print(mdcorealign.__version__)"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok_install = completed.returncode == 0
            install_output = (completed.stdout + completed.stderr).strip()
        rows.append(
            _row(
                "package installs with pip --target",
                "pass" if ok_install else "fail",
                "blocker",
                install_output,
                "fix pyproject/package metadata",
            )
        )

    help_commands = [
        [sys.executable, "-m", "mdcorealign", "--help"],
        [sys.executable, "-m", "mdcorealign", "align", "--help"],
        [sys.executable, "-m", "mdcorealign", "flexibility", "--help"],
        [sys.executable, "-m", "mdcorealign", "diagnostics", "--help"],
        [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--help"],
    ]
    for command in help_commands:
        ok, output = _run(command)
        rows.append(_row(" ".join(command[-2:]) if len(command) > 3 else "mdcorealign --help", "pass" if ok else "fail", "blocker", output))

    required_docs = [
        "README.md",
        "docs/support_matrix.md",
        "docs/recommended_workflow.md",
        "docs/troubleshooting.md",
        "docs/output_formats.md",
        "docs/private_alpha_advisor_test.md",
    ]
    for relpath in required_docs:
        path = ROOT / relpath
        rows.append(
            _row(
                f"required doc {relpath}",
                "pass" if _has_nonempty(path) else "fail",
                "blocker",
                "present" if _has_nonempty(path) else "missing or empty",
                "add missing user-facing documentation",
            )
        )

    docs_text = "\n".join((ROOT / relpath).read_text(errors="ignore") for relpath in required_docs if (ROOT / relpath).exists())
    fit_apply_ok = "--fit-selection" in docs_text and "--apply-selection" in docs_text
    rows.append(
        _row(
            "fit/apply selection documented",
            "pass" if fit_apply_ok else "fail",
            "blocker",
            "fit/apply selection terms found" if fit_apply_ok else "missing fit/apply selection documentation",
            "document separate fit and apply selection semantics",
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "dry_run"
        ok, output = _run(
            [
                sys.executable,
                "-m",
                "mdcorealign",
                "align",
                "--input",
                str(ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"),
                "--input-format",
                "ca-pdb",
                "--output-dir",
                str(out),
                "--dry-run",
            ]
        )
        expected = [out / "summary.json", out / "input_audit.tsv", out / "input_audit.json", out / "chain_mapping_report.tsv", out / "run_metadata.json"]
        rows.append(
            _row(
                "align dry-run writes audit outputs",
                "pass" if ok and all(_has_nonempty(path) for path in expected) else "fail",
                "blocker",
                output if not ok else "dry-run audit outputs present",
                "fix dry-run/audit output generation",
            )
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ok, output = _run(["bash", str(ROOT / "examples" / "toy_md" / "run_quickstart.sh"), str(Path(tmpdir) / "toy")])
        rows.append(_row("toy quickstart", "pass" if ok else "fail", "blocker", output, "fix bundled toy workflow"))

    real_input = ROOT / "examples" / "real_md_small" / "trajectory_small.pdb"
    if real_input.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, output = _run(["bash", str(ROOT / "examples" / "real_md_small" / "run_recommended_workflow.sh"), str(Path(tmpdir) / "real")], timeout=180)
        rows.append(_row("real_md_small workflow", "pass" if ok else "fail", "warning", output, "fix or document small real-MD example"))
    else:
        rows.append(_row("real_md_small workflow", "warning", "warning", "trajectory_small.pdb not bundled", "document how to generate the example"))

    leaks = _local_path_leaks()
    rows.append(
        _row(
            "no local absolute paths in public docs/examples",
            "pass" if not leaks else "fail",
            "blocker",
            "; ".join(leaks) if leaks else "no local path leaks detected",
            "replace local paths with portable placeholders",
        )
    )

    large = _large_example_files()
    rows.append(
        _row(
            "no large example trajectories",
            "pass" if not large else "warning",
            "warning",
            "; ".join(large) if large else "no example file larger than 5 MB",
            "remove large files before public GitHub release",
        )
    )

    bundle_manifest = OUT / "private_alpha_bundle" / "private_alpha_bundle_manifest.tsv"
    rows.append(
        _row(
            "private alpha bundle manifest",
            "pass" if _has_nonempty(bundle_manifest) else "warning",
            "warning",
            "present" if _has_nonempty(bundle_manifest) else "run scripts/package_private_alpha_bundle.py",
        )
    )

    test_status, test_details = _test_status_from_report(OUT / "test_report.txt")
    rows.append(_row("full pytest report", test_status, "blocker" if test_status == "fail" else "warning", test_details, "run python -m pytest -q tests"))

    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    _write_tsv(OUT / "private_alpha_readiness_checklist.tsv", rows)
    blockers = [row for row in rows if row["status"] == "fail" and row["severity"] == "blocker"]
    warnings = [row for row in rows if row["status"] != "pass" and row["severity"] != "blocker"]
    if blockers:
        recommendation = "C: not ready for private alpha until blockers are fixed"
    elif warnings:
        recommendation = "B: ready for limited private alpha with documented warnings"
    else:
        recommendation = "A: ready for private alpha testing"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blockers": [row["check"] for row in blockers],
        "warnings": [row["check"] for row in warnings],
        "recommendation": recommendation,
    }
    (OUT / "blockers.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = [
        "# Private Alpha Readiness Summary",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Recommendation: `{recommendation}`",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Notes",
        "",
        "- This audit does not rerun official scientific benchmark systems.",
        "- Missing MDLovoFit/THESEUS is not a blocker for core commands.",
        "- The small real-MD example is for usability only, not a benchmark claim.",
    ]
    if blockers:
        summary.extend(["", "## Blockers", ""])
        summary.extend(f"- {row['check']}: {row['details']}" for row in blockers)
    if warnings:
        summary.extend(["", "## Warnings", ""])
        summary.extend(f"- {row['check']}: {row['details']}" for row in warnings)
    (OUT / "private_alpha_readiness_summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
