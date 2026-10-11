"""Audit input-format support upgrade outputs and claims."""

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
OUT = ROOT / "results" / "input_format_support_upgrade" / "final_audit"
FIELDS = ["check", "status", "severity", "details", "recommended_action"]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(command: list[str], timeout: int = 120) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode == 0, (completed.stdout + "\n" + completed.stderr).strip()[-1200:]


def _relativize(text: str) -> str:
    """Avoid local absolute paths in audit detail text."""

    return text.replace(str(ROOT) + "/", "")


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


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _docs_text() -> str:
    paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
    return "\n".join(path.read_text(errors="ignore") for path in paths if path.exists())


def _test_status() -> tuple[str, str]:
    report = ROOT / "results" / "input_format_support_upgrade" / "test_report.txt"
    if not report.exists():
        return "warning", "test report not generated yet"
    text = report.read_text(errors="ignore")
    if re.search(r"\bfailed\b|ERROR|Traceback", text, re.IGNORECASE):
        return "fail", "test report contains failure/error markers"
    match = re.search(r"(\d+) passed", text)
    if match:
        return "pass", f"{match.group(1)} tests passed"
    return "warning", "could not parse pytest pass count"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required_docs = [
        "docs/input_formats.md",
        "docs/support_matrix.md",
        "docs/mdlovofit_comparison.md",
        "docs/conversion_guidance.md",
        "docs/non_protein_systems.md",
        "docs/output_formats.md",
    ]
    for relpath in required_docs:
        rows.append(
            _row(
                f"required doc {relpath}",
                "pass" if _nonempty(ROOT / relpath) else "fail",
                "blocker",
                "present" if _nonempty(ROOT / relpath) else "missing or empty",
                "create/update required documentation",
            )
        )
    readme = (ROOT / "README.md").read_text(errors="ignore")
    rows.append(
        _row(
            "README support matrix summary",
            "pass" if "MDCoreAlign is primarily validated" in readme and "docs/support_matrix.md" in readme else "fail",
            "blocker",
            "README contains support boundary wording" if "MDCoreAlign is primarily validated" in readme else "missing support boundary wording",
        )
    )
    ok, output = _run([sys.executable, "-m", "mdcorealign", "formats", "--out", str(OUT / "supported_formats.tsv")])
    rows.append(_row("mdcorealign formats works", "pass" if ok else "fail", "blocker", _relativize(output), "fix formats CLI"))
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "audit"
        ok, output = _run(
            [
                sys.executable,
                "-m",
                "mdcorealign",
                "audit-preprocessing",
                "--input",
                str(ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"),
                "--input-format",
                "ca-pdb",
                "--output-dir",
                str(out),
            ]
        )
        readiness_present = False
        if ok:
            audit = (out / "input_audit.tsv").read_text(errors="ignore")
            readiness_present = "input_readiness_status" in audit
        rows.append(
            _row(
                "input readiness status appears in audit output",
                "pass" if ok and readiness_present else "fail",
                "blocker",
                "readiness columns present" if readiness_present else output,
                "write readiness fields to audit outputs",
            )
        )
    paper_dir = ROOT / "results" / "input_format_support_upgrade" / "paper_text"
    snippets = list(paper_dir.glob("*.md"))
    rows.append(
        _row(
            "paper text snippets exist",
            "pass" if len(snippets) >= 6 and all(path.stat().st_size > 50 for path in snippets) else "fail",
            "blocker",
            f"{len(snippets)} snippet files",
            "create concise paper text snippets",
        )
    )
    docs_text = _docs_text().lower()
    prohibited = [
        ("unsupported all formats claim", "supports all md formats"),
        ("unsupported all systems claim", "supports all molecular systems"),
        ("unsupported RNA/DNA supported claim", "rna/dna supported"),
        ("unsupported MDLovoFit direct binary trajectory claim", "mdlovofit directly supports xtc"),
        ("unsupported all atom selections formal claim", "formally validated for all atom selections"),
    ]
    for check, phrase in prohibited:
        rows.append(
            _row(
                check,
                "pass" if phrase not in docs_text else "fail",
                "blocker",
                "not found" if phrase not in docs_text else f"found phrase: {phrase}",
                "remove unsupported claim",
            )
        )
    test_status, test_details = _test_status()
    rows.append(
        _row(
            "full test suite",
            test_status,
            "blocker" if test_status == "fail" else "warning",
            test_details,
            "run PYTHONPATH=src python -m pytest -q tests",
        )
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    _write_tsv(OUT / "input_format_support_checklist.tsv", rows)
    blockers = [row for row in rows if row["status"] == "fail" and row["severity"] == "blocker"]
    warnings = [row for row in rows if row["status"] != "pass" and row["severity"] != "blocker"]
    if blockers:
        recommendation = "D. Not ready; unsupported claims or usability blockers remain"
    elif warnings:
        recommendation = "C. Almost ready; documentation warnings remain"
    else:
        recommendation = "B. Ready for public documentation after author review"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blockers": [row["check"] for row in blockers],
        "warnings": [row["check"] for row in warnings],
        "recommendation": recommendation,
    }
    (OUT / "blockers.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = [
        "# Input Format Support Upgrade Audit",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Recommendation: `{recommendation}`",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "This audit checks documentation, CLI behavior, readiness reporting, and conservative support-boundary wording.",
    ]
    if blockers:
        summary.extend(["", "## Blockers", ""])
        summary.extend(f"- {row['check']}: {row['details']}" for row in blockers)
    if warnings:
        summary.extend(["", "## Warnings", ""])
        summary.extend(f"- {row['check']}: {row['details']}" for row in warnings)
    (OUT / "input_format_support_summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
