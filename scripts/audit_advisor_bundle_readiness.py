"""Final readiness audit for the private-alpha advisor bundle."""

from __future__ import annotations

import csv
import json
import re
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist_advisor"
OUT = ROOT / "results" / "advisor_bundle"
ZIP_PATH = DIST / "MDCoreAlign_private_alpha_v0.1.0_advisor.zip"
TAR_PATH = DIST / "MDCoreAlign_private_alpha_v0.1.0_advisor.tar.gz"
FIELDS = ["check", "status", "severity", "details", "recommended_action"]
MAX_FILE_MB = 25.0


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


def _parse_test_report() -> tuple[str, str]:
    path = OUT / "test_report.txt"
    if not path.exists():
        return "warning", "test report not found"
    text = path.read_text(errors="ignore")
    if re.search(r"\bfailed\b|ERROR|Traceback", text, re.IGNORECASE):
        return "fail", "test report contains failure markers"
    match = re.search(r"(\d+) passed", text)
    if match:
        return "pass", f"{match.group(1)} tests passed"
    return "warning", "could not parse test report"


def _bundle_files() -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            files.extend((info.filename, info.file_size) for info in archive.infolist() if not info.is_dir())
    return files


def _scan_bundle_text() -> tuple[list[str], list[str], list[str], list[str]]:
    local_paths: list[str] = []
    public_claims: list[str] = []
    unsupported_claims: list[str] = []
    fake_urls: list[str] = []
    if not ZIP_PATH.exists():
        return local_paths, public_claims, unsupported_claims, fake_urls
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".md", ".txt", ".tsv", ".json", ".sh", ".py")):
                continue
            text = archive.read(info).decode("utf-8", errors="ignore")
            if any(marker in text for marker in ["/storage/", "/home/", "MDAnalysis_data"]):
                local_paths.append(info.filename)
            lower = text.lower()
            if "this is a public release" in lower or "ready for public release" in lower:
                public_claims.append(info.filename)
            if "supports all md formats" in lower or "supports all molecular systems" in lower or "rna/dna supported" in lower:
                unsupported_claims.append(info.filename)
            if re.search(r"https?://(github|zenodo|testpypi)[^\s`)]*", lower):
                fake_urls.append(info.filename)
    return local_paths, public_claims, unsupported_claims, fake_urls


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required_docs = [
        "README_FOR_ADVISOR.md",
        "INSTALLATION.md",
        "QUICKSTART.md",
        "ADVISOR_TESTING_GUIDE.md",
        "FEEDBACK_FORM.tsv",
        "FEEDBACK_FORM.md",
        "OUTPUT_INTERPRETATION.md",
        "SUPPORT_MATRIX.md",
        "ADVISOR_MANUSCRIPT_BRIEF.md",
    ]
    for rel in required_docs:
        rows.append(
            _row(
                f"advisor file {rel}",
                "pass" if _nonempty(DIST / rel) else "fail",
                "blocker",
                "present" if _nonempty(DIST / rel) else "missing or empty",
                "create advisor-facing file",
            )
        )
    for path, label in [(ZIP_PATH, "bundle zip"), (TAR_PATH, "bundle tar.gz")]:
        rows.append(_row(label, "pass" if _nonempty(path) else "fail", "blocker", str(path.relative_to(ROOT)), "run package script"))
    for rel in ["bundle_manifest.tsv", "bundle_manifest.json", "checksums.tsv", "bundle_summary.md", "version_manifest.json", "version_manifest.md"]:
        rows.append(
            _row(
                f"result file {rel}",
                "pass" if _nonempty(OUT / rel) else "fail",
                "blocker",
                "present" if _nonempty(OUT / rel) else "missing or empty",
                "run package script",
            )
        )
    validation = OUT / "advisor_bundle_validation_report.json"
    validation_status = "missing"
    if validation.exists():
        validation_status = json.loads(validation.read_text()).get("status", "unknown")
    rows.append(
        _row(
            "bundle validation passed",
            "pass" if validation_status == "pass" else "fail",
            "blocker",
            validation_status,
            "run scripts/validate_advisor_bundle.py",
        )
    )
    files = _bundle_files()
    large = [f"{name} ({size / (1024 * 1024):.1f} MB)" for name, size in files if size / (1024 * 1024) > MAX_FILE_MB]
    rows.append(
        _row(
            "no large files included",
            "pass" if not large else "fail",
            "blocker",
            "; ".join(large) if large else f"no files above {MAX_FILE_MB:.1f} MB",
            "remove large files from bundle",
        )
    )
    local_paths, public_claims, unsupported_claims, fake_urls = _scan_bundle_text()
    rows.append(_row("no local absolute paths", "pass" if not local_paths else "fail", "blocker", "; ".join(local_paths) if local_paths else "none"))
    rows.append(_row("no public-release claims", "pass" if not public_claims else "fail", "blocker", "; ".join(public_claims) if public_claims else "none"))
    rows.append(_row("no unsupported all-format/RNA-DNA claims", "pass" if not unsupported_claims else "fail", "blocker", "; ".join(unsupported_claims) if unsupported_claims else "none"))
    rows.append(_row("no fake GitHub/Zenodo/TestPyPI URLs", "pass" if not fake_urls else "fail", "blocker", "; ".join(fake_urls) if fake_urls else "none"))
    test_status, test_details = _parse_test_report()
    rows.append(_row("tests pass", test_status, "blocker" if test_status == "fail" else "warning", test_details, "run full pytest suite"))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    _write_tsv(OUT / "final_advisor_readiness_checklist.tsv", rows)
    blockers = [row for row in rows if row["status"] == "fail" and row["severity"] == "blocker"]
    warnings = [row for row in rows if row["status"] != "pass" and row["severity"] != "blocker"]
    if blockers:
        recommendation = "C. Not ready; usability or packaging blockers remain"
    elif warnings:
        recommendation = "B. Almost ready; minor documentation fixes needed"
    else:
        recommendation = "A. Ready to send to advisor"
    payload = {
        "recommendation": recommendation,
        "blockers": [row["check"] for row in blockers],
        "warnings": [row["check"] for row in warnings],
    }
    (OUT / "blockers.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Final Advisor Bundle Readiness Summary",
        "",
        f"- Recommendation: `{recommendation}`",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
    ]
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {row['check']}: {row['details']}" for row in blockers)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {row['check']}: {row['details']}" for row in warnings)
    (OUT / "final_advisor_readiness_summary.md").write_text("\n".join(lines) + "\n")
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
