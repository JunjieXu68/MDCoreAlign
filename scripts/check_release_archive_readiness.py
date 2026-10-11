"""Check GitHub release, archive, build, and Zenodo-readiness status."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import (  # noqa: E402
    FINAL_SUBMISSION_ROOT,
    env_value,
    file_nonempty,
    git_snapshot,
    has_local_absolute_path,
    run_command,
    timestamp,
    write_json,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "release_archive"
FIELDS = ["check", "status", "evidence", "notes"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--github-repository-url", default=env_value("GITHUB_REPOSITORY_URL"))
    parser.add_argument("--zenodo-doi", default=env_value("ZENODO_DOI"))
    return parser.parse_args()


def add(rows: list[dict[str, object]], check: str, status: str, evidence: str, notes: str) -> None:
    """Append one checklist row."""

    rows.append({"check": check, "status": status, "evidence": evidence, "notes": notes})


def run_build(output_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Run python -m build and twine check."""

    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    for path in ROOT.glob("*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)
    build = run_command([sys.executable, "-m", "build"], timeout=240)
    (output_dir / "build_log.txt").write_text(str(build["stdout"]) + "\n" + str(build["stderr"]))
    twine = run_command([sys.executable, "-m", "twine", "check", "dist/*"], timeout=120)
    if twine["returncode"] != 0:
        # subprocess does not expand globs without a shell; retry with explicit dist paths.
        dist_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "dist").glob("*"))
        if dist_files:
            twine = run_command([sys.executable, "-m", "twine", "check", *dist_files], timeout=120)
    (output_dir / "twine_check_log.txt").write_text(str(twine["stdout"]) + "\n" + str(twine["stderr"]))
    return build, twine


def main() -> int:
    """Run release archive readiness checks."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = pyproject["project"]["version"]
    snapshot = git_snapshot()
    dirty = run_command(["git", "status", "--short", "--", "mdlovofit"], cwd=ROOT.parent, timeout=20)
    dirty_lines = [line for line in str(dirty["stdout"]).splitlines() if line.strip()]
    build, twine = run_build(args.output_dir)

    add(rows, "release-candidate version", "pass" if "rc" in version else "warning", version, "explicit rc version expected")
    add(rows, "git working tree", "pass" if not dirty_lines else "warning", f"{len(dirty_lines)} dirty paths", "dirty paths are listed in blockers.json")
    remote = snapshot.get("origin_remote") or args.github_repository_url
    if remote and remote != "pending":
        add(rows, "GitHub remote", "pass", str(remote), "remote or provided repository URL found")
    else:
        add(rows, "GitHub remote", "warning", "pending", "set real GitHub remote before final public release")
    add(rows, "GitHub Actions workflow", "pass" if (ROOT / ".github" / "workflows" / "ci.yml").exists() else "blocker", ".github/workflows/ci.yml", "CI workflow expected")
    for name in ["LICENSE", "CITATION.cff", "README.md"]:
        add(rows, name, "pass" if file_nonempty(ROOT / name) else "blocker", name, "required release file")
    readme = (ROOT / "README.md").read_text()
    add(rows, "README quickstart", "pass" if "Quick Start" in readme else "blocker", "README.md", "README should include quickstart")
    add(rows, "python -m build", "pass" if build["returncode"] == 0 else "blocker", "build_log.txt", "build must pass")
    add(rows, "twine check", "pass" if twine["returncode"] == 0 else "blocker", "twine_check_log.txt", "twine check must pass")
    add(rows, "release checklist", "pass" if "Release Checklist" in (ROOT / "docs" / "release.md").read_text() else "blocker", "docs/release.md", "release checklist expected")
    doi_text = (ROOT / "CITATION.cff").read_text()
    fake_doi = "10.0000" in doi_text or "fake" in doi_text.lower()
    add(rows, "Zenodo DOI status", "blocker" if fake_doi else ("warning" if args.zenodo_doi == "pending" else "pass"), args.zenodo_doi, "pending is acceptable; fabricated DOI is blocker")
    public_docs = [ROOT / "README.md", ROOT / "docs" / "release.md", ROOT / "docs" / "installation.md", ROOT / "docs" / "quickstart.md", ROOT / "CITATION.cff", ROOT / "pyproject.toml"]
    bad_paths = [str(path.relative_to(ROOT)) for path in public_docs if has_local_absolute_path(path.read_text())]
    add(rows, "no local absolute paths", "blocker" if bad_paths else "pass", ",".join(bad_paths), "public release metadata should not expose local paths")

    table = args.output_dir / "release_archive_checklist.tsv"
    summary = args.output_dir / "release_archive_summary.md"
    blockers = [row for row in rows if row["status"] == "blocker"]
    warnings = [row for row in rows if row["status"] == "warning"]
    write_table(table, rows, FIELDS)
    write_json(
        args.output_dir / "blockers.json",
        {
            "generated_at": timestamp(),
            "blockers": [row["check"] for row in blockers],
            "warnings": [row["check"] for row in warnings],
            "dirty_paths": dirty_lines,
            "build_returncode": build["returncode"],
            "twine_returncode": twine["returncode"],
        },
    )
    summary.write_text(
        "\n".join(
            [
                "# Release Archive Readiness",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Package version: `{version}`",
                f"- Build status: `{'pass' if build['returncode'] == 0 else 'blocker'}`",
                f"- Twine check status: `{'pass' if twine['returncode'] == 0 else 'blocker'}`",
                f"- Blockers: `{[row['check'] for row in blockers]}`",
                f"- Warnings: `{[row['check'] for row in warnings]}`",
                "",
                "No package was published by this script.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/check_release_archive_readiness.py",
        inputs=[ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "CITATION.cff", ROOT / "docs" / "release.md"],
        outputs=[
            table,
            summary,
            args.output_dir / "blockers.json",
            args.output_dir / "build_log.txt",
            args.output_dir / "twine_check_log.txt",
        ],
        commands=[build, twine, dirty],
        extra={"version": version, "github_repository_url": args.github_repository_url, "zenodo_doi": args.zenodo_doi},
    )
    print(f"Wrote release archive readiness checks to {args.output_dir}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
