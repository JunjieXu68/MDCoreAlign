"""Update release metadata when real GitHub/TestPyPI/Zenodo URLs are available."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import (  # noqa: E402
    FINAL_SUBMISSION_ROOT,
    env_value,
    has_local_absolute_path,
    managed_section,
    safe_status,
    timestamp,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "release_metadata"
FIELDS = ["file", "field", "status", "value", "notes"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-repository-url", default=env_value("GITHUB_REPOSITORY_URL"))
    parser.add_argument("--github-issues-url", default=env_value("GITHUB_ISSUES_URL"))
    parser.add_argument("--github-documentation-url", default=env_value("GITHUB_DOCUMENTATION_URL"))
    parser.add_argument("--release-tag", default=env_value("RELEASE_TAG", "v0.1.0rc1"))
    parser.add_argument("--zenodo-doi", default=env_value("ZENODO_DOI"))
    parser.add_argument("--testpypi-url", default=env_value("TESTPYPI_URL"))
    parser.add_argument("--date-released", default=env_value("DATE_RELEASED", str(date.today())))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def version_from_tag(tag: str) -> str:
    """Return version string from a release tag."""

    return tag[1:] if tag.startswith("v") else tag


def update_pyproject(repo: str, issues: str, docs: str) -> bool:
    """Update project URLs when a real repository URL is provided."""

    path = SCRIPT_DIR.parent / "pyproject.toml"
    text = path.read_text()
    if repo == "pending":
        return False
    data = tomllib.loads(text)
    project = data.get("project", {})
    homepage = repo
    docs_url = docs if docs != "pending" else repo
    issues_url = issues if issues != "pending" else f"{repo.rstrip('/')}/issues"
    urls_block = "\n".join(
        [
            "[project.urls]",
            f'Homepage = "{homepage}"',
            f'Repository = "{repo}"',
            f'Documentation = "{docs_url}"',
            f'Issues = "{issues_url}"',
            "",
        ]
    )
    lines = text.splitlines()
    filtered: list[str] = []
    in_urls = False
    for line in lines:
        if line.strip() == "[project.urls]":
            in_urls = True
            continue
        if in_urls and line.startswith("[") and line.strip() != "[project.urls]":
            in_urls = False
        if not in_urls:
            filtered.append(line)
    insertion = "\n".join(filtered).rstrip() + "\n\n" + urls_block
    path.write_text(insertion)
    return bool(project.get("name"))


def update_citation(repo: str, tag: str, doi: str, released: str) -> None:
    """Update CITATION.cff fields conservatively."""

    path = SCRIPT_DIR.parent / "CITATION.cff"
    lines = path.read_text().splitlines()
    keys = {
        "version": f"version: {version_from_tag(tag)}",
        "date-released": f"date-released: {released}",
    }
    if repo != "pending":
        keys["repository-code"] = f'repository-code: "{repo}"'
    if doi != "pending":
        keys["doi"] = f'doi: "{doi}"'
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.split(":", 1)[0].strip()
        if key in keys:
            output.append(keys[key])
            seen.add(key)
        elif key in {"repository-code", "doi"} and key not in keys:
            continue
        else:
            output.append(line)
    for key, line in keys.items():
        if key not in seen:
            output.append(line)
    path.write_text("\n".join(output).rstrip() + "\n")


def update_markdown(repo: str, issues: str, docs: str, tag: str, doi: str, testpypi: str) -> None:
    """Update public Markdown release metadata sections."""

    repo_text = safe_status(repo)
    issues_text = safe_status(issues if issues != "pending" else (f"{repo.rstrip('/')}/issues" if repo != "pending" else "pending"))
    docs_text = safe_status(docs if docs != "pending" else repo)
    doi_text = safe_status(doi)
    testpypi_text = safe_status(testpypi)
    body = "\n".join(
        [
            "## Release Metadata",
            "",
            f"- Repository: `{repo_text}`",
            f"- Issues: `{issues_text}`",
            f"- Documentation: `{docs_text}`",
            f"- Release tag: `{tag}`",
            f"- Zenodo DOI: `{doi_text}`",
            f"- TestPyPI URL: `{testpypi_text}`",
        ]
    )
    for rel in ["README.md", "docs/installation.md", "docs/quickstart.md"]:
        path = SCRIPT_DIR.parent / rel
        path.write_text(managed_section(path.read_text(), "MDCOREALIGN_RELEASE_METADATA", body))

    release_body = "\n".join(
        [
            "## Release Metadata Status",
            "",
            f"- GitHub repository URL: `{repo_text}`",
            f"- GitHub issues URL: `{issues_text}`",
            f"- Documentation URL: `{docs_text}`",
            f"- Release tag: `{tag}`",
            f"- Zenodo DOI: `{doi_text}`",
            f"- TestPyPI URL: `{testpypi_text}`",
            "",
            "Release managers should not fabricate a DOI or TestPyPI URL. Use `pending` until the corresponding external service has created the real record.",
        ]
    )
    release = SCRIPT_DIR.parent / "docs" / "release.md"
    release.write_text(managed_section(release.read_text(), "MDCOREALIGN_RELEASE_METADATA", release_body))

    changelog = SCRIPT_DIR.parent / "CHANGELOG.md"
    changelog_body = "\n".join(
        [
            "## Release-Candidate Metadata",
            "",
            f"- Current release candidate: `{tag}`",
            f"- Repository: `{repo_text}`",
            f"- Zenodo DOI: `{doi_text}`",
            f"- TestPyPI URL: `{testpypi_text}`",
            "- This release candidate includes package installation, CLI, toy quickstart, clean-install validation, build/twine checks, and manuscript-readiness audit artifacts.",
        ]
    )
    changelog.write_text(managed_section(changelog.read_text(), "MDCOREALIGN_RELEASE_METADATA", changelog_body))


def build_rows(args: argparse.Namespace, pyproject_updated: bool) -> list[dict[str, object]]:
    """Build metadata audit rows."""

    rows = [
        {
            "file": "pyproject.toml",
            "field": "project.urls",
            "status": "pass" if pyproject_updated else "warning",
            "value": args.github_repository_url,
            "notes": "updated only when a real GitHub repository URL is provided",
        },
        {
            "file": "CITATION.cff",
            "field": "repository-code",
            "status": "pass" if args.github_repository_url != "pending" else "warning",
            "value": args.github_repository_url,
            "notes": "pending is acceptable before GitHub repository creation",
        },
        {
            "file": "CITATION.cff",
            "field": "doi",
            "status": "pass" if args.zenodo_doi != "pending" else "warning",
            "value": args.zenodo_doi,
            "notes": "pending is acceptable; fake DOI would be a blocker",
        },
        {
            "file": "README/docs/CHANGELOG",
            "field": "release metadata section",
            "status": "pass",
            "value": args.release_tag,
            "notes": "managed release metadata sections inserted or refreshed",
        },
    ]
    public_paths = [
        SCRIPT_DIR.parent / "README.md",
        SCRIPT_DIR.parent / "docs" / "release.md",
        SCRIPT_DIR.parent / "docs" / "installation.md",
        SCRIPT_DIR.parent / "docs" / "quickstart.md",
        SCRIPT_DIR.parent / "CHANGELOG.md",
        SCRIPT_DIR.parent / "CITATION.cff",
        SCRIPT_DIR.parent / "pyproject.toml",
    ]
    bad = [str(path.relative_to(SCRIPT_DIR.parent)) for path in public_paths if has_local_absolute_path(path.read_text())]
    rows.append(
        {
            "file": "public metadata/docs",
            "field": "local absolute path scan",
            "status": "blocker" if bad else "pass",
            "value": ",".join(bad),
            "notes": "metadata/docs should not include local machine paths",
        }
    )
    return rows


def main() -> int:
    """Update release metadata and write audit outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pyproject_updated = update_pyproject(args.github_repository_url, args.github_issues_url, args.github_documentation_url)
    update_citation(args.github_repository_url, args.release_tag, args.zenodo_doi, args.date_released)
    update_markdown(
        args.github_repository_url,
        args.github_issues_url,
        args.github_documentation_url,
        args.release_tag,
        args.zenodo_doi,
        args.testpypi_url,
    )
    rows = build_rows(args, pyproject_updated)
    table = args.output_dir / "release_metadata_audit.tsv"
    summary = args.output_dir / "release_metadata_summary.md"
    write_table(table, rows, FIELDS)
    summary.write_text(
        "\n".join(
            [
                "# Release Metadata Audit",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- GitHub repository URL: `{args.github_repository_url}`",
                f"- GitHub issues URL: `{args.github_issues_url}`",
                f"- Documentation URL: `{args.github_documentation_url}`",
                f"- Release tag: `{args.release_tag}`",
                f"- Zenodo DOI: `{args.zenodo_doi}`",
                f"- TestPyPI URL: `{args.testpypi_url}`",
                "",
                "If any value is `pending`, the release metadata is intentionally conservative and should be updated after the real external record exists.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/update_release_metadata.py",
        inputs=[
            SCRIPT_DIR.parent / "pyproject.toml",
            SCRIPT_DIR.parent / "CITATION.cff",
            SCRIPT_DIR.parent / "README.md",
            SCRIPT_DIR.parent / "docs" / "release.md",
            SCRIPT_DIR.parent / "docs" / "installation.md",
            SCRIPT_DIR.parent / "docs" / "quickstart.md",
            SCRIPT_DIR.parent / "CHANGELOG.md",
        ],
        outputs=[table, summary],
        extra={
            "github_repository_url": args.github_repository_url,
            "github_issues_url": args.github_issues_url,
            "github_documentation_url": args.github_documentation_url,
            "release_tag": args.release_tag,
            "zenodo_doi": args.zenodo_doi,
            "testpypi_url": args.testpypi_url,
        },
    )
    print(f"Wrote release metadata audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
