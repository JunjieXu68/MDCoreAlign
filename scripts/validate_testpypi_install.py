"""Validate TestPyPI installation when a TestPyPI package is available."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import venv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import (  # noqa: E402
    FINAL_SUBMISSION_ROOT,
    env_value,
    run_command,
    timestamp,
    write_json,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "testpypi_validation"
TOY_FIELDS = ["output_file", "status", "size_bytes", "notes"]
EXPECTED = ["aligned_trajectory.pdb", "weights.csv", "flexibility_profile.csv", "flexible_segments.tsv", "summary.json"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--package-name", default=env_value("TESTPYPI_PACKAGE_NAME"))
    parser.add_argument("--package-url", default=env_value("TESTPYPI_PACKAGE_URL"))
    parser.add_argument("--keep-venv", action="store_true")
    return parser.parse_args()


def _python(venv_dir: Path) -> Path:
    """Return venv Python path."""

    return venv_dir / "bin" / "python"


def _command_env(venv_dir: Path) -> dict[str, str]:
    """Return environment using venv bin first."""

    env = os.environ.copy()
    env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def check_outputs(output_dir: Path) -> list[dict[str, object]]:
    """Check expected toy quickstart outputs."""

    rows = []
    for name in EXPECTED:
        path = output_dir / name
        rows.append(
            {
                "output_file": name,
                "status": "present" if path.exists() and path.stat().st_size > 0 else "missing",
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "notes": "toy quickstart expected output",
            }
        )
    return rows


def main() -> int:
    """Run TestPyPI validation or write pending status."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "testpypi_validation_report.json"
    report_md = args.output_dir / "testpypi_validation_report.md"
    toy_table = args.output_dir / "toy_outputs_check.tsv"
    commands: list[dict[str, object]] = []
    package_available = args.package_name != "pending" or args.package_url != "pending"
    if not package_available:
        write_table(toy_table, [], TOY_FIELDS)
        payload = {
            "status": "not_applicable_pending_upload",
            "package_name": args.package_name,
            "package_url": args.package_url,
            "external_tools_required_for_core_commands": False,
        }
        write_json(report_json, payload)
        report_md.write_text(
            "\n".join(
                [
                    "# TestPyPI Validation",
                    "",
                    f"- Generated at: `{timestamp()}`",
                    "- Status: `not_applicable_pending_upload`",
                    "- No TestPyPI install was attempted because no TestPyPI package name or URL was provided.",
                    "- Core commands do not require MDLovoFit or THESEUS.",
                    "",
                ]
            )
        )
        write_manifest(args.output_dir, script="scripts/validate_testpypi_install.py", outputs=[report_json, report_md, toy_table], extra=payload)
        print(f"Wrote pending TestPyPI validation to {args.output_dir}")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="mdcorealign-testpypi-"))
    venv_dir = tmp / "venv"
    toy_output = tmp / "toy_output"
    status = "pass"
    try:
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        install_target = args.package_url if args.package_url != "pending" else args.package_name
        install = run_command(
            [
                str(_python(venv_dir)),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
                install_target,
            ],
            timeout=240,
        )
        commands.append(install)
        checks = [
            [str(_python(venv_dir)), "-c", "import mdcorealign; print(mdcorealign.__version__)"],
            [str(venv_dir / "bin" / "mdcorealign"), "--help"],
            [str(venv_dir / "bin" / "mdcorealign"), "align", "--help"],
            [str(venv_dir / "bin" / "mdcorealign"), "flexibility", "--help"],
            [str(venv_dir / "bin" / "mdcorealign"), "diagnostics", "--help"],
            [str(venv_dir / "bin" / "mdcorealign"), "audit-preprocessing", "--help"],
            ["bash", str(ROOT / "examples" / "toy_md" / "run_quickstart.sh"), str(toy_output)],
        ]
        for command in checks:
            result = run_command(command, timeout=180)
            commands.append(result)
        if any(command["returncode"] != 0 for command in commands):
            status = "blocker"
    finally:
        rows = check_outputs(toy_output)
        if any(row["status"] != "present" for row in rows):
            status = "blocker"
        write_table(toy_table, rows, TOY_FIELDS)
        if not args.keep_venv:
            shutil.rmtree(tmp, ignore_errors=True)

    payload = {
        "status": status,
        "package_name": args.package_name,
        "package_url": args.package_url,
        "commands": commands,
        "external_tools_required_for_core_commands": False,
    }
    write_json(report_json, payload)
    report_md.write_text(
        "\n".join(
            [
                "# TestPyPI Validation",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Status: `{status}`",
                f"- Package name: `{args.package_name}`",
                f"- Package URL: `{args.package_url}`",
                "- Core commands do not require MDLovoFit or THESEUS.",
                "",
            ]
        )
    )
    write_manifest(args.output_dir, script="scripts/validate_testpypi_install.py", outputs=[report_json, report_md, toy_table], commands=commands, extra=payload)
    print(f"Wrote TestPyPI validation to {args.output_dir}")
    return 0 if status != "blocker" else 1


if __name__ == "__main__":
    raise SystemExit(main())
