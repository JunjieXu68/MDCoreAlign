"""Validate the private-alpha advisor bundle in a clean temporary environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "advisor_bundle"
ARCHIVE = ROOT / "dist_advisor" / "MDCoreAlign_private_alpha_v0.1.0_advisor.zip"
PACKAGE_ROOT_NAME = "MDCoreAlign_private_alpha_v0.1.0_advisor"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int = 240) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _venv_bin(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / name


def _venv_bin_dir(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def _unpack_bundle(archive: Path, dest: Path) -> Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)
    elif archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive) as handle:
            handle.extractall(dest)
    else:
        raise ValueError(f"unsupported bundle archive: {archive}")
    root = dest / PACKAGE_ROOT_NAME
    if not root.exists():
        raise FileNotFoundError(f"bundle root was not found after unpacking: {root}")
    return root


def _check_outputs(output_dir: Path, names: list[str]) -> dict[str, str]:
    return {
        name: "present" if (output_dir / name).exists() and (output_dir / name).stat().st_size > 0 else "missing"
        for name in names
    }


def _sanitize(value: object, replacements: dict[str, str]) -> object:
    """Remove local absolute paths from persisted validation reports."""

    if isinstance(value, dict):
        return {key: _sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        sanitized = value
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        return sanitized
    return value


def validate() -> dict[str, object]:
    """Run clean bundle validation and return a report payload."""

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    steps: list[dict[str, object]] = []
    if not ARCHIVE.exists():
        return {
            "status": "fail",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": f"bundle archive missing: {ARCHIVE}",
            "steps": steps,
        }

    with tempfile.TemporaryDirectory(prefix="mdcorealign-advisor-validate-") as tmpdir:
        tmp = Path(tmpdir)
        unpacked = _unpack_bundle(ARCHIVE, tmp / "unpacked")
        venv = tmp / "venv"
        steps.append(_run([sys.executable, "-m", "venv", str(venv)], cwd=unpacked, timeout=120))
        python = _venv_bin(venv, "python")
        mdcorealign = _venv_bin(venv, "mdcorealign")
        env = os.environ.copy()
        env["PATH"] = f"{_venv_bin_dir(venv)}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        if not all(step["success"] for step in steps):
            status = "fail"
        else:
            commands = [
                [str(python), "-m", "pip", "install", "--upgrade", "pip"],
                [str(python), "-m", "pip", "install", "-e", ".[dev]"],
                [str(python), "-c", "import mdcorealign; print(mdcorealign.__version__)"],
                [str(mdcorealign), "--help"],
                [str(mdcorealign), "formats"],
                [str(mdcorealign), "align", "--help"],
                [str(mdcorealign), "flexibility", "--help"],
                [str(mdcorealign), "audit-preprocessing", "--help"],
            ]
            for command in commands:
                steps.append(_run(command, cwd=unpacked, env=env, timeout=300))
            toy_out = tmp / "toy_output"
            real_out = tmp / "real_output"
            steps.append(_run(["bash", "examples/toy_md/run_quickstart.sh", str(toy_out)], cwd=unpacked, env=env, timeout=180))
            steps.append(_run(["bash", "examples/real_md_small/run_recommended_workflow.sh", str(real_out)], cwd=unpacked, env=env, timeout=180))
            steps.append(_run([str(python), "-m", "pytest", "-q", "tests/test_mdcorealign_release_readiness.py", "tests/test_formats_cli.py"], cwd=unpacked, env=env, timeout=240))
            toy_outputs = _check_outputs(
                toy_out,
                [
                    "aligned_trajectory.pdb",
                    "weights.csv",
                    "flexibility_profile.csv",
                    "flexible_segments.tsv",
                    "summary.json",
                    "run_metadata.json",
                ],
            )
            real_outputs = _check_outputs(
                real_out,
                [
                    "aligned_trajectory.pdb",
                    "weights.csv",
                    "flexibility_profile.csv",
                    "flexible_segments.tsv",
                    "input_audit.tsv",
                    "input_audit.json",
                    "chain_mapping_report.tsv",
                    "summary.json",
                    "run_metadata.json",
                ],
            )
            output_success = all(value == "present" for value in [*toy_outputs.values(), *real_outputs.values()])
            status = "pass" if all(step["success"] for step in steps) and output_success else "fail"

    report = {
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "steps": steps,
        "toy_outputs": toy_outputs if "toy_outputs" in locals() else {},
        "real_md_small_outputs": real_outputs if "real_outputs" in locals() else {},
    }
    replacements = {
        str(ROOT): "<repo>",
        str(Path.home()): "<home>",
    }
    if "tmpdir" in locals():
        replacements[str(tmpdir)] = "<tmp>"
    return _sanitize(report, replacements)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = validate()
    (OUT / "advisor_bundle_validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    failed = [step for step in report.get("steps", []) if not step.get("success")]
    lines = [
        "# Advisor Bundle Validation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Archive: `{report.get('archive', str(ARCHIVE.relative_to(ROOT)))}`",
        f"- Started: `{report['started_at']}`",
        f"- Finished: `{report['finished_at']}`",
        f"- Failed command count: `{len(failed)}`",
        "",
        "## Output Checks",
        "",
        f"- Toy outputs: `{report.get('toy_outputs', {})}`",
        f"- Real-MD small outputs: `{report.get('real_md_small_outputs', {})}`",
    ]
    if failed:
        lines.extend(["", "## Failed Commands", ""])
        lines.extend(f"- `{step['command']}` returned `{step['returncode']}`" for step in failed)
    (OUT / "advisor_bundle_validation_report.md").write_text("\n".join(lines) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
