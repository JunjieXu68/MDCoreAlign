"""Single full-product release contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from mdcorealign.cli import run

ROOT = Path(__file__).resolve().parents[2]


def _metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_one_wheel_declares_complete_base_product() -> None:
    metadata = _metadata()
    assert metadata["name"] == "mdcorealign"
    assert metadata["dependencies"] == [
        "numpy>=1.24",
        "scipy>=1.10",
        "MDAnalysis>=2.6",
        "gemmi>=0.6.7",
    ]
    extras = metadata["optional-dependencies"]
    assert extras["io"] == []
    assert extras["external"] == []
    assert "gpu" not in extras
    assert extras["cuda12"] == ["cupy-cuda12x>=13,<14"]


def test_doctor_json_has_complete_readiness_schema(capsys) -> None:
    assert run(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "cuda_visible",
        "cupy_installed",
        "detected_gpu",
        "gemmi_version",
        "gpu_backend_ready",
        "gpu_detail",
        "mdanalysis_version",
        "mdcorealign_version",
        "numpy_version",
        "python_version",
        "recommended_backend",
    }
    assert report["recommended_backend"] in {"cpu", "gpu"}
    assert report["recommended_backend"] == (
        "gpu" if report["gpu_backend_ready"] else "cpu"
    )


def test_doctor_is_first_class_cli_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mdcorealign", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "doctor" in completed.stdout
    doctor = subprocess.run(
        [sys.executable, "-m", "mdcorealign", "doctor", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CPU/GPU readiness" in doctor.stdout


def test_normal_user_docs_do_not_require_compatibility_extras() -> None:
    documents = [
        ROOT / "README.md",
        ROOT / "docs" / "installation.md",
        ROOT / "docs" / "quickstart.md",
        ROOT / "docs" / "gpu_installation.md",
    ]
    text = "\n".join(path.read_text() for path in documents)
    assert "mdcorealign[io]" not in text
    assert "mdcorealign[external]" not in text
    assert "mdcorealign[gpu]" not in text
