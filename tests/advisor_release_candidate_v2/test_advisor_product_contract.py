"""Focused checks for the advisor-facing product contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from mdcorealign.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]


def test_readme_leads_with_throughput_and_keeps_exact_audit() -> None:
    text = (ROOT / "README.md").read_text()
    first_screen = text.split("## Installation", 1)[0]
    assert first_screen.index(
        "mdcorealign align --topology system.gro"
    ) < first_screen.index("--mode exact --backend cpu")
    assert "throughput is the default" in first_screen.lower()
    parser = build_parser()
    assert parser.parse_args(["align", "--input", "toy.npy"]).mode == "throughput"


def test_standard_dependencies_and_compatibility_aliases() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    dependencies = project["dependencies"]
    assert "numpy>=1.24" in dependencies
    assert "MDAnalysis>=2.6" in dependencies
    assert "gemmi>=0.6.7" in dependencies
    assert extras["external"] == []
    assert extras["io"] == []
    assert "gpu" not in extras


def test_public_help_keeps_gpu_explicit_and_ordinary_rmsf_opt_in() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "align",
            "--input",
            "toy.npy",
            "--mode",
            "throughput",
            "--backend",
            "gpu",
        ]
    )
    assert args.backend == "gpu"
    assert args.ordinary_rmsf is False
