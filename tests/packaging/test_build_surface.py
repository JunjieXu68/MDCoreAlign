"""Regression checks for the modules actually copied by setuptools."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import pytest
import setuptools
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {
    "mdcorealign/benchmark_minimal.py",
    "mdcorealign/experimental_gpu_batch.py",
    "mdcorealign/experimental_stream.py",
    "mdcorealign/experimental_throughput_v2.py",
    "mdcorealign/guarded.py",
}


def test_build_py_copies_exactly_the_stable_package_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy operation must honor exclusions and retain every other module."""

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    hook = runpy.run_path(str(ROOT / "setup.py"))["StableBuildPy"]
    source = tmp_path / "src"
    for package in ("mdcorealign", "mdsuperpose"):
        shutil.copytree(
            ROOT / "src" / package, source / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    original = {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
    assert EXCLUDED <= original
    distribution = Distribution({
        "packages": ["mdcorealign", "mdsuperpose"],
        "package_dir": {"": str(source)},
        "script_name": str(ROOT / "setup.py"),
    })
    command = hook(distribution)
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "lib")
    command.compile = False
    command.run()
    copied = {
        path.relative_to(command.build_lib).as_posix()
        for path in Path(command.build_lib).rglob("*.py")
    }
    assert copied == original - EXCLUDED
    # Building distributions must also leave the research source intact.
    assert original == {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
