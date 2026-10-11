"""Release metadata tests for the pip-first VT02.2 package."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _project() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_base_dependencies_cover_cpu_and_io_runtime() -> None:
    project = _project()
    dependencies = [str(value).lower() for value in project["dependencies"]]
    for requirement in ("numpy", "scipy", "mdanalysis", "gemmi"):
        assert any(value.startswith(requirement) for value in dependencies)
    assert not any("cupy" in value for value in dependencies)
    assert not any(
        tool in value
        for value in dependencies
        for tool in ("pytest", "ruff", "mypy", "twine", "build")
    )


def test_cuda12_extra_is_specific_and_bounded() -> None:
    extras = _project()["optional-dependencies"]
    assert extras["cuda12"] == ["cupy-cuda12x>=13,<14"]
    assert "gpu" not in extras
    all_requirements = [
        requirement
        for values in extras.values()
        for requirement in values
    ]
    assert "cupy-cuda12x>=13,<14" in all_requirements
    assert not any(
        requirement.startswith(("cupy-cuda11", "cupy-cuda13"))
        for requirement in all_requirements
    )


def test_release_identity_and_schema_are_independent() -> None:
    project = _project()
    assert project["version"] == "0.1.0rc6"
    namespace = (ROOT / "src/mdcorealign/__init__.py").read_text()
    schema = (ROOT / "src/mdcorealign/output_schema.py").read_text()
    assert '__version__ = "0.1.0rc6"' in namespace
    assert 'SCHEMA_VERSION = "1.1.0"' in schema


def test_readme_is_pip_first_and_conda_optional() -> None:
    readme = (ROOT / "README.md").read_text()
    pip_heading = readme.index("### Pip and venv (primary)")
    conda_heading = readme.index("### Conda (optional)")
    assert pip_heading < conda_heading
    assert "Conda is optional" in readme
    assert "mdcorealign-0.1.0rc6-py3-none-any.whl[cuda12]" in readme
