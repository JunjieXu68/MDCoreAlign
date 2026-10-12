"""Exercise collection boundaries in separate, minimal pytest processes."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def layered_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"
    tests = suite / "tests"
    tests.mkdir(parents=True)
    shutil.copyfile(TESTS / "conftest.py", tests / "conftest.py")
    (suite / "pytest.ini").write_text("[pytest]\nmarkers =\n    archive_qa: frozen QA\n")
    (tests / "test_layer_manifest.json").write_text(json.dumps({
        "research_files": [{"path": "tests/test_research.py"}],
        "research_nodes": [{"nodeid": "tests/test_mixed.py::test_research"}],
        "archive_files": [{"path": "tests/test_archive.py"}],
    }))
    (tests / "archive_qa_nodes.json").write_text(json.dumps({"tests": [
        {"nodeid": "tests/test_archive.py::test_archive"},
        {"nodeid": "tests/test_mixed.py::test_archive"},
    ]}))
    (tests / "test_product.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_product(value):\n    assert value > 0\n"
    )
    (tests / "test_research.py").write_text("def test_research():\n    assert True\n")
    (tests / "test_archive.py").write_text("def test_archive():\n    assert True\n")
    (tests / "test_mixed.py").write_text(
        "def test_product():\n    assert True\n"
        "def test_research():\n    assert True\n"
        "def test_archive():\n    assert True\n"
    )
    return suite


def _pytest(suite: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args], cwd=suite,
        env=env, capture_output=True, text=True, check=False,
    )


def test_product_boundary_prevents_source_and_archive_imports(layered_suite: Path) -> None:
    for name in ("research", "archive"):
        (layered_suite / "tests" / f"test_{name}.py").write_text(
            "raise RuntimeError('must not import excluded module')\n"
        )
    result = _pytest(layered_suite, "tests")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert "must not import excluded module" not in result.stdout


def test_explicit_research_executes_failing_assertions(layered_suite: Path) -> None:
    (layered_suite / "tests/test_research.py").write_text(
        "def test_research():\n    assert 1 == 2\n"
    )
    result = _pytest(layered_suite, "--test-layer=research", "tests")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed" in result.stdout
    assert "assert 1 == 2" in result.stdout


def test_archive_missing_fixture_is_a_failure_not_a_product_skip(layered_suite: Path) -> None:
    (layered_suite / "tests/test_archive.py").write_text(
        "from pathlib import Path\ndef test_archive():\n"
        "    assert Path('missing_frozen_fixture.tsv').read_text()\n"
    )
    result = _pytest(layered_suite, "--test-layer=archive", "tests")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FileNotFoundError" in result.stdout
    assert "1 failed, 1 passed" in result.stdout


def test_layer_collections_are_disjoint_and_cover_parameterized_union(layered_suite: Path) -> None:
    nodes = {}
    for layer in ("product", "research", "archive", "all"):
        result = _pytest(layered_suite, f"--test-layer={layer}", "--collect-only", "tests")
        assert result.returncode == 0, result.stdout + result.stderr
        nodes[layer] = {line for line in result.stdout.splitlines() if "::test_" in line}
    assert [len(nodes[k]) for k in ("product", "research", "archive", "all")] == [3, 2, 2, 7]
    assert not nodes["product"] & nodes["research"]
    assert not nodes["product"] & nodes["archive"]
    assert not nodes["research"] & nodes["archive"]
    assert nodes["all"] == nodes["product"] | nodes["research"] | nodes["archive"]


def test_explicit_source_file_requires_layer_before_import(layered_suite: Path) -> None:
    (layered_suite / "tests/test_research.py").write_text("raise RuntimeError('not imported')\n")
    result = _pytest(layered_suite, "tests/test_research.py")
    assert result.returncode == 4
    assert "use --test-layer=research" in result.stderr
    assert "RuntimeError" not in result.stdout + result.stderr


def test_archive_file_classification_cannot_hide_product_nodes() -> None:
    policy = json.loads((TESTS / "test_layer_manifest.json").read_text())
    archive = {row["nodeid"] for row in json.loads((TESTS / "archive_qa_nodes.json").read_text())["tests"]}
    for row in policy["archive_files"]:
        path = TESTS.parent / row["path"]
        tree = ast.parse(path.read_text())
        nodes = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                nodes.add(f"{row['path']}::{node.name}")
            if isinstance(node, ast.ClassDef):
                nodes.update(
                    f"{row['path']}::{node.name}::{method.name}" for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")
                )
        assert nodes and nodes <= archive, (path, nodes - archive)
