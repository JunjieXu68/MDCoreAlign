"""Select installed-product, source-research, or frozen-artifact tests explicitly.

Research-only modules are excluded before import. Mixed files retain their product
nodes; imports needed only by research nodes belong inside those test functions.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--test-layer", choices=("product", "research", "archive", "all"), default="product",
        help="product (default): installed runtime; research: source helpers; archive: frozen QA",
    )


def _policy() -> tuple[set[str], set[str], set[str], set[str]]:
    manifest = json.loads((TESTS / "test_layer_manifest.json").read_text())
    archive = json.loads((TESTS / "archive_qa_nodes.json").read_text())
    return (
        {row["path"] for row in manifest["research_files"]},
        {row["nodeid"] for row in manifest["research_nodes"]},
        {row["nodeid"] for row in archive["tests"]},
        {row["path"] for row in manifest["archive_files"]},
    )


def _node_layer(nodeid: str) -> str:
    research_files, research_nodes, archive_nodes, _ = _policy()
    base = nodeid.split("[", 1)[0]
    if base in archive_nodes:
        return "archive"
    if base in research_nodes or base.split("::", 1)[0] in research_files:
        return "research"
    return "product"


def pytest_configure(config: pytest.Config) -> None:
    layer = config.getoption("test_layer")
    os.environ["MDCOREALIGN_TEST_LAYER"] = layer
    # Tests also exercise shipped script helpers. The repository root exposes
    # that namespace, but never the src-layout runtime packages.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for name in ("product", "research", "archive"):
        config.addinivalue_line("markers", f"test_layer_{name}: explicitly classified {name} test")
    research_files, _, _, archive_files = _policy()
    if layer == "product":
        for argument in config.args:
            path = Path(argument.split("::", 1)[0]).resolve()
            try:
                relative = path.relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if relative in research_files:
                raise pytest.UsageError(
                    f"{relative} is source research; use --test-layer=research with source dependencies"
                )
            if relative in archive_files:
                raise pytest.UsageError(f"{relative} requires --test-layer=archive and frozen fixtures")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.suffix != ".py":
        return None
    try:
        relative = collection_path.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    research_files, research_nodes, archive_nodes, archive_files = _policy()
    layer = config.getoption("test_layer")
    if layer == "product" and relative in research_files | archive_files:
        return True
    if layer == "research":
        return relative not in research_files | {n.split("::", 1)[0] for n in research_nodes}
    if layer == "archive":
        return relative not in {n.split("::", 1)[0] for n in archive_nodes}
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected, deselected = [], []
    layer = config.getoption("test_layer")
    for item in items:
        relative = item.path.relative_to(ROOT).as_posix()
        nodeid = relative + "::" + item.nodeid.split("::", 1)[1]
        actual = _node_layer(nodeid)
        item.add_marker(getattr(pytest.mark, f"test_layer_{actual}"))
        item.user_properties.append(("test_layer", actual))
        if actual == "archive":
            item.add_marker(pytest.mark.archive_qa)
        (selected if layer == "all" or layer == actual else deselected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config: pytest.Config) -> str:
    return f"MDCoreAlign test layer: {config.getoption('test_layer')} (explicit collection policy)"
