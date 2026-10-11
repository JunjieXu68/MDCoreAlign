"""Keep portable software tests separate from frozen research-artifact QA.

No test is skipped here. Explicit pytest marker selection controls the tier;
running without ``-m`` still executes both, including their original assertions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    manifest = json.loads(Path(__file__).with_name("archive_qa_nodes.json").read_text())
    nodes = {row["nodeid"] for row in manifest["tests"]}
    for item in items:
        if item.nodeid in nodes:
            item.add_marker(pytest.mark.archive_qa)
